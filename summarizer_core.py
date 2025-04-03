from dotenv import load_dotenv
load_dotenv()

import asyncio
import hashlib
import io
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from llmrouter import LLMRouter
from lemonfox_whisper import transcribe_audio
from cache import get_cached_summary, set_cached_summary

# Predefined summary length percentages
PREDEFINED_PERCENTAGES = {
    'nano': 0.01,
    'micro': 0.05,
    'very_short': 0.1,
    'shorter': 0.15,
    'short': 0.23,
    'medium_short': 0.33,
    'medium': 0.4,
    'medium_long': 0.62,
    'long': 0.8
}

# --- Rate Limiter Implementation ---
class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        """
        Allows up to max_calls every period (in seconds).
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            # Remove timestamps older than the period
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                await asyncio.sleep(sleep_time)
            self.calls.append(time.monotonic())

# Global rate limiter: 2000 requests per 60 seconds.
RATE_LIMITER = RateLimiter(max_calls=2000, period=60)

# --- FastAPISummarizer Class ---
class FastAPISummarizer:
    def __init__(
        self,
        file_stream: io.BytesIO,
        input_filename: str,
        input_type: str = "file",
        summary_length: str = "medium",
        custom_percentage: Optional[float] = None,
        lang: str = "es",
        easy: bool = False,
        model: str = "gemini-2.0-flash-exp",
        caching: bool = False,
        debug: bool = False,
    ):
        """
        Initialize the FastAPISummarizer with the provided parameters.
        
        Args:
            file_stream: BytesIO object containing the file to summarize
            input_filename: Name of the input file
            input_type: Type of input (file, url, audio)
            summary_length: Predefined summary length (nano, micro, very_short, shorter, short, medium_short, medium, medium_long, long)
            custom_percentage: Custom percentage for summary length (overrides summary_length if provided)
            lang: Language for summarization/transcription
            easy: Whether to generate easy-to-understand summaries
            model: LLM model to use for summarization
            caching: Whether to enable caching for summarization
            debug: Whether to enable debug mode with detailed logging
        """
        self.file_stream = file_stream
        self.input_filename = input_filename
        self.input_type = input_type
        self.summary_length = summary_length
        self.custom_percentage = custom_percentage
        self.lang = lang
        self.easy = easy
        self.model = model
        self.caching = caching
        self.debug = debug
        
        # Parameters for length control
        self.target_percentages = {
            "nano": 0.01,        # 1%
            "micro": 0.05,       # 5%
            "very_short": 0.10,  # 10%
            "shorter": 0.15,     # 15%
            "short": 0.23,       # 23%
            "medium_short": 0.33,  # 33%
            "medium": 0.40,      # 40%
            "medium_long": 0.62, # 62%
            "long": 0.80         # 80%
        }
        
        # Length control parameters
        self.total_word_count = 0
        self.tokens_per_section_summary = 600  # Estimated tokens per section summary
        self.words_per_token = 0.75  # Estimated words per token ratio
        self.length_wiggle_room = 0.08  # Acceptable deviation from target (8%)
        self.target_subdivisions = 0  # Will be calculated based on content
        self.batch_size = 10  # Number of sections to process in each batch
        
        # Diagnostics
        self.length_diagnostics = {
            "iterations": [],
            "section_counts": [],
            "estimated_word_count": 0,
            "target_word_count": 0,
            "deviation": 0,
            "within_bounds": False,
            "final_word_count": 0,  # Added to track final word count
            "target_reached": False  # Added to track if target was reached
        }

        # Batch processing configuration
        self.batch_size = 10

        # Always skip repetition evaluation (as per requirements)
        self.skip_repetition = True

        # Determine target percentage (either custom or predefined)
        if self.custom_percentage is not None:
            self.target_percentage = self.custom_percentage
        else:
            self.target_percentage = PREDEFINED_PERCENTAGES.get(summary_length, 0.4)

        # For iteration and length estimation
        self.tokens_per_summary = 515
        if 'claude' in self.model.lower():
            self.words_per_token = 0.36
            self.temperature = 0.95
        elif 'gpt' in self.model.lower():
            self.words_per_token = 2.9
            self.temperature = 1.75
        elif 'gemini' in self.model.lower():
            self.words_per_token = 0.247
            self.temperature = 1.75
        else:
            self.words_per_token = 0.36
            self.temperature = 0.95

        # LLM router – all prompts remain exactly as in the original scripts.
        gemini_model_name = "gemini-2.0-flash" if self.model.startswith("gemini") else "gemini-2.0-flash"
        self.llm_router = LLMRouter(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            deepinfra_api_key=os.getenv("DEEPINFRA_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            gemini_model_name=gemini_model_name,
        )

        # Attributes for summarization
        self.total_word_count = 0
        self.sections: List[Dict[str, str]] = []
        self.summarized_sections: List[Dict[str, str]] = []
        # Instead of keeping the last 200 words, we discard them.
        self.context_words = ""
        self.subdivision_factor = 1.0
        self.max_subdivision_factor = 50
        self.wiggle_room = 0.06
        self.current_subdivisions = 0
        
        # For developer output – each section will have a tag and boundary info.
        self.dev_sections: List[Dict[str, str]] = []

        # Regex patterns for transition markers
        self.transition_patterns = [
            r"(?i)Rewritten\s+beginning\s*:",
            r"(?i)Rewritten\s+start\s*:",
            r"(?i)Modified\s+beginning\s*:",
            r"(?i)New\s+beginning\s*:",
            r"(?i)Updated\s+beginning\s*:",
            r"(?i)\*\*Rewritten\s+beginning\*\*\s*:",
            r"(?i)\*\*Rewritten\s+start\*\*\s*:",
            r"(?i)\*\*Modified\s+beginning\*\*\s*:",
            r"(?i)\*\*New\s+beginning\*\*\s*:",
            r"(?i)\*\*Updated\s+beginning\*\*\s*:",
            r"(?i)__Rewritten\s+beginning__\s*:",
            r"(?i)__Rewritten\s+start__\s*:",
            r"(?i)__Modified\s+beginning__\s*:",
            r"(?i)__New\s+beginning__\s*:",
            r"(?i)__Updated\s+beginning__\s*:"
        ]
        
        # Paragraph break markers
        self.paragraph_markers = [
            r"\n\n",
            r"\r\n\r\n",
            r"\n\s*\n",
            r"\r\n\s*\r\n"
        ]

    async def summarize(self) -> (str, str):
        # Read and (if needed) transcribe file content
        if self.input_type == "audio":
            file_content = await transcribe_audio(self.file_stream, self.lang)
        else:
            file_content = self.file_stream.read().decode("utf-8", errors="replace")

        if not file_content:
            raise ValueError("No content found in file.")

        self.total_word_count = len(file_content.split())
        
        # Update diagnostics
        self.length_diagnostics["original_word_count"] = self.total_word_count

        # Check cache (if enabled)
        file_hash = hashlib.md5(file_content.encode()).hexdigest()
        if self.caching:
            cached = get_cached_summary(file_hash)
            if cached:
                return cached.get("sanitized_text"), cached.get("dev_text")

        # Determine if it's a markdown or plain text file
        is_plain_text = not self.is_markdown(file_content)

        # Calculate the target word count for the summary based on the desired percentage of the original content.
        target_word_count = int(self.total_word_count * self.target_percentage)
        self.length_diagnostics["target_word_count"] = target_word_count
        
        # Define reasonable wiggle room (smaller for smaller summaries, larger for larger ones)
        wiggle_percentage = max(0.05, min(0.10, 0.07 + (1.0 - self.target_percentage) * 0.05))
        lower_bound = int(target_word_count * (1 - wiggle_percentage))
        upper_bound = int(target_word_count * (1 + wiggle_percentage))
        
        # Log the target summary length information if debug is enabled
        if self.debug:
            print(f"Target word count: {target_word_count} (allowed range: {lower_bound}-{upper_bound})")
            print(f"Wiggle room: ±{wiggle_percentage*100:.1f}%")
        
        # Generate subdivisions from each section's content
        all_subdivisions = []

        if is_plain_text:
            # Direct subdivision of plain text using the refactored method
            if self.debug: print("Processing as plain text, using direct calculation for subdivision size.")
            all_subdivisions = self.subdivide_plain_text(file_content)
            
            # --- PLAIN TEXT PATH ENDS HERE for subdivision calculation ---
            # We directly use the subdivisions from subdivide_plain_text
            # No further iterative refinement needed for plain text based on estimation.
            
            # Estimate length based on final subdivisions for diagnostics
            estimation_result = self.estimate_summary_length(all_subdivisions)
            estimated_words = estimation_result["estimated_words"]

        else: # It's Markdown
            if self.debug: print("Processing as Markdown, using section-based subdivision and iterative refinement.")
            # --- START EXISTING MARKDOWN LOGIC ---
            # Divide based on Markdown structure first
            self.sections = self.divide_markdown(file_content)
            self.sections = self.adjust_sections(self.sections) # Adjusts very large sections

            # Subdivide each markdown section
            for section in self.sections:
                subdivisions = self.subdivide_section(section['content']) # Uses paragraph/sentence logic
                all_subdivisions.extend(subdivisions)

            # Apply initial merge/split adjustments based on TARGET subdivisions (heuristic)
            # Determine the target number of subdivisions using tokens_per_summary and words_per_token.
            # This target is primarily for the initial adjustment heuristic for Markdown
            self.target_subdivisions = max(1, int(target_word_count / (self.tokens_per_summary * self.words_per_token))) if (self.tokens_per_summary * self.words_per_token) > 0 else 1
            self.length_diagnostics["target_subdivisions"] = self.target_subdivisions # Log this for markdown path

            if self.debug: print(f"Markdown initial target subdivisions: {self.target_subdivisions}")

            if len(all_subdivisions) > self.target_subdivisions:
                all_subdivisions = self.merge_subdivisions(all_subdivisions, self.target_subdivisions)
            elif len(all_subdivisions) < self.target_subdivisions:
                all_subdivisions = self.adjust_subdivisions_count(all_subdivisions) # Tries splitting

            # Verify the subdivision count and estimate the summary length AFTER initial adjustments
            subdivision_count = len(all_subdivisions)
            estimation_result = self.estimate_summary_length(all_subdivisions)
            estimated_words = estimation_result["estimated_words"]

            # Update diagnostics for initial estimate
            self.length_diagnostics["initial_markdown_subdivisions"] = subdivision_count
            self.length_diagnostics["initial_markdown_estimated_words"] = estimated_words

            if self.debug:
                print(f"Markdown after initial adjustments: Subdivisions: {subdivision_count}, Estimated words: {estimated_words}")

            # --- START ITERATIVE REFINEMENT LOGIC (MARKDOWN ONLY) ---
            # If estimated words are outside the acceptable range, refine the subdivision process iteratively
            iteration = 0
            max_iterations = 5 # Keep a reasonable limit

            # Store the initial state for diagnostics
            initial_subdivision_count = len(all_subdivisions) # Count after initial adjust
            initial_estimated_words = estimated_words # Estimate after initial adjust

            while (not (lower_bound <= estimated_words <= upper_bound)) and iteration < max_iterations:
                iteration += 1
                old_subdivision_count = len(all_subdivisions) # Record count before this iteration's adjustment

                adjustment_type = "none" # Track adjustment type

                # Store iteration diagnostics - BEFORE adjustment
                iteration_info = {
                    "iteration": iteration,
                    "old_subdivisions": old_subdivision_count,
                    "target_word_count": target_word_count,
                    "estimated_words": estimated_words, # Estimation before adjustment
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "path": "markdown" # Indicate this is the markdown refinement loop
                }

                # --- This block is now ONLY for Markdown refinement ---
                if estimated_words < lower_bound:
                    if self.debug: print(f"Iteration {iteration} (Markdown): Too short ({estimated_words} < {lower_bound}). Attempting to split.")
                    all_subdivisions = self.adjust_subdivisions_count(all_subdivisions) # Tries splitting longest
                    adjustment_type = "split attempt"
                    iteration_info["adjustment"] = adjustment_type

                elif estimated_words > upper_bound:
                    if self.debug: print(f"Iteration {iteration} (Markdown): Too long ({estimated_words} > {upper_bound}). Attempting to merge.")
                    merge_target_count = max(1, int(old_subdivision_count * (target_word_count / max(1, estimated_words)))) # Avoid div by zero
                    if self.debug: print(f"  Merge target count: {merge_target_count}")
                    all_subdivisions = self.merge_subdivisions(all_subdivisions, merge_target_count)
                    adjustment_type = "merge"
                    iteration_info["adjustment"] = adjustment_type
                    iteration_info["merge_target_count"] = merge_target_count

                # Recalculate the estimated words AFTER adjustment
                new_subdivision_count = len(all_subdivisions)
                estimation_result = self.estimate_summary_length(all_subdivisions)
                estimated_words = estimation_result["estimated_words"]

                # Update diagnostics AFTER adjustment
                iteration_info["new_subdivisions"] = new_subdivision_count
                iteration_info["new_estimated"] = estimated_words

                self.length_diagnostics["iterations"].append(iteration_info)

                if self.debug:
                    print(f"Iteration {iteration} (Markdown): Adjustment: {adjustment_type}")
                    print(f"  Subdivisions: {old_subdivision_count} -> {new_subdivision_count}")
                    print(f"  Estimated words: {iteration_info['estimated_words']} -> {estimated_words}")

                if new_subdivision_count == old_subdivision_count and adjustment_type != "none":
                     if self.debug: print(f"Iteration {iteration} (Markdown): Subdivision count ({new_subdivision_count}) unchanged after {adjustment_type}. Breaking loop.")
                     break
                # --- END ITERATIVE REFINEMENT LOGIC ---
                # --- END EXISTING MARKDOWN LOGIC ---

        # --- COMMON LOGIC AFTER SUBDIVISION ---
        # Final update to diagnostics after all subdivision logic (plain or markdown)
        self.length_diagnostics["actual_subdivisions"] = len(all_subdivisions)
        # Use the 'estimated_words' calculated from the appropriate path (plain direct or markdown refined)
        final_estimation = self.estimate_summary_length(all_subdivisions) # Re-estimate based on final subdivisions
        self.length_diagnostics["final_estimated_word_count_before_summary"] = final_estimation["estimated_words"] # Estimate based on final subs
        self.length_diagnostics["final_deviation_before_summary"] = final_estimation["deviation"]
        self.length_diagnostics["final_within_bounds_before_summary"] = final_estimation["within_bounds"]

        if self.debug:
            print(f"Final subdivisions before summarization: {len(all_subdivisions)}")
            print(f"Final estimated words before summarization: {final_estimation['estimated_words']} (Target: {target_word_count}, Bounds: {lower_bound}-{upper_bound})")

        # Update sections with the final properly sized subdivisions, each becoming a separate section.
        self.sections = [{"header": "", "content": sub} for sub in all_subdivisions]
        # --- End Enhanced Length Control Logic --- # Note: This comment might be slightly misplaced now but harmless

        # Process sections in batches
        batched_sections = [self.sections[i:i + self.batch_size] for i in range(0, len(self.sections), self.batch_size)]
        sanitized_batches = []
        dev_info = {
            "generated_at": datetime.utcnow().isoformat(), 
            "sections": [],
            "length_diagnostics": self.length_diagnostics
        }
        
        for batch in batched_sections:
            batch_summaries = await self._summarize_batch(batch)
            sanitized_batches.append(" ".join([sec["content"] for sec in batch_summaries]))
            dev_info["sections"].extend(batch_summaries)

        # Apply final verification to ensure text integrity
        sanitized_text = self.verify_text_integrity("\n\n".join(sanitized_batches))
        
        # Count final summary length for diagnostics
        final_word_count = len(sanitized_text.split())
        self.length_diagnostics["final_word_count"] = final_word_count
        self.length_diagnostics["target_reached"] = lower_bound <= final_word_count <= upper_bound
        
        if self.debug:
            actual_percentage = final_word_count / self.total_word_count
            print(f"Final summary: {final_word_count} words ({actual_percentage:.1%} of original)")
            print(f"Target was: {target_word_count} words ({self.target_percentage:.1%} of original)")
            print(f"Difference: {final_word_count - target_word_count} words " +
                  f"({(actual_percentage - self.target_percentage) * 100:.1f}% points)")
            print(f"Target reached: {'Yes' if lower_bound <= final_word_count <= upper_bound else 'No'}")
        
        dev_text = json.dumps(dev_info, indent=2, ensure_ascii=False)

        # If caching is enabled, store result
        if self.caching:
            set_cached_summary(file_hash, {"sanitized_text": sanitized_text, "dev_text": dev_text})

        return sanitized_text, dev_text

    async def _summarize_batch(self, batch_sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process a batch of sections using smart merging strategy."""
        # Primero resumir todas las secciones individualmente
        individual_summaries = []
        for idx, section in enumerate(batch_sections):
            # Get summary for current section
            summary = await self.ask_for_summary(section["content"])
            summary = self.clean_transition_markers(summary)  # Remove any transition markers
            
            individual_summaries.append({
                "tag": section.get("header").strip() or f"Section {idx+1}",
                "content": summary,
                "first_50_words": " ".join(summary.split()[:50]),
                "last_50_words": " ".join(summary.split()[-50:])
            })
        
        # Luego proceder con la fusión inteligente
        batch_summaries = []
        i = 0
        while i < len(individual_summaries):
            current = individual_summaries[i]
            
            # Mirar adelante para ver si debemos combinar con la siguiente sección
            if i + 1 < len(individual_summaries):
                next_section = individual_summaries[i + 1]
                should_merge = await self.should_merge_sections(
                    current["content"],
                    next_section["content"],
                    similarity_threshold=0.75  # Umbral ligeramente más alto para evitar fusiones incorrectas
                )
                
                if should_merge:
                    # Reescribir transición
                    next_content = await self.rewrite_transition(current["content"], next_section["content"])
                    
                    # Combinar secciones con transición adecuada
                    merged_content = current["content"]
                    # Asegurarse de que hay un separador de párrafo apropiado
                    if not re.search(r'\n\n$', merged_content):
                        merged_content += "\n\n"
                    merged_content += next_content
                    
                    # Actualizar sección actual y saltarse la siguiente
                    current = {
                        "tag": current["tag"],
                        "content": merged_content,
                        "first_50_words": current["first_50_words"],
                        "last_50_words": " ".join(next_content.split()[-50:])
                    }
                    i += 2  # Saltar la siguiente sección
                else:
                    batch_summaries.append(current)
                    i += 1
            else:
                # Última sección
                batch_summaries.append(current)
                i += 1
                
        return batch_summaries

    async def rewrite_transition(self, prev_summary: str, curr_summary: str) -> str:
        # Usar más contexto del final de la sección anterior y del inicio de la actual
        prev_context = " ".join(prev_summary.split()[-40:])  # Aumentado de 20 a 40 palabras
        curr_start = " ".join(curr_summary.split()[:40])     # Aumentado de 20 a 40 palabras
        
        language_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'it': 'Italian',
        }
        language = language_map.get(self.lang, 'Spanish')
        
        prompt = (
            f"Rewrite the following first 40 words of the current section so that it seamlessly follows the previous section. "
            f"Do not add any title, header, or section marker. Do not change the meaning, just make the transition smooth.\n\n"
            f"Previous section ending: {prev_context}\n"
            f"Current section starting: {curr_start}\n\n"
            f"Rewritten beginning (in {language}):"
        )
        messages = [{"role": "user", "content": prompt}]
        system_prompt = (
            f"You are a helpful and knowledgeable assistant specialized in summarizing texts in {language}. "
            f"Ensure that the rewritten text is seamless and stylistically consistent. "
            f"Answer in markdown. Maintain all important information while improving flow."
        )
        # Enforce rate limit before calling LLM.
        await RATE_LIMITER.acquire()
        rewritten = await self.llm_router.generate(
            model=self.model,
            messages=messages,
            max_tokens=150,  # Aumentado para permitir transiciones más completas
            temperature=self.temperature,
            top_p=0.9,
            stop_sequences=["User:", "Human:", "Assistant:"],
            system=system_prompt
        )
        curr_words = curr_summary.split()
        new_beginning = rewritten.strip()
        if len(new_beginning.split()) < 25:  # Asegurarse de que la transición sea sustancial
            return curr_summary
        
        # Reemplazar solo el inicio manteniendo el resto intacto
        new_summary = new_beginning + " " + " ".join(curr_words[40:])
        return new_summary

    async def ask_for_summary(self, text: str) -> str:
        structured_outline = await self.generate_structured_outline(text)
        
        # Detectar si el texto original contiene listas
        has_lists = bool(re.search(r'^\s*[-*•]\s+|\s*\d+\.\s+', text, re.MULTILINE))
        list_instruction = (
            "Preserve all list structures (numbered lists, bullet points) exactly as they appear in the original text. "
            "Maintain the hierarchy and complete content of lists even when summarizing other parts."
        ) if has_lists else ""
        
        messages = [
            {
                "role": "user",
                "content": (
                    f"## Structured Outline: {structured_outline}\n\n"
                    f"## Text to summarize: {text}\n\n"
                    f"## Instructions: Summarize the text above. Maintain the most important information and ensure the summary is substantive. "
                    f"{list_instruction} "
                    f"Do not write lists or numbered items unless they are explicitly written in the given text. "
                    f"Do not truncate paragraphs mid-sentence. Ensure each section has a logical conclusion. "
                    f"Write a cohesive text. Output in markdown."
                )
            }
        ]
        language_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'it': 'Italian',
        }
        language = language_map.get(self.lang, 'Spanish')
        easy_to_understand_text = (
            "but easy to understand. You must easily explain complex topics in such way that they are understandable to anyone. "
            "That means you must not use technical jargon or complex words, you must use simple words and explain the topics in a way that is easy to understand. "
            "Also, explain in great detail each topic and subtopic, possibly even stop just to explain a single word or concept if it's necessary. "
            "Write thorough explanations that take as much space as necessary for explaining these concepts in a way as such that everyone can understand it. "
            "You must divide the answer in two parts: Summary/Resumen/Résumé and Explanation/Explicación/Explication. Only write in the language it is needed, "
            "if it is in English:Summary / Explanation, if it is in Spanish: Resumen / Explicación, if it is in French: Résumé / Explication. "
            "Use Header 2 (##) To indicate this divide. The first part must be a summary of the text, the second part must be an explanation of the text. "
            "The summary must be a concise summary of the text, the explanation must be a detailed explanation of the text. The summary must be in the same language as the text, "
            "the explanation must be in the same language as the text."
        ) if self.easy else (
            "with rigorous academic methodology. Analyze scholarly discourse, maintaining intellectual depth and critical perspective. Explore topics across diverse disciplines, emphasizing systematic research approaches, nuanced interpretation, and interdisciplinary connections. Use precise academic language, structured argumentation, and evidence-based reasoning. Demonstrate intellectual curiosity through comprehensive examination of complex subjects, whether in humanities, sciences, social sciences, or emerging interdisciplinary fields. Prioritize scholarly objectivity while revealing innovative analytical insights."
        )
        max_tokens = self.tokens_per_summary * 2 if self.easy else self.tokens_per_summary
        import random
        max_tokens = int(max_tokens * (1 + (random.uniform(-0.1, 0.1))))
        
        # Styling guidelines based on language
        style_guidelines = ""
        if language == "Spanish":
            style_guidelines = (
                "Use Spanish quotation marks («»). Maintain technical and academic language. Avoid section separation and integrate all elements into a cohesive analysis. "
                "Avoid both gratuitous praise and unjustified criticism. Avoid overuse of connectors. "
                "Use long subordinate phrases. Don't include introductions or conclusions in paragraphs or sections—get straight to the point."
            )
        elif language == "English":
            style_guidelines = (
                "Use English quotation marks. Maintain technical and academic language. Avoid section separation and integrate all elements into a cohesive analysis. "
                "Avoid both gratuitous praise and unjustified criticism. Use em dashes (—). Avoid overuse of connectors. "
                "Use long subordinate phrases. Don't include introductions or conclusions in paragraphs or sections—get straight to the point."
            )
        elif language == "French":
            style_guidelines = (
                "Use French quotation marks («»). Maintain technical and academic language. Avoid section separation and integrate all elements into a cohesive analysis. "
                "Avoid both gratuitous praise and unjustified criticism. Avoid overuse of connectors. "
                "Use long subordinate phrases. Don't include introductions or conclusions in paragraphs or sections—get straight to the point."
            )
        elif language == "Italian":
            style_guidelines = (
                "Use Italian quotation marks («»). Maintain technical and academic language. Avoid section separation and integrate all elements into a cohesive analysis. "
                "Avoid both gratuitous praise and unjustified criticism. Avoid overuse of connectors. "
                "Use long subordinate phrases. Don't include introductions or conclusions in paragraphs or sections—get straight to the point."
            )
        
        system_prompt = (
            f"You are a helpful and knowledgeable assistant specialized in summarizing texts. "
            f"You will understand the given text and its underlying syntax, which might not be evident. "
            f"You will keep the most essential information and optimize for space, this may include specific names, theories, dates, lists and definitions. "
            f"Follow the provided structured outline to maintain the organization of ideas, you are only talking about the text provided, so you must not mention anything outside of it or in the outline, "
            f"this is given to you so that you know what not to write about. You must answer in precise and technically perfect {language}, {easy_to_understand_text} "
            f"{style_guidelines}\n\n"
            f"You must make no mention to the fact that you are summarizing anything or that you've been given any text but rather you must write the text as one that exists by itself. "
            f"You must not reference the document. Keep in mind the context if provided and then make a seamless transition and do not repeat yourself if possible. "
            f"Do not include titles, headers, or section markers in your summary. Write it as continuous prose without separating it into distinct parts or sections. "
            f"Never leave any paragraph, list, or section incomplete. Ensure each section has proper closure. "
            f"You must do all of this in markdown. Remember to answer in {language}. Take a deep breath and think step by step."
        )
        # Enforce rate limit before calling LLM.
        await RATE_LIMITER.acquire()
        summary_text = await self.llm_router.generate(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=0.9,
            stop_sequences=["User:", "Human:", "Assistant:"],
            system=system_prompt
        )
        return summary_text

    async def generate_structured_outline(self, text: str) -> str:
        first_50_words = ' '.join(text.split()[:50])
        text_hash = hashlib.md5(first_50_words.encode()).hexdigest()
        cache_file = Path("outline_cache.json")
        if cache_file.exists():
            with open(cache_file, "r") as f:
                cache = json.load(f)
        else:
            cache = {}
        if text_hash in cache:
            return cache[text_hash]
        outline_prompt = f"Generate a brief structured outline for the following text. The outline should capture the main topics and subtopics, providing a clear organization of ideas:\n\n{text}"
        outline_messages = [{"role": "user", "content": outline_prompt}]
        # Enforce rate limit before calling LLM.
        await RATE_LIMITER.acquire()
        outline = await self.llm_router.generate(
            model=self.model,
            messages=outline_messages,
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            system="You are an expert at creating concise, well-structured outlines. Generate a brief outline that captures the main topics and subtopics of the given text, providing a clear organization of ideas."
        )
        cache[text_hash] = outline
        with open(cache_file, "w") as f:
            json.dump(cache, f)
        return outline

    def is_markdown(self, text: str) -> bool:
        return bool(re.search(r'^#{1,6}\s', text, re.MULTILINE))

    def divide_markdown(self, text: str) -> List[Dict[str, str]]:
        sections = []
        lines = text.split('\n')
        current_section = {'header': '', 'content': ''}
        for line in lines:
            if re.match(r'^#{1,6}\s', line):
                if current_section['content']:
                    sections.append(current_section)
                    current_section = {'header': '', 'content': ''}
                current_section['header'] = line
            elif re.match(r'^\d+\.\s', line):
                if current_section['content']:
                    sections.append(current_section)
                    current_section = {'header': '', 'content': ''}
                current_section['header'] = 'List'
                current_section['content'] += line + '\n'
            else:
                current_section['content'] += line + '\n'
        if current_section['content']:
            sections.append(current_section)
        return sections

    def adjust_sections(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        adjusted_sections = []
        for section in sections:
            content_words = section['content'].split()
            word_count = len(content_words)
            if word_count <= 4000:
                adjusted_sections.append(section)
            else:
                subsections = self.divide_into_subsections(section)
                adjusted_sections.extend(subsections)
        return adjusted_sections

    def divide_into_subsections(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        content = section['content']
        lines = content.split('\n')
        subsections = []
        current_subsection = {'header': '', 'content': ''}
        header_pattern = re.compile(r'^(#{1,6}) (.+)')
        for line in lines:
            header_match = header_pattern.match(line)
            if header_match:
                if current_subsection['content']:
                    subsections.append(current_subsection)
                    current_subsection = {'header': '', 'content': ''}
                current_subsection['header'] = line
            else:
                current_subsection['content'] += line + '\n'
        if current_subsection['content']:
            subsections.append(current_subsection)
        return subsections

    # Versión mejorada para preservar la integridad de los párrafos y respetar límites de oraciones
    def subdivide_section(self, section_content):
        """
        Subdivide a section into smaller chunks, ensuring chunks end on complete sentences
        and match the target subdivision count needed for the desired summary length.
        """
        # First, determine if input is a plain text file
        is_plain_text = not self.is_markdown(section_content)
        
        # For plain text files, we'll use a more precise mathematical approach
        if is_plain_text:
            return self.subdivide_plain_text(section_content)
        
        # For markdown, we'll use the existing paragraph-based approach with improvements
        # Dividir primero por párrafos para preservar su integridad
        paragraphs = re.split(r'(\n\n|\r\n\r\n)', section_content)
        
        subdivisions = []
        current_subdivision = ""
        current_word_count = 0
        # Calculate target words per subdivision based on the *current* overall target subdivisions
        target_words_per_sub = max(150, len(section_content.split()) // self.target_subdivisions) if self.target_subdivisions else 150
        
        # Procesar párrafos y sus separadores
        i = 0
        while i < len(paragraphs):
            # Obtener el párrafo y su separador (si existe)
            paragraph = paragraphs[i]
            separator = paragraphs[i+1] if i+1 < len(paragraphs) else ""
            
            paragraph_word_count = len(paragraph.split())
            
            # Si añadir este párrafo excedería mucho el tamaño objetivo, comenzar nueva subdivisión
            if current_word_count > 0 and current_word_count + paragraph_word_count > target_words_per_sub * 1.5:
                subdivisions.append(current_subdivision.strip())
                current_subdivision = ""
                current_word_count = 0
            
            current_subdivision += paragraph + separator
            current_word_count += paragraph_word_count
            
            # Si alcanzamos aproximadamente el tamaño objetivo, finalizar subdivisión
            if current_word_count >= target_words_per_sub:
                subdivisions.append(current_subdivision.strip())
                current_subdivision = ""
                current_word_count = 0
            
            # Avanzar al siguiente par (párrafo + separador)
            i += 2
        
        # Añadir última subdivisión si quedó contenido
        if current_subdivision:
            subdivisions.append(current_subdivision.strip())
        
        # Verificar si hay listas de elementos y mantenerlas juntas
        final_subdivisions = []
        processed_indices = set() # Keep track of subdivisions already merged into a list
        
        for idx, subdivision in enumerate(subdivisions):
            if idx in processed_indices:
                continue

            # Detectar si contiene inicios de lista
            list_items = re.findall(r'^\s*[-*•]\s+|\s*\d+\.\s+', subdivision, re.MULTILINE)
            
            if list_items and len(list_items) < 3:  # If few items, likely incomplete
                merged_list_subdivision = subdivision
                # Look ahead for subsequent subdivisions that might contain the rest of the list
                for next_idx in range(idx + 1, len(subdivisions)):
                    next_sub = subdivisions[next_idx]
                    if re.search(r'^\s*[-*•]\s+|\s*\d+\.\s+', next_sub, re.MULTILINE):
                        merged_list_subdivision += "\n\n" + next_sub
                        processed_indices.add(next_idx) # Mark as processed
                    else:
                         # Stop merging if the next subdivision doesn't look like a list item
                        break 
                final_subdivisions.append(merged_list_subdivision)
                processed_indices.add(idx)
            else:
                final_subdivisions.append(subdivision)
        
        return final_subdivisions

    def merge_subdivisions(self, subdivisions, target_subdivisions):
        """Merges subdivisions if the current count exceeds the target."""
        if self.debug:
            print(f"Attempting to merge {len(subdivisions)} subdivisions into {target_subdivisions}")
        
        if len(subdivisions) <= target_subdivisions:
            return subdivisions # No merging needed

        # Calculate average target words per subdivision for merging
        total_words = sum(len(s.split()) for s in subdivisions)
        target_words_per_merged_sub = total_words / target_subdivisions

        merged = []
        current_merged_sub = []
        current_merged_words = 0

        for subdivision in subdivisions:
            subdivision_words = len(subdivision.split())

            # If adding this subdivision would significantly exceed the target word count
            # and we haven't reached the desired number of merged sections yet,
            # finalize the current merged section and start a new one.
            # The `* 1.2` adds some flexibility. Check `len(merged) < target_subdivisions - 1` ensures the last section can absorb remaining content.
            if current_merged_words > 0 and \
               current_merged_words + subdivision_words > target_words_per_merged_sub * 1.2 and \
               len(merged) < target_subdivisions - 1:
                merged.append("\n\n".join(current_merged_sub)) # Join with paragraph breaks
                current_merged_sub = [subdivision]
                current_merged_words = subdivision_words
            else:
                # Otherwise, add the current subdivision to the ongoing merged section
                current_merged_sub.append(subdivision)
                current_merged_words += subdivision_words

        # Add the last collected merged subdivision
        if current_merged_sub:
            merged.append("\n\n".join(current_merged_sub))

        if self.debug:
            print(f"Merged into {len(merged)} subdivisions.")
        return merged

    def adjust_subdivisions_count(self, subdivisions):
        """Adjusts subdivisions if the current count is less than the target, by splitting the longest ones."""
        target_count = self.target_subdivisions
        if self.debug:
            print(f"Attempting to adjust {len(subdivisions)} subdivisions to reach target {target_count}")

        if len(subdivisions) >= target_count:
            return subdivisions # No adjustment needed or already too many

        # Sort subdivisions by length (longest first) to prioritize splitting them
        sorted_indices = sorted(range(len(subdivisions)), key=lambda k: len(subdivisions[k].split()), reverse=True)
        
        current_subdivisions = list(subdivisions) # Create a mutable copy

        # Keep track of which original subdivisions have been split to avoid infinite loops if splitting fails
        split_attempted_indices = set() 

        while len(current_subdivisions) < target_count:
            made_a_split = False
            # Iterate through sorted indices to find the next longest subdivision to split
            for original_idx in sorted_indices:
                 # Find the current index of this subdivision in the potentially modified list
                try:
                    current_idx = -1
                    # Need to find the *content* in the current list as indices shift
                    original_content = subdivisions[original_idx]
                    for i, sub in enumerate(current_subdivisions):
                        if sub == original_content:
                            current_idx = i
                            break
                    
                    if current_idx == -1 or current_idx in split_attempted_indices:
                        continue # Already tried splitting this one or it's gone

                    longest_subdivision_content = current_subdivisions[current_idx]
                    split_attempted_indices.add(current_idx) # Mark as attempted

                    # Attempt to split using the same subdivide_section logic
                    # Temporarily reduce target_subdivisions to encourage splitting this section
                    original_target = self.target_subdivisions
                    self.target_subdivisions = 2 # Aim to split into at least 2
                    newly_split_parts = self.subdivide_section(longest_subdivision_content)
                    self.target_subdivisions = original_target # Restore original target

                    if len(newly_split_parts) > 1: 
                        # Successful split: remove the original and insert the new parts
                        current_subdivisions.pop(current_idx)
                        current_subdivisions.insert(current_idx, newly_split_parts[0]) # Insert first part at original location
                        current_subdivisions.extend(newly_split_parts[1:]) # Append the rest
                        made_a_split = True
                        # Recalculate sorted_indices as lengths/positions changed
                        sorted_indices = sorted(range(len(current_subdivisions)), key=lambda k: len(current_subdivisions[k].split()), reverse=True)
                        # Reset attempted indices for the new list structure
                        split_attempted_indices = set() 
                        break # Go back to the while loop condition check
                    #else:
                        # If subdivide_section returned only 1 part, it couldn't be split further.
                        # Leave it as is and try the next longest.

                except IndexError:
                    # This can happen if indices get out of sync, break and return current best effort
                    break

            if not made_a_split:
                # If we went through all subdivisions and couldn't split any further, break the loop
                break
        
        if self.debug:
             print(f"Adjusted count to {len(current_subdivisions)} subdivisions.")
        return current_subdivisions

    def subdivide_plain_text(self, text):
        """
        Subdivide plain text content into optimal sections to achieve the target summary length.
        Uses mathematical calculation to determine the ideal section size.
        
        Args:
            text (str): The plain text content to be subdivided
        
        Returns:
            list: A list of text sections optimized for the target summary length
        """
        if self.debug:
            self.length_diagnostics["iterations"].append("Starting plain text subdivision")
        
        # Calculate the target summary word count based on percentage
        percentage = self.get_percentage_from_summary_length()
        target_word_count = int(self.total_word_count * percentage)
        self.length_diagnostics["target_word_count"] = target_word_count
        
        # Split text into sentences for proper boundary handling
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Calculate optimal section size using the direct formula
        words_per_section = 250 # Default fallback

        # Calculate optimal section size directly based on target length
        if target_word_count <= 0: # Avoid division by zero
            # If target is very small/zero, aim for fewer, larger sections
            # Use total words / a small number (e.g., 2) to get large sections
            words_per_section = max(250, int(self.total_word_count / 2)) 
            if self.debug: print(f"Plain text: Target word count {target_word_count} too low, setting large words_per_section: {words_per_section}")
        else:
            # Formula: target_summary_words = (total_original_words / words_per_section) * tokens_per_summary * words_per_token
            # Rearranged to find words_per_section:
            # words_per_section = (total_original_words * tokens_per_summary * words_per_token) / target_summary_words
            
            # Ensure divisors are not zero
            tokens_per_summary = self.tokens_per_summary if self.tokens_per_summary > 0 else 515 # Use default if zero
            words_per_token = self.words_per_token if self.words_per_token > 0 else 0.3 # Use default if zero
            
            denominator = target_word_count
            numerator = self.total_word_count * tokens_per_summary * words_per_token
            
            if denominator <= 0:
                 words_per_section = 4000 # Assign a large value if target_word_count is non-positive
                 if self.debug: print(f"Plain text: Denominator (target_word_count) is zero or negative, setting large words_per_section: {words_per_section}")
            else:
                # Estimate number of sections needed first to avoid huge intermediate numbers if possible
                estimated_sections_needed = target_word_count / max(1e-6, (tokens_per_summary * words_per_token))
                words_per_section = int(self.total_word_count / max(1, estimated_sections_needed))

        # --- HARDCODED ADJUSTMENT FOR PLAIN TEXT --- 
        # Empirically observed ~122% overestimation in final word count compared to target for plain text.
        # Multiply calculated words_per_section by 2.22 to create fewer, larger sections, aiming to compensate.
        adjustment_factor = 2.7 # Changed from 1.23 based on latest test results
        adjusted_words_per_section = int(words_per_section * adjustment_factor)
        if self.debug:
            print(f"Plain text: Applying adjustment factor {adjustment_factor}. Original words_per_section: {words_per_section}, Adjusted: {adjusted_words_per_section}")
        words_per_section = adjusted_words_per_section # Use the adjusted value
        # --- END HARDCODED ADJUSTMENT ---

        # Clamp within reasonable bounds
        words_per_section = max(50, min(4000, words_per_section)) # Min 50 words, Max 4000 words per section

        if self.debug:
             print(f"Plain text direct calculation: target={target_word_count}, total_words={self.total_word_count}, calculated words_per_section={words_per_section}")
             self.length_diagnostics["iterations"].append({
                 "type": "plain_text_direct_calc",
                 "calculated_words_per_section": words_per_section,
                 "target_word_count": target_word_count,
             })
        
        # Now create the actual sections based on the calculated words_per_section
        sections = []
        current_section = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed the section size and we already have content,
            # finalize the current section and start a new one
            if current_word_count > 0 and current_word_count + sentence_words > words_per_section:
                sections.append(" ".join(current_section))
                current_section = [sentence]
                current_word_count = sentence_words
            else:
                current_section.append(sentence)
                current_word_count += sentence_words
        
        # Add the last section if it has any content
        if current_section:
            sections.append(" ".join(current_section))
        
        if self.debug:
            self.length_diagnostics["section_counts"].append({
                "type": "plain_text",
                "final_sections": len(sections),
                "words_per_section": words_per_section,
                "estimated_total_summary_words": len(sections) * self.tokens_per_section_summary * self.words_per_token
            })
        
        return sections

    def estimate_summary_length(self, subdivisions):
        """
        Estimate the length of the summary based on subdivision count and model parameters.
        
        Args:
            subdivisions (list): The list of content subdivisions
        
        Returns:
            dict: Estimation data including expected word count and deviation from target
        """
        # Calculate the target summary word count
        percentage = self.get_percentage_from_summary_length()
        target_word_count = int(self.total_word_count * percentage)
        
        # Estimate tokens and words based on subdivision count
        estimated_summary_tokens = len(subdivisions) * self.tokens_per_section_summary
        estimated_summary_words = int(estimated_summary_tokens * self.words_per_token)
        
        # Calculate deviation from target
        deviation = abs(estimated_summary_words - target_word_count) / target_word_count
        within_bounds = deviation <= self.length_wiggle_room
        
        # Update diagnostics
        if self.debug:
            self.length_diagnostics["estimated_word_count"] = estimated_summary_words
            self.length_diagnostics["target_word_count"] = target_word_count
            self.length_diagnostics["deviation"] = deviation
            self.length_diagnostics["within_bounds"] = within_bounds
        
        return {
            "estimated_words": estimated_summary_words,
            "target_words": target_word_count,
            "deviation": deviation,
            "within_bounds": within_bounds
        }

    def get_percentage_from_summary_length(self):
        """
        Get the percentage value for the current summary length setting.
        
        Returns:
            float: Percentage value (0.0-1.0) for the current summary length
        """
        # If a custom percentage is provided, use that
        if self.custom_percentage is not None:
            return min(1.0, max(0.01, self.custom_percentage))
        
        # Otherwise, get the percentage from the predefined settings
        if self.summary_length in self.target_percentages:
            return self.target_percentages[self.summary_length]
        
        # Default to medium (40%) if the summary_length is not recognized
        return self.target_percentages["medium"]

    def verify_text_integrity(self, text: str) -> str:
        """Verifica y corrige problemas de integridad en el texto final."""
        
        # Corregir párrafos truncados (terminados abruptamente sin puntuación)
        paragraphs = text.split('\n\n')
        for i, paragraph in enumerate(paragraphs):
            if i < len(paragraphs) - 1 and paragraph and not paragraph.strip().endswith(('.', '!', '?', ':', '"', '»', ')')):
                # Si el párrafo no termina con puntuación, verificar si el siguiente párrafo 
                # comienza con minúscula (posible continuación)
                if paragraphs[i+1] and len(paragraphs[i+1]) > 0 and not paragraphs[i+1][0].isupper():
                    # Unir con el siguiente párrafo
                    paragraphs[i] = paragraph + ' ' + paragraphs[i+1]
                    paragraphs[i+1] = ''
        
        # Eliminar párrafos vacíos
        paragraphs = [p for p in paragraphs if p.strip()]
        
        # Restaurar listas fragmentadas
        processed_paragraphs = []
        in_list = False
        current_list = []
        
        for paragraph in paragraphs:
            # Detectar inicio de lista
            if re.match(r'^\s*[-*•]\s+|\s*\d+\.\s+', paragraph):
                if not in_list:
                    in_list = True
                    current_list = [paragraph]
                else:
                    current_list.append(paragraph)
            else:
                if in_list:
                    # Finalizar lista anterior
                    processed_paragraphs.append('\n'.join(current_list))
                    in_list = False
                    current_list = []
                processed_paragraphs.append(paragraph)
        
        # No olvidar la última lista si existe
        if in_list:
            processed_paragraphs.append('\n'.join(current_list))
        
        return '\n\n'.join(processed_paragraphs)

    async def check_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Check semantic similarity between two text sections using the LLM.
        Returns a similarity score between 0 and 1.
        """
        prompt = (
            f"Rate the semantic similarity between these two text sections on a scale from 0 to 1, "
            f"where 0 means completely unrelated and 1 means highly related. "
            f"Only respond with a number between 0 and 1.\n\n"
            f"Text 1: {text1}\n\n"
            f"Text 2: {text2}"
        )
        messages = [{"role": "user", "content": prompt}]
        
        await RATE_LIMITER.acquire()
        response = await self.llm_router.generate(
            model=self.model,
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            top_p=0.9,
            system="You are a helpful assistant that rates semantic similarity between texts."
        )
        
        try:
            similarity = float(response.strip())
            return max(0.0, min(1.0, similarity))
        except (ValueError, TypeError):
            return 0.5  # Default to medium similarity if parsing fails

    def clean_transition_markers(self, text: str) -> str:
        """Remove transition markers from the text."""
        cleaned = text
        for pattern in self.transition_patterns:
            cleaned = re.sub(pattern, "", cleaned)
        return cleaned.strip()

    def get_paragraph_breaks(self, text: str) -> List[int]:
        """Get indices of paragraph breaks in the text."""
        breaks = set()
        for marker in self.paragraph_markers:
            for match in re.finditer(marker, text):
                breaks.add(match.start())
        return sorted(list(breaks))

    async def should_merge_sections(self, prev_section: str, curr_section: str, similarity_threshold: float = 0.7) -> bool:
        """
        Determine if two sections should be merged based on:
        1. Semantic similarity
        2. Presence of paragraph breaks
        3. Length considerations
        """
        # Don't merge if either section is too long
        if len(prev_section.split()) > 200 or len(curr_section.split()) > 200:
            return False
            
        # Check for natural paragraph breaks
        prev_breaks = self.get_paragraph_breaks(prev_section)
        curr_breaks = self.get_paragraph_breaks(curr_section)
        
        # If both sections have multiple paragraph breaks, don't merge
        if len(prev_breaks) > 1 and len(curr_breaks) > 1:
            return False
            
        # Check semantic similarity
        similarity = await self.check_semantic_similarity(prev_section, curr_section)
        return similarity >= similarity_threshold