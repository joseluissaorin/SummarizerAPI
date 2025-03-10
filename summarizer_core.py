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
    ):
        self.file_stream = file_stream
        self.input_filename = input_filename
        self.input_type = input_type
        self.summary_length = summary_length
        self.custom_percentage = custom_percentage
        self.lang = lang
        self.easy = easy
        self.model = model
        self.caching = caching

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
        gemini_model_name = self.model if self.model.startswith("gemini") else "gemini-2.0-flash-lite"
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
        self.target_subdivisions = 0
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

        # Check cache (if enabled)
        file_hash = hashlib.md5(file_content.encode()).hexdigest()
        if self.caching:
            cached = get_cached_summary(file_hash)
            if cached:
                return cached.get("sanitized_text"), cached.get("dev_text")

        # Divide text into sections (using the original section dividing logic)
        if self.is_markdown(file_content):
            self.sections = self.divide_markdown(file_content)
        else:
            self.sections = [{"header": "", "content": file_content}]
        self.sections = self.adjust_sections(self.sections)

        # --- Begin Adaptive Subdivision Logic from new_summarize.py ---
        # Calculate the target word count for the summary based on the desired percentage of the original content.
        target_word_count = int(self.total_word_count * self.target_percentage)
        
        # Determine the target number of subdivisions using tokens_per_summary and words_per_token.
        self.target_subdivisions = max(1, int(target_word_count / (self.tokens_per_summary * self.words_per_token)))
        
        # Define an acceptable range of word count with a buffer of 350 words.
        lower_bound = target_word_count - 350
        upper_bound = target_word_count + 350

        # Generate subdivisions from each section's content using the subdivide_section helper.
        all_subdivisions = []
        for section in self.sections:
            subdivisions = self.subdivide_section(section['content'])
            all_subdivisions.extend(subdivisions)

        # If we have more subdivisions than needed, merge them to match the target.
        if len(all_subdivisions) > self.target_subdivisions:
            all_subdivisions = self.merge_subdivisions(all_subdivisions, self.target_subdivisions)
        # If we have too few subdivisions, iteratively split the longest subdivision until reaching the target.
        elif len(all_subdivisions) < self.target_subdivisions:
            while len(all_subdivisions) < self.target_subdivisions:
                longest_idx = max(range(len(all_subdivisions)), key=lambda i: len(all_subdivisions[i].split()))
                longest_subdivision = all_subdivisions.pop(longest_idx)
                new_subdivisions = self.subdivide_section(longest_subdivision)
                if len(new_subdivisions) == 1:  # Cannot subdivide further
                    all_subdivisions.insert(longest_idx, longest_subdivision)
                    break
                all_subdivisions.extend(new_subdivisions)

        # Estimate the combined word count of all subdivisions.
        estimated_words = self.estimate_summary_length(all_subdivisions, self.target_subdivisions)
        
        # If the estimated words are not within the target range, refine subdivisions iteratively.
        if not (lower_bound <= estimated_words <= upper_bound):
            iteration = 0
            max_iterations = 15
            while iteration < max_iterations:
                iteration += 1
                self.summarized_sections = []
                all_subdivisions = []
                for section in self.sections:
                    content = section['content']
                    # Generate subdivisions from the section
                    subdivisions = self.subdivide_section(content)
                    # If subdivisions are too few, reformat the content as a list and subdivide again
                    if len(subdivisions) < self.target_subdivisions:
                        formatted_content = self.format_section_as_list(content)
                        subdivisions = self.subdivide_section(formatted_content)
                    # Adjust subdivision factor if still under target
                    while len(subdivisions) < self.target_subdivisions and self.subdivision_factor > 0.1:
                        self.subdivision_factor *= 0.5
                        subdivisions = self.subdivide_section(content)
                    # Merge subdivisions if there are too many
                    if len(subdivisions) > self.target_subdivisions:
                        subdivisions = self.merge_subdivisions(subdivisions, self.target_subdivisions)
                    all_subdivisions.extend(subdivisions)
                estimated_words = self.estimate_summary_length(all_subdivisions, self.target_subdivisions)
                if lower_bound <= estimated_words <= upper_bound:
                    break
        
        # Update sections with the final properly sized subdivisions, each becoming a separate section.
        self.sections = [{"header": "", "content": sub} for sub in all_subdivisions]
        # --- End Adaptive Subdivision Logic ---

        # Process sections in batches of 10
        batched_sections = [self.sections[i:i + self.batch_size] for i in range(0, len(self.sections), self.batch_size)]
        sanitized_batches = []
        dev_info = {"generated_at": datetime.utcnow().isoformat(), "sections": []}
        
        for batch in batched_sections:
            batch_summaries = await self._summarize_batch(batch)
            sanitized_batches.append(" ".join([sec["content"] for sec in batch_summaries]))
            dev_info["sections"].extend(batch_summaries)

        # Apply final verification to ensure text integrity
        sanitized_text = self.verify_text_integrity("\n\n".join(sanitized_batches))
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

    # Versión mejorada para preservar la integridad de los párrafos
    def subdivide_section(self, section_content):
        # Dividir primero por párrafos para preservar su integridad
        paragraphs = re.split(r'(\n\n|\r\n\r\n)', section_content)
        
        subdivisions = []
        current_subdivision = ""
        current_word_count = 0
        target_word_count = max(150, len(section_content.split()) // self.target_subdivisions) if self.target_subdivisions else 150
        
        # Procesar párrafos y sus separadores
        i = 0
        while i < len(paragraphs):
            # Obtener el párrafo y su separador (si existe)
            paragraph = paragraphs[i]
            separator = paragraphs[i+1] if i+1 < len(paragraphs) else ""
            
            paragraph_word_count = len(paragraph.split())
            
            # Si añadir este párrafo excedería mucho el tamaño objetivo, comenzar nueva subdivisión
            if current_word_count > 0 and current_word_count + paragraph_word_count > target_word_count * 1.5:
                subdivisions.append(current_subdivision.strip())
                current_subdivision = ""
                current_word_count = 0
            
            current_subdivision += paragraph + separator
            current_word_count += paragraph_word_count
            
            # Si alcanzamos aproximadamente el tamaño objetivo, finalizar subdivisión
            if current_word_count >= target_word_count:
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
        for subdivision in subdivisions:
            # Detectar si contiene inicios de lista
            list_items = re.findall(r'^\s*[-*•]\s+|\s*\d+\.\s+', subdivision, re.MULTILINE)
            
            if list_items and len(list_items) < 3:  # Si hay pocos elementos, probablemente esté incompleta
                # Buscar la siguiente subdivisión que podría contener el resto de la lista
                for i, next_sub in enumerate(subdivisions):
                    if next_sub != subdivision and re.findall(r'^\s*[-*•]\s+|\s*\d+\.\s+', next_sub, re.MULTILINE):
                        # Combinar con la siguiente subdivisión que contiene elementos de lista
                        combined = subdivision + "\n\n" + next_sub
                        final_subdivisions.append(combined)
                        subdivisions.remove(next_sub)  # Evitar procesarla de nuevo
                        break
                else:
                    final_subdivisions.append(subdivision)
            else:
                final_subdivisions.append(subdivision)
        
        return final_subdivisions if final_subdivisions else subdivisions

    # Helper method: merge_subdivisions
    def merge_subdivisions(self, subdivisions, target_subdivisions):
        # Calculate the total number of words in all subdivisions
        total_words = sum(len(s.split()) for s in subdivisions)
        # Determine optimal merge size per subdivision
        optimal_merge_size = total_words // target_subdivisions if target_subdivisions > 0 else total_words
        merged = []
        temp = []
        current_words = 0
        for subdivision in subdivisions:
            word_count = len(subdivision.split())
            temp.append(subdivision)
            current_words += word_count
            # Once the accumulated words meet the optimal merge size, merge these subdivisions
            if current_words >= optimal_merge_size:
                merged.append(" ".join(temp))
                temp = []
                current_words = 0
        if temp:
            merged.append(" ".join(temp))
        return merged

    # Helper method: estimate_summary_length
    def estimate_summary_length(self, subdivisions, target_subdivisions):
        # Estimate final summary word count assuming each subdivision is condensed to
        # tokens_per_summary tokens, each token roughly representing words_per_token words.
        # This gives an estimated summary length of:
        estimated_words_per_subdivision = self.tokens_per_summary * self.words_per_token
        return len(subdivisions) * estimated_words_per_subdivision

    # Helper method: format_section_as_list
    def format_section_as_list(self, text):
        # Split text into sentences and format as markdown list items
        sentences = text.split('. ')
        formatted = "\n".join(f"- {s.strip()}" for s in sentences if s.strip())
        return formatted

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