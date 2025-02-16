import asyncio
import hashlib
import io
import json
import math
import os
import re
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
        self.llm_router = LLMRouter(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            deepinfra_api_key=os.getenv("DEEPINFRA_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
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

    async def summarize(self) -> (str, str):
        # Read and (if needed) transcribe file content
        if self.input_type == "audio":
            # Use Lemonfox API for transcription
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
                return cached.get("sanitized_text", ""), cached.get("dev_text", "")

        # Divide text into sections (using the original section dividing logic)
        if self.is_markdown(file_content):
            self.sections = self.divide_markdown(file_content)
        else:
            self.sections = [{"header": "", "content": file_content}]
        self.sections = self.adjust_sections(self.sections)

        # Process sections in batches of 5
        batched_sections = [self.sections[i:i + 5] for i in range(0, len(self.sections), 5)]
        sanitized_batches = []
        dev_info = {"generated_at": datetime.utcnow().isoformat(), "sections": []}

        for batch in batched_sections:
            batch_summaries = await self._summarize_batch(batch)
            sanitized_batches.append(" ".join([sec["content"] for sec in batch_summaries]))
            dev_info["sections"].extend(batch_summaries)

        sanitized_text = "\n\n".join(sanitized_batches)
        dev_text = json.dumps(dev_info, indent=2, ensure_ascii=False)

        # If caching is enabled, store result
        if self.caching:
            set_cached_summary(file_hash, {"sanitized_text": sanitized_text, "dev_text": dev_text})

        return sanitized_text, dev_text

    async def _summarize_batch(self, batch_sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        batch_summaries = []
        previous_summary = ""
        for idx, section in enumerate(batch_sections):
            summary = await self.ask_for_summary(section["content"])
            # For all but the first section, adjust the beginning for a seamless transition.
            if idx > 0:
                summary = await self.rewrite_transition(previous_summary, summary)
            # Capture section tag – if header exists, use it; otherwise use section number.
            tag = section.get("header", "").strip() or f"Section {idx+1}"
            # Also store first and last 20 words of the summary for developer info.
            words = summary.split()
            first_20 = " ".join(words[:20])
            last_20 = " ".join(words[-20:])
            batch_summaries.append({
                "tag": tag,
                "content": summary,
                "first_20_words": first_20,
                "last_20_words": last_20,
            })
            previous_summary = summary
        return batch_summaries

    async def rewrite_transition(self, prev_summary: str, curr_summary: str) -> str:
        # Get last 20 words of previous summary and first 20 words of current summary
        prev_context = " ".join(prev_summary.split()[-20:])
        curr_start = " ".join(curr_summary.split()[:20])
        # Craft the prompt in the same style as original prompts
        prompt = (
            f"Rewrite the following first 20 words of the current section so that it seamlessly follows the previous section. "
            f"Do not change the meaning, just make the transition smooth.\n\n"
            f"Previous section ending: {prev_context}\n"
            f"Current section starting: {curr_start}\n\n"
            f"Rewritten beginning (first 20 words):"
        )
        messages = [{"role": "user", "content": prompt}]
        system_prompt = (
            "You are a helpful and knowledgeable assistant specialized in summarizing texts. "
            "Ensure that the rewritten text is seamless and stylistically consistent. "
            "Answer in markdown."
        )
        rewritten = await self.llm_router.generate(
            model=self.model,
            messages=messages,
            max_tokens=100,
            temperature=self.temperature,
            top_p=0.9,
            stop_sequences=["User:", "Human:", "Assistant:"],
            system=system_prompt
        )
        # Replace the beginning of the current summary with the rewritten text (preserving the rest)
        curr_words = curr_summary.split()
        new_beginning = rewritten.strip()
        # If the rewritten text does not have 20 words, fallback to original.
        if len(new_beginning.split()) < 15:
            return curr_summary
        new_summary = new_beginning + " " + " ".join(curr_words[20:])
        return new_summary

    async def ask_for_summary(self, text: str) -> str:
        # Instead of using last 200 words context, we do not include them.
        # We generate a structured outline as in the original.
        structured_outline = await self.generate_structured_outline(text)
        messages = [
            {
                "role": "user",
                "content": (
                    f"## Structured Outline: {structured_outline}\n\n"
                    f"## Text to summarize: {text}\n\n"
                    f"## Instructions: Summarize the text above. Maintain the most important information and ensure the summary is substantive. "
                    f"Do not write lists or numbered items unless they are explicitly written in the given text. Write a cohesive text. Output in markdown."
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
            "that that would be used in academia"
        )
        max_tokens = self.tokens_per_summary * 2 if self.easy else self.tokens_per_summary
        # Preserve slight randomness as in original
        import random
        max_tokens = int(max_tokens * (1 + (random.uniform(-0.1, 0.1))))
        system_prompt = (
            f"You are a helpful and knowledgeable assistant specialized in summarizing texts. "
            f"You will understand the given text and its underlying syntax, which might not be evident. "
            f"You will keep the most essential information and optimize for space, this may include specific names, theories, dates, lists and definitions. "
            f"Follow the provided structured outline to maintain the organization of ideas, you are only talking about the text provided, so you must not mention anything outside of it or in the outline, "
            f"this is given to you so that you know what not to write about. You must answer in precise and technically perfect Spanish, {easy_to_understand_text}. "
            f"You must make no mention to the fact that you are summarizing anything or that you've been given any text but rather you must write the text as one that exists by itself. "
            f"You must not reference the document. Keep in mind the context if provided and then make a seamless transition and do not repeat yourself if possible. "
            f"You must do all of this in markdown. Remember to answer in {language}. Take a deep breath and think step by step."
        )
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
        # Use a cache file for outlines (same logic as original)
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
