from pyzerox import zerox
import os
import json
import asyncio
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

custom_system_prompt = """
    Convert the following PDF page to markdown.
    Return only the markdown with no explanation text.
    Do not exclude any content from the page.
    Do not extract the text from the pages that are crossed in red, in a single image, there may only be one section that is crossed in red, extract the text from the remaining parts of the image.
    The images may include diagrams, try to interpret them in plain markwdown or adapt them to a markdown table.
    """

def get_pdf_page_count(file_path: str) -> int:
    try:
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            return len(pdf.pages)
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return 0

# (The actual OCR processing is handled via the async call to zerox in the API endpoint.)
