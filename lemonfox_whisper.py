from dotenv import load_dotenv
load_dotenv()

import os
import httpx
import asyncio

# Use the correct endpoint per Lemonfox docs.
LEMONFOX_API_URL = os.getenv("LEMONFOX_API_URL", "https://api.lemonfox.ai/v1/audio/transcriptions")
LEMONFOX_API_KEY = os.getenv("LEMONFOX_API_KEY", "your_lemonfox_api_key_here")

async def transcribe_audio(file_stream, language: str = "english") -> str:
    """
    Transcribe an audio file using the Lemonfox API for Whisper.
    Expects a file-like object (e.g. io.BytesIO).
    """
    files = {
        "file": ("audio_file", file_stream, "application/octet-stream")
    }
    data = {"language": language, "response_format": "json"}
    headers = {"Authorization": f"Bearer {LEMONFOX_API_KEY}"}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            LEMONFOX_API_URL, 
            data=data, 
            files=files, 
            headers=headers, 
            timeout=300
        )
        response.raise_for_status()
        json_response = response.json()
        return json_response.get("text", "")
