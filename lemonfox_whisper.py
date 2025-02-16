import os
import httpx
import asyncio
from fastapi import UploadFile

# Replace the URL and headers below with the actual Lemonfox API endpoint details.
LEMONFOX_API_URL = os.getenv("LEMONFOX_API_URL", "https://api.lemonfox.ai/whisper")
LEMONFOX_API_KEY = os.getenv("LEMONFOX_API_KEY", "your_lemonfox_api_key_here")


async def transcribe_audio(file_stream, language: str = "es") -> str:
    """
    Transcribe an audio file using the Lemonfox API for Whisper.
    Expects a file-like object (e.g. io.BytesIO).
    """
    # Build multipart data for the file upload
    files = {
        "file": ("audio_file", file_stream, "application/octet-stream")
    }
    data = {"language": language}
    headers = {"Authorization": f"Bearer {LEMONFOX_API_KEY}"}
    async with httpx.AsyncClient() as client:
        response = await client.post(LEMONFOX_API_URL, data=data, files=files, headers=headers, timeout=300)
        response.raise_for_status()
        json_response = response.json()
        # Assume the API returns JSON with a "text" field for transcription.
        return json_response.get("text", "")
