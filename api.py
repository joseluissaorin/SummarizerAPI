import io
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional

from summarizer_core import FastAPISummarizer
from lemonfox_whisper import transcribe_audio
from docx_converter import MarkdownToDocxConverter
from ocr import get_pdf_page_count, custom_system_prompt  # See ocr.py for details
from ocr_processor import OCRProcessor  # Import OCRProcessor class
from pyzerox import zerox  # For OCR processing
import asyncio
import glob

router = APIRouter()


@router.post("/summarize", summary="Summarize uploaded files")
async def summarize(
    files: List[UploadFile] = File(..., description="One or more text/audio files"),
    input_type: str = Form("file", description="Type of input: file, url, or audio"),
    summary_length: str = Form("medium", description="Predefined summary length (nano, micro, very_short, shorter, short, medium_short, medium, medium_long, long)"),
    custom_percentage: Optional[float] = Form(None, description="Custom summary percentage (0.0-1.0) – if provided, overrides predefined lengths"),
    lang: str = Form("es", description="Language for summarization/transcription"),
    easy: bool = Form(False, description="Generate easy-to-understand summaries"),
    model: str = Form("gemini-2.0-flash-exp", description="LLM model to use for summarization"),
    caching: bool = Form(False, description="Enable caching for summarization"),
):
    results = []
    for uploaded_file in files:
        try:
            content = await uploaded_file.read()
            file_bytes = io.BytesIO(content)
            summarizer = FastAPISummarizer(
                file_stream=file_bytes,
                input_filename=uploaded_file.filename,
                input_type=input_type,
                summary_length=summary_length,
                custom_percentage=custom_percentage,
                lang=lang,
                easy=easy,
                model=model,
                caching=caching,
            )
            sanitized_text, dev_text = await summarizer.summarize()
            results.append({
                "filename": uploaded_file.filename,
                "sanitized_text": sanitized_text,
                "dev_text": dev_text,
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content={"results": results})


@router.post("/transcribe", summary="Transcribe audio using Lemonfox Whisper")
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    lang: str = Form("es", description="Language code (default: es)")
):
    try:
        content = await file.read()
        file_bytes = io.BytesIO(content)
        transcription = await transcribe_audio(file_bytes, lang)
        return JSONResponse(content={"filename": file.filename, "transcription": transcription})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr", summary="Perform OCR on a document (PDF/image) using pyzerox")
async def ocr_endpoint(
    file: UploadFile = File(..., description="PDF or image file to perform OCR on"),
    model: str = Form("gemini-2.0-flash-lite-preview-02-05", description="Vision model to use for OCR")
):
    tmp_path = None
    try:
        # Save the uploaded file to a temporary file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Use the shared OCR processor
        processor = OCRProcessor(model=model)
        results = await processor.process_file(tmp_path)

        if not results["success"]:
            raise HTTPException(
                status_code=500,
                detail=results["error"] or "OCR processing failed"
            )

        return JSONResponse(content={
            "filename": file.filename,
            "ocr_result": results["text"],
            "pages_processed": results["pages_processed"],
            "processing_time": results["processing_time"]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary input file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@router.post("/convert-markdown-to-docx", summary="Convert a Markdown file to DOCX")
async def convert_markdown_to_docx(
    file: UploadFile = File(..., description="Markdown file to convert to DOCX")
):
    try:
        content = await file.read()
        # Save uploaded Markdown file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_md:
            tmp_md.write(content)
            tmp_md_path = tmp_md.name

        # Create a temporary directory for output
        output_dir = tempfile.mkdtemp()
        converter = MarkdownToDocxConverter(input_path=tmp_md_path, output_folder=output_dir)
        converter.convert()

        # The converter creates a zip file of the converted DOCX files.
        # We assume the zip file is in the output directory and its name is based on the output folder.
        import glob
        zip_files = glob.glob(os.path.join(output_dir, "*.zip"))
        if not zip_files:
            raise HTTPException(status_code=500, detail="Conversion failed; no ZIP file generated.")
        zip_file_path = zip_files[0]
        # Return the zip file as a downloadable response.
        return FileResponse(path=zip_file_path, filename=os.path.basename(zip_file_path), media_type="application/zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
