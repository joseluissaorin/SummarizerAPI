from fastapi import FastAPI
from api import router as summarizer_router

app = FastAPI(
    title="Summarizer API",
    description="An API for summarizing text, transcribing audio, performing OCR and converting Markdown to Word.",
    version="1.0.0"
)

app.include_router(summarizer_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
