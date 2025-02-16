# Summarizer API

The Summarizer API is a robust, feature-rich service built with FastAPI that provides comprehensive text and document processing capabilities. With this API, you can:

- **Summarize** long-form content (e.g., articles, research papers, reports) into concise, refined summaries.
- **Transcribe** audio files using the Lemonfox API for Whisper.
- **Perform OCR** on PDFs and images, converting them into Markdown.
- **Convert Markdown to DOCX**, generating Word documents packaged as ZIP files.

In addition to returning a polished summary ("Sanitized Text"), the API also provides detailed developer metadata ("Dev Text") that includes section tags, boundaries, and the first/last 20 words of each section. This metadata makes it easy to extract and further process specific sections.

---

## Table of Contents

- [Features](#features)
- [Documentation](#documentation)
- [Endpoints](#endpoints)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the API](#running-the-api)
  - [Example Requests](#example-requests)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Summarization**: Condenses long documents (6K-10K words) into summaries with seamless section transitions.  
- **Audio Transcription**: Converts audio files to text using the Lemonfox API.  
- **OCR**: Extracts text from PDFs and images, returning the output as Markdown.  
- **Markdown to DOCX Conversion**: Transforms Markdown files into Word documents (DOCX) and packages them into ZIP files for download.  
- **Caching**: Optionally caches summarization results based on file content, reducing redundant processing.  
- **Fully Configurable**: All parameters (input type, summary length, custom percentage, language, style, model, etc.) can be adjusted via form-data fields.

---

## Documentation

For detailed technical documentation, including internal logic, workflow, and comprehensive examples, please refer to our [Technical Documentation](DOCUMENTATION.md).

---

## Endpoints

### POST `/api/summarize`
- **Description**: Generates a summary from one or more files.
- **Outputs**:
  - **Sanitized Text**: The final, polished summary.
  - **Dev Text**: Detailed metadata for each summarized section.

### POST `/api/transcribe`
- **Description**: Transcribes an audio file using the Lemonfox API for Whisper.

### POST `/api/ocr`
- **Description**: Performs OCR on PDF or image files and returns the extracted text in Markdown format.

### POST `/api/convert-markdown-to-docx`
- **Description**: Converts a Markdown file to a DOCX file and returns a ZIP file containing the converted document.

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/summarizer-api.git
   cd summarizer-api
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Before running the API, set the following environment variables:

- **LLM API Keys** (for Anthropic, OpenAI, DeepInfra, Gemini, etc.):
  - `ANTHROPIC_API_KEY`
  - `OPENAI_API_KEY`
  - `DEEPINFRA_API_KEY`
  - `GEMINI_API_KEY`
  - `GROQ_API_KEY`
  - `DEEPSEEK_API_KEY`
- **Lemonfox API Settings**:
  - `LEMONFOX_API_URL` (default: `https://api.lemonfox.ai/whisper`)
  - `LEMONFOX_API_KEY`

You can create a `.env` file in the root directory with the required variables. For example:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
DEEPINFRA_API_KEY=your_deepinfra_api_key
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
LEMONFOX_API_URL=https://api.lemonfox.ai/whisper
LEMONFOX_API_KEY=your_lemonfox_api_key
```

---

## Usage

### Running the API

Start the API server with Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### Example Requests

#### Summarization Example (Python)

```python
import requests

url = "http://localhost:8000/api/summarize"

files = {
    "files": open("example.txt", "rb")
}

data = {
    "input_type": "file",
    "summary_length": "medium",
    "lang": "en",
    "easy": "false",
    "model": "gemini-2.0-flash-exp",
    "caching": "true"
}

response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    result = response.json()
    print("Sanitized Text:")
    print(result["results"][0]["sanitized_text"])
    print("\nDeveloper Text:")
    print(result["results"][0]["dev_text"])
else:
    print(f"Error: {response.status_code} - {response.text}")
```

#### Transcription Example

```python
import requests

url = "http://localhost:8000/api/transcribe"

files = {"file": open("meeting_audio.mp3", "rb")}
data = {"lang": "en"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

#### OCR Example

```python
import requests

url = "http://localhost:8000/api/ocr"

files = {"file": open("document.pdf", "rb")}
data = {"model": "gpt-4o"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

#### Markdown-to-DOCX Conversion Example

```python
import requests

url = "http://localhost:8000/api/convert-markdown-to-docx"

files = {"file": open("blog_post.md", "rb")}

response = requests.post(url, files=files)
if response.ok:
    with open("converted_doc.zip", "wb") as f:
        f.write(response.content)
else:
    print("Conversion failed:", response.text)
```

---

## Directory Structure

```
summarizer-api/
├── main.py                     # FastAPI entry point
├── api.py                      # API route definitions
├── summarizer_core.py          # Core summarization logic
├── lemonfox_whisper.py         # Audio transcription using Lemonfox API
├── docx_converter.py           # Markdown-to-DOCX conversion functionality
├── ocr.py                      # OCR functionality using pyzerox and PyPDF2
├── cache.py                    # Caching logic for summarization results
├── llmrouter.py                # LLM Router integrating various LLM providers
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation (this file)
```

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

---

## License

This project is licensed under the [MIT License](LICENSE).

---