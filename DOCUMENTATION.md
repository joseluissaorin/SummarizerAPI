# Summarizer API Documentation

The Summarizer API is a robust, newly created service built with FastAPI that provides comprehensive text, audio, and document processing capabilities. It enables you to generate refined summaries, transcribe audio files, perform OCR on documents, and convert Markdown files into Word documents (DOCX). For summarization requests, the API returns two outputs:

- **Sanitized Text**: A polished, final summary.
- **Developer Text**: Detailed metadata for each summarized section—including tags, boundaries, and the first/last 20 words—to facilitate targeted inspection and extraction.

Additional capabilities include:
- **Audio Transcription**: Utilizing the Lemonfox API for Whisper to transcribe audio files.
- **Seamless Section Transitions**: Processing text in batches of 5 sections with LLM-powered rewriting to ensure smooth transitions between sections.
- **Caching**: Optionally cache results based on file content for faster repeat requests.
- **Complete Parameter Control**: Adjust every parameter (summary length, custom percentages, language, model selection, style, etc.) via form-data fields.

---

## Table of Contents

- [Endpoint Overview](#endpoint-overview)
- [Parameters Description](#parameters-description)
- [Internal Logic and Workflow](#internal-logic-and-workflow)
- [Examples](#examples)
  - [Python Example](#python-example)
  - [JavaScript Example (Next.js)](#javascript-example-nextjs)
  - [TypeScript Example (Next.js)](#typescript-example-nextjs)
- [Response Format](#response-format)
- [Additional Notes](#additional-notes)
- [Summary Length Control Features](#summary-length-control-features)
- [Key Length Control Features](#key-length-control-features)
- [Debug Mode](#debug-mode)
- [Length Control Parameters](#length-control-parameters)
- [Technical Details](#technical-details)

---

## Endpoint Overview

### POST `/api/summarize`

Generates a summary from one or more files and returns both the refined summary and detailed developer metadata.

- **URL**: `/api/summarize`
- **Method**: `POST`
- **Content Type**: `multipart/form-data`

### POST `/api/transcribe`

Transcribes an audio file using the Lemonfox API for Whisper.

- **URL**: `/api/transcribe`
- **Method**: `POST`
- **Content Type**: `multipart/form-data`

### POST `/api/ocr`

Performs OCR on PDF or image files, converting the extracted text into Markdown format.

- **URL**: `/api/ocr`
- **Method**: `POST`
- **Content Type**: `multipart/form-data`

### POST `/api/convert-markdown-to-docx`

Converts a Markdown file to a Word document (DOCX) and returns a ZIP file containing the converted document.

- **URL**: `/api/convert-markdown-to-docx`
- **Method**: `POST`
- **Content Type**: `multipart/form-data`

---

## Parameters Description

All parameters are passed as form-data fields. Below is a detailed breakdown:

| Parameter             | Type              | Default Value               | Description                                                                                                                                                                                                   |
|-----------------------|-------------------|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `files`               | File (one or more)| **Required**                | Files to be processed. Accepts plain text, markdown, or audio files (when `input_type` is set to `"audio"`).                                                                                                  |
| `input_type`          | String            | `"file"`                    | Type of input. Options: `"file"`, `"url"`, or `"audio"`. When set to `"audio"`, the file is transcribed using the Lemonfox API.                                                                                |
| `summary_length`      | String            | `"medium"`                  | Predefined summary length. Options include: `nano`, `micro`, `very_short`, `shorter`, `short`, `medium_short`, `medium`, `medium_long`, `long`. Determines the proportion of text to be summarized.      |
| `custom_percentage`   | Float             | `None` (optional)           | Custom summary length as a percentage (0.0 to 1.0). If provided, it overrides the predefined `summary_length`. For instance, `0.5` aims for a summary that is 50% of the original text's length.           |
| `lang`                | String            | `"es"`                      | Language code for summarization or transcription. Examples: `"es"`, `"en"`, `"fr"`, `"it"`.                                                                                                                   |
| `easy`                | Boolean           | `False`                     | If `true`, generates a summary in an easy-to-understand style with detailed explanations.                                                                                                                    |
| `model`               | String            | `"gemini-2.0-flash-exp"`    | Specifies the LLM model to use for processing. Options include models such as Claude, GPT, Gemini, etc. Internal parameters (words-per-token ratio, temperature) are adjusted accordingly.                  |
| `caching`             | Boolean           | `False`                     | When enabled, caches the summarization result based on file content, ensuring faster responses for repeated requests with identical content.                                                                 |
| `debug`               | Boolean           | `False`                     | When enabled, provides detailed logging about the length control process and includes comprehensive diagnostics in the developer output.                                                                      |

Endpoints dedicated to transcription, OCR, or Markdown conversion require only the relevant fields.

---

## Internal Logic and Workflow

1. **Input Processing**:
   - For **audio files** (when `input_type` is `"audio"`), the file is transcribed using the Lemonfox API.
   - For **text files**, the content is read with appropriate encoding (UTF-8 with fallbacks).

2. **Section Division and Batching**:
   - **Sectioning**: Text is divided into sections based on Markdown headers and list patterns (or treated as a single section for plain text).
   - **Subdivision**: Sections exceeding 4000 words are split into smaller subsections.
   - **Batching**: Sections are processed in batches of 5. For every section in a batch (except the first), the first 20 words are rewritten by the LLM to ensure smooth transitions from the previous section.

3. **LLM-Based Summarization**:
   - A structured outline is generated for each section.
   - The summary is generated with a detailed prompt that ensures the output meets specific style requirements.
   - **Repetition evaluation is disabled by default.**

4. **Output Generation**:
   - **Sanitized Text**: A concatenated, polished summary of the document.
   - **Developer Text**: A JSON-formatted string containing metadata for each section, including tags, and the first and last 20 words for easy reference.

5. **Caching**:
   - When enabled, a hash of the file content is computed to store and retrieve cached results, reducing redundant processing.

6. **Additional Endpoints**:
   - **Transcription**: Directly returns the transcription of an audio file.
   - **OCR**: Extracts text from PDFs or images using OCR and returns the result in Markdown.
   - **Markdown-to-DOCX Conversion**: Converts Markdown files into DOCX format and packages them into a ZIP file for download.

---

## Examples

### Python Example

Below is an example of how to call the Summarizer API using Python's `requests` library:

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

### JavaScript Example (Next.js)

Using Next.js with JavaScript, you can call the API as follows:

```javascript
import React, { useState } from 'react';

export default function ApiClient() {
  const [result, setResult] = useState(null);

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('files', file);
    formData.append('input_type', 'file');
    formData.append('summary_length', 'medium');
    formData.append('lang', 'en');
    formData.append('easy', 'false');
    formData.append('model', 'gemini-2.0-flash-exp');
    formData.append('caching', 'true');

    const response = await fetch('http://localhost:8000/api/summarize', {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      setResult(data.results[0]);
    } else {
      console.error("API Error:", await response.text());
    }
  };

  return (
    <div>
      <h1>Summarizer API Client</h1>
      <input type="file" onChange={handleUpload} />
      {result && (
        <div>
          <h2>Sanitized Text</h2>
          <pre>{result.sanitized_text}</pre>
          <h2>Developer Text</h2>
          <pre>{result.dev_text}</pre>
        </div>
      )}
    </div>
  );
}
```

### TypeScript Example (Next.js)

Below is an example written in TypeScript for a Next.js page:

```typescript
import React, { useState, ChangeEvent } from 'react';

interface ApiResult {
  filename: string;
  sanitized_text: string;
  dev_text: string;
}

const ApiClient: React.FC = () => {
  const [result, setResult] = useState<ApiResult | null>(null);

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) return;
    const file = event.target.files[0];

    const formData = new FormData();
    formData.append('files', file);
    formData.append('input_type', 'file');
    formData.append('summary_length', 'medium');
    formData.append('lang', 'en');
    formData.append('easy', 'false');
    formData.append('model', 'gemini-2.0-flash-exp');
    formData.append('caching', 'true');

    const response = await fetch('http://localhost:8000/api/summarize', {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      setResult(data.results[0]);
    } else {
      console.error("API Error:", await response.text());
    }
  };

  return (
    <div>
      <h1>Summarizer API Client (TypeScript)</h1>
      <input type="file" onChange={handleUpload} />
      {result && (
        <div>
          <h2>Sanitized Text</h2>
          <pre>{result.sanitized_text}</pre>
          <h2>Developer Text</h2>
          <pre>{result.dev_text}</pre>
        </div>
      )}
    </div>
  );
};

export default ApiClient;
```

---

## Response Format

A successful request to the `/api/summarize` endpoint returns a JSON object structured as follows:

```json
{
  "results": [
    {
      "filename": "example.txt",
      "sanitized_text": "This is the final refined summary of the document...",
      "dev_text": "{\n  \"generated_at\": \"2025-02-16T12:34:56.789Z\",\n  \"sections\": [\n    {\n      \"tag\": \"Section 1\",\n      \"content\": \"...\",\n      \"first_20_words\": \"...\",\n      \"last_20_words\": \"...\"\n    },\n    ...\n  ]\n}"
    }
  ]
}
```

- **`filename`**: Name of the original file.
- **`sanitized_text`**: The polished final summary.
- **`dev_text`**: A JSON string containing detailed metadata for each section (tags, first/last 20 words, etc.).

Other endpoints (transcription, OCR, Markdown conversion) return their respective processed outputs—either in JSON format or as a downloadable file (ZIP for DOCX conversion).

---

## Additional Notes

- **Configuration**: Every parameter is configurable via form-data fields, offering granular control over processing behavior.
- **Seamless Transitions**: The API rewrites section transitions in batches of 5 to ensure a cohesive summary.
- **Caching**: When enabled, results are cached based on file content to speed up repeated requests.
- **Extensibility**: The API's modular design facilitates easy integration of new features and endpoints.

This documentation provides all the information needed to integrate and use the Summarizer API across various applications, whether using Python, JavaScript, or TypeScript with Next.js.

# Summary Length Control Features

The API now includes enhanced controls for summary length, ensuring that the generated summaries closely match the desired target length specified by either `summary_length` or `custom_percentage` parameters.

## Key Length Control Features

1. **Improved Length Precision**: 
   - Summaries now more accurately match the requested length percentage
   - Adaptive subdivision process ensures consistent summary length across different document types
   - Proper handling of sentence boundaries to avoid truncated content

2. **Mathematical Length Estimation**: 
   - The system uses a sophisticated algorithm to estimate summary length before processing
   - This predictive approach eliminates the need for trial-and-error summarization
   - Adaptive iteration refines the subdivision strategy as needed

3. **Plain Text Handling**: 
   - Special treatment for plain text documents to ensure accurate length control
   - Enhanced sentence detection to maintain coherent paragraphs
   - Balanced subdivision based on model token characteristics

4. **Diagnostic Information**: 
   - When debug mode is enabled, detailed length control diagnostics are included in the dev_text output
   - This includes target vs. actual length statistics, iteration data, and adjustment information

## Debug Mode

Setting the `debug` parameter to `true` provides enhanced logging and diagnostics during the summarization process. This is particularly useful for:

- Monitoring target vs. actual summary length
- Seeing subdivision adjustments in real-time
- Understanding how the system is calibrating to meet length requirements
- Getting detailed length statistics in the output

## Length Control Parameters

| Parameter             | Type              | Default Value               | Description                                                                                                                                                                                                   |
|-----------------------|-------------------|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `summary_length`      | String            | `"medium"`                  | Predefined summary length. Options include: `nano` (1%), `micro` (5%), `very_short` (10%), `shorter` (15%), `short` (23%), `medium_short` (33%), `medium` (40%), `medium_long` (62%), `long` (80%).      |
| `custom_percentage`   | Float             | `None` (optional)           | Custom summary length as a percentage (0.0 to 1.0). If provided, it overrides the predefined `summary_length`. For instance, `0.5` aims for a summary that is 50% of the original text's length.           |
| `debug`               | Boolean           | `False`                     | When enabled, provides detailed logging about the length control process and includes comprehensive diagnostics in the developer output.                                                                      |

## Technical Details

The length control system works by:

1. Calculating the target word count based on original document length and requested percentage
2. Determining the optimal number of subdivisions needed to achieve that length
3. Splitting content into appropriate subdivisions, respecting sentence boundaries
4. Estimating the expected summary length before processing
5. Iteratively adjusting the subdivision strategy if the estimate is outside acceptable bounds
6. Applying final verification to ensure text coherence and integrity

This approach ensures that the summary closely matches the requested length while maintaining the quality and coherence of the content.
