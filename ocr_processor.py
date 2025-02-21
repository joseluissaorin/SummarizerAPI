import os
import tempfile
import logging
import asyncio
import time
from datetime import datetime
from typing import Optional, Dict, List, Union
from pathlib import Path

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pyzerox import zerox

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OCRProcessor:
    """
    Shared OCR processing class that handles both API and test use cases.
    """
    def __init__(
        self,
        model: str = "gemini-2.0-flash-lite-preview-02-05",
        custom_system_prompt: Optional[str] = None
    ):
        self.model = model
        self.custom_system_prompt = custom_system_prompt or """
            Convert the following PDF page to markdown.
            Return only the markdown with no explanation text.
            Do not exclude any content from the page.
            Do not extract the text from the pages that are crossed in red, in a single image, there may only be one section that is crossed in red, extract the text from the remaining parts of the image.
            The images may include diagrams, try to interpret them in plain markwdown or adapt them to a markdown table.
            Start directly with the markdown, do not include any other text.
        """
        # Set poppler path
        self.poppler_path = "/usr/bin"
        os.environ["PATH"] = f"{os.environ.get('PATH', '')}:/usr/bin"
        # Check dependencies on initialization
        self.check_dependencies()

    @staticmethod
    def check_dependencies():
        """Check for required dependencies and provide installation instructions."""
        missing_deps = []
        installation_instructions = []
        PDFINFO_PATH = "/usr/bin/pdfinfo"

        # Check for PyPDF2
        try:
            import PyPDF2
            logger.info("PyPDF2 is installed")
        except ImportError:
            missing_deps.append("PyPDF2")
            installation_instructions.append("pip install PyPDF2")

        # Check for pdf2image and poppler
        try:
            from pdf2image import pdfinfo_from_path
            # Try to run pdfinfo to check if poppler is installed
            import subprocess
            try:
                if not os.path.exists(PDFINFO_PATH):
                    raise FileNotFoundError(f"pdfinfo not found at {PDFINFO_PATH}")
                subprocess.run([PDFINFO_PATH, '-v'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.info("pdf2image and poppler are installed")
            except FileNotFoundError as e:
                logger.error(f"Poppler error: {e}")
                missing_deps.append("poppler-utils")
                installation_instructions.append("sudo apt-get install -y poppler-utils")
        except ImportError:
            missing_deps.append("pdf2image")
            installation_instructions.append("pip install pdf2image")

        # Check for pikepdf
        try:
            import pikepdf
            logger.info("pikepdf is installed")
        except ImportError:
            missing_deps.append("pikepdf")
            installation_instructions.append("pip install pikepdf")

        if missing_deps:
            warning_msg = (
                f"\nWarning: The following dependencies are missing: {', '.join(missing_deps)}\n"
                "This may affect PDF processing capabilities.\n"
                "To install the missing dependencies, run the following commands:\n"
                f"{chr(10).join(installation_instructions)}\n"
                "For Ubuntu/Debian systems, you might need to run the apt commands with sudo.\n"
                "After installing poppler-utils, you may need to restart your application."
            )
            logger.warning(warning_msg)
            return False
        
        logger.info("All PDF processing dependencies are installed")
        return True

    def get_pdf_info(self, file_path: str) -> Dict:
        """Get information about the PDF file using multiple fallback methods."""
        info = {
            "num_pages": 0,
            "metadata": None,
            "is_encrypted": False
        }
        
        logger.info(f"Attempting to get PDF info for: {file_path}")
        
        # Method 1: Try using pdf2image's get_pdf_info with explicit path
        try:
            import subprocess
            PDFINFO_PATH = "/usr/bin/pdfinfo"
            result = subprocess.run([PDFINFO_PATH, file_path], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse the output to get page count
                for line in result.stdout.split('\n'):
                    if line.startswith('Pages:'):
                        info["num_pages"] = int(line.split(':')[1].strip())
                        logger.info(f"Successfully got PDF info using pdfinfo command: {info}")
                        return info
        except Exception as e:
            logger.warning(f"pdfinfo command failed: {e}")
        
        # Method 2: Try using PyPDF2
        try:
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                info = {
                    "num_pages": len(pdf.pages),
                    "metadata": pdf.metadata,
                    "is_encrypted": pdf.is_encrypted,
                }
                logger.info(f"Successfully got PDF info using PyPDF2: {info}")
                return info
        except Exception as e:
            logger.warning(f"PyPDF2 method failed: {e}")

        # Method 3: Try using pikepdf (if available)
        try:
            import pikepdf
            with pikepdf.open(file_path) as pdf:
                info["num_pages"] = len(pdf.pages)
                info["metadata"] = pdf.docinfo
                logger.info(f"Successfully got PDF info using pikepdf: {info}")
                return info
        except ImportError:
            logger.warning("pikepdf not available")
        except Exception as e:
            logger.warning(f"pikepdf method failed: {e}")

        # Method 4: Assume at least one page if we can open the file
        try:
            with open(file_path, 'rb') as file:
                # Check if it's a PDF by reading the magic number
                if file.read(4) == b'%PDF':
                    info["num_pages"] = 1
                    logger.warning("Could not determine exact page count, assuming 1 page")
                    return info
        except Exception as e:
            logger.error(f"Failed to read file: {e}")

        raise ValueError("Could not determine PDF information using any available method")

    async def process_file(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        cleanup: bool = True
    ) -> Dict:
        """
        Process a file and return OCR results.
        
        Args:
            file_path: Path to the file to process
            output_dir: Optional directory for output files
            cleanup: Whether to clean up temporary files after processing
            
        Returns:
            Dictionary containing:
                - success: bool
                - text: str
                - processing_time: float
                - pages_processed: int
                - error: Optional[str]
        """
        start_time = time.time()
        tmp_output_dir = None
        logger.info(f"Starting OCR process for: {file_path}")

        try:
            # Validate PDF
            pdf_info = self.get_pdf_info(file_path)
            if pdf_info["num_pages"] == 0:
                raise ValueError("PDF has no pages")

            # Create temporary output directory if none provided
            if output_dir is None:
                tmp_output_dir = tempfile.mkdtemp()
                output_dir = tmp_output_dir
                logger.info(f"Created temporary output directory: {output_dir}")

            # Process PDF with zerox
            select_pages = list(range(1, pdf_info["num_pages"] + 1))
            logger.info(f"Processing {len(select_pages)} pages with zerox")

            # Process with zerox
            logger.info(f"Starting zerox processing with model: {self.model}")
            logger.info(f"Using poppler path: {self.poppler_path}")
            ocr_result = await zerox(
                file_path=file_path,
                model=self.model,
                output_dir=output_dir,
                custom_system_prompt=self.custom_system_prompt,
                select_pages=select_pages,
                poppler_path=self.poppler_path  # Add poppler path to zerox call
            )
            logger.info(f"Zerox processing completed. Output directory: {output_dir}")
            logger.info(f"Zerox output type: {type(ocr_result)}")
            logger.info(f"Zerox output attributes: {dir(ocr_result)}")
            logger.info(f"String representation length: {len(str(ocr_result))}")
            logger.info(f"String representation preview: {str(ocr_result)[:200]}")  # First 200 chars

            # Initialize results
            results = {
                "success": False,
                "text": "",
                "processing_time": 0,
                "pages_processed": 0,
                "error": None
            }

            # Try to read from markdown files first
            md_files = sorted(Path(output_dir).glob("*.md"))  # Sort to maintain page order
            logger.info(f"Found {len(md_files)} markdown files in output directory")

            if md_files:
                combined_text = []
                for md_file in md_files:
                    try:
                        with open(md_file, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                combined_text.append(content)
                                logger.info(f"Successfully read content from {md_file}")
                            else:
                                logger.warning(f"Empty content in {md_file}")
                    except Exception as e:
                        logger.error(f"Error reading markdown file {md_file}: {e}")

                if combined_text:
                    results["text"] = "\n\n".join(combined_text)
                    results["success"] = True
                    results["pages_processed"] = len(md_files)
                    logger.info(f"Successfully extracted text from {len(md_files)} markdown files")
                else:
                    logger.warning("No content found in any markdown files")

            # If no markdown content, try to extract from zerox output object
            if not results["text"] and ocr_result:
                logger.info("No markdown content found, falling back to zerox output object")
                try:
                    if hasattr(ocr_result, 'pages') and ocr_result.pages:
                        # Extract text from pages
                        text_content = []
                        for i, page in enumerate(ocr_result.pages):
                            logger.info(f"Processing page {i+1}")
                            if hasattr(page, 'text') and page.text:
                                text_content.append(page.text.strip())
                                logger.info(f"Successfully extracted text from page {i+1}")
                            else:
                                logger.warning(f"No text attribute or empty text in page {i+1}")
                        
                        if text_content:
                            results["text"] = "\n\n".join(text_content)
                            results["success"] = True
                            results["pages_processed"] = len(text_content)
                            logger.info(f"Successfully extracted text from {len(text_content)} pages in zerox output")
                        else:
                            logger.warning("No text content found in any pages")
                    elif hasattr(ocr_result, 'text') and ocr_result.text:
                        # Direct text attribute
                        results["text"] = ocr_result.text.strip()
                        results["success"] = True
                        results["pages_processed"] = pdf_info["num_pages"]
                        logger.info("Successfully extracted text from zerox output text attribute")
                    else:
                        # Try to get something useful from string representation
                        str_repr = str(ocr_result).strip()
                        if str_repr and str_repr != "None":
                            results["text"] = str_repr
                            results["success"] = True
                            results["pages_processed"] = pdf_info["num_pages"]
                            logger.info("Successfully extracted text from zerox string representation")
                        else:
                            logger.warning("No usable text found in zerox output")
                except Exception as e:
                    logger.error(f"Error extracting text from zerox output: {e}")
                    results["error"] = f"Failed to extract text from zerox output: {str(e)}"

            # Calculate processing time
            results["processing_time"] = time.time() - start_time
            
            if not results["text"]:
                error_msg = "No text extracted from the document"
                logger.error(error_msg)
                results["error"] = error_msg
            else:
                logger.info(f"Successfully processed document in {results['processing_time']:.2f} seconds")
            
            return results

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in OCR process: {error_msg}")
            return {
                "success": False,
                "text": "",
                "processing_time": time.time() - start_time,
                "pages_processed": 0,
                "error": error_msg
            }

        finally:
            # Clean up temporary output directory if we created it
            if cleanup and tmp_output_dir and os.path.exists(tmp_output_dir):
                try:
                    for file in Path(tmp_output_dir).glob("*"):
                        try:
                            os.remove(file)
                            logger.info(f"Cleaned up file: {file}")
                        except Exception as e:
                            logger.error(f"Error removing file {file}: {e}")
                    os.rmdir(tmp_output_dir)
                    logger.info(f"Cleaned up temporary directory: {tmp_output_dir}")
                except Exception as e:
                    logger.error(f"Error cleaning up temporary directory: {e}")

    @staticmethod
    async def process_url(url: str) -> Dict:
        """
        Process a file from a URL.
        Downloads the file to a temporary location and processes it.
        """
        import requests
        logger.info(f"Processing URL: {url}")
        
        try:
            # Download the file
            response = requests.get(url)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
                logger.info(f"Downloaded file saved to: {tmp_path}")
            
            try:
                # Process the file
                processor = OCRProcessor()
                results = await processor.process_file(tmp_path)
                return results
                
            finally:
                # Clean up temporary file
                try:
                    os.remove(tmp_path)
                    logger.info(f"Cleaned up temporary file: {tmp_path}")
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {e}")
                    
        except Exception as e:
            error_msg = f"Error processing URL {url}: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "text": "",
                "processing_time": 0,
                "pages_processed": 0,
                "error": error_msg
            } 