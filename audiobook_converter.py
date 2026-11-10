#!/usr/bin/env python3
"""
Qwen-Based Audiobook Converter
Converts PDFs, EPUBs, DOCX, DOC, TXT files into audiobooks using Qwen Voice API

Author: Rewritten for Qwen Voice Model
License: MIT
"""

import os
import shutil
import logging
import hashlib
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys
import zipfile
import xml.etree.ElementTree as ET
from html import unescape
import re
from datetime import datetime
import PyPDF2
import ebooklib
from ebooklib import epub
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from gradio_client import Client, handle_file

# Fix Windows console encoding for emoji/unicode
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# =============================================================================
# HARDCODED CONFIGURATION
# =============================================================================

# Qwen API Configuration
QWEN_API_URL = "http://127.0.0.1:7860"
API_TIMEOUT = 300
MAX_RETRIES = 3

# Hardcoded Voice Settings (Always use 1.7B model)
CUSTOM_VOICE_SPEAKER = "Ryan"
CUSTOM_VOICE_LANGUAGE = "English"
CUSTOM_VOICE_INSTRUCT = "Speak naturally and clearly, as if reading a dramatic book to an adult audience."
CUSTOM_VOICE_MODEL_SIZE = "1.7B"  # Always use 1.7B
CUSTOM_VOICE_SEED = -1
CUSTOM_VOICE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Voice Clone Settings (Always use 1.7B model)
VOICE_CLONE_LANGUAGE = "English"
VOICE_CLONE_USE_XVECTOR_ONLY = False
VOICE_CLONE_MODEL_SIZE = "1.7B"  # Always use 1.7B
VOICE_CLONE_MAX_CHUNK_CHARS = 200
VOICE_CLONE_CHUNK_GAP = 0
VOICE_CLONE_SEED = -1

# Processing Settings
BOOKS_FOLDER = "book_to_convert"  # Input folder
AUDIOBOOKS_FOLDER = "audiobooks"  # Output folder
CHUNK_SIZE_WORDS = 1500  # Increased to reduce number of chunks and speed up processing
MAX_WORKERS = 1  # Keep at 1 to avoid rate limiting
AUDIO_FORMAT = "mp3"
AUDIO_BITRATE = "128k"
MIN_DELAY_BETWEEN_CHUNKS = 1  # Reduced delay

# Optional imports with fallbacks
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import docx2txt
    DOC_AVAILABLE = True
except ImportError:
    DOC_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class QwenAudiobookConverter:
    """Audiobook converter using Qwen Voice API"""

    def __init__(self, voice_mode: str = "custom_voice", voice_clone_ref_audio: Optional[str] = None):
        self.voice_mode = voice_mode
        self.voice_clone_ref_audio = voice_clone_ref_audio
        self.voice_clone_ref_text = ""
        self.setup_logging()
        self.setup_directories()
        self.validate_configuration()
        self.client = None
        self.api_info: Dict[str, Any] = {}
        self.init_qwen_client()

    def setup_logging(self):
        """Setup logging configuration"""
        Path("logs").mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/audiobook_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create necessary directories"""
        directories = [BOOKS_FOLDER, AUDIOBOOKS_FOLDER, "chunks", "cache/audio_chunks", "logs"]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file using Qwen's Whisper transcription"""
        transcribe_api = self._resolve_api_name("/transcribe_audio", "/run_transcribe_audio")
        try:
            self.logger.info(f"Transcribing audio: {audio_path}")
            result = self.client.predict(
                audio=handle_file(audio_path),
                api_name=transcribe_api
            )
            transcribed_text = result if isinstance(result, str) else str(result)
            self.logger.info(f"Transcription complete: {transcribed_text[:100]}...")
            return transcribed_text.strip()
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise

    def validate_configuration(self):
        """Validate configuration settings"""
        if self.voice_mode == "voice_clone":
            if not self.voice_clone_ref_audio:
                print("[ERROR] Configuration Error!")
                print("Voice Clone mode requires a reference audio file.")
                print("Use --voice-sample <path> to specify the reference audio.")
                sys.exit(1)
            
            if not Path(self.voice_clone_ref_audio).exists():
                print("[ERROR] Configuration Error!")
                print(f"Reference audio file not found: {self.voice_clone_ref_audio}")
                sys.exit(1)
            
            # Transcribe the audio if client is available (will be done after init)
            # For now, we'll transcribe it in init_qwen_client if needed

    def init_qwen_client(self):
        """Initialize Qwen Gradio client"""
        try:
            self.logger.info(f"Connecting to Qwen API at {QWEN_API_URL}...")
            # Suppress gradio_client's print statements that cause encoding issues on Windows
            import io
            old_stdout = sys.stdout
            sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding='utf-8', errors='replace')
            try:
                self.client = Client(QWEN_API_URL)
            finally:
                sys.stdout = old_stdout
            self.api_info = self._load_api_info()
            self.logger.info("Connected to Qwen API")
            print("[OK] Connected to Qwen API")
            
            # If voice clone mode, transcribe the reference audio
            if self.voice_mode == "voice_clone" and self.voice_clone_ref_audio:
                print("[INFO] Transcribing reference audio for voice cloning...")
                self.voice_clone_ref_text = self.transcribe_audio(self.voice_clone_ref_audio)
                print(f"[OK] Transcription: {self.voice_clone_ref_text[:100]}...")
        except Exception as e:
            print("[ERROR] Qwen API initialization failed!")
            print(f"API endpoint: {QWEN_API_URL}")
            print("Make sure:")
            print("1. Qwen Gradio server is running")
            print("2. The server is accessible at the configured URL")
            print("3. The endpoint URL is correct")
            print("4. Your installed Qwen3-TTS version matches this converter's API expectations")
            print(f"Error: {e}")
            sys.exit(1)

    def _load_api_info(self) -> Dict[str, Any]:
        """Load available API metadata from Gradio app."""
        try:
            return self.client.view_api(return_format="dict")
        except Exception as exc:
            self.logger.warning(f"Unable to read API metadata: {exc}")
            return {}

    def _resolve_api_name(self, *candidates: str) -> str:
        """Return the first available api_name from candidate list."""
        named_endpoints = self.api_info.get("named_endpoints", {})
        for candidate in candidates:
            if candidate in named_endpoints:
                return candidate
        return candidates[0]

    def _endpoint_accepts_param(self, api_name: str, param_name: str) -> bool:
        """Check whether endpoint input schema includes the given parameter."""
        endpoint = self.api_info.get("named_endpoints", {}).get(api_name, {})
        parameters = endpoint.get("parameters", [])
        return any(parameter.get("parameter_name") == param_name for parameter in parameters)

    def generate_chunk_via_qwen(self, text: str, chunk_num: int) -> Optional[str]:
        """Generate audio chunk using Qwen API"""
        try:
            # Check cache first
            cache_path = self.get_cache_path(text)
            if cache_path.exists():
                output_path = Path("chunks") / f"chunk_{chunk_num:04d}.wav"
                shutil.copy2(cache_path, output_path)
                self.logger.debug(f"Using cached audio for chunk {chunk_num}")
                return str(output_path)

            # Generate audio based on selected mode
            if self.voice_mode == "custom_voice":
                result = self._generate_custom_voice(text)
            elif self.voice_mode == "voice_clone":
                result = self._generate_voice_clone(text)
            else:
                raise ValueError(f"Unknown voice mode: {self.voice_mode}")

            if not result or len(result) < 2:
                raise RuntimeError("Qwen API returned invalid result")

            audio_path = result[0]  # First element is the audio file path
            status = result[1] if len(result) > 1 else ""

            if not audio_path or not Path(audio_path).exists():
                raise RuntimeError(f"Generated audio file not found: {audio_path}")

            # Copy to chunks directory
            output_path = Path("chunks") / f"chunk_{chunk_num:04d}.wav"
            shutil.copy2(audio_path, output_path)

            # Cache the result
            shutil.copy2(output_path, cache_path)

            self.logger.debug(f"Chunk {chunk_num} generated successfully")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Qwen chunk processing failed for chunk {chunk_num}: {e}")
            return None

    def _generate_custom_voice(self, text: str) -> Tuple:
        """Generate audio using CustomVoice mode"""
        custom_api = self._resolve_api_name("/run_custom_voice", "/generate_custom_voice")
        payload = dict(
            text=text,
            language=CUSTOM_VOICE_LANGUAGE,
            speaker=CUSTOM_VOICE_SPEAKER,
            instruct=CUSTOM_VOICE_INSTRUCT,
        )
        if self._endpoint_accepts_param(custom_api, "model_id_cv"):
            payload["model_id_cv"] = CUSTOM_VOICE_MODEL_ID
        elif self._endpoint_accepts_param(custom_api, "model_size"):
            payload["model_size"] = CUSTOM_VOICE_MODEL_SIZE

        if self._endpoint_accepts_param(custom_api, "seed"):
            payload["seed"] = CUSTOM_VOICE_SEED

        return self.client.predict(**payload, api_name=custom_api)

    def _generate_voice_clone(self, text: str) -> Tuple:
        """Generate audio using Voice Clone mode"""
        if not Path(self.voice_clone_ref_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {self.voice_clone_ref_audio}")

        if not self.voice_clone_ref_text:
            raise ValueError("Reference text is required for voice cloning. Transcription may have failed.")

        return self.client.predict(
            ref_audio=handle_file(self.voice_clone_ref_audio),
            ref_text=self.voice_clone_ref_text,
            target_text=text,
            language=VOICE_CLONE_LANGUAGE,
            use_xvector_only=VOICE_CLONE_USE_XVECTOR_ONLY,
            model_size=VOICE_CLONE_MODEL_SIZE,
            max_chunk_chars=VOICE_CLONE_MAX_CHUNK_CHARS,
            chunk_gap=VOICE_CLONE_CHUNK_GAP,
            seed=VOICE_CLONE_SEED,
            api_name="/generate_voice_clone"
        )

    def process_chunk_with_retry(self, args: Tuple[int, str]) -> bool:
        """Process chunk with retry logic and rate limiting"""
        chunk_num, text = args

        # Small delay between chunks to avoid rate limiting (only if not first chunk)
        if chunk_num > 1:
            time.sleep(MIN_DELAY_BETWEEN_CHUNKS)

        for attempt in range(MAX_RETRIES):
            try:
                result = self.generate_chunk_via_qwen(text, chunk_num)
                if result and Path(result).exists():
                    return True
                else:
                    self.logger.warning(f"Chunk {chunk_num} attempt {attempt + 1} failed")
            except Exception as e:
                self.logger.warning(f"Chunk {chunk_num} attempt {attempt + 1} error: {e}")

            if attempt < MAX_RETRIES - 1:
                sleep_time = 5 + (2 ** attempt)
                self.logger.info(f"Waiting {sleep_time}s before retry...")
                time.sleep(sleep_time)

        self.logger.error(f"Chunk {chunk_num} failed after {MAX_RETRIES} attempts")
        return False

    def get_cache_path(self, text: str) -> Path:
        """Get cache path for text chunk"""
        content = f"{text}_{self.voice_mode}_{CUSTOM_VOICE_SPEAKER if self.voice_mode == 'custom_voice' else Path(self.voice_clone_ref_audio).name if self.voice_clone_ref_audio else ''}"
        hash_obj = hashlib.md5(content.encode())
        return Path("cache/audio_chunks") / f"{hash_obj.hexdigest()}.wav"

    def extract_text_from_epub(self, file_path: Path) -> str:
        """Extract text from EPUB with fallback methods"""
        methods = [
            self._extract_epub_ebooklib,
            self._extract_epub_zipfile,
            self._extract_epub_manual
        ]

        for method in methods:
            try:
                text = method(file_path)
                if text and text.strip():
                    self.logger.info(f"EPUB extraction successful: {len(text)} characters")
                    return text
            except Exception as e:
                self.logger.warning(f"EPUB method failed: {e}")
                continue

        raise RuntimeError("All EPUB extraction methods failed")

    def _extract_epub_ebooklib(self, file_path: Path) -> str:
        """Extract using ebooklib"""
        book = epub.read_epub(str(file_path))
        text_parts = []

        for item_id, linear in book.spine:
            try:
                item = book.get_item_by_id(item_id)
                if item and isinstance(item, ebooklib.ITEM_DOCUMENT):
                    content = item.get_body_content()
                    if content:
                        if isinstance(content, bytes):
                            content = content.decode('utf-8', errors='ignore')
                        clean_text = self._clean_html(str(content))
                        if clean_text.strip():
                            text_parts.append(clean_text)
            except Exception:
                continue

        return '\n\n'.join(text_parts)

    def _extract_epub_zipfile(self, file_path: Path) -> str:
        """Extract using zipfile parsing"""
        text_parts = []
        with zipfile.ZipFile(file_path, 'r') as epub_zip:
            for file_name in epub_zip.namelist():
                if file_name.lower().endswith(('.html', '.xhtml', '.htm')):
                    try:
                        content = epub_zip.read(file_name).decode('utf-8', errors='ignore')
                        clean_text = self._clean_html(content)
                        if clean_text.strip():
                            text_parts.append(clean_text)
                    except Exception:
                        continue
        return '\n\n'.join(text_parts)

    def _extract_epub_manual(self, file_path: Path) -> str:
        """Manual extraction fallback"""
        text_parts = []
        with zipfile.ZipFile(file_path, 'r') as epub_zip:
            for file_name in epub_zip.namelist():
                if not any(file_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.css', '.js']):
                    try:
                        content = epub_zip.read(file_name).decode('utf-8', errors='ignore')
                        if '<' in content and len(content.strip()) > 100:
                            clean_text = self._clean_html(content)
                            if clean_text:
                                text_parts.append(clean_text)
                    except Exception:
                        continue
        return '\n\n'.join(text_parts)

    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content"""
        if not html_content:
            return ""

        if BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                return ' '.join(chunk for chunk in chunks if chunk)
            except Exception:
                pass

        # Fallback regex cleaning
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<[^>]+>', ' ', html_content)
        html_content = unescape(html_content)
        html_content = re.sub(r'\s+', ' ', html_content)
        return html_content.strip()

    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from various file formats"""
        extension = file_path.suffix.lower()

        if extension == '.txt':
            return self._extract_txt(file_path)
        elif extension == '.pdf':
            return self._extract_pdf(file_path)
        elif extension == '.epub':
            return self.extract_text_from_epub(file_path)
        elif extension == '.docx' and DOCX_AVAILABLE:
            return self._extract_docx(file_path)
        elif extension == '.doc' and DOC_AVAILABLE:
            return self._extract_doc(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def _extract_txt(self, file_path: Path) -> str:
        """Extract from TXT with encoding detection"""
        for encoding in ['utf-8-sig', 'utf-16', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return self._clean_text(f.read())
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file")

    def _extract_pdf(self, file_path: Path) -> str:
        """Extract from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            self.logger.info(f"PDF has {total_pages} pages")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n\n{page_text}"
                    if page_num % 10 == 0:
                        self.logger.debug(f"Extracted {page_num}/{total_pages} pages")
                except Exception as e:
                    self.logger.warning(f"Failed to extract page {page_num}: {e}")
                    continue
            
            self.logger.info(f"Extracted text from {total_pages} pages, {len(text)} characters total")
        return self._clean_text(text)

    def _extract_docx(self, file_path: Path) -> str:
        """Extract from DOCX"""
        doc = Document(file_path)
        text = '\n\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
        return self._clean_text(text)

    def _extract_doc(self, file_path: Path) -> str:
        """Extract from DOC"""
        text = docx2txt.process(str(file_path))
        return self._clean_text(text) if text else ""

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ')
        text = re.sub(r'\b\d{1,3}\b(?=\s|$)', '', text)
        return text.strip()

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into manageable chunks"""
        if not text.strip():
            return []

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if sentence_words > CHUNK_SIZE_WORDS:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_words = 0

                # Split long sentences
                parts = re.split(r'[,;:]', sentence)
                for part in parts:
                    part_words = len(part.split())
                    if current_words + part_words <= CHUNK_SIZE_WORDS:
                        current_chunk += part + " "
                        current_words += part_words
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + " "
                        current_words = part_words
            else:
                if current_words + sentence_words <= CHUNK_SIZE_WORDS:
                    current_chunk += sentence + " "
                    current_words += sentence_words
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                    current_words = sentence_words

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if chunk.strip()]

    def combine_chunks(self, total_chunks: int, output_path: Path, results: Optional[Dict[int, bool]] = None) -> bool:
        """Combine audio chunks into final audiobook"""
        try:
            combined = AudioSegment.empty()
            successful = 0
            missing_chunks = []

            for i in range(1, total_chunks + 1):
                # Skip chunks that failed if we have results tracking
                if results is not None and not results.get(i, False):
                    missing_chunks.append(i)
                    continue
                    
                chunk_file = Path("chunks") / f"chunk_{i:04d}.wav"
                if chunk_file.exists():
                    try:
                        chunk_audio = AudioSegment.from_wav(str(chunk_file))
                        combined += chunk_audio
                        successful += 1
                        if successful % 10 == 0:
                            self.logger.info(f"Combined {successful} chunks")
                    except Exception as e:
                        self.logger.warning(f"Failed to load chunk {i}: {e}")
                        missing_chunks.append(i)
                else:
                    self.logger.warning(f"Chunk file not found: {chunk_file}")
                    missing_chunks.append(i)

            if successful == 0:
                raise RuntimeError("No valid chunks found")

            if missing_chunks:
                self.logger.warning(f"Missing chunks: {missing_chunks}")

            combined.export(str(output_path), format=AUDIO_FORMAT, bitrate=AUDIO_BITRATE)
            self.logger.info(f"Audiobook saved: {output_path} ({successful}/{total_chunks} chunks)")
            print(f"[INFO] Saved audiobook: {output_path.name} ({successful}/{total_chunks} chunks)")
            if missing_chunks:
                print(f"[WARNING] Missing chunks: {missing_chunks}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to combine chunks: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def cleanup_chunks(self):
        """Remove temporary chunk files and cache"""
        try:
            # Clean up chunks folder
            chunk_count = 0
            for chunk_file in Path("chunks").glob("chunk_*.wav"):
                try:
                    chunk_file.unlink()
                    chunk_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete {chunk_file}: {e}")
            
            # Clean up cache folder
            cache_count = 0
            cache_dir = Path("cache/audio_chunks")
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.wav"):
                    try:
                        cache_file.unlink()
                        cache_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to delete cache file {cache_file}: {e}")
            
            if chunk_count > 0 or cache_count > 0:
                self.logger.info(f"Cleaned up {chunk_count} chunk files and {cache_count} cache files")
                print(f"[INFO] Cleaned up {chunk_count} chunk files and {cache_count} cache files")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    def convert_book(self, file_path: Path) -> bool:
        """Convert a single book to audiobook using Qwen API"""
        self.logger.info(f"Converting: {file_path.name}")
        start_time = time.time()

        try:
            # Extract text
            self.logger.info("Extracting text...")
            text = self.extract_text_from_file(file_path)
            if not text.strip():
                self.logger.error("No text extracted")
                return False

            self.logger.info(f"Extracted {len(text)} characters ({len(text.split())} words)")

            # Split into chunks
            chunks = self.split_into_chunks(text)
            total_chunks = len(chunks)
            if total_chunks == 0:
                self.logger.error("No chunks created")
                return False

            # Log chunk info
            chunk_sizes = [len(chunk.split()) for chunk in chunks]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            self.logger.info(f"Split into {total_chunks} chunks (avg {avg_chunk_size:.0f} words per chunk)")
            print(f"[INFO] Processing {total_chunks} chunks via Qwen API...")
            print(f"[INFO] Estimated time: ~{total_chunks * 4} minutes (4 min per chunk)")

            # Process chunks - process in order to ensure correct naming
            chunk_args = [(i + 1, chunk) for i, chunk in enumerate(chunks)]

            print(f"\n{'=' * 50}")
            print(f"PROCESSING {total_chunks} CHUNKS")
            print(f"{'=' * 50}")

            # Track results by chunk number
            results = {}  # chunk_num -> success (bool)
            
            # Process chunks sequentially to ensure correct order and naming
            # This ensures chunks are named 1, 2, 3, 4... in order
            for chunk_num, chunk_text in chunk_args:
                try:
                    result = self.process_chunk_with_retry((chunk_num, chunk_text))
                    results[chunk_num] = result
                    
                    if result:
                        print(f"[OK] Chunk {chunk_num:3d}/{total_chunks} completed")
                        self.logger.info(f"+ Chunk {chunk_num}/{total_chunks} completed")
                    else:
                        print(f"[FAIL] Chunk {chunk_num:3d}/{total_chunks} FAILED")
                        self.logger.error(f"- Chunk {chunk_num}/{total_chunks} failed")
                        
                except Exception as e:
                    results[chunk_num] = False
                    print(f"[ERROR] Chunk {chunk_num:3d}/{total_chunks} ERROR: {e}")
                    self.logger.error(f"- Chunk {chunk_num}/{total_chunks} error: {e}")

            successful_chunks = sum(1 for v in results.values() if v)
            print(f"\n{'=' * 50}")
            print(f"CHUNK PROCESSING COMPLETE")
            print(f"Successful: {successful_chunks}/{total_chunks}")
            print(f"{'=' * 50}")
            self.logger.info(f"Qwen processing completed: {successful_chunks}/{total_chunks} chunks")

            if successful_chunks == 0:
                self.logger.error("No chunks were successfully processed")
                self.cleanup_chunks()  # Cleanup even on failure
                return False

            if successful_chunks < total_chunks:
                self.logger.warning(f"Only {successful_chunks}/{total_chunks} chunks succeeded. Proceeding with partial audiobook.")

            # Combine chunks (only the successful ones)
            output_path = Path(AUDIOBOOKS_FOLDER) / f"{file_path.stem}.{AUDIO_FORMAT}"
            success = self.combine_chunks(total_chunks, output_path, results)

            if success:
                duration = time.time() - start_time
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                self.logger.info(f"Conversion completed in {minutes}m {seconds}s: {output_path}")
                print(f"[SUCCESS] Conversion completed in {minutes}m {seconds}s")
            else:
                self.logger.error("Failed to combine chunks into final audiobook")

            # Always cleanup, even on failure
            self.cleanup_chunks()
            return success

        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Cleanup on exception
            self.cleanup_chunks()
            return False

    def run(self):
        """Main conversion process"""
        print("=" * 70)
        print("QWEN-BASED AUDIOBOOK CONVERTER")
        print("=" * 70)
        print(f"Books folder: {BOOKS_FOLDER}")
        print(f"Output folder: {AUDIOBOOKS_FOLDER}")
        print(f"Qwen API endpoint: {QWEN_API_URL}")
        print(f"Voice mode: {self.voice_mode}")
        print(f"Model size: 1.7B (always)")
        if self.voice_mode == "custom_voice":
            print(f"Speaker: {CUSTOM_VOICE_SPEAKER}")
            print(f"Language: {CUSTOM_VOICE_LANGUAGE}")
        elif self.voice_mode == "voice_clone":
            print(f"Reference audio: {Path(self.voice_clone_ref_audio).name}")
            print(f"Language: {VOICE_CLONE_LANGUAGE}")
        print(f"Output format: {AUDIO_FORMAT}")
        print(f"Max workers: {MAX_WORKERS}")
        print("=" * 70)

        # Check for books
        books_dir = Path(BOOKS_FOLDER)
        supported_formats = ['.txt', '.pdf', '.epub']
        if DOCX_AVAILABLE:
            supported_formats.append('.docx')
        if DOC_AVAILABLE:
            supported_formats.append('.doc')

        book_files = [f for f in books_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in supported_formats]

        if not book_files:
            print(f"[INFO] No supported files found in {BOOKS_FOLDER}")
            print(f"Supported formats: {', '.join(supported_formats)}")

            # Create sample file
            sample_file = books_dir / "sample.txt"
            with open(sample_file, 'w') as f:
                f.write("This is a sample audiobook for testing the Qwen-based converter. "
                        "The system will send this text to the Qwen API for voice generation. "
                        "You can replace this file with your own books to convert.")
            print(f"[INFO] Created sample file: {sample_file}")
            return

        print(f"[INFO] Found {len(book_files)} books to convert")

        # Convert each book
        results = {}
        for book_file in book_files:
            try:
                success = self.convert_book(book_file)
                results[book_file.name] = success
            except KeyboardInterrupt:
                print("\n[WARNING] Conversion interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                results[book_file.name] = False

        # Print summary
        successful = sum(results.values())
        total = len(results)

        print("\n" + "=" * 70)
        print("CONVERSION SUMMARY")
        print("=" * 70)
        print(f"Total: {total} | Success: {successful} | Failed: {total - successful}")
        print("=" * 70)

        for filename, success in results.items():
            status = "[OK]" if success else "[FAIL]"
            print(f"{status} {filename}")

        if successful > 0:
            print(f"\n[INFO] Audiobooks saved to: {AUDIOBOOKS_FOLDER}/")


def main():
    """Entry point with argparse"""
    parser = argparse.ArgumentParser(
        description="Convert books to audiobooks using Qwen Voice Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use custom voice (default - Ryan speaker)
  python audiobook_converter.py

  # Use voice cloning with reference audio
  python audiobook_converter.py --voice-clone --voice-sample path/to/reference.wav
        """
    )
    
    parser.add_argument(
        "--voice-clone",
        action="store_true",
        help="Use voice cloning mode instead of custom voice (requires --voice-sample)"
    )
    
    parser.add_argument(
        "--voice-sample",
        type=str,
        help="Path to reference audio file for voice cloning (WAV format). Audio will be automatically transcribed."
    )
    
    args = parser.parse_args()
    
    # Determine voice mode
    if args.voice_clone:
        if not args.voice_sample:
            print("[ERROR] --voice-clone requires --voice-sample")
            print("Usage: python audiobook_converter.py --voice-clone --voice-sample <path>")
            sys.exit(1)
        voice_mode = "voice_clone"
        voice_clone_ref_audio = args.voice_sample
    else:
        voice_mode = "custom_voice"
        voice_clone_ref_audio = None
    
    try:
        converter = QwenAudiobookConverter(
            voice_mode=voice_mode,
            voice_clone_ref_audio=voice_clone_ref_audio
        )
        converter.run()
    except KeyboardInterrupt:
        print("\n[WARNING] Shutdown requested by user")
    except Exception as e:
        print(f"[FATAL] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
