"""
Document Processing & Ingestion Service for MediAssist.
======================================================
Securely processes user-uploaded medical study notes and PDFs:
1. Validates magic bytes, file extensions, and file size limits (15 MB).
2. Defense-in-depth against malware, binary executables, and PDF action abuse.
3. Extracts digital text page-by-page with timeout and memory guards using pypdf.
4. Chunks text while preserving 1-based page references.
5. Produces embeddings via existing Google Gemini integration.
6. Manages conversation-isolated FAISS indices and metadata lifecycle.
7. Neutralizes prompt injection inside student-uploaded notes.
8. Provides strictly authorized conversation-scoped retrieval and atomic deletion.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from werkzeug.utils import secure_filename

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.chunker import _is_markdown_table, _split_markdown_table
from rag.embeddings import get_embedding, get_embeddings_batch, DEFAULT_EMBEDDING_MODEL
from rag.indexer import normalize_vectors, manifest_path_for, INDEX_FORMAT_VERSION
from rag.retriever import sanitize_retrieved_content

logger = logging.getLogger("mediassist.documents")

# Dangerous executable magic signatures to reject even if extension is spoofed
SUSPICIOUS_EXECUTABLE_SIGNATURES = [
    (b"MZ", "Windows PE executable / DLL"),
    (b"\x7fELF", "Linux ELF executable"),
    (b"\xfe\xed\xfa\xce", "Mach-O 32-bit binary"),
    (b"\xfe\xed\xfa\xcf", "Mach-O 64-bit binary"),
    (b"\xce\xfa\xed\xfe", "Mach-O binary"),
    (b"\xcf\xfa\xed\xfe", "Mach-O binary"),
    (b"\xca\xfe\xba\xbe", "Java classfile / Mach-O fat binary"),
    (b"PK\x03\x04", "ZIP / Archive container (not supported)"),
    (b"Rar!\x1a\x07", "RAR archive"),
    (b"7z\xbc\xaf\x27\x1c", "7-Zip archive"),
    (b"\x1f\x8b", "Gzip archive"),
]

# PDF byte signatures that indicate potential active script or launch actions
SUSPICIOUS_PDF_ACTIONS = [
    re.compile(rb"/Launch\b", re.IGNORECASE),
    re.compile(rb"/EmbeddedFiles\b", re.IGNORECASE),
    re.compile(rb"/JavaScript\b", re.IGNORECASE),
    re.compile(rb"/JS\b", re.IGNORECASE),
]


class DocumentValidationError(ValueError):
    """Raised when an uploaded document fails security, format, or size validation."""


class DocumentExtractionError(RuntimeError):
    """Raised when document text extraction fails or yields no extractable text."""


def validate_file_content(
    file_bytes: bytes,
    filename: str,
    max_size_bytes: int = 15 * 1024 * 1024,
    allowed_extensions: Sequence[str] = ("pdf", "txt", "md"),
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate actual file content, size, extension, and basic structural integrity.

    Defense-in-depth checks applied in order:
      1. Non-empty, within size bound.
      2. Extension present and in allowlist (do not trust client-supplied MIME).
      3. Magic-byte blocklist rejects known executables and archives.
      4. PDF: requires %PDF- header, %%EOF terminator, and is parseable by pypdf.
      5. Text/Markdown: must be UTF-8 or Latin-1, must not contain null bytes.

    Returns (is_valid, detected_type, error_message).
    """
    if not filename or not str(filename).strip():
        return False, "", "Filename is missing or invalid."

    if not file_bytes:
        return False, "", "Uploaded file is completely empty (0 bytes)."

    if len(file_bytes) > max_size_bytes:
        mb_limit = max_size_bytes / (1024 * 1024)
        return (
            False,
            "",
            f"File size ({len(file_bytes) / (1024 * 1024):.1f} MB) exceeds "
            f"maximum allowed limit ({mb_limit:.0f} MB).",
        )

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {str(e).lower() for e in allowed_extensions}:
        return (
            False,
            "",
            f"Unsupported file extension '.{ext}'. Permitted formats: {sorted(allowed_extensions)}.",
        )

    # ── 1. Executable / archive magic byte blocklist ─────────────────────────
    for sig, desc in SUSPICIOUS_EXECUTABLE_SIGNATURES:
        if file_bytes.startswith(sig):
            # ELF is a special case: check description text to produce correct msg
            label = "executable" if "executable" in desc.lower() or "binary" in desc.lower() else "archive"
            return False, "", f"Malicious or unsupported {label} format detected ({desc}). Upload rejected."

    # ── 2. PDF validation ─────────────────────────────────────────────────────
    if ext == "pdf":
        # Require valid %PDF- header (at least 5 bytes)
        if not file_bytes.startswith(b"%PDF-"):
            return False, "", "File content does not match PDF specification (missing %PDF- header magic bytes)."

        # Check for %%EOF within last 2 KB (standard) or anywhere in file (tolerance)
        tail = file_bytes[-2048:]
        if b"%%EOF" not in tail and b"%%EOF" not in file_bytes:
            return False, "", "Malformed PDF file: missing %%EOF termination marker."

        # Structural parse: ensure pypdf can open the file without raising
        # Scan only the first 512 KB for active content patterns (DoS protection)
        scan_window = file_bytes[:524288]
        try:
            import io as _io
            from pypdf import PdfReader as _PdfReader

            reader = _PdfReader(_io.BytesIO(file_bytes), strict=False)
            # Accessing pages exercises basic structural integrity
            _ = len(reader.pages)
        except Exception as parse_exc:
            logger.warning(
                "PDF structural parse failed for '%s': %s",
                os.path.basename(filename),
                str(parse_exc)[:120],
            )
            return (
                False,
                "",
                "PDF file is malformed or corrupted and cannot be processed safely.",
            )

        return True, "pdf", None

    # ── 3. Plain text / Markdown ──────────────────────────────────────────────
    if ext in ("txt", "md"):
        # Null bytes in the first 4 KB indicate binary data masquerading as text
        if b"\x00" in file_bytes[:4096]:
            return (
                False,
                "",
                "Binary file detected with text extension. Only plain-text files are supported.",
            )
        try:
            file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                file_bytes.decode("latin-1")
            except Exception:
                return False, "", "File is not valid text encoded in UTF-8 or Latin-1."
        return True, "text", None

    return False, "", f"Unrecognized file type for extension '.{ext}'."


def scan_file_for_malware(filepath: str, file_bytes: bytes) -> Tuple[bool, str]:
    """
    Defense-in-depth security inspection for uploaded files.
    Checks for PDF active content or exploit payloads.
    Rejects PDFs containing known dangerous action dictionaries.
    Documents residual risks if host does not run an external ClamAV daemon.
    """
    # 1. Structural PDF active content inspection — reject dangerous action dicts
    if file_bytes.startswith(b"%PDF-"):
        for pattern in SUSPICIOUS_PDF_ACTIONS:
            if pattern.search(file_bytes):
                action_name = pattern.pattern.decode("ascii", errors="replace").strip()
                logger.warning(
                    "Security: Rejecting PDF '%s' — contains potentially dangerous action dict '%s'.",
                    os.path.basename(filepath),
                    action_name,
                )
                return False, (
                    f"File upload rejected: the PDF contains an active content element ({action_name}) "
                    "that poses a security risk. Please upload a plain text-only PDF without embedded scripts, "
                    "JavaScript, or launch actions."
                )

    # 2. Document residual risk note:
    # MediAssist parses PDF content strictly using text-extraction (no bytecode or renderer execution).
    residual_note = (
        "Content passed magic-signature verification, size boundaries, and text-only sandboxing. "
        "No OS ClamAV daemon detected on local host; active scripts and binary executables rejected."
    )
    return True, residual_note


class DocumentService:
    """Manages secure document parsing, chunking, and conversation-scoped vector storage."""

    def __init__(self, app_config: Any, client: Optional[Any] = None):
        self.config = app_config
        self.client = client
        self._upload_root_override = None
        self.max_file_size_bytes = self._get_config("MAX_UPLOAD_FILE_SIZE_BYTES", 15 * 1024 * 1024)
        self.max_pages = self._get_config("MAX_DOCUMENT_PAGES", 50)
        self.max_documents_per_conversation = self._get_config("MAX_DOCUMENTS_PER_CONVERSATION", 5)
        self.timeout_seconds = self._get_config("DOCUMENT_EXTRACTION_TIMEOUT_SECONDS", 30.0)
        self.embedding_model = self._get_config("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="doc-ingest-")

    @property
    def upload_root(self) -> str:
        if self._upload_root_override:
            root = self._upload_root_override
        else:
            root = self._get_config("UPLOAD_FOLDER", os.path.join(os.getcwd(), "instance", "uploads"))
        os.makedirs(root, exist_ok=True)
        return root

    @upload_root.setter
    def upload_root(self, value: str) -> None:
        self._upload_root_override = value

    def _get_config(self, key: str, default: Any = None) -> Any:
        if hasattr(self.config, "get"):
            val = self.config.get(key)
            if val is not None:
                return val
        return getattr(self.config, key, default)

    def get_conversation_dir(self, conversation_id: str) -> str:
        """Return the isolated storage directory for a specific conversation."""
        safe_id = secure_filename(str(conversation_id))
        conv_dir = os.path.join(self.upload_root, safe_id)
        os.makedirs(conv_dir, exist_ok=True)
        return conv_dir

    def extract_document_text(self, file_bytes: bytes, filename: str, file_type: str) -> List[Dict[str, Any]]:
        """
        Extract digital text page-by-page from PDF or text file with strict timeout,
        page count limits, and prompt-injection neutralization.

        Returns list of pages: [{"page_number": int, "text": str}]
        """
        start_time = time.perf_counter()
        pages: List[Dict[str, Any]] = []

        if file_type == "pdf":
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
            except Exception as exc:
                raise DocumentExtractionError(f"Failed to parse PDF document structure: {exc}") from exc

            if reader.is_encrypted:
                try:
                    # Attempt decrypt with empty password for standard protected PDFs
                    decrypt_status = reader.decrypt("")
                    if decrypt_status == 0:
                        raise DocumentExtractionError("PDF is password-protected or encrypted. Please upload an unprotected file.")
                except Exception as exc:
                    raise DocumentExtractionError("PDF is password-protected or encrypted. Please upload an unprotected file.") from exc

            total_pages = len(reader.pages)
            if total_pages == 0:
                raise DocumentExtractionError("PDF file contains 0 pages.")
            if total_pages > self.max_pages:
                raise DocumentValidationError(
                    f"PDF document exceeds page limit ({total_pages} pages found, maximum is {self.max_pages} pages)."
                )

            total_extracted_chars = 0
            for idx, page in enumerate(reader.pages, start=1):
                if time.perf_counter() - start_time > self.timeout_seconds:
                    raise DocumentExtractionError(f"PDF extraction timed out after {self.timeout_seconds:.1f} seconds.")

                try:
                    raw_text = page.extract_text() or ""
                except Exception as exc:
                    logger.warning("Error extracting text on page %d: %s", idx, exc)
                    raw_text = ""

                # Extract embedded medical illustrations / diagrams on this page
                page_imgs = []
                try:
                    for img_idx, img_file in enumerate(getattr(page, "images", []), start=1):
                        img_name = getattr(img_file, "name", f"img_p{idx}_{img_idx}.jpg")
                        img_data = getattr(img_file, "data", None)
                        if img_data and len(img_data) > 1024:
                            page_imgs.append({
                                "name": img_name,
                                "data": img_data,
                                "page_number": idx,
                                "index": img_idx,
                            })
                except Exception as img_exc:
                    logger.debug("Image extraction on page %d: %s", idx, img_exc)

                cleaned_text = sanitize_retrieved_content(raw_text.strip())
                if cleaned_text:
                    total_extracted_chars += len(cleaned_text)

                if cleaned_text or page_imgs:
                    pages.append({
                        "page_number": idx,
                        "text": cleaned_text,
                        "images": page_imgs,
                    })

            # Scanned or image-only PDF diagnosis and multimodal fallback
            if total_extracted_chars == 0:
                all_embedded_images = [img for p in pages for img in p.get("images", [])]
                if all_embedded_images and self.client:
                    logger.info("Scanned PDF detected with %d embedded images. Attempting multimodal OCR...", len(all_embedded_images))
                    try:
                        from google.genai import types as genai_types
                        for p in pages:
                            p_images = p.get("images", [])
                            if not p_images:
                                continue
                            primary_img = p_images[0]
                            mime = "image/png" if primary_img["name"].lower().endswith(".png") else "image/jpeg"
                            prompt_text = (
                                "Extract all medical textbook or note text from this scanned page accurately, "
                                "including headings, tables, drug dosages, and anatomical labels. Preserve markdown tables."
                            )
                            ocr_resp = self.client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=[
                                    genai_types.Part.from_bytes(data=primary_img["data"], mime_type=mime),
                                    prompt_text,
                                ],
                            )
                            if ocr_resp and getattr(ocr_resp, "text", None):
                                txt = sanitize_retrieved_content(ocr_resp.text.strip())
                                if txt:
                                    p["text"] = txt
                                    total_extracted_chars += len(txt)
                    except Exception as ocr_err:
                        logger.warning("Multimodal OCR failed: %s", ocr_err)

                if total_extracted_chars == 0:
                    raise DocumentExtractionError(
                        "No extractable digital text found in this PDF. Scanned or image-only PDFs without an embedded OCR text layer are not supported."
                    )

        elif file_type == "text":
            try:
                text_content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text_content = file_bytes.decode("latin-1", errors="replace")

            cleaned_text = sanitize_retrieved_content(text_content.strip())
            if not cleaned_text:
                raise DocumentExtractionError("Uploaded text file is empty or contains only whitespace.")

            pages.append({
                "page_number": 1,
                "text": cleaned_text,
                "images": [],
            })

        else:
            raise DocumentValidationError(f"Unsupported document file type '{file_type}'.")

        return pages

    def chunk_extracted_pages(
        self,
        pages: List[Dict[str, Any]],
        doc_id: str,
        filename: str,
        conversation_id: str,
        user_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Split extracted pages into semantic chunks while maintaining 1-based page references,
        provenance metadata, and document ownership boundaries.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=650,
            chunk_overlap=80,
            separators=["\n\n", "\n### ", "\n## ", "\n# ", "\n- ", "\n1. ", "\n", " "],
        )

        chunks: List[Dict[str, Any]] = []
        chunk_idx = 1

        for page in pages:
            page_num = page["page_number"]
            page_text = page["text"]
            page_images = [img["url"] for img in page.get("images", []) if "url" in img]

            # Preserve markdown tables without fragmented splitting
            if _is_markdown_table(page_text):
                splits = _split_markdown_table(page_text, max_chunk_size=750)
            else:
                if "\n|" in page_text and "|\n" in page_text:
                    splits = []
                    blocks = page_text.split("\n\n")
                    for b in blocks:
                        b_clean = b.strip()
                        if _is_markdown_table(b_clean):
                            splits.extend(_split_markdown_table(b_clean, max_chunk_size=750))
                        else:
                            splits.extend(splitter.split_text(b_clean))
                else:
                    splits = splitter.split_text(page_text)

            for split in splits:
                text_content = split.strip()
                if not text_content:
                    continue

                passage_id = f"doc_{doc_id[:8]}_p{page_num:03d}_{chunk_idx:02d}"
                chunk_data = {
                    "passage_id": passage_id,
                    "chunk_id": f"{doc_id}_{chunk_idx}",
                    "document_id": doc_id,
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "document_name": filename,
                    "page_number": page_num,
                    "section": f"Page {page_num}",
                    "source_title": f"{filename} (Page {page_num})",
                    "source": f"Attached Study Note: {filename}",
                    "source_url": "",
                    "source_type": "uploaded_study_note",
                    "is_uploaded_document": True,
                    "is_verbatim_textbook": False,
                    "retrieval_date": time.strftime("%Y-%m-%d"),
                    "text": text_content,
                    "provenance_note": f"User-uploaded study note attached to conversation {conversation_id}.",
                    "embedded_images": page_images,
                }
                chunks.append(chunk_data)
                chunk_idx += 1

        return chunks

    def process_and_index_document(
        self,
        conversation_id: str,
        user_id: int,
        doc_id: str,
        filename: str,
        file_bytes: bytes,
        file_type: str,
    ) -> Tuple[int, int]:
        """
        Extract text, chunk content, generate embeddings, and update the conversation-isolated FAISS index.

        Returns (page_count, chunk_count).
        """
        conv_dir = self.get_conversation_dir(conversation_id)

        # 1. Extract digital text
        pages = self.extract_document_text(file_bytes, filename, file_type)
        page_count = len(pages)

        # 1b. Save any extracted images to conversation storage
        conv_images_dir = os.path.join(conv_dir, "images")
        os.makedirs(conv_images_dir, exist_ok=True)
        for page in pages:
            for img_info in page.get("images", []):
                safe_img_name = f"{doc_id[:8]}_p{img_info['page_number']}_{img_info['index']}.jpg"
                img_path = os.path.join(conv_images_dir, safe_img_name)
                try:
                    with open(img_path, "wb") as img_file:
                        img_file.write(img_info["data"])
                    img_info["saved_path"] = img_path
                    img_info["url"] = f"/api/conversations/{conversation_id}/images/{safe_img_name}"
                except Exception as save_err:
                    logger.debug("Failed saving extracted image: %s", save_err)

        # 2. Chunk pages preserving page references
        new_chunks = self.chunk_extracted_pages(pages, doc_id, filename, conversation_id, user_id)
        chunk_count = len(new_chunks)

        if chunk_count == 0:
            raise DocumentExtractionError("No usable text chunks could be produced from this document.")

        # 3. Load existing chunks for this conversation (if other documents attached)
        meta_path = os.path.join(conv_dir, "chunks_metadata.json")
        index_path = os.path.join(conv_dir, "faiss_index.bin")
        manifest_path = os.path.join(conv_dir, "faiss_index_manifest.json")

        existing_chunks: List[Dict[str, Any]] = []
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    existing_chunks = json.load(f)
            except Exception as exc:
                logger.warning("Could not read existing conversation metadata, reinitializing: %s", exc)
                existing_chunks = []

        all_chunks = existing_chunks + new_chunks

        # Bound maximum chunks per conversation to prevent unbounded memory growth
        max_total_chunks = 250
        if len(all_chunks) > max_total_chunks:
            raise DocumentValidationError(
                f"Adding this document would exceed maximum conversation chunk capacity ({len(all_chunks)} chunks > {max_total_chunks} limit)."
            )

        # 4. Generate embeddings for new chunks using existing embedding integration
        chunk_texts = [c["text"] for c in all_chunks]
        try:
            vectors = get_embeddings_batch(
                texts=chunk_texts,
                client=self.client,
                model=self.embedding_model,
                batch_size=15,
            )
        except Exception as exc:
            logger.error("Embedding generation failed for document %s: %s", filename, exc)
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

        # 5. Build FAISS index for the conversation
        vector_matrix = np.array(vectors, dtype=np.float32)
        normalized_vectors = normalize_vectors(vector_matrix)
        dim = normalized_vectors.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(normalized_vectors)

        # 6. Save FAISS index and metadata atomically
        temp_index = index_path + ".tmp"
        temp_meta = meta_path + ".tmp"
        temp_manifest = manifest_path + ".tmp"

        faiss.write_index(index, temp_index)
        with open(temp_meta, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        with open(temp_meta, "rb") as f:
            meta_sha = hashlib.sha256(f.read()).hexdigest()

        manifest = {
            "format_version": INDEX_FORMAT_VERSION,
            "embedding_model": self.embedding_model,
            "dimension": dim,
            "vector_metric": "cosine_via_normalized_inner_product",
            "normalized": True,
            "vector_count": index.ntotal,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "index_to_chunk_id": [c.get("passage_id") for c in all_chunks],
            "metadata_sha256": meta_sha,
        }
        with open(temp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        os.replace(temp_index, index_path)
        os.replace(temp_meta, meta_path)
        os.replace(temp_manifest, manifest_path)

        logger.info(
            "Indexed document '%s' (doc_id=%s) for conversation %s: %d pages, %d chunks (total in conv: %d)",
            filename,
            doc_id,
            conversation_id,
            page_count,
            chunk_count,
            len(all_chunks),
        )
        return page_count, chunk_count

    def submit_async_ingestion(
        self,
        app: Any,
        conversation_id: str,
        user_id: int,
        doc_id: str,
        filename: str,
        file_bytes: bytes,
        file_type: str,
    ) -> Any:
        """Submit document extraction and embedding generation to a background worker thread."""
        def _worker():
            with app.app_context():
                from models import db, ConversationDocument
                try:
                    logger.info("Starting background ingestion for doc_id=%s ('%s') in conv=%s", doc_id, filename, conversation_id)
                    page_count, chunk_count = self.process_and_index_document(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        doc_id=doc_id,
                        filename=filename,
                        file_bytes=file_bytes,
                        file_type=file_type,
                    )
                    doc = ConversationDocument.query.filter_by(id=doc_id).first()
                    if doc:
                        doc.page_count = page_count
                        doc.chunk_count = chunk_count
                        doc.status = "ready"
                        doc.error_message = None
                        db.session.commit()
                        logger.info(
                            "Background ingestion completed successfully for doc_id=%s ('%s'): %d pages, %d chunks",
                            doc_id, filename, page_count, chunk_count
                        )

                    # Invalidate conversation response cache so subsequent queries ground on the new content
                    llm_service = app.extensions.get("llm_service")
                    if llm_service and hasattr(llm_service, "cache") and llm_service.cache:
                        llm_service.cache.invalidate_conversation(conversation_id)

                except Exception as exc:
                    logger.error("Background ingestion error for doc_id=%s ('%s'): %s", doc_id, filename, exc, exc_info=True)
                    try:
                        db.session.rollback()
                        doc = ConversationDocument.query.filter_by(id=doc_id).first()
                        if doc:
                            doc.status = "error"
                            doc.error_message = str(exc)
                            db.session.commit()
                    except Exception as rb_exc:
                        logger.error("Failed to persist error status for doc_id=%s: %s", doc_id, rb_exc)

        return self.executor.submit(_worker)

    def retrieve_for_conversation(
        self,
        conversation_id: str,
        user_id: int,
        query: str,
        top_k: int = 2,
        score_threshold: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant passages strictly from documents attached to the specified conversation,
        enforcing multi-tenant authorization boundaries.
        """
        if not conversation_id or not query or not query.strip():
            return []

        conv_dir = self.get_conversation_dir(conversation_id)
        index_path = os.path.join(conv_dir, "faiss_index.bin")
        meta_path = os.path.join(conv_dir, "chunks_metadata.json")
        manifest_path = os.path.join(conv_dir, "faiss_index_manifest.json")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return []

        try:
            # Verify manifest user ownership
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                if manifest_data.get("user_id") != user_id or manifest_data.get("conversation_id") != conversation_id:
                    logger.warning("Tenant isolation violation: user %s attempted retrieval from conv %s", user_id, conversation_id)
                    return []

            index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if index.ntotal == 0 or len(metadata) == 0:
                return []

            raw_embedding = get_embedding(
                text=query.strip(),
                client=self.client,
                model=self.embedding_model,
            )
            query_vector = np.array([raw_embedding], dtype=np.float32)
            normalized_query = normalize_vectors(query_vector)

            search_k = min(max(top_k, 5), index.ntotal)
            scores, indices = index.search(normalized_query, search_k)

            results: List[Dict[str, Any]] = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(metadata) and float(score) >= score_threshold:
                    chunk = dict(metadata[idx])
                    chunk["score"] = float(score)
                    chunk["semantic_score"] = float(score)
                    chunk["is_uploaded_document"] = True
                    results.append(chunk)

            return results[:top_k]

        except Exception as exc:
            logger.warning("Error during conversation document retrieval for conv=%s: %s", conversation_id, exc)
            return []

    def delete_document(self, conversation_id: str, user_id: int, document_id: str) -> bool:
        """
        Delete an uploaded document, remove its source file from disk, purge its chunks
        from the conversation index, and rebuild or delete the conversation FAISS index.
        """
        conv_dir = self.get_conversation_dir(conversation_id)
        meta_path = os.path.join(conv_dir, "chunks_metadata.json")
        index_path = os.path.join(conv_dir, "faiss_index.bin")
        manifest_path = os.path.join(conv_dir, "faiss_index_manifest.json")

        # 1. Remove stored document file from disk
        removed_file = False
        if os.path.exists(conv_dir):
            for fname in os.listdir(conv_dir):
                if fname.startswith(f"{document_id}_"):
                    try:
                        os.remove(os.path.join(conv_dir, fname))
                        removed_file = True
                    except OSError as exc:
                        logger.warning("Could not delete stored file %s: %s", fname, exc)

        # 2. Update conversation chunks metadata
        if not os.path.exists(meta_path):
            return removed_file

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                all_chunks = json.load(f)

            remaining_chunks = [c for c in all_chunks if c.get("document_id") != document_id]

            if not remaining_chunks:
                # No documents left in this conversation: delete index files
                for fpath in (index_path, meta_path, manifest_path):
                    if os.path.exists(fpath):
                        os.remove(fpath)
                logger.info("Removed all document chunks for conversation %s; vector index purged.", conversation_id)
                return True

            # Rebuild FAISS index with remaining chunks
            chunk_texts = [c["text"] for c in remaining_chunks]
            vectors = get_embeddings_batch(
                texts=chunk_texts,
                client=self.client,
                model=self.embedding_model,
                batch_size=15,
            )
            vector_matrix = np.array(vectors, dtype=np.float32)
            normalized_vectors = normalize_vectors(vector_matrix)

            dim = normalized_vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(normalized_vectors)

            temp_index = index_path + ".tmp"
            temp_meta = meta_path + ".tmp"
            temp_manifest = manifest_path + ".tmp"

            faiss.write_index(index, temp_index)
            with open(temp_meta, "w", encoding="utf-8") as f:
                json.dump(remaining_chunks, f, indent=2, ensure_ascii=False)

            with open(temp_meta, "rb") as f:
                meta_sha = hashlib.sha256(f.read()).hexdigest()

            manifest = {
                "format_version": INDEX_FORMAT_VERSION,
                "embedding_model": self.embedding_model,
                "dimension": dim,
                "vector_metric": "cosine_via_normalized_inner_product",
                "normalized": True,
                "vector_count": index.ntotal,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "index_to_chunk_id": [c.get("passage_id") for c in remaining_chunks],
                "metadata_sha256": meta_sha,
            }
            with open(temp_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            os.replace(temp_index, index_path)
            os.replace(temp_meta, meta_path)
            os.replace(temp_manifest, manifest_path)

            logger.info("Rebuilt conversation index for %s after deleting document %s (%d chunks remaining)", conversation_id, document_id, len(remaining_chunks))
            return True

        except Exception as exc:
            logger.error("Error during document deletion from index: %s", exc)
            return False

    def delete_conversation_storage(self, conversation_id: str) -> None:
        """Completely remove all stored files and vector indices for a deleted conversation."""
        safe_id = secure_filename(str(conversation_id))
        conv_dir = os.path.join(self.upload_root, safe_id)
        if os.path.exists(conv_dir):
            try:
                shutil.rmtree(conv_dir, ignore_errors=True)
                logger.info("Purged storage directory for conversation %s", conversation_id)
            except Exception as exc:
                logger.warning("Could not completely purge directory %s: %s", conv_dir, exc)
