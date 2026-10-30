"""
Document Chunker Module
=======================
Parses Markdown files with frontmatter and splits text into semantic,
manageable chunks with metadata attribution.
"""

import os
import re
import hashlib
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def normalize_text(text: str) -> str:
    """Normalize source text while preserving paragraph and heading boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _make_chunk(
    text: str,
    metadata: Dict[str, Any],
    doc_id: str,
    passage_number: int,
    section: str,
) -> Dict[str, Any]:
    normalized_text = normalize_text(text)
    passage_id = f"{doc_id}:p{passage_number:04d}"
    chunk = {
        "chunk_id": f"{doc_id}_{passage_number}",
        "document_id": doc_id,
        "passage_id": passage_id,
        "source_title": metadata.get("title", ""),
        "source": metadata.get("source", ""),
        "source_url": metadata.get("url", ""),
        "retrieval_date": metadata.get("retrieval_date", ""),
        "version": metadata.get("version", "1"),
        "category": metadata.get("category", ""),
        "section": section,
        "text": normalized_text,
        "content_hash": hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:16],
    }
    # Pass through verifiable MBBS curriculum provenance fields
    for field in (
        "mbbs_subject",
        "source_type",
        "license",
        "author_organization",
        "edition_version",
        "chapter_section",
        "is_verbatim_textbook",
        "provenance_note",
        "page_number",
    ):
        if field in metadata:
            chunk[field] = metadata[field]
    return chunk


def _is_markdown_table(text: str) -> bool:
    """Check if the text block represents a markdown table."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return len(lines) >= 2 and all(line.startswith("|") and line.endswith("|") for line in lines)


def _split_markdown_table(table_text: str, max_chunk_size: int = 1200) -> List[str]:
    """Split oversized markdown tables row-by-row while preserving header and separator rows."""
    lines = [line.strip() for line in table_text.strip().splitlines() if line.strip()]
    if len(lines) <= 2:
        return [table_text]
    header = lines[0]
    separator = lines[1] if lines[1].startswith("|") and "-" in lines[1] else "| " + " | ".join(["---"] * (len(header.split("|")) - 2)) + " |"
    data_rows = lines[2:] if lines[1].startswith("|") and "-" in lines[1] else lines[1:]

    table_chunks = []
    current_rows = []
    current_size = len(header) + len(separator) + 2

    for row in data_rows:
        row_len = len(row) + 1
        if current_rows and (current_size + row_len > max_chunk_size):
            chunk_content = "\n".join([header, separator] + current_rows)
            table_chunks.append(chunk_content)
            current_rows = [row]
            current_size = len(header) + len(separator) + 2 + row_len
        else:
            current_rows.append(row)
            current_size += row_len

    if current_rows:
        chunk_content = "\n".join([header, separator] + current_rows)
        table_chunks.append(chunk_content)

    return table_chunks


def parse_markdown_document(file_path: str) -> Dict[str, Any]:
    """
    Parse a Markdown file, extracting metadata frontmatter (YAML-style)
    and returning metadata along with body text.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    metadata = {
        "title": os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").title(),
        "source": "Curated Medical Source",
        "url": "",
        "retrieval_date": "",
        "category": "General Health",
    }

    # Match frontmatter between --- and ---
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if frontmatter_match:
        yaml_text = frontmatter_match.group(1)
        body_text = normalize_text(frontmatter_match.group(2))

        for line in yaml_text.splitlines():
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().lower()
                val = val.strip().strip('"\'')
                if val.lower() == "false":
                    metadata[key] = False
                elif val.lower() == "true":
                    metadata[key] = True
                else:
                    metadata[key] = val
    else:
        body_text = normalize_text(content)

    return {
        "metadata": metadata,
        "body": body_text,
        "file_name": os.path.basename(file_path),
    }


def chunk_text(
    text: str,
    metadata: Dict[str, Any],
    doc_id: str,
    max_chunk_size: int = 650,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Split body text into paragraph-aware chunks.
    Preserves heading context, markdown tables, mechanisms, and lists.
    """
    chunks = []
    text = normalize_text(text)
    
    # Split text by Markdown section headers (## or ###)
    sections = re.split(r"\n(?=#{1,3}\s)", text)
    chunk_counter = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract current section header if available
        header_match = re.match(r"^(#{1,3}\s+[^\n]+)", section)
        current_header = header_match.group(1).lstrip("#").strip() if header_match else metadata.get("title", "")

        # If section is small enough, keep as single chunk
        if len(section) <= max_chunk_size:
            chunk_counter += 1
            chunks.append(_make_chunk(section, metadata, doc_id, chunk_counter, current_header))
            continue

        # Otherwise split into paragraphs while keeping tables and mechanisms intact
        paragraphs = section.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Handle Markdown tables specially: never break table rows into random sentence pieces
            if _is_markdown_table(para):
                # Flush existing text chunk first
                if current_chunk:
                    chunk_counter += 1
                    chunks.append(_make_chunk(current_chunk, metadata, doc_id, chunk_counter, current_header))
                    current_chunk = ""

                # If table fits within expanded table threshold, keep intact
                if len(para) <= max(max_chunk_size, 1200):
                    chunk_counter += 1
                    chunks.append(_make_chunk(para, metadata, doc_id, chunk_counter, current_header))
                else:
                    # Split table cleanly by rows while preserving headers
                    sub_tables = _split_markdown_table(para, max_chunk_size=max_chunk_size)
                    for sub_tab in sub_tables:
                        chunk_counter += 1
                        chunks.append(_make_chunk(sub_tab, metadata, doc_id, chunk_counter, current_header))
                continue

            if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunk_counter += 1
                    chunks.append(_make_chunk(current_chunk, metadata, doc_id, chunk_counter, current_header))
                    # Add overlap from previous chunk
                    current_chunk = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                    current_chunk = f"{current_chunk} {para}".strip()
                else:
                    # Single paragraph exceeds max_chunk_size, split by sentences
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= max_chunk_size:
                            current_chunk = f"{current_chunk} {sent}".strip()
                        else:
                            if current_chunk:
                                chunk_counter += 1
                                chunks.append(_make_chunk(current_chunk, metadata, doc_id, chunk_counter, current_header))
                            current_chunk = sent

        if current_chunk:
            chunk_counter += 1
            chunks.append(_make_chunk(current_chunk, metadata, doc_id, chunk_counter, current_header))

    return chunks


def load_and_chunk_documents(
    knowledge_dir: str,
    max_chunk_size: int = 650,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Scan knowledge_dir for .md and .txt files, parse their metadata,
    and split them into structured chunks.
    """
    if not os.path.exists(knowledge_dir):
        return []

    all_chunks = []
    files = sorted([f for f in os.listdir(knowledge_dir) if f.endswith((".md", ".txt"))])

    for f in files:
        file_path = os.path.join(knowledge_dir, f)
        doc = parse_markdown_document(file_path)
        doc_id = os.path.splitext(f)[0]
        chunks = chunk_text(
            text=doc["body"],
            metadata=doc["metadata"],
            doc_id=doc_id,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)

    return all_chunks


def split_text_with_langchain(
    text: str,
    metadata: Dict[str, Any],
    doc_id: str,
    chunk_size: int = 650,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """Split medical body text using LangChain RecursiveCharacterTextSplitter with metadata preservation."""
    normalized_text = normalize_text(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n|", "\n1. ", "\n- ", "\n", ". ", " ", ""],
    )
    raw_splits = splitter.split_text(normalized_text)
    chunks = []
    for idx, split_text in enumerate(raw_splits, start=1):
        # Extract section if split starts with heading
        header_match = re.match(r"^(#{1,3}\s+[^\n]+)", split_text)
        current_header = header_match.group(1).lstrip("#").strip() if header_match else metadata.get("title", "")
        chunks.append(_make_chunk(split_text, metadata, doc_id, idx, current_header))
    return chunks


def load_and_chunk_langchain_documents(
    knowledge_dir: str,
    chunk_size: int = 650,
    chunk_overlap: int = 100,
) -> List[Document]:
    """Parse medical source documents into standard LangChain Document instances."""
    if not os.path.exists(knowledge_dir):
        return []

    documents = []
    files = sorted([f for f in os.listdir(knowledge_dir) if f.endswith((".md", ".txt"))])

    for f in files:
        file_path = os.path.join(knowledge_dir, f)
        doc = parse_markdown_document(file_path)
        doc_id = os.path.splitext(f)[0]
        chunks = split_text_with_langchain(
            text=doc["body"],
            metadata=doc["metadata"],
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for chunk in chunks:
            content = chunk.get("text", "")
            meta = {k: v for k, v in chunk.items() if k != "text"}
            documents.append(Document(page_content=content, metadata=meta))

    return documents
