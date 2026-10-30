"""
MediAssist RAG (Retrieval-Augmented Generation) Module
======================================================
Provides document chunking, embedding generation using Gemini,
FAISS vector indexing, and semantic similarity search.
"""

from .chunker import load_and_chunk_documents, parse_markdown_document
from .embeddings import get_embedding, get_embeddings_batch
from .indexer import build_and_save_index, load_index
from .retriever import MedicalRetriever

__all__ = [
    "load_and_chunk_documents",
    "parse_markdown_document",
    "get_embedding",
    "get_embeddings_batch",
    "build_and_save_index",
    "load_index",
    "MedicalRetriever",
]
