"""
FAISS Vector Indexer Module
===========================
Builds, persists, and loads FAISS vector indices along with
associated chunk metadata for similarity search.
"""

import os
import json
import logging
import hashlib
import uuid
from typing import List, Dict, Any, Tuple, Optional
import faiss
import numpy as np

logger = logging.getLogger("mediassist.rag.indexer")
INDEX_FORMAT_VERSION = 2
SIMILARITY_METRIC = "cosine_via_normalized_inner_product"


def manifest_path_for(index_path: str) -> str:
    """Return the sidecar manifest path for a persisted FAISS index."""
    return os.path.join(os.path.dirname(os.path.abspath(index_path)), "faiss_index_manifest.json")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors so that Inner Product (IP) corresponds to Cosine Similarity.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def build_and_save_index(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    index_path: str,
    metadata_path: str,
    embedding_model: str = "",
) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP from chunk embeddings, L2-normalize them,
    and persist both the binary index and the JSON metadata to disk.
    """
    if not chunks:
        raise ValueError("Cannot build FAISS index with empty chunks list.")
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})."
        )

    seen_keys = set()
    unique_chunks = []
    unique_embeddings = []
    for chunk, embedding in zip(chunks, embeddings):
        key = chunk.get("content_hash") or hashlib.sha256(
            chunk.get("text", "").strip().encode("utf-8")
        ).hexdigest()
        if key in seen_keys:
            logger.warning("Skipping duplicate chunk during indexing: %s", chunk.get("chunk_id", key))
            continue
        seen_keys.add(key)
        unique_chunks.append(chunk)
        unique_embeddings.append(embedding)

    chunks = unique_chunks
    embeddings = unique_embeddings
    vectors = np.array(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty two-dimensional matrix.")
    normalized = normalize_vectors(vectors)

    dim = normalized.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalized)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)

    metadata_bytes = json.dumps(chunks, indent=2, ensure_ascii=False).encode("utf-8")
    manifest = {
        "format_version": INDEX_FORMAT_VERSION,
        "embedding_model": embedding_model,
        "dimension": int(dim),
        "vector_metric": SIMILARITY_METRIC,
        "normalized": True,
        "vector_count": int(index.ntotal),
        "index_to_chunk_id": [chunk.get("passage_id", chunk.get("chunk_id", "")) for chunk in chunks],
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    }

    index_temp = f"{index_path}.{uuid.uuid4().hex}.tmp"
    metadata_temp = f"{metadata_path}.{uuid.uuid4().hex}.tmp"
    manifest_path = manifest_path_for(index_path)
    manifest_temp = f"{manifest_path}.{uuid.uuid4().hex}.tmp"
    try:
        faiss.write_index(index, index_temp)
        with open(metadata_temp, "wb") as handle:
            handle.write(metadata_bytes)
        with open(manifest_temp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        os.replace(index_temp, index_path)
        os.replace(metadata_temp, metadata_path)
        os.replace(manifest_temp, manifest_path)
    finally:
        for temporary_path in (index_temp, metadata_temp, manifest_temp):
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    logger.info("Successfully saved FAISS index (%d vectors, dim %d) to %s", index.ntotal, dim, index_path)
    logger.info("Successfully saved chunk metadata to %s", metadata_path)

    return index


def load_index(
    index_path: str,
    metadata_path: str,
    expected_embedding_model: str = "",
    expected_index_version: int = INDEX_FORMAT_VERSION,
) -> Tuple[Optional[faiss.Index], List[Dict[str, Any]]]:
    """
    Load a persisted FAISS index and chunk metadata from disk.
    Returns (None, []) gracefully if files do not exist or are corrupted.
    """
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        logger.warning("FAISS index or metadata not found at %s / %s", index_path, metadata_path)
        return None, []

    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        manifest_path = manifest_path_for(index_path)
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("format_version") != expected_index_version:
                raise ValueError("FAISS index format version mismatch")
            if manifest.get("dimension") != index.d:
                raise ValueError("FAISS index dimension mismatch")
            if manifest.get("vector_count") != index.ntotal or index.ntotal != len(metadata):
                raise ValueError("FAISS index and metadata count mismatch")
            if expected_embedding_model and manifest.get("embedding_model") != expected_embedding_model:
                raise ValueError("FAISS embedding model mismatch")
            mapped_ids = manifest.get("index_to_chunk_id", [])
            metadata_ids = [item.get("passage_id", item.get("chunk_id", "")) for item in metadata]
            if mapped_ids != metadata_ids:
                raise ValueError("FAISS position-to-metadata mapping mismatch")
            with open(metadata_path, "rb") as f:
                metadata_bytes = f.read()
            if manifest.get("metadata_sha256") != hashlib.sha256(metadata_bytes).hexdigest():
                raise ValueError("FAISS metadata checksum mismatch")
        else:
            logger.warning("Loading legacy FAISS index without a manifest: %s", index_path)

        if index.ntotal != len(metadata):
            raise ValueError("FAISS index and metadata count mismatch")
        logger.info("Loaded FAISS index (%d vectors) and %d metadata records.", index.ntotal, len(metadata))
        return index, metadata
    except Exception as exc:
        logger.error("Failed to load FAISS index or metadata: %s", exc)
        return None, []
