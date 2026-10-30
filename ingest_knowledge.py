"""
MediAssist — Knowledge Ingestion & Indexing Pipeline
====================================================
Parses curated medical documents from the knowledge/ directory, splits them
into manageable chunks, generates vector embeddings using Google Gemini's
official API, and constructs a FAISS vector index.

Usage:
    python ingest_knowledge.py [--knowledge-dir knowledge] [--output-dir knowledge]
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Load environment before importing Gemini
load_dotenv()

from google import genai
from rag.chunker import load_and_chunk_documents
from rag.embeddings import get_embeddings_batch, DEFAULT_EMBEDDING_MODEL
from rag.indexer import build_and_save_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mediassist.ingest")


def run_ingestion(
    knowledge_dir: str = "knowledge",
    output_dir: str = "knowledge",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    max_chunk_size: int = 650,
    chunk_overlap: int = 100,
    embedding_batch_size: int = 15,
    embedding_cache: bool = True,
) -> bool:
    """
    Execute the end-to-end ingestion pipeline:
    1. Scan & parse medical documents from knowledge_dir
    2. Chunk documents with semantic metadata
    3. Generate embeddings using Gemini
    4. Build & save FAISS IndexFlatIP + metadata JSON
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set. Cannot generate embeddings.")
        sys.exit(1)

    print("=" * 65)
    print("  MediAssist Knowledge Ingestion & FAISS Indexing Pipeline")
    print(f"  Source Directory : {knowledge_dir}")
    print(f"  Embedding Model  : {embedding_model}")
    print("=" * 65)

    # 1. Load and chunk documents
    logger.info("Step 1: Reading and chunking curated documents from '%s'...", knowledge_dir)
    chunks = load_and_chunk_documents(
        knowledge_dir,
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        logger.error("No documents or chunks found in '%s'. Ingestion aborted.", knowledge_dir)
        return False

    logger.info("Found and created %d chunks across medical topics.", len(chunks))
    for i, ch in enumerate(chunks[:3], 1):
        logger.info("  Sample Chunk %d: [%s] '%s' (%d chars)", i, ch['source_title'], ch['section'], len(ch['text']))

    # 2. Generate embeddings
    logger.info("Step 2: Generating embeddings via Gemini API (%s)...", embedding_model)
    client = genai.Client(api_key=api_key)
    texts_to_embed = [ch["text"] for ch in chunks]

    try:
        embeddings = get_embeddings_batch(
            texts=texts_to_embed,
            client=client,
            model=embedding_model,
            batch_size=embedding_batch_size,
            cache_path=os.path.join(output_dir, "embedding_cache.json") if embedding_cache else None,
        )
    except Exception as exc:
        logger.error("Failed to generate embeddings: %s", exc)
        return False

    logger.info("Generated %d embeddings (dimension = %d).", len(embeddings), len(embeddings[0]))

    # 3. Build & save FAISS index
    index_path = os.path.join(output_dir, "faiss_index.bin")
    metadata_path = os.path.join(output_dir, "chunks_metadata.json")

    logger.info("Step 3: Building FAISS index and saving to '%s'...", output_dir)
    try:
        build_and_save_index(
            chunks=chunks,
            embeddings=embeddings,
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_model=embedding_model,
        )
    except Exception as exc:
        logger.error("Failed to build FAISS index: %s", exc)
        return False

    from rag.embeddings import get_embedding_metrics
    emb_metrics = get_embedding_metrics()
    print("\n" + "=" * 65)
    print("  Ingestion complete! Knowledge base is ready for RAG retrieval.")
    print(f"  Total chunks indexed : {len(chunks)}")
    print(f"  Embedding cache hits : {emb_metrics['cache_hits']} (avoided re-embedding unchanged docs)")
    print(f"  API embedding calls  : {emb_metrics['api_calls']}")
    print(f"  Index file           : {index_path}")
    print(f"  Metadata file        : {metadata_path}")
    print("=" * 65)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest medical documents and build FAISS vector index.")
    parser.add_argument("--knowledge-dir", default="knowledge", help="Directory containing source markdown documents")
    parser.add_argument("--output-dir", default="knowledge", help="Directory where FAISS index and metadata will be saved")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Gemini embedding model name")
    parser.add_argument("--max-chunk-size", type=int, default=650)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--embedding-batch-size", type=int, default=15)
    parser.add_argument("--no-embedding-cache", action="store_true")
    args = parser.parse_args()

    success = run_ingestion(
        knowledge_dir=args.knowledge_dir,
        output_dir=args.output_dir,
        embedding_model=args.model,
        max_chunk_size=args.max_chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_batch_size=args.embedding_batch_size,
        embedding_cache=not args.no_embedding_cache,
    )
    sys.exit(0 if success else 1)
