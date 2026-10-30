"""
Embedding Generator Module
==========================
Generates text embeddings using Google Gemini's official API
via the google-genai SDK, with safe query caching, unchanged-document
cache reuse, rate-limit resilience, and telemetry tracking.
"""

import time
import logging
import hashlib
import json
import os
import uuid
from typing import List, Dict, Any, Optional
from google import genai

logger = logging.getLogger("mediassist.rag.embeddings")

DEFAULT_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "gemini-embedding-2")

# In-memory session cache (partitioned strictly by model & normalized text hash;
# never stores user identities, session IDs, or private health conversations).
_IN_MEMORY_CACHE: Dict[str, List[float]] = {}
_CACHES_PRELOADED = False

def _persist_to_query_cache(model: str, cache_key: str, embedding: List[float]) -> None:
    if not str(model).startswith("gemini-"):
        return
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        query_cache_path = os.path.join(base_dir, "knowledge", "query_embedding_cache.json")
        if os.path.exists(query_cache_path):
            with open(query_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
        data[cache_key] = embedding
        with open(query_cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as exc:
        logger.debug("Could not persist query embedding to disk: %s", exc)

# Telemetry tracking for API costs, latencies, retries, and cache efficacy
_METRICS: Dict[str, Any] = {
    "cache_hits": 0,
    "cache_misses": 0,
    "api_calls": 0,
    "api_retries": 0,
    "api_errors": 0,
    "latencies_ms": [],
}


def get_cache_key(model: str, text: str) -> str:
    """Generate a deterministic, privacy-preserving cache key from normalized text."""
    cleaned = text.strip()
    return hashlib.sha256(f"{model}\n{cleaned}".encode("utf-8")).hexdigest()


def _preload_known_caches() -> None:
    """Preload persisted knowledge base chunk and query caches into memory once."""
    global _CACHES_PRELOADED
    if _CACHES_PRELOADED:
        return
    _CACHES_PRELOADED = True

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_paths = [
        os.path.join(base_dir, "knowledge", "embedding_cache.json"),
        os.path.join(base_dir, "knowledge", "query_embedding_cache.json"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, list) and v:
                            _IN_MEMORY_CACHE[k] = v
                logger.info("Preloaded %d cached embedding vectors from %s", len(data), os.path.basename(path))
            except Exception as exc:
                logger.warning("Could not preload embedding cache from %s: %s", path, exc)


def get_embedding_metrics() -> Dict[str, Any]:
    """Return embedding metrics including cache hit rate and p95 latency."""
    total_requests = _METRICS["cache_hits"] + _METRICS["cache_misses"]
    hit_rate = (_METRICS["cache_hits"] / total_requests) if total_requests > 0 else 0.0
    latencies = sorted(_METRICS["latencies_ms"])
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0

    return {
        "cache_hits": _METRICS["cache_hits"],
        "cache_misses": _METRICS["cache_misses"],
        "total_requests": total_requests,
        "hit_rate": hit_rate,
        "api_calls": _METRICS["api_calls"],
        "api_retries": _METRICS["api_retries"],
        "api_errors": _METRICS["api_errors"],
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_mean_ms": mean_lat,
    }


def reset_embedding_metrics() -> None:
    """Reset telemetry counters for a fresh benchmark run."""
    _METRICS["cache_hits"] = 0
    _METRICS["cache_misses"] = 0
    _METRICS["api_calls"] = 0
    _METRICS["api_retries"] = 0
    _METRICS["api_errors"] = 0
    _METRICS["latencies_ms"] = []


def clear_in_memory_cache() -> None:
    """Clear memory cache (for testing cache miss scenarios)."""
    global _CACHES_PRELOADED
    _IN_MEMORY_CACHE.clear()
    _CACHES_PRELOADED = False


def get_embedding(
    text: str,
    client: Optional[genai.Client] = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    cache_path: Optional[str] = None,
    use_cache: bool = True,
) -> List[float]:
    """
    Generate an embedding vector for a single text query or document chunk.
    Utilizes multi-tier caching (in-memory & sidecar JSON) where safe to avoid
    costly network trips and quota exhaustion.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Cannot generate embedding for empty text.")

    _preload_known_caches()

    cache_key = get_cache_key(model, cleaned_text)
    # Check memory cache
    if use_cache:
        if cache_key in _IN_MEMORY_CACHE:
            _METRICS["cache_hits"] += 1
            _METRICS["latencies_ms"].append(0.05)
            return _IN_MEMORY_CACHE[cache_key]
        if cleaned_text in _IN_MEMORY_CACHE:
            _METRICS["cache_hits"] += 1
            _METRICS["latencies_ms"].append(0.05)
            return _IN_MEMORY_CACHE[cleaned_text]

    # Check sidecar disk cache if provided
    if use_cache and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                disk_cache = json.load(handle)
                if cache_key in disk_cache:
                    _IN_MEMORY_CACHE[cache_key] = disk_cache[cache_key]
                    _METRICS["cache_hits"] += 1
                    _METRICS["latencies_ms"].append(0.1)
                    return disk_cache[cache_key]
                if cleaned_text in disk_cache:
                    _IN_MEMORY_CACHE[cache_key] = disk_cache[cleaned_text]
                    _METRICS["cache_hits"] += 1
                    _METRICS["latencies_ms"].append(0.1)
                    return disk_cache[cleaned_text]
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read disk cache at %s", cache_path)

    # Cache miss
    _METRICS["cache_misses"] += 1

    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("API key or client required for uncached embedding.")
        client = genai.Client(api_key=api_key)

    start_time = time.perf_counter()
    _METRICS["api_calls"] += 1

    # Retry loop with backoff for rate limits / transient API failures
    for attempt in range(4):
        try:
            res = client.models.embed_content(
                model=model,
                contents=cleaned_text,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _METRICS["latencies_ms"].append(elapsed_ms)
            embedding = res.embeddings[0].values

            # Populate in-memory cache
            _IN_MEMORY_CACHE[cache_key] = embedding
            _persist_to_query_cache(model, cache_key, embedding)
            return embedding
        except Exception as exc:
            exc_str = str(exc)
            if "RESOURCE_EXHAUSTED" in exc_str or "quota" in exc_str.lower():
                _METRICS["api_errors"] += 1
                logger.warning("Daily embedding quota exhausted: %s. Using cached vector if available.", exc_str[:80])
                if cache_key in _IN_MEMORY_CACHE:
                    return _IN_MEMORY_CACHE[cache_key]
                if cleaned_text in _IN_MEMORY_CACHE:
                    return _IN_MEMORY_CACHE[cleaned_text]
                if model != "gemini-embedding-2":
                    try:
                        logger.info("Attempting embedding fallback to gemini-embedding-2...")
                        res = client.models.embed_content(
                            model="gemini-embedding-2",
                            contents=cleaned_text,
                        )
                        embedding = res.embeddings[0].values
                        _IN_MEMORY_CACHE[cache_key] = embedding
                        _persist_to_query_cache("gemini-embedding-2", cache_key, embedding)
                        return embedding
                    except Exception as fallback_exc:
                        logger.warning("Fallback to gemini-embedding-2 failed: %s", fallback_exc)
                raise

            _METRICS["api_retries"] += 1
            if attempt == 3:
                _METRICS["api_errors"] += 1
                logger.error("Failed to generate embedding after 4 attempts: %s", exc)
                if cache_key in _IN_MEMORY_CACHE:
                    return _IN_MEMORY_CACHE[cache_key]
                raise
            wait_time = (attempt + 1) * 1.5
            logger.warning("Transient embedding error (%s). Retrying in %.1fs...", exc, wait_time)
            time.sleep(wait_time)

    raise RuntimeError("Embedding generation failed.")


def get_embeddings_batch(
    texts: List[str],
    client: Optional[genai.Client] = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 15,
    delay_between_batches: float = 0.5,
    cache_path: Optional[str] = None,
    use_cache: bool = True,
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts sequentially or in batches,
    respecting rate limits, reusing unchanged documents from cache,
    and persisting newly embedded items.
    """
    _preload_known_caches()
    all_embeddings: List[List[float]] = []
    disk_cache: Dict[str, List[float]] = {}

    if use_cache and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                disk_cache = json.load(handle)
        except (OSError, json.JSONDecodeError):
            logger.warning("Embedding cache could not be read; rebuilding missing entries.")

    def persist_cache() -> None:
        if not cache_path or not disk_cache:
            return
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        temporary_path = f"{cache_path}.{uuid.uuid4().hex}.tmp"
        try:
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(disk_cache, handle)
            os.replace(temporary_path, cache_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if api_key:
            client = genai.Client(api_key=api_key)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info("Processing embedding batch %d to %d of %d chunks...", i + 1, min(i + batch_size, len(texts)), len(texts))

        for item in batch:
            cleaned = item.strip()
            if not cleaned:
                cleaned = "empty"
            cache_key = get_cache_key(model, cleaned)

            # 1. Check in-memory & disk caches
            if use_cache:
                if cache_key in _IN_MEMORY_CACHE:
                    all_embeddings.append(_IN_MEMORY_CACHE[cache_key])
                    _METRICS["cache_hits"] += 1
                    continue
                if cache_key in disk_cache:
                    emb = disk_cache[cache_key]
                    _IN_MEMORY_CACHE[cache_key] = emb
                    all_embeddings.append(emb)
                    _METRICS["cache_hits"] += 1
                    continue

            # 2. Need API call
            if client is None:
                raise RuntimeError("API client required for uncached texts in get_embeddings_batch")

            _METRICS["cache_misses"] += 1
            _METRICS["api_calls"] += 1
            start_time = time.perf_counter()

            for attempt in range(4):
                try:
                    res = client.models.embed_content(
                        model=model,
                        contents=cleaned,
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    _METRICS["latencies_ms"].append(elapsed_ms)
                    embedding = res.embeddings[0].values
                    disk_cache[cache_key] = embedding
                    _IN_MEMORY_CACHE[cache_key] = embedding
                    all_embeddings.append(embedding)
                    break
                except Exception as exc:
                    exc_str = str(exc)
                    if "RESOURCE_EXHAUSTED" in exc_str or "quota" in exc_str.lower():
                        _METRICS["api_errors"] += 1
                        logger.warning("Daily embedding quota exhausted: %s. Using cached vector.", exc_str[:80])
                        if cache_key in _IN_MEMORY_CACHE:
                            all_embeddings.append(_IN_MEMORY_CACHE[cache_key])
                            break
                        raise

                    _METRICS["api_retries"] += 1
                    if attempt == 3:
                        _METRICS["api_errors"] += 1
                        logger.error("Failed to embed chunk after 4 attempts: %s", exc)
                        raise
                    wait_time = (attempt + 1) * 2.0
                    logger.warning("Transient embedding error (%s). Retrying in %.1fs...", exc, wait_time)
                    time.sleep(wait_time)

        if i + batch_size < len(texts) and delay_between_batches > 0:
            time.sleep(delay_between_batches)
        if use_cache:
            persist_cache()

    return all_embeddings

