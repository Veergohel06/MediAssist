"""Configurable post-retrieval rerankers.

Reranker scores are ordering signals only. They are not probabilities,
clinical confidence, or evidence of medical certainty.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger("mediassist.rag.reranker")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class Reranker(Protocol):
    """Stable interface for post-retrieval candidate reranking."""

    name: str

    def rerank(self, query: str, candidates: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Return at most top_k candidates in reranked order."""


class RerankerError(RuntimeError):
    """Raised when a configured reranker cannot score candidates."""


class TokenOverlapReranker:
    """Small local reranker for deterministic operation and offline evaluation."""

    name = "token-overlap"

    def __init__(self, max_passage_chars: int = 4000):
        self.max_passage_chars = max(max_passage_chars, 100)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(_TOKEN_PATTERN.findall(text.lower()))

    def rerank(self, query: str, candidates: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        query_tokens = self._tokens(query)
        if not query_tokens:
            return [dict(candidate) for candidate in candidates[:top_k]]

        reranked = []
        for position, candidate in enumerate(candidates):
            item = dict(candidate)
            passage = str(item.get("text", ""))[: self.max_passage_chars]
            overlap = len(query_tokens & self._tokens(passage)) / len(query_tokens)
            item["rerank_score"] = float(overlap)
            item["reranker"] = self.name
            item["rerank_source_rank"] = position + 1
            reranked.append(item)
        reranked.sort(key=lambda item: (item["rerank_score"], item.get("rrf_score", 0.0)), reverse=True)
        return reranked[:top_k]


class SentenceTransformerReranker:
    """Optional local Sentence Transformers CrossEncoder reranker.

    The model is loaded once when this object is constructed. It is not loaded
    during individual requests. Model scores are ranking signals only.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_passage_chars: int = 4000,
        batch_size: int = 16,
        timeout_seconds: Optional[float] = None,
        model: Any = None,
    ):
        self.name = model_name
        self.max_passage_chars = max(max_passage_chars, 100)
        self.batch_size = max(batch_size, 1)
        self.timeout_seconds = timeout_seconds
        if model is not None:
            self.model = model
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "sentence-transformers is not installed; use TokenOverlapReranker or install dependencies"
            ) from exc
        try:
            self.model = CrossEncoder(model_name, device=device, max_length=512)
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(f"Unable to load reranker model '{model_name}'") from exc

    def rerank(self, query: str, candidates: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        pairs = [
            [query, str(candidate.get("text", ""))[: self.max_passage_chars]]
            for candidate in candidates
        ]
        started = time.perf_counter()
        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            raise RerankerError("Reranker inference failed") from exc
        elapsed = time.perf_counter() - started
        if self.timeout_seconds is not None and elapsed > self.timeout_seconds:
            raise RerankerError("Reranker inference exceeded configured timeout")

        reranked = []
        for position, (candidate, score) in enumerate(zip(candidates, scores)):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            item["reranker"] = self.name
            item["rerank_source_rank"] = position + 1
            reranked.append(item)
        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[:top_k]
