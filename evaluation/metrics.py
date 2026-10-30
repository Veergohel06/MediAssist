"""Reusable, non-clinical evaluation metrics for MediAssist experiments."""

import math
from typing import Any, Dict, Iterable, List, Sequence


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    selected = list(retrieved)[:k]
    if not relevant_set:
        return 1.0 if not selected else 0.0
    return sum(item in relevant_set for item in selected) / k if k else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0 if not list(retrieved)[:k] else 0.0
    return len(set(retrieved[:k]) & relevant_set) / len(relevant_set)


def mrr(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for rank, item in enumerate(retrieved, 1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevance_grades: Dict[str, int], k: int) -> float:
    def dcg(items: Sequence[str]) -> float:
        return sum((2 ** relevance_grades.get(item, 0) - 1) / math.log2(rank + 1) for rank, item in enumerate(items[:k], 1))

    actual = dcg(retrieved)
    ideal = dcg(sorted(relevance_grades, key=relevance_grades.get, reverse=True))
    return actual / ideal if ideal else 1.0


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((percentile_value / 100) * (len(ordered) - 1))))
    return float(ordered[index])


def latency_summary(values: Sequence[float]) -> Dict[str, float | None]:
    return {
        "mean_ms": sum(values) / len(values) if values else None,
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
    }


def citation_summary(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    audited = [item.get("citation_validation", {}) for item in predictions if item.get("citation_validation") is not None]
    if not audited:
        return {"validity_rate": None, "invalid_id_count": 0, "unknown_url_count": 0}
    return {
        "validity_rate": sum(bool(item.get("all_citations_valid")) for item in audited) / len(audited),
        "invalid_id_count": sum(len(item.get("invalid_citation_ids", [])) for item in audited),
        "unknown_url_count": sum(len(item.get("unknown_urls", [])) for item in audited),
    }


def abstention_summary(predictions: Sequence[Dict[str, Any]]) -> Dict[str, float | None]:
    measured = [item for item in predictions if item.get("actual_abstention") is not None]
    if not measured:
        return {"appropriate_abstention_rate": None, "unnecessary_refusal_rate": None}
    expected = [item for item in measured if item.get("expected_abstention")]
    non_expected = [item for item in measured if not item.get("expected_abstention")]
    return {
        "appropriate_abstention_rate": sum(item["actual_abstention"] for item in expected) / len(expected) if expected else None,
        "unnecessary_refusal_rate": sum(item["actual_abstention"] for item in non_expected) / len(non_expected) if non_expected else None,
    }


def retrieval_summary(predictions: Sequence[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    precisions = [item["precision_at_k"] for item in predictions]
    recalls = [item["recall_at_k"] for item in predictions]
    reciprocal_ranks = [item["mrr"] for item in predictions]
    ndcgs = [item["ndcg_at_k"] for item in predictions]
    latencies = [item["latency_ms"] for item in predictions]
    return {
        "case_count": len(predictions),
        "k": k,
        "precision_at_k": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "ndcg_at_k": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "latency_ms": latency_summary(latencies),
        "failure_count": sum(bool(item.get("failure")) for item in predictions),
    }
