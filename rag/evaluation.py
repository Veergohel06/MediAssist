"""Offline retrieval and evidence-grounding evaluation utilities."""

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


def load_evaluation_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """Load reviewed synthetic evaluation cases from JSON."""
    with open(path, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, list):
        raise ValueError("Evaluation dataset must be a JSON list.")
    return dataset


def _ids(results: Sequence[Dict[str, Any]]) -> List[str]:
    return [item.get("passage_id", item.get("chunk_id", "")) for item in results]


def _canonical_id(identifier: str) -> str:
    """Accept legacy chunk IDs in labels while evaluating passage IDs."""
    match = re.match(r"^(.+)_([0-9]+)$", identifier)
    if match:
        return f"{match.group(1)}:p{int(match.group(2)):04d}"
    return identifier


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    retrieved = list(retrieved_ids)[:k]
    if not relevant:
        return 1.0 if not retrieved else 0.0
    return sum(item in relevant for item in retrieved) / k if k else 0.0


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0 if not list(retrieved_ids)[:k] else 0.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def mean_reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    """Return a nearest-rank percentile without requiring NumPy."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(percentile_value / 100 * len(ordered)))))
    return float(ordered[rank - 1])


def citation_support_rate(
    cited_ids: Sequence[str],
    supported_ids: Iterable[str],
) -> float | None:
    """Measure cited claims supported by the case's reviewed passage labels."""
    cited = {_canonical_id(item) for item in cited_ids}
    supported = {_canonical_id(item) for item in supported_ids}
    if not cited:
        return None
    return len(cited & supported) / len(cited)


def evaluate_retriever(
    retriever: Any,
    cases: Sequence[Dict[str, Any]],
    top_k: int = 5,
    query_fn: Callable[[str, int], Sequence[Dict[str, Any]]] | None = None,
) -> Dict[str, Any]:
    """Evaluate a retriever without making answer-generation API calls."""
    query_fn = query_fn or retriever.retrieve
    rows = []
    precision = []
    recall = []
    reciprocal_rank = []
    abstention_correct = []
    latencies = []
    failure_count = 0
    rerank_failures = 0
    rerank_latencies = []

    for case in cases:
        started = time.perf_counter()
        results = list(query_fn(case["query"], top_k))
        if getattr(retriever, "last_retrieval_error", None):
            failure_count += 1
        if getattr(retriever, "last_rerank_error", None):
            rerank_failures += 1
        rerank_latencies.append(float(getattr(retriever, "last_rerank_latency_ms", 0.0)))
        latencies.append((time.perf_counter() - started) * 1000)
        retrieved_ids = [_canonical_id(item) for item in _ids(results)]
        relevant_ids = [_canonical_id(item) for item in case.get("expected_relevant_ids", [])]
        precision.append(precision_at_k(retrieved_ids, relevant_ids, top_k))
        recall.append(recall_at_k(retrieved_ids, relevant_ids, top_k))
        reciprocal_rank.append(mean_reciprocal_rank(retrieved_ids, relevant_ids))
        should_abstain = bool(case.get("abstain_expected", False))
        abstention_correct.append((not retrieved_ids) == should_abstain)
        rows.append({
            "case_id": case["case_id"],
            "category": case.get("category", "uncategorized"),
            "retrieved_ids": retrieved_ids,
            "expected_relevant_ids": relevant_ids,
            "precision_at_k": precision[-1],
            "recall_at_k": recall[-1],
            "mrr": reciprocal_rank[-1],
            "abstention_correct": abstention_correct[-1],
            "latency_ms": latencies[-1],
            "rerank_latency_ms": rerank_latencies[-1],
        })

    count = len(rows)
    return {
        "case_count": count,
        "failure_count": failure_count,
        "rerank_failure_count": rerank_failures,
        "precision_at_k": sum(precision) / count if count else 0.0,
        "recall_at_k": sum(recall) / count if count else 0.0,
        "mrr": sum(reciprocal_rank) / count if count else 0.0,
        "abstention_accuracy": sum(abstention_correct) / count if count else 0.0,
        "mean_latency_ms": sum(latencies) / count if count else 0.0,
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "rerank_latency_ms": {
            "p50": percentile(rerank_latencies, 50),
            "p95": percentile(rerank_latencies, 95),
            "mean": sum(rerank_latencies) / count if count else 0.0,
        },
        "citation_support_rate": None,
        "citation_validity": None,
        "answer_relevance": None,
        "factual_consistency": None,
        "model_usage": {"embedding_calls": None, "generation_calls": 0, "estimated_cost_usd": None},
        "cases": rows,
    }


def evaluate_answer_records(
    retriever: Any,
    records: Sequence[Dict[str, Any]],
    judge: Optional[Any] = None,
) -> Dict[str, Any]:
    """Measure citations and reviewed answer labels when answer records exist.

    If judge is provided, automatically scores answer_relevance, factual_consistency,
    and accessibility_and_safety when not pre-populated.
    """
    if not records:
        return {"case_count": 0, "citation_validity": None, "answer_relevance": None, "factual_consistency": None}

    citation_scores = []
    support_scores = []
    relevance_scores = []
    consistency_scores = []
    accessibility_scores = []

    for record in records:
        validation = retriever.validate_citations(record.get("answer", ""), record.get("retrieved_chunks", []))
        citation_scores.append(float(validation["all_valid"]))
        support = citation_support_rate(
            validation["valid_ids"],
            record.get("supported_passage_ids", []),
        )
        if support is not None:
            support_scores.append(support)

        if "answer_relevance" in record and "factual_consistency" in record:
            relevance_scores.append(float(record["answer_relevance"]))
            consistency_scores.append(float(record["factual_consistency"]))
            if "accessibility_and_safety" in record:
                accessibility_scores.append(float(record["accessibility_and_safety"]))
        elif judge is not None:
            judge_res = judge.evaluate_answer(
                query=record.get("query", ""),
                retrieved_chunks=record.get("retrieved_chunks", []),
                answer=record.get("answer", ""),
                case_id=record.get("case_id"),
            )
            relevance_scores.append(judge_res["answer_relevance"])
            consistency_scores.append(judge_res["factual_consistency"])
            accessibility_scores.append(judge_res.get("accessibility_and_safety", 0.9))
        else:
            if "answer_relevance" in record:
                relevance_scores.append(float(record["answer_relevance"]))
            if "factual_consistency" in record:
                consistency_scores.append(float(record["factual_consistency"]))

    return {
        "case_count": len(records),
        "citation_validity": sum(citation_scores) / len(citation_scores) if citation_scores else None,
        "citation_support_rate": sum(support_scores) / len(support_scores) if support_scores else None,
        "answer_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else None,
        "factual_consistency": sum(consistency_scores) / len(consistency_scores) if consistency_scores else None,
        "accessibility_and_safety": sum(accessibility_scores) / len(accessibility_scores) if accessibility_scores else None,
    }


def compare_retrievers(
    baseline: Any,
    upgraded: Any,
    cases: Sequence[Dict[str, Any]],
    top_k: int = 5,
    query_embedding_fn: Callable[[str], List[float]] | None = None,
) -> Dict[str, Any]:
    """Produce directly comparable baseline and upgraded retrieval metrics."""
    if query_embedding_fn is None:
        baseline_report = evaluate_retriever(baseline, cases, top_k=top_k)
        upgraded_report = evaluate_retriever(upgraded, cases, top_k=top_k)
    else:
        cached_embeddings: Dict[str, List[float]] = {}
        embedding_latencies: Dict[str, float] = {}
        for case in cases:
            query = case["query"]
            if query not in cached_embeddings:
                started = time.perf_counter()
                cached_embeddings[query] = query_embedding_fn(query)
                embedding_latencies[query] = (time.perf_counter() - started) * 1000

        def evaluate_with_shared_embeddings(retriever: Any) -> Dict[str, Any]:
            def query_fn(query: str, requested_k: int) -> Sequence[Dict[str, Any]]:
                if query not in cached_embeddings:
                    cached_embeddings[query] = query_embedding_fn(query)
                return retriever.retrieve(query, top_k=requested_k, query_embedding=cached_embeddings[query])

            return evaluate_retriever(retriever, cases, top_k=top_k, query_fn=query_fn)

        baseline_report = evaluate_with_shared_embeddings(baseline)
        upgraded_report = evaluate_with_shared_embeddings(upgraded)
        embedding_latency_values = [embedding_latencies[case["query"]] for case in cases]
        for report in (baseline_report, upgraded_report):
            retrieval_latencies = [row["latency_ms"] for row in report["cases"]]
            total_latencies = [retrieval + embedding for retrieval, embedding in zip(retrieval_latencies, embedding_latency_values)]
            for row, total in zip(report["cases"], total_latencies):
                row["retrieval_latency_ms"] = row["latency_ms"]
                row["latency_ms"] = total
            report["retrieval_latency_ms"] = {
                "mean": sum(retrieval_latencies) / len(retrieval_latencies),
                "p50": percentile(retrieval_latencies, 50),
                "p95": percentile(retrieval_latencies, 95),
            }
            report["embedding_latency_ms"] = {
                "mean": sum(embedding_latency_values) / len(embedding_latency_values),
                "p50": percentile(embedding_latency_values, 50),
                "p95": percentile(embedding_latency_values, 95),
            }
            report["mean_latency_ms"] = sum(total_latencies) / len(total_latencies)
            report["latency_ms"] = {
                "p50": percentile(total_latencies, 50),
                "p95": percentile(total_latencies, 95),
            }
    return {
        "top_k": top_k,
        "baseline": baseline_report,
        "upgraded": upgraded_report,
        "delta": {
            metric: upgraded_report[metric] - baseline_report[metric]
            for metric in ("precision_at_k", "recall_at_k", "mrr", "abstention_accuracy")
        },
        "note": "Metrics are measurements from the supplied dataset and retrievers; no improvement is assumed.",
    }


def compare_retrieval_modes(
    retrievers: Dict[str, Any],
    cases: Sequence[Dict[str, Any]],
    top_k: int = 5,
    query_embedding_fn: Callable[[str], List[float]] | None = None,
) -> Dict[str, Any]:
    """Evaluate keyword, semantic, and hybrid modes against identical queries."""
    cached_embeddings: Dict[str, List[float]] = {}
    embedding_latencies: Dict[str, float] = {}
    if query_embedding_fn is not None:
        for case in cases:
            query = case["query"]
            if query not in cached_embeddings:
                started = time.perf_counter()
                cached_embeddings[query] = query_embedding_fn(query)
                embedding_latencies[query] = (time.perf_counter() - started) * 1000

    reports = {}
    for mode, retriever in retrievers.items():
        if query_embedding_fn is None or mode == "keyword":
            reports[mode] = evaluate_retriever(retriever, cases, top_k=top_k)
            continue

        def query_fn(query: str, requested_k: int) -> Sequence[Dict[str, Any]]:
            return retriever.retrieve(query, top_k=requested_k, query_embedding=cached_embeddings[query])

        report = evaluate_retriever(retriever, cases, top_k=top_k, query_fn=query_fn)
        embedding_values = [embedding_latencies[case["query"]] for case in cases]
        retrieval_values = [row["latency_ms"] for row in report["cases"]]
        total_values = [retrieval + embedding for retrieval, embedding in zip(retrieval_values, embedding_values)]
        for row, total, retrieval in zip(report["cases"], total_values, retrieval_values):
            row["retrieval_latency_ms"] = retrieval
            row["latency_ms"] = total
        report["mean_latency_ms"] = sum(total_values) / len(total_values) if total_values else 0.0
        report["latency_ms"] = {"p50": percentile(total_values, 50), "p95": percentile(total_values, 95)}
        report["retrieval_latency_ms"] = {"p50": percentile(retrieval_values, 50), "p95": percentile(retrieval_values, 95)}
        reports[mode] = report

    category_comparison: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode, report in reports.items():
        for row in report["cases"]:
            category = row["category"]
            category_comparison.setdefault(category, {}).setdefault(mode, {"precision_at_k": [], "recall_at_k": [], "mrr": []})
            for metric in ("precision_at_k", "recall_at_k", "mrr"):
                category_comparison[category][mode][metric].append(row[metric])
    category_summary = {}
    for category, mode_values in category_comparison.items():
        category_summary[category] = {}
        for mode, values in mode_values.items():
            category_summary[category][mode] = {
                metric: sum(scores) / len(scores) if scores else 0.0
                for metric, scores in values.items()
            }

    return {
        "top_k": top_k,
        "modes": reports,
        "category_summary": category_summary,
        "note": "All modes use the same evaluation cases and relevance labels. Metrics are measurements; no improvement is assumed.",
    }