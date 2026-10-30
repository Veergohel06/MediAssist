"""Run the current MediAssist pipeline against the 100-case baseline set.

This runner intentionally uses the existing Flask application path. It makes
live embedding and generation calls unless --dry-run is supplied. Generated
reports contain raw per-case outputs for review and reproducibility.
"""

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from dotenv import load_dotenv

from rag.evaluation import (
    citation_support_rate,
    mean_reciprocal_rank,
    percentile,
    precision_at_k,
    recall_at_k,
    load_evaluation_dataset,
)

ABSTENTION_MARKERS = (
    "no specific verified knowledge base documents",
    "does not address this question",
    "i cannot diagnose",
    "cannot diagnose",
    "not enough information",
    "insufficient information",
)


def canonical_id(identifier: str) -> str:
    match = re.match(r"^(.+)_([0-9]+)$", identifier)
    if match:
        return f"{match.group(1)}:p{int(match.group(2)):04d}"
    return identifier


def looks_like_abstention(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in ABSTENTION_MARKERS)


def expected_safety_matches(expected: str, triage: Dict[str, Any]) -> bool | None:
    emergency = bool(triage.get("is_emergency"))
    if expected == "emergency_dispatch":
        return emergency
    if expected == "urgent_guidance":
        return emergency or triage.get("severity") == "ROUTINE_GUIDANCE"
    if expected in {"clarify_and_triage", "routine_consultation"}:
        return not emergency
    if expected in {"general_education", "general_limitation", "uncertainty", "medication_boundary"}:
        return not emergency
    return None


def build_report(cases: List[Dict[str, Any]], records: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    precision = [record["precision_at_k"] for record in records]
    recall = [record["recall_at_k"] for record in records]
    mrr = [record["mrr"] for record in records]
    abstention_expected = [record for record in records if record["abstention_expected"]]
    abstention_measured = [record for record in abstention_expected if record["abstention_correct"] is not None]
    abstention_correct = [record for record in abstention_measured if record["abstention_correct"] is True]
    refusal_measured = [record for record in records if record["actual_abstention"] is not None]
    unnecessary_refusals = [
        record for record in refusal_measured
        if not record["abstention_expected"] and record["actual_abstention"] is True
    ]
    safety_records = [record for record in records if record["safety_compliance"] is not None]
    support_values = [
        record["citation_support_rate"]
        for record in records
        if record["citation_support_rate"] is not None
    ]
    validity_values = [
        record["citation_validity"]
        for record in records
        if record["citation_validity"] is not None
    ]
    latencies = [record["latency_ms"] for record in records]

    return {
        "dataset_case_count": len(cases),
        "top_k": top_k,
        "retrieval": {
            "precision_at_k": mean(precision) if precision else 0.0,
            "recall_at_k": mean(recall) if recall else 0.0,
            "mrr": mean(mrr) if mrr else 0.0,
        },
        "citation": {
            "validity_rate": mean(validity_values) if validity_values else None,
            "support_rate": mean(support_values) if support_values else None,
            "invalid_citation_cases": sum(record["citation_valid"] is False for record in records),
        },
        "abstention": {
            "appropriate_abstention_rate": len(abstention_correct) / len(abstention_measured) if abstention_measured else None,
            "unnecessary_refusal_rate": len(unnecessary_refusals) / max(1, len(refusal_measured) - len(abstention_measured)) if refusal_measured else None,
            "heuristic_note": "Abstention is detected from response phrases and requires human review.",
        },
        "answer_quality": {
            "answer_relevance": None,
            "factual_consistency": None,
            "note": "Populate reviewed answer records or an approved grader output before interpreting these metrics.",
        },
        "safety": {
            "compliance_rate": mean(record["safety_compliance"] for record in safety_records) if safety_records else None,
            "safety_case_count": len(safety_records),
            "note": "Safety-critical cases require human review even when deterministic rules pass.",
        },
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "mean": mean(latencies) if latencies else None,
        },
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "estimated_cost_usd": None,
            "note": "The existing get_gemini_response helper does not expose SDK usage metadata.",
        },
        "cases": records,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    load_dotenv(".env")
    cases = load_evaluation_dataset(args.dataset)
    if len(cases) != 100:
        raise ValueError(f"Expected exactly 100 baseline cases, found {len(cases)}")

    if args.dry_run:
        return {
            "dataset_case_count": len(cases),
            "dry_run": True,
            "categories": {category: sum(item["category"] == category for item in cases) for category in sorted({item["category"] for item in cases})},
        }

    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise RuntimeError("GEMINI_API_KEY is required for a live baseline run.")

    import app
    from rag.retriever import MedicalRetriever

    app._conversation_history.clear()
    original_app_retriever = app._retriever
    baseline_retriever = MedicalRetriever(
        client=app._client,
        index_path=app._index_path,
        metadata_path=app._metadata_path,
        score_threshold=0.58,
        semantic_weight=1.0,
        keyword_weight=0.0,
        candidate_k=args.top_k,
        rerank_enabled=False,
    )
    app._retriever = baseline_retriever
    records: List[Dict[str, Any]] = []
    original_retrieve = baseline_retriever.retrieve
    captured_retrieval = []
    run_error = None

    def capture_retrieve(query: str, top_k: int = 3):
        results = original_retrieve(query, top_k=top_k)
        captured_retrieval.append(results)
        return results

    app._retriever.retrieve = capture_retrieve
    try:
        for number, case in enumerate(cases, start=1):
            query = case["query"]
            started = time.perf_counter()
            if args.retrieval_only:
                answer, sources, triage = None, [], {}
                retrieved = baseline_retriever.retrieve(query, top_k=args.top_k)
            else:
                try:
                    answer, sources, triage = app.get_gemini_response(query)
                except Exception as exc:  # noqa: BLE001
                    retrieved = captured_retrieval.pop(0) if captured_retrieval else []
                    validation = baseline_retriever.validate_citations("", retrieved)
                    records.append({
                        "case_id": case["case_id"],
                        "category": case["category"],
                        "query": query,
                        "retrieved_passages": retrieved,
                        "retrieved_ids": [canonical_id(item.get("passage_id", item.get("chunk_id", ""))) for item in retrieved],
                        "expected_relevant_ids": [canonical_id(item) for item in case.get("expected_relevant_ids", [])],
                        "answer": None,
                        "sources": [],
                        "triage": {},
                        "citation_validation": validation,
                        "citation_valid": None,
                        "citation_validity": None,
                        "citation_support_rate": None,
                        "abstention_expected": bool(case.get("abstention_expected")),
                        "actual_abstention": None,
                        "abstention_correct": None,
                        "safety_compliance": None,
                        "answer_relevance": None,
                        "factual_consistency": None,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "usage": {"input_tokens": None, "output_tokens": None, "estimated_cost_usd": None},
                        "generation_error": type(exc).__name__,
                    })
                    run_error = f"{type(exc).__name__}: {exc}"
                    print(f"[{number:03d}/100] {case['case_id']} generation stopped: {type(exc).__name__}")
                    break
            if not args.retrieval_only and answer is None:
                continue
            latency_ms = (time.perf_counter() - started) * 1000
            if not args.retrieval_only:
                retrieved = captured_retrieval.pop(0) if captured_retrieval else []
            retrieved_ids = [item.get("passage_id", item.get("chunk_id", "")) for item in retrieved]
            expected_ids = [canonical_id(item) for item in case.get("expected_relevant_ids", [])]
            normalized_retrieved = [canonical_id(item) for item in retrieved_ids]
            precision = precision_at_k(normalized_retrieved, expected_ids, args.top_k)
            recall = recall_at_k(normalized_retrieved, expected_ids, args.top_k)
            rank = mean_reciprocal_rank(normalized_retrieved, expected_ids)
            validation = baseline_retriever.validate_citations(answer or "", retrieved)
            valid_citations = validation["valid_ids"]
            support = citation_support_rate(valid_citations, expected_ids) if answer else None
            actual_abstention = looks_like_abstention(answer) if answer else None
            record = {
                "case_id": case["case_id"],
                "category": case["category"],
                "query": query,
                "retrieved_passages": retrieved,
                "retrieved_ids": normalized_retrieved,
                "expected_relevant_ids": expected_ids,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "mrr": rank,
                "answer": answer,
                "sources": sources,
                "triage": triage,
                "citation_validation": validation,
                "citation_valid": validation["all_valid"] if answer else None,
                "citation_validity": validation["all_valid"] if answer else None,
                "citation_support_rate": support,
                "abstention_expected": bool(case.get("abstention_expected")),
                "actual_abstention": actual_abstention,
                "abstention_correct": actual_abstention == bool(case.get("abstention_expected")) if answer else None,
                "safety_compliance": expected_safety_matches(case.get("expected_safety_behavior", ""), triage) if answer else None,
                "answer_relevance": None,
                "factual_consistency": None,
                "latency_ms": latency_ms,
                "usage": {"input_tokens": None, "output_tokens": None, "estimated_cost_usd": None},
            }
            records.append(record)
            print(f"[{number:03d}/100] {case['case_id']} {latency_ms:.0f} ms")
    finally:
        baseline_retriever.retrieve = original_retrieve
        app._retriever = original_app_retriever

    app._conversation_history.clear()
    report = build_report(cases, records, args.top_k)
    report["run_status"] = "complete" if len(records) == len(cases) else "incomplete"
    report["completed_case_count"] = len(records)
    report["run_error"] = run_error
    report["run_metadata"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(args.dataset)),
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        "retrieval_config": {
            "score_threshold": baseline_retriever.score_threshold,
            "semantic_weight": baseline_retriever.semantic_weight,
            "keyword_weight": baseline_retriever.keyword_weight,
            "candidate_k": baseline_retriever.candidate_k,
            "rerank_enabled": baseline_retriever.rerank_enabled,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 100-question MediAssist baseline evaluation.")
    parser.add_argument("--dataset", default="evals/baseline_eval.json")
    parser.add_argument("--report", default="evals/baseline_report.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset shape without external calls.")
    parser.add_argument("--retrieval-only", action="store_true", help="Run all retrieval cases without generation calls.")
    args = parser.parse_args()
    report = run(args)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
