"""Repeatable MediAssist evaluation runner.

Examples from the repository root:
  python evaluation/run_eval.py --mode mock
  python evaluation/run_eval.py --mode hybrid --dataset evaluation/dataset.jsonl
  python evaluation/run_eval.py --mode full_generation --dataset evaluation/dataset.jsonl

API-backed modes are opt-in. Mock mode is deterministic and intended for CI;
its metrics describe the mock, not production quality.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from rag.embeddings import get_embedding
from evaluation.metrics import (
    abstention_summary,
    citation_summary,
    ndcg_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
    retrieval_summary,
)
from rag.reranker import TokenOverlapReranker
from rag.retriever import MedicalRetriever


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("id") or "question" not in row:
            raise ValueError(f"Invalid evaluation row at {path}:{line_number}")
        rows.append(row)
    return rows


def build_retriever(mode: str, client: Any = None, settings: Dict[str, Any] | None = None) -> MedicalRetriever:
    if client is None:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    settings = settings or {}
    kwargs = {
        "client": client,
        "index_path": str(ROOT / "knowledge" / "faiss_index.bin"),
        "metadata_path": str(ROOT / "knowledge" / "chunks_metadata.json"),
        "embedding_model": os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        "retrieval_mode": "hybrid" if mode == "hybrid_reranked" else mode,
        "semantic_candidate_k": settings.get("semantic_candidate_k", 20 if mode == "hybrid_reranked" else 12),
        "keyword_candidate_k": settings.get("keyword_candidate_k", 20 if mode == "hybrid_reranked" else 12),
        "candidate_k": settings.get("candidate_k", 20 if mode == "hybrid_reranked" else 12),
        "score_threshold": settings.get("score_threshold", 0.58),
    }
    if mode == "hybrid_reranked":
        kwargs.update({
            "rerank_enabled": True,
            "reranker": TokenOverlapReranker(),
            "rerank_candidate_k": settings.get("rerank_candidate_k", 20),
        })
    return MedicalRetriever(**kwargs)


def run_retrieval(cases: List[Dict[str, Any]], mode: str, settings: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("API-backed evaluation requires GEMINI_API_KEY")
    retriever = build_retriever(mode, settings=settings)
    query_embeddings: Dict[str, List[float]] = {}
    predictions = []
    for case in cases:
        question = case["question"]
        started = time.perf_counter()
        query_embedding = None
        if mode in {"semantic", "hybrid", "hybrid_reranked"}:
            if question not in query_embeddings:
                query_embeddings[question] = get_embedding(question, retriever.client, retriever.embedding_model)
            query_embedding = query_embeddings[question]
        results = retriever.retrieve(question, top_k=(settings or {}).get("top_k", 5), query_embedding=query_embedding)
        retrieved_ids = [item.get("passage_id", item.get("chunk_id", "")) for item in results]
        latency_ms = (time.perf_counter() - started) * 1000
        predictions.append({
            "id": case["id"],
            "question": question,
            "mode": mode,
            "retrieved_ids": retrieved_ids,
            "retrieved_passages": results,
            "expected_relevant_ids": case.get("relevant_chunk_ids", []),
            "precision_at_k": precision_at_k(retrieved_ids, case.get("relevant_chunk_ids", []), 5),
            "recall_at_k": recall_at_k(retrieved_ids, case.get("relevant_chunk_ids", []), 5),
            "mrr": mrr(retrieved_ids, case.get("relevant_chunk_ids", [])),
            "ndcg_at_k": ndcg_at_k(retrieved_ids, case.get("relevance_grades", {}), 5),
            "latency_ms": latency_ms,
            "rerank_latency_ms": getattr(retriever, "last_rerank_latency_ms", 0.0),
            "failure": getattr(retriever, "last_retrieval_error", None),
            "rerank_failure": getattr(retriever, "last_rerank_error", None),
            "actual_abstention": None,
            "citation_validation": None,
        })
    return predictions


def run_mock(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic smoke path; deliberately marked as non-production mock data."""
    predictions = []
    for case in cases:
        retrieved_ids = list(case.get("relevant_chunk_ids", []))[:5]
        predictions.append({
            "id": case["id"],
            "question": case["question"],
            "mode": "mock",
            "retrieved_ids": retrieved_ids,
            "retrieved_passages": [],
            "expected_relevant_ids": case.get("relevant_chunk_ids", []),
            "precision_at_k": precision_at_k(retrieved_ids, case.get("relevant_chunk_ids", []), 5),
            "recall_at_k": recall_at_k(retrieved_ids, case.get("relevant_chunk_ids", []), 5),
            "mrr": mrr(retrieved_ids, case.get("relevant_chunk_ids", [])),
            "ndcg_at_k": ndcg_at_k(retrieved_ids, case.get("relevance_grades", {}), 5),
            "latency_ms": 0.0,
            "failure": None,
            "actual_abstention": bool(not retrieved_ids),
            "citation_validation": None,
        })
    return predictions


def run_full_generation(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Opt-in production generation capture; answer graders remain unimplemented."""
    load_dotenv(ROOT / ".env")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Full generation evaluation requires GEMINI_API_KEY")
    import app
    predictions = []
    for case in cases:
        started = time.perf_counter()
        answer, sources, triage = app.get_gemini_response(case["question"])
        retrieved_ids = [source.get("passage_id", "") for source in sources]
        predictions.append({
            "id": case["id"],
            "question": case["question"],
            "mode": "full_generation",
            "answer": answer,
            "sources": sources,
            "triage": triage,
            "retrieved_ids": retrieved_ids,
            "expected_relevant_ids": case.get("relevant_chunk_ids", []),
            "citation_validation": triage.get("citation_validation"),
            "answer_relevance": None,
            "factual_support": None,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "failure": None,
        })
    return predictions


def write_outputs(args: argparse.Namespace, cases: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_dir)
    prediction_path = output_root / "predictions" / f"{run_id}_{args.mode}.jsonl"
    report_path = output_root / "reports" / f"{run_id}_{args.mode}.json"
    markdown_path = output_root / "reports" / f"{run_id}_{args.mode}.md"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in predictions) + "\n", encoding="utf-8")

    report = {
        "run_id": run_id,
        "mode": args.mode,
        "dataset": str(Path(args.dataset)),
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "mocked": args.mode == "mock",
        "production_claim": False if args.mode == "mock" else True,
        "retrieval": retrieval_summary(predictions, k=5),
        "citation": citation_summary(predictions),
        "abstention": abstention_summary(predictions),
        "generation": {
            "answer_relevance": None,
            "factual_support": None,
            "token_usage": None,
            "estimated_cost_usd": None,
            "note": "Answer metrics and usage require reviewed/generated records; automated structure is not ground truth.",
        },
        "configuration": {
            "embedding_model": os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            "index": "knowledge/faiss_index.bin",
            "reranker": args.mode == "hybrid_reranked",
            "memory_usage_mb": None,
        },
        "artifacts": {
            "predictions": str(prediction_path),
            "json_report": str(report_path),
            "markdown_report": str(markdown_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(
        f"# MediAssist Evaluation Run\n\n"
        f"- Mode: `{args.mode}`\n- Cases: {len(cases)}\n- Mocked: `{args.mode == 'mock'}`\n\n"
        f"## Retrieval\n\n```json\n{json.dumps(report['retrieval'], indent=2)}\n```\n\n"
        "Answer quality, token usage, cost, and memory are not inferred when they were not measured.\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeatable MediAssist experiments.")
    parser.add_argument("--dataset", default=str(ROOT / "evaluation" / "dataset.jsonl"))
    parser.add_argument("--mode", choices=["mock", "keyword", "semantic", "hybrid", "hybrid_reranked", "full_generation"], default="mock")
    parser.add_argument("--output-dir", default=str(ROOT / "evaluation"))
    args = parser.parse_args()
    cases = load_jsonl(Path(args.dataset))
    if args.mode == "mock":
        predictions = run_mock(cases)
    elif args.mode == "full_generation":
        predictions = run_full_generation(cases)
    else:
        predictions = run_retrieval(cases, args.mode)
    report = write_outputs(args, cases, predictions)
    print(json.dumps(report["retrieval"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
