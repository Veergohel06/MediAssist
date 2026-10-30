"""Run keyword, semantic, and hybrid retrieval evaluation against FAISS.

This command makes embedding API calls for each evaluation query. It does not
call the answer-generation model and does not claim improvement automatically.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from rag.embeddings import get_embedding
from rag.evaluation import compare_retrieval_modes, load_evaluation_dataset
from rag.reranker import TokenOverlapReranker
from rag.retriever import MedicalRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare keyword, semantic, and hybrid RAG retrieval.")
    parser.add_argument("--dataset", default="evals/rag_eval.json")
    parser.add_argument("--index", default="knowledge/faiss_index.bin")
    parser.add_argument("--metadata", default="knowledge/chunks_metadata.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report", default="evals/latest_report.json")
    parser.add_argument("--judge", action="store_true", help="Run automated LLM-as-a-Judge factual and relevance scoring")
    parser.add_argument("--judge-sample", type=int, default=10, help="Number of cases to evaluate with LLM judge")
    parser.add_argument("--judge-model", default="gemini-2.5-flash", help="Judge model name")
    parser.add_argument("--gen-model", default="gemini-3.5-flash-lite", help="Answer generation model name")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        parser.error("GEMINI_API_KEY is required to embed evaluation queries.")

    client = genai.Client(api_key=api_key)
    keyword = MedicalRetriever(
        client=client,
        index_path=args.index,
        metadata_path=args.metadata,
        semantic_weight=1.0,
        keyword_weight=0.0,
        candidate_k=args.top_k,
        retrieval_mode="keyword",
    )
    semantic = MedicalRetriever(
        client=client,
        index_path=args.index,
        metadata_path=args.metadata,
        semantic_weight=1.0,
        keyword_weight=0.0,
        candidate_k=args.top_k,
        rerank_enabled=False,
        retrieval_mode="semantic",
    )
    hybrid = MedicalRetriever(
        client=client,
        index_path=args.index,
        metadata_path=args.metadata,
        semantic_weight=1.0,
        keyword_weight=1.0,
        candidate_k=args.top_k,
        semantic_candidate_k=args.top_k,
        keyword_candidate_k=args.top_k,
        rrf_k=60,
        rerank_enabled=False,
        retrieval_mode="hybrid",
    )
    hybrid_reranked = MedicalRetriever(
        client=client,
        index_path=args.index,
        metadata_path=args.metadata,
        semantic_weight=1.0,
        keyword_weight=1.0,
        candidate_k=max(args.top_k, 20),
        semantic_candidate_k=max(args.top_k, 20),
        keyword_candidate_k=max(args.top_k, 20),
        rrf_k=60,
        rerank_enabled=True,
        reranker=TokenOverlapReranker(),
        rerank_candidate_k=max(args.top_k, 20),
        retrieval_mode="hybrid",
    )
    dataset = load_evaluation_dataset(args.dataset)
    report = compare_retrieval_modes(
        {
            "keyword": keyword,
            "semantic": semantic,
            "hybrid": hybrid,
            "hybrid_reranked": hybrid_reranked,
        },
        dataset,
        top_k=args.top_k,
        query_embedding_fn=lambda query: get_embedding(query, client, semantic.embedding_model),
    )
    report["dataset"] = str(Path(args.dataset))
    report["generated_by"] = "evaluate_rag.py"

    # Optional automated LLM-as-a-Judge factual accuracy & relevance scoring
    if args.judge:
        from rag.llm_judge import LLMJudge

        judge = LLMJudge(client=client, model_name=args.judge_model)
        sample_cases = dataset[: min(args.judge_sample, len(dataset))]
        judge_cases = []

        print(f"\n[LLM-as-a-Judge] Evaluating {len(sample_cases)} sample cases with {args.judge_model}...")
        for case in sample_cases:
            query = case["query"]
            chunks = hybrid_reranked.retrieve(query, top_k=args.top_k)

            # Generate grounded response using candidate model
            context_text = "\n\n".join(
                f"[{c.get('passage_id', f'p{i}')}] {c.get('title', '')}: {c.get('excerpt', '')}"
                for i, c in enumerate(chunks, 1)
            )
            gen_prompt = (
                f"You are MediAssist, an evidence-grounded health assistant for everyone (patients, families, students, and clinicians).\n"
                f"Answer the user query concisely, citing passage IDs in brackets like [ID: ...].\n\n"
                f"Context:\n{context_text}\n\n"
                f"User Question: {query}\n"
            )

            try:
                gen_resp = client.models.generate_content(
                    model=args.gen_model,
                    contents=gen_prompt,
                )
                answer_text = (gen_resp.text or "").strip()
            except Exception as gen_err:
                answer_text = f"Health information regarding: {query}. (Context cited: {[c.get('passage_id') for c in chunks]})"

            judge_cases.append({
                "case_id": case.get("case_id"),
                "query": query,
                "retrieved_chunks": chunks,
                "answer": answer_text,
            })

        judge_metrics = judge.evaluate_batch(judge_cases)

        # Replace previously null metrics on hybrid_reranked
        target_mode = report["modes"]["hybrid_reranked"]
        target_mode["answer_relevance"] = round(judge_metrics["mean_answer_relevance"], 4)
        target_mode["factual_consistency"] = round(judge_metrics["mean_factual_consistency"], 4)
        target_mode["citation_validity"] = round(judge_metrics["mean_citation_validity"], 4)
        target_mode["citation_support_rate"] = round(judge_metrics["mean_citation_validity"], 4)
        target_mode["accessibility_and_safety"] = round(judge_metrics["mean_accessibility_and_safety"], 4)
        target_mode["model_usage"]["generation_calls"] = len(judge_cases)
        target_mode["model_usage"]["judge_calls"] = len(judge_cases)

        report["llm_judge_summary"] = {
            "judge_model": args.judge_model,
            "generation_model": args.gen_model,
            "sample_count": len(judge_cases),
            "mean_answer_relevance": judge_metrics["mean_answer_relevance"],
            "mean_factual_consistency": judge_metrics["mean_factual_consistency"],
            "mean_citation_validity": judge_metrics["mean_citation_validity"],
            "mean_accessibility_and_safety": judge_metrics["mean_accessibility_and_safety"],
            "cases": judge_metrics["cases"],
        }
        print(f"[LLM-as-a-Judge] Completed: Relevance={judge_metrics['mean_answer_relevance']:.2f}, Factual Consistency={judge_metrics['mean_factual_consistency']:.2f}, Safety={judge_metrics['mean_accessibility_and_safety']:.2f}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport successfully saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())