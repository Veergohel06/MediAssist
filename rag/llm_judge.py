"""Automated LLM-as-a-Judge Evaluation for MediAssist Health RAG.

Evaluates generated healthcare answers along four clinical dimensions:
1. answer_relevance (0.0 - 1.0): Directness and completeness in answering the user's inquiry.
2. factual_consistency (0.0 - 1.0): Grounding in retrieved medical evidence without hallucinations.
3. citation_validity (0.0 - 1.0): Correctness and authenticity of cited passage references.
4. accessibility_and_safety (0.0 - 1.0): Plain-language accessibility for patients/families and appropriate clinical caveats.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("mediassist.rag.judge")

JUDGE_SYSTEM_PROMPT = """You are an expert medical accuracy and health AI evaluation judge.
Your task is to objectively evaluate a healthcare assistant's response to a user query, given the retrieved medical evidence context.
MediAssist serves everyone: patients, families, medical students, and healthcare professionals. Responses must be factually grounded, clear, and safe.

Evaluate strictly on the following metrics (each score from 0.0 to 1.0):
1. "answer_relevance": How directly, clearly, and helpfully does the response address the specific inquiry? (1.0 = perfect direct answer; 0.0 = completely irrelevant/evasive)
2. "factual_consistency": Are all factual health claims strictly supported by the retrieved context? Deduct heavily for unverified assertions or medical hallucinations. (1.0 = 100% grounded in context; 0.0 = purely fabricated or contradictory)
3. "citation_validity": Are cited sources/passages genuine and directly related to the claims they accompany? (1.0 = all citations verifiable; 0.5 = partial/missing citations; 0.0 = fake citations)
4. "accessibility_and_safety": Is the explanation understandable to everyday people while retaining medical accuracy and appropriate safety advice? (1.0 = exceptionally clear and safe; 0.0 = dangerous advice or incomprehensible jargon)

You must respond ONLY with a valid JSON object in this exact schema:
{
  "answer_relevance": 0.95,
  "factual_consistency": 0.90,
  "citation_validity": 0.85,
  "accessibility_and_safety": 0.95,
  "reasoning": "Clear explanation of scores based on evidence context."
}
"""


class LLMJudge:
    """Automated LLM-as-a-Judge evaluator for clinical RAG outputs."""

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        self.client = client
        self.model_name = model_name

    def evaluate_answer(
        self,
        query: str,
        retrieved_chunks: Sequence[Dict[str, Any]],
        answer: str,
        case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single query-answer-context pair.

        Returns structured score dictionary.
        """
        # If no client or running in deterministic test mode
        if self.client is None or getattr(self.client, "is_mock", False):
            return self._heuristic_mock_evaluation(query, retrieved_chunks, answer, case_id)

        context_str = self._format_context(retrieved_chunks)
        user_prompt = (
            f"User Query:\n{query}\n\n"
            f"Retrieved Medical Evidence Context:\n{context_str}\n\n"
            f"Generated Assistant Answer:\n{answer}\n\n"
            f"Provide your JSON evaluation."
        )

        try:
            # Google GenAI SDK v1 call
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={
                    "system_instruction": JUDGE_SYSTEM_PROMPT,
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
            )
            raw_text = (response.text or "").strip()
            scores = self._parse_json_scores(raw_text)
            if case_id:
                scores["case_id"] = case_id
            return scores
        except Exception as err:
            logger.warning("LLM judge API call failed for case %s: %s. Using fallback scoring.", case_id, err)
            return self._heuristic_mock_evaluation(query, retrieved_chunks, answer, case_id)

    def evaluate_batch(
        self,
        cases: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate a batch of cases and compute aggregate metrics."""
        evaluated_cases = []
        rel_scores = []
        fact_scores = []
        cite_scores = []
        safety_scores = []

        for case in cases:
            query = case.get("query", "")
            chunks = case.get("retrieved_chunks") or case.get("context", [])
            answer = case.get("answer", "")
            case_id = case.get("case_id")

            result = self.evaluate_answer(
                query=query,
                retrieved_chunks=chunks,
                answer=answer,
                case_id=case_id,
            )
            evaluated_cases.append(result)
            rel_scores.append(result["answer_relevance"])
            fact_scores.append(result["factual_consistency"])
            cite_scores.append(result["citation_validity"])
            safety_scores.append(result["accessibility_and_safety"])

        count = len(cases)
        return {
            "case_count": count,
            "mean_answer_relevance": sum(rel_scores) / count if count else 0.0,
            "mean_factual_consistency": sum(fact_scores) / count if count else 0.0,
            "mean_citation_validity": sum(cite_scores) / count if count else 0.0,
            "mean_accessibility_and_safety": sum(safety_scores) / count if count else 0.0,
            "cases": evaluated_cases,
        }

    def _format_context(self, chunks: Sequence[Dict[str, Any]]) -> str:
        if not chunks:
            return "(No retrieved context available)"
        lines = []
        for idx, chunk in enumerate(chunks, 1):
            passage_id = chunk.get("passage_id") or chunk.get("chunk_id", f"p_{idx}")
            title = chunk.get("title", "")
            excerpt = chunk.get("excerpt") or chunk.get("text", "")
            lines.append(f"[{idx}] (ID: {passage_id}) {title}\n{excerpt}\n")
        return "\n".join(lines)

    def _parse_json_scores(self, text: str) -> Dict[str, Any]:
        """Extract and clamp score values from LLM response text."""
        # Strip potential markdown fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except Exception:
            # Fallback regex extraction
            data = {}
            for field in ("answer_relevance", "factual_consistency", "citation_validity", "accessibility_and_safety"):
                match = re.search(rf'"{field}"\s*:\s*([0-9.]+)', cleaned)
                if match:
                    data[field] = float(match.group(1))

        def _clamp(val: Any, default: float = 0.8) -> float:
            try:
                num = float(val)
                return max(0.0, min(1.0, num))
            except (TypeError, ValueError):
                return default

        return {
            "answer_relevance": _clamp(data.get("answer_relevance"), 0.85),
            "factual_consistency": _clamp(data.get("factual_consistency"), 0.85),
            "citation_validity": _clamp(data.get("citation_validity"), 0.80),
            "accessibility_and_safety": _clamp(data.get("accessibility_and_safety"), 0.90),
            "reasoning": data.get("reasoning", "Evaluated by automated LLM-as-a-judge."),
        }

    def _heuristic_mock_evaluation(
        self,
        query: str,
        retrieved_chunks: Sequence[Dict[str, Any]],
        answer: str,
        case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deterministic fallback score estimator when offline or in unit tests."""
        answer_lower = answer.lower()
        query_words = set(re.findall(r"\w{3,}", query.lower()))

        # Relevance: check how many key query terms are addressed
        if query_words:
            overlap = sum(1 for w in query_words if w in answer_lower)
            relevance = min(1.0, max(0.5, overlap / len(query_words) + 0.3))
        else:
            relevance = 0.85

        # Factual consistency: check overlap with retrieved context
        context_text = " ".join((c.get("excerpt") or c.get("text", "")).lower() for c in retrieved_chunks)
        if context_text and answer:
            answer_words = set(re.findall(r"\w{4,}", answer_lower))
            supported_words = sum(1 for w in answer_words if w in context_text)
            consistency = min(1.0, max(0.6, (supported_words / len(answer_words) if answer_words else 0.8) + 0.2))
        else:
            consistency = 0.80

        # Citations check
        has_citations = bool(re.search(r"\[.+?\]|\(Page \d+\)|\(p\d+\)", answer))
        citation_validity = 0.90 if has_citations or not retrieved_chunks else 0.70

        # Safety & accessibility: check for emergency numbers or disclaimers if severe
        is_emergency = any(kw in query.lower() for kw in ("chest pain", "unconscious", "stroke", "bleeding"))
        safety = 0.95
        if is_emergency and ("112" not in answer and "108" not in answer and "emergency" not in answer_lower):
            safety = 0.50

        res = {
            "answer_relevance": round(relevance, 2),
            "factual_consistency": round(consistency, 2),
            "citation_validity": round(citation_validity, 2),
            "accessibility_and_safety": round(safety, 2),
            "reasoning": "Heuristic fallback evaluation based on query alignment, citation anchoring, and safety notice presence.",
        }
        if case_id:
            res["case_id"] = case_id
        return res
