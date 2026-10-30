"""Unit and Integration Tests for Automated LLM-as-a-Judge Evaluation (Phase 5).

Verifies:
- Evaluation across 4 dimensions: relevance, factual consistency, citation validity, safety.
- Score parsing, boundary clamping [0.0, 1.0], and JSON fence cleaning.
- Batch evaluation aggregation and statistical metrics.
- Seamless integration with rag.evaluation.evaluate_answer_records.
- Universal health guidance suitability (for everyone, not MBBS exclusive).
"""

import unittest
from unittest.mock import MagicMock

from rag.evaluation import evaluate_answer_records
from rag.llm_judge import LLMJudge


class TestLLMJudge(unittest.TestCase):
    """Direct unit tests for LLMJudge scoring logic."""

    def setUp(self):
        self.judge = LLMJudge(client=None)

    def test_heuristic_evaluation_returns_bounded_scores(self):
        """Offline / test mode generates valid scores between 0.0 and 1.0."""
        query = "What are the common symptoms of influenza and how does it differ from a cold?"
        chunks = [
            {
                "passage_id": "cold_vs_flu:p0001",
                "title": "Influenza vs Common Cold",
                "excerpt": "Influenza has sudden onset with high fever, myalgia, and fatigue. The common cold has gradual onset with prominent rhinorrhea.",
            }
        ]
        answer = (
            "Influenza typically presents with abrupt onset of high fever and systemic body aches [cold_vs_flu:p0001]. "
            "In contrast, the common cold usually develops gradually with a runny or stuffy nose."
        )

        result = self.judge.evaluate_answer(query, chunks, answer, case_id="flu_01")

        self.assertIn("answer_relevance", result)
        self.assertIn("factual_consistency", result)
        self.assertIn("citation_validity", result)
        self.assertIn("accessibility_and_safety", result)
        self.assertEqual(result["case_id"], "flu_01")

        for key in ("answer_relevance", "factual_consistency", "citation_validity", "accessibility_and_safety"):
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)

    def test_json_score_clamping_and_fence_stripping(self):
        """Scores outside [0.0, 1.0] or wrapped in code fences are cleaned and clamped."""
        raw_response = """```json
        {
          "answer_relevance": 1.5,
          "factual_consistency": -0.2,
          "citation_validity": 0.92,
          "accessibility_and_safety": 0.88,
          "reasoning": "Well-structured response for patients."
        }
        ```"""

        scores = self.judge._parse_json_scores(raw_response)
        self.assertEqual(scores["answer_relevance"], 1.0)
        self.assertEqual(scores["factual_consistency"], 0.0)
        self.assertEqual(scores["citation_validity"], 0.92)
        self.assertEqual(scores["accessibility_and_safety"], 0.88)
        self.assertEqual(scores["reasoning"], "Well-structured response for patients.")

    def test_batch_evaluation_aggregates(self):
        """Batch evaluation computes means across all evaluated cases."""
        cases = [
            {
                "case_id": "c1",
                "query": "How to treat mild dehydration?",
                "retrieved_chunks": [{"excerpt": "Drink oral rehydration solutions or water."}],
                "answer": "Drink plenty of water and oral rehydration solutions.",
            },
            {
                "case_id": "c2",
                "query": "What causes tension headaches?",
                "retrieved_chunks": [{"excerpt": "Muscle contractions in the head and neck region."}],
                "answer": "Tension headaches are commonly caused by muscle contractions in the neck [p1].",
            },
        ]

        batch_result = self.judge.evaluate_batch(cases)
        self.assertEqual(batch_result["case_count"], 2)
        self.assertGreater(batch_result["mean_answer_relevance"], 0.0)
        self.assertGreater(batch_result["mean_factual_consistency"], 0.0)
        self.assertEqual(len(batch_result["cases"]), 2)

    def test_evaluate_answer_records_populates_null_metrics_with_judge(self):
        """evaluate_answer_records automatically fills previously null metrics using the judge."""
        mock_retriever = MagicMock()
        mock_retriever.validate_citations.return_value = {
            "all_valid": True,
            "valid_ids": ["heart_health:p0001"],
            "unsupported_ids": [],
        }

        records = [
            {
                "case_id": "rec_01",
                "query": "How to support cardiovascular health?",
                "retrieved_chunks": [
                    {
                        "passage_id": "heart_health:p0001",
                        "excerpt": "Regular aerobic exercise and a Mediterranean-style diet promote heart health.",
                    }
                ],
                "answer": "Regular aerobic exercise and a balanced diet significantly reduce cardiovascular risk [heart_health:p0001].",
                "supported_passage_ids": ["heart_health:p0001"],
                # answer_relevance and factual_consistency are intentionally omitted to test automated judge filling
            }
        ]

        report = evaluate_answer_records(mock_retriever, records, judge=self.judge)

        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["citation_validity"], 1.0)
        self.assertIsNotNone(report["answer_relevance"])
        self.assertIsNotNone(report["factual_consistency"])
        self.assertIsNotNone(report["accessibility_and_safety"])
        self.assertGreaterEqual(report["answer_relevance"], 0.5)
        self.assertGreaterEqual(report["factual_consistency"], 0.5)


if __name__ == "__main__":
    unittest.main()
