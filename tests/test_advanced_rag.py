"""Offline tests for upgraded retrieval, evidence IDs, and evaluation metrics."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.chunker import chunk_text, normalize_text
from rag.evaluation import (
    evaluate_answer_records,
    evaluate_retriever,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from rag.retriever import MedicalRetriever


class TestEvidenceMetadata(unittest.TestCase):
    def test_normalization_and_stable_passage_metadata(self):
        chunks = chunk_text(
            "##  Symptoms\r\n\r\nFever   and\t chills.",
            {"title": "Example", "source": "Test", "url": "https://example.test"},
            "example",
        )

        self.assertEqual(normalize_text(" a  b\r\n\r\n\r\n c "), "a b\n\nc")
        self.assertEqual(chunks[0]["passage_id"], "example:p0001")
        self.assertEqual(chunks[0]["document_id"], "example")
        self.assertEqual(chunks[0]["section"], "Symptoms")
        self.assertIn("content_hash", chunks[0])


class TestCitationValidation(unittest.TestCase):
    def test_only_retrieved_evidence_ids_are_valid(self):
        chunks = [{"passage_id": "doc:p0001"}, {"passage_id": "doc:p0002"}]

        result = MedicalRetriever.validate_citations(
            "Supported [doc:p0001], invented [other:p0009].", chunks
        )

        self.assertEqual(result["valid_ids"], ["doc:p0001"])
        self.assertEqual(result["invalid_ids"], ["other:p0009"])
        self.assertFalse(result["all_valid"])


class TestEvaluationMetrics(unittest.TestCase):
    def test_ranking_metrics(self):
        retrieved = ["a", "b", "c"]
        relevant = ["b", "c"]

        self.assertAlmostEqual(precision_at_k(retrieved, relevant, 3), 2 / 3)
        self.assertEqual(recall_at_k(retrieved, relevant, 2), 0.5)
        self.assertEqual(mean_reciprocal_rank(retrieved, relevant), 0.5)

    def test_evaluation_report_is_reproducible_without_external_calls(self):
        class FakeRetriever:
            def retrieve(self, query, top_k):
                if "sleep" in query:
                    return [{"chunk_id": "sleep_1"}]
                return []

        cases = [
            {"case_id": "supported", "query": "sleep help", "expected_relevant_ids": ["sleep_1"]},
            {"case_id": "unsupported", "query": "programming", "expected_relevant_ids": [], "abstain_expected": True},
        ]
        report = evaluate_retriever(FakeRetriever(), cases, top_k=1)

        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["precision_at_k"], 1.0)
        self.assertEqual(report["recall_at_k"], 1.0)
        self.assertEqual(report["abstention_accuracy"], 1.0)
        self.assertEqual(report["model_usage"]["generation_calls"], 0)

    def test_answer_metrics_are_only_computed_from_supplied_records(self):
        chunks = [{"passage_id": "doc:p0001"}]
        records = [{
            "answer": "The answer cites [doc:p0001].",
            "retrieved_chunks": chunks,
            "answer_relevance": 1.0,
            "factual_consistency": 0.75,
        }]

        report = evaluate_answer_records(MedicalRetriever, records)

        self.assertEqual(report["citation_validity"], 1.0)
        self.assertEqual(report["answer_relevance"], 1.0)
        self.assertEqual(report["factual_consistency"], 0.75)


class TestEvaluationDataset(unittest.TestCase):
    def test_dataset_contains_required_scenario_types(self):
        with open(os.path.join("evals", "rag_eval.json"), encoding="utf-8") as handle:
            cases = json.load(handle)

        case_ids = {case["case_id"] for case in cases}
        self.assertIn("ambiguous_fever", case_ids)
        self.assertIn("out_of_scope_programming", case_ids)
        self.assertIn("safety_chest_pain", case_ids)
        self.assertIn("conflicting_fever_claim", case_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)