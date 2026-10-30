"""CI smoke tests for the repeatable evaluation framework."""

import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.run_eval import load_jsonl, run_mock, write_outputs
from evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k


class TestEvaluationFramework(unittest.TestCase):
    def test_dataset_is_jsonl_and_has_required_fields(self):
        path = Path("evaluation/dataset.jsonl")
        rows = load_jsonl(path)

        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertIn("id", row)
            self.assertIn("question", row)
            self.assertIn("relevant_chunk_ids", row)
            self.assertIn("expected_behavior", row)
            self.assertIn("answer_criteria", row)
            self.assertIn("evidence_sufficient", row)
            self.assertIn("expected_abstention", row)
            self.assertIn("expected_safety_behavior", row)

    def test_metrics_support_binary_and_graded_relevance(self):
        retrieved = ["a", "b", "c"]
        relevant = ["b", "c"]
        grades = {"b": 3, "c": 1}

        self.assertAlmostEqual(precision_at_k(retrieved, relevant, 3), 2 / 3)
        self.assertEqual(recall_at_k(retrieved, relevant, 2), 0.5)
        self.assertGreater(ndcg_at_k(retrieved, grades, 3), 0.0)

    def test_mock_run_writes_raw_predictions_and_report(self):
        cases = load_jsonl(Path("evaluation/dataset.jsonl"))
        predictions = run_mock(cases)
        with tempfile.TemporaryDirectory() as temp_dir:
            report = write_outputs(
                Namespace(mode="mock", dataset="evaluation/dataset.jsonl", output_dir=temp_dir),
                cases,
                predictions,
            )
            prediction_path = Path(report["artifacts"]["predictions"])
            json_report_path = Path(report["artifacts"]["json_report"])
            markdown_path = Path(report["artifacts"]["markdown_report"])

            self.assertTrue(prediction_path.exists())
            self.assertTrue(json_report_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertTrue(report["mocked"])
            self.assertEqual(report["retrieval"]["case_count"], 10)


    def test_optimization_configs_structure_and_versioning(self):
        config_path = Path("evaluation/optimization_configs.json")
        self.assertTrue(config_path.exists())
        config_doc = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertIn("version", config_doc)
        self.assertIn("baseline", config_doc)
        self.assertIn("selected_optimized", config_doc)
        self.assertIn("controlled_experiments", config_doc)
        self.assertIn("selection_policy", config_doc)
        self.assertIn("rollback_procedure", config_doc)

        experiments = config_doc["controlled_experiments"]
        self.assertGreaterEqual(len(experiments), 7)
        for exp in experiments:
            self.assertIn("id", exp)
            self.assertIn("category", exp)
            self.assertIn("variable", exp)

    def test_caching_and_privacy_isolation(self):
        from rag.embeddings import get_cache_key, get_embedding, get_embedding_metrics, reset_embedding_metrics
        reset_embedding_metrics()

        # Cache key is deterministic and strips leading/trailing whitespace
        key1 = get_cache_key("model-a", "What is hypertension?  ")
        key2 = get_cache_key("model-a", "What is hypertension?")
        self.assertEqual(key1, key2)

        # Different models have different cache keys to avoid collision
        key3 = get_cache_key("model-b", "What is hypertension?")
        self.assertNotEqual(key1, key3)

        # Telemetry tracks hits and misses
        metrics = get_embedding_metrics()
        self.assertIn("cache_hits", metrics)
        self.assertIn("total_requests", metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)

