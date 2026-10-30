"""Offline tests for the 100-case baseline report aggregation."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluate_baseline import build_report
from rag.evaluation import load_evaluation_dataset


class TestBaselineDataset(unittest.TestCase):
    def test_requested_category_counts(self):
        cases = load_evaluation_dataset("evals/baseline_eval.json")
        counts = {}
        for case in cases:
            counts[case["category"]] = counts.get(case["category"], 0) + 1

        self.assertEqual(len(cases), 100)
        self.assertEqual(counts["direct_factual"], 20)
        self.assertEqual(counts["paraphrased"], 15)
        self.assertEqual(counts["multi_part"], 15)
        self.assertEqual(counts["ambiguous"], 10)
        self.assertEqual(counts["unsupported"], 10)
        self.assertEqual(counts["conflicting_outdated"], 10)
        self.assertEqual(counts["citation_focused"], 10)
        self.assertEqual(counts["safety_critical"], 10)


class TestBaselineReport(unittest.TestCase):
    def test_report_keeps_unreviewed_answer_metrics_unknown(self):
        cases = [{"case_id": "case_1"}]
        records = [{
            "case_id": "case_1",
            "precision_at_k": 1.0,
            "recall_at_k": 1.0,
            "mrr": 1.0,
            "abstention_expected": False,
            "abstention_correct": True,
            "actual_abstention": False,
            "citation_validity": True,
            "citation_valid": True,
            "citation_support_rate": 1.0,
            "safety_compliance": True,
            "latency_ms": 10.0,
        }]

        report = build_report(cases, records, top_k=5)

        self.assertEqual(report["retrieval"]["precision_at_k"], 1.0)
        self.assertEqual(report["citation"]["support_rate"], 1.0)
        self.assertIsNone(report["answer_quality"]["answer_relevance"])
        self.assertIsNone(report["usage"]["input_tokens"])
        self.assertEqual(report["latency_ms"]["p50"], 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)