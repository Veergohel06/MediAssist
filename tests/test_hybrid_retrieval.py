"""Unit tests for BM25, semantic, and RRF retrieval modes."""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.indexer import build_and_save_index
from rag.retriever import MedicalRetriever


class KeywordFixtureClient:
    def __init__(self):
        self.models = SimpleNamespace(embed_content=self.embed_content)
        self.calls = []

    def embed_content(self, model, contents):
        self.calls.append(contents)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0])])


def build_fixture(temp_dir):
    chunks = [
        {
            "chunk_id": "one_1", "passage_id": "one:p0001", "document_id": "one",
            "source_title": "One", "source_url": "https://one.test", "section": "One",
            "text": "sleep schedule and consistent bedtime", "content_hash": "one",
        },
        {
            "chunk_id": "two_1", "passage_id": "two:p0001", "document_id": "two",
            "source_title": "Two", "source_url": "https://two.test", "section": "Two",
            "text": "hydration and drinking water", "content_hash": "two",
        },
        {
            "chunk_id": "three_1", "passage_id": "three:p0001", "document_id": "three",
            "source_title": "Three", "source_url": "https://three.test", "section": "Three",
            "text": "sleep hygiene and dark room", "content_hash": "three",
        },
    ]
    index_path = os.path.join(temp_dir, "hybrid.faiss")
    metadata_path = os.path.join(temp_dir, "hybrid.json")
    build_and_save_index(
        chunks,
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
        index_path,
        metadata_path,
        embedding_model="fixture-model",
    )
    return index_path, metadata_path


class TestHybridRetrieval(unittest.TestCase):
    def make_retriever(self, temp_dir, mode):
        client = KeywordFixtureClient()
        index_path, metadata_path = build_fixture(temp_dir)
        retriever = MedicalRetriever(
            client=client,
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_model="fixture-model",
            retrieval_mode=mode,
            score_threshold=0.0,
            semantic_candidate_k=3,
            keyword_candidate_k=3,
            rrf_k=1,
        )
        return retriever, client

    def test_keyword_only_uses_bm25_without_embedding_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, client = self.make_retriever(temp_dir, "keyword")
            results = retriever.retrieve("hydration drinking water", top_k=2)

        self.assertEqual(results[0]["passage_id"], "two:p0001")
        self.assertEqual(results[0]["retrieval_mode"], "keyword")
        self.assertEqual(client.calls, [])

    def test_hybrid_rrf_deduplicates_and_rewards_both_rankers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, _ = self.make_retriever(temp_dir, "hybrid")
            results = retriever.retrieve("sleep schedule", top_k=3, query_embedding=[1.0, 0.0])

        ids = [item["passage_id"] for item in results]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "one:p0001")
        self.assertTrue(results[0]["rrf_score"] > results[-1]["rrf_score"])

    def test_rrf_merges_overlapping_ranked_lists_once(self):
        retriever = object.__new__(MedicalRetriever)
        retriever.metadata = [
            {"passage_id": "a:p0001"},
            {"passage_id": "b:p0001"},
            {"passage_id": "c:p0001"},
        ]
        retriever.rrf_k = 1
        results = retriever._rrf_rank([(0, 0.9), (1, 0.8)], [(1, 4.0), (2, 3.0)])

        self.assertEqual([item[0] for item in results], [1, 0, 2])
        self.assertEqual(len(results), 3)

    def test_missing_keyword_results_do_not_break_hybrid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever, _ = self.make_retriever(temp_dir, "hybrid")
            with patch.object(retriever, "_keyword_scores", return_value={}):
                results = retriever.retrieve("unmatched", top_k=2, query_embedding=[1.0, 0.0])

        self.assertTrue(results)
        self.assertEqual(results[0]["semantic_score"], 1.0)

    def test_empty_query_returns_no_results_in_every_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for mode in ("keyword", "semantic", "hybrid"):
                retriever, client = self.make_retriever(temp_dir, mode)
                self.assertEqual(retriever.retrieve("  "), [])
                self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
