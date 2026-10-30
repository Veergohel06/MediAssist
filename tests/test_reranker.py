"""Unit and integration tests for the configurable reranking stage."""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.indexer import build_and_save_index
from rag.reranker import RerankerError, SentenceTransformerReranker, TokenOverlapReranker
from rag.retriever import MedicalRetriever


class RerankerFixtureClient:
    def __init__(self):
        self.models = SimpleNamespace(embed_content=self.embed_content)

    def embed_content(self, model, contents):
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0])])


class MockReranker:
    name = "mock-model"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def rerank(self, query, candidates, top_k):
        self.calls.append((query, len(candidates), top_k))
        if self.fail:
            raise RerankerError("mock failure")
        return [dict(candidate, rerank_score=float(100 - index), reranker=self.name) for index, candidate in enumerate(reversed(candidates))][:top_k]


def make_retriever(temp_dir, reranker, rerank_enabled=True):
    chunks = [
        {"chunk_id": "a_1", "passage_id": "a:p0001", "document_id": "a", "text": "sleep schedule", "source_title": "A", "source_url": "https://a.test", "content_hash": "a"},
        {"chunk_id": "b_1", "passage_id": "b:p0001", "document_id": "b", "text": "sleep hygiene", "source_title": "B", "source_url": "https://b.test", "content_hash": "b"},
        {"chunk_id": "c_1", "passage_id": "c:p0001", "document_id": "c", "text": "sleep and exercise", "source_title": "C", "source_url": "https://c.test", "content_hash": "c"},
    ]
    index_path = os.path.join(temp_dir, "rerank.faiss")
    metadata_path = os.path.join(temp_dir, "rerank.json")
    build_and_save_index(chunks, [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], index_path, metadata_path, embedding_model="fixture-model")
    return MedicalRetriever(
        client=RerankerFixtureClient(),
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model="fixture-model",
        retrieval_mode="semantic",
        score_threshold=0.0,
        semantic_candidate_k=3,
        rerank_enabled=rerank_enabled,
        reranker=reranker,
        rerank_candidate_k=3,
    )


class TestRerankerImplementations(unittest.TestCase):
    def test_cross_encoder_uses_mocked_scores_and_preserves_ids(self):
        class FakeCrossEncoder:
            def predict(self, pairs, batch_size, show_progress_bar):
                self.pairs = pairs
                return [0.2, 0.9]

        reranker = SentenceTransformerReranker(model=FakeCrossEncoder())
        candidates = [{"passage_id": "a:p0001", "text": "one"}, {"passage_id": "b:p0001", "text": "two"}]
        results = reranker.rerank("query", candidates, top_k=2)

        self.assertEqual(results[0]["passage_id"], "b:p0001")
        self.assertEqual(results[0]["reranker"], reranker.name)
        self.assertEqual(results[0]["rerank_score"], 0.9)

    def test_token_reranker_handles_empty_and_long_passages(self):
        reranker = TokenOverlapReranker(max_passage_chars=100)
        results = reranker.rerank("sleep", [{"passage_id": "a:p0001", "text": "sleep " * 1000}], top_k=1)

        self.assertEqual(results[0]["passage_id"], "a:p0001")
        self.assertGreater(results[0]["rerank_score"], 0.0)
        self.assertEqual(reranker.rerank("", results, top_k=1)[0]["passage_id"], "a:p0001")


class TestRerankerIntegration(unittest.TestCase):
    def test_candidate_pool_is_reranked_and_reduced(self):
        mock = MockReranker()
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever = make_retriever(temp_dir, mock)
            results = retriever.retrieve("sleep", top_k=1, query_embedding=[1.0, 0.0])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["passage_id"], "c:p0001")
        self.assertEqual(mock.calls[0][1], 3)
        self.assertEqual(mock.calls[0][2], 1)

    def test_reranker_failure_falls_back_to_retrieval_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever = make_retriever(temp_dir, MockReranker(fail=True))
            results = retriever.retrieve("sleep", top_k=2, query_embedding=[1.0, 0.0])

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["passage_id"], "a:p0001")
        self.assertEqual(retriever.last_rerank_error, "RerankerError")

    def test_disabled_reranker_does_not_change_retrieval(self):
        mock = MockReranker()
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever = make_retriever(temp_dir, mock, rerank_enabled=False)
            results = retriever.retrieve("sleep", top_k=1, query_embedding=[1.0, 0.0])

        self.assertEqual(results[0]["passage_id"], "a:p0001")
        self.assertEqual(mock.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
