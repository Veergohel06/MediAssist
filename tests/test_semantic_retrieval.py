"""Deterministic semantic-retrieval tests with mocked embeddings."""

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.indexer import build_and_save_index, manifest_path_for
from rag.retriever import MedicalRetriever


class FixtureClient:
    def __init__(self):
        self.models = SimpleNamespace(embed_content=self.embed_content)
        self.calls = []

    def embed_content(self, model, contents):
        self.calls.append((model, contents))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0])])


def fixture_chunks():
    return [
        {
            "chunk_id": "doc_1",
            "document_id": "doc",
            "passage_id": "doc:p0001",
            "source_title": "Fixture One",
            "source_url": "https://example.test/one",
            "retrieval_date": "2026-10-21",
            "version": "v1",
            "section": "One",
            "text": "sleep schedule and consistent bedtime",
            "content_hash": "fixture-one",
        },
        {
            "chunk_id": "doc_2",
            "document_id": "doc",
            "passage_id": "doc:p0002",
            "source_title": "Fixture Two",
            "source_url": "https://example.test/two",
            "retrieval_date": "2026-10-21",
            "version": "v1",
            "section": "Two",
            "text": "hydration and drinking water",
            "content_hash": "fixture-two",
        },
    ]


class TestSemanticRetrieval(unittest.TestCase):
    def make_retriever(self, temp_dir, client=None):
        client = client or FixtureClient()
        index_path = os.path.join(temp_dir, "fixture.faiss")
        metadata_path = os.path.join(temp_dir, "fixture.json")
        build_and_save_index(
            fixture_chunks(),
            [[1.0, 0.0], [0.0, 1.0]],
            index_path,
            metadata_path,
            embedding_model="fixture-model",
        )
        return MedicalRetriever(
            client=client,
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_model="fixture-model",
            score_threshold=0.5,
            retrieval_mode="semantic",
            candidate_k=2,
        )

    def test_top_k_semantic_retrieval_uses_normalized_inner_product(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FixtureClient()
            retriever = self.make_retriever(temp_dir, client)
            results = retriever.retrieve("sleep question", top_k=1)

        self.assertEqual([item["passage_id"] for item in results], ["doc:p0001"])
        self.assertAlmostEqual(results[0]["score"], 1.0)
        self.assertEqual(client.calls[0][0], "fixture-model")

    def test_empty_query_does_not_call_embedding_api(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FixtureClient()
            retriever = self.make_retriever(temp_dir, client)
            self.assertEqual(retriever.retrieve("   "), [])

        self.assertEqual(client.calls, [])

    def test_embedding_failure_returns_empty_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever = self.make_retriever(temp_dir)
            with patch("rag.retriever.get_embedding", side_effect=RuntimeError("fixture failure")):
                self.assertEqual(retriever.retrieve("sleep question"), [])

    def test_duplicate_chunks_are_not_indexed_twice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks = fixture_chunks()
            chunks.append(dict(chunks[0]))
            index_path = os.path.join(temp_dir, "fixture.faiss")
            metadata_path = os.path.join(temp_dir, "fixture.json")
            index = build_and_save_index(
                chunks,
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                index_path,
                metadata_path,
                embedding_model="fixture-model",
            )

            self.assertEqual(index.ntotal, 2)
            with open(metadata_path, encoding="utf-8") as handle:
                self.assertEqual(len(json.load(handle)), 2)

    def test_manifest_version_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            retriever = self.make_retriever(temp_dir)
            manifest_path = manifest_path_for(retriever.index_path)
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["format_version"] += 1
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)

            self.assertFalse(retriever.reload())
            self.assertFalse(retriever.is_ready())


if __name__ == "__main__":
    unittest.main(verbosity=2)
