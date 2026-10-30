"""
Automated Test Suite for MediAssist RAG Architecture
====================================================
Tests document parsing, chunking, FAISS index creation, vector normalization,
retrieval thresholding, empty results handling, and API response formatting.
"""

import os
import sys
import unittest
from unittest.mock import patch
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from rag.chunker import parse_markdown_document, chunk_text, load_and_chunk_documents
from rag.indexer import normalize_vectors, build_and_save_index, load_index
from rag.retriever import MedicalRetriever


class TestDocumentChunking(unittest.TestCase):
    """Test markdown parsing and semantic chunking."""

    def test_parse_markdown_frontmatter(self):
        sample_path = os.path.join("knowledge", "sleep_hygiene_and_disorders.md")
        self.assertTrue(os.path.exists(sample_path), "Sample medical document must exist")
        doc = parse_markdown_document(sample_path)
        self.assertIn("title", doc["metadata"])
        self.assertIn("source", doc["metadata"])
        self.assertIn("url", doc["metadata"])
        self.assertIn("retrieval_date", doc["metadata"])
        self.assertTrue(len(doc["body"]) > 100)

    def test_chunk_text_structure(self):
        metadata = {
            "title": "Test Title",
            "source": "WHO",
            "url": "https://who.int",
            "retrieval_date": "2026-10-21",
            "category": "Test",
        }
        text = "## Section 1\nThis is paragraph one.\n\nThis is paragraph two.\n\n## Section 2\nThis is section two."
        chunks = chunk_text(text, metadata, "doc_test", max_chunk_size=100)
        self.assertTrue(len(chunks) >= 2)
        for ch in chunks:
            self.assertIn("chunk_id", ch)
            self.assertEqual(ch["source_title"], "Test Title")
            self.assertEqual(ch["source_url"], "https://who.int")
            self.assertTrue(len(ch["text"]) > 0)

    def test_load_all_knowledge_documents(self):
        chunks = load_and_chunk_documents("knowledge")
        self.assertTrue(len(chunks) >= 10, "Expected at least 10 chunks from curated knowledge base")


class TestVectorIndexing(unittest.TestCase):
    """Test vector normalization and FAISS index persistence."""

    def test_vector_normalization(self):
        raw = np.array([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        normalized = normalize_vectors(raw)
        norms = np.linalg.norm(normalized, axis=1)
        for n in norms:
            self.assertAlmostEqual(n, 1.0, places=5)

    def test_missing_index_handled_gracefully(self):
        index, meta = load_index("nonexistent_index.bin", "nonexistent_meta.json")
        self.assertIsNone(index)
        self.assertEqual(meta, [])


class TestRetrievalFlow(unittest.TestCase):
    """Test semantic retrieval and threshold filtering using live index."""

    @classmethod
    def setUpClass(cls):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise unittest.SkipTest("GEMINI_API_KEY not configured in environment")
        cls.client = genai.Client(api_key=api_key)
        cls.retriever = MedicalRetriever(
            client=cls.client,
            index_path=os.path.join("knowledge", "faiss_index.bin"),
            metadata_path=os.path.join("knowledge", "chunks_metadata.json"),
            score_threshold=0.58,
        )

    def test_index_is_ready(self):
        self.assertTrue(self.retriever.is_ready(), "MedicalRetriever must have loaded index and metadata")
        self.assertTrue(self.retriever.index.ntotal > 0)

    def test_relevant_query_retrieves_chunks_and_sources(self):
        query = "What causes tension headaches and how can I ease the pain?"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertTrue(len(results) > 0, "Expected at least 1 relevant chunk for tension headaches")
        
        sources = self.retriever.extract_sources(results)
        self.assertTrue(len(sources) > 0)
        self.assertIn("url", sources[0])
        self.assertTrue(sources[0]["url"].startswith("http"))

    def test_unrelated_query_yields_empty_results(self):
        unrelated = "How do I implement quicksort in Python with recursion?"
        results = self.retriever.retrieve(unrelated, top_k=3)
        self.assertEqual(len(results), 0, "Unrelated query should yield 0 results with threshold 0.58")
        sources = self.retriever.extract_sources(results)
        self.assertEqual(len(sources), 0)

    def test_format_context_for_prompt(self):
        results = self.retriever.retrieve("How can I improve sleep?", top_k=2)
        context_str = self.retriever.format_context_for_prompt(results)
        self.assertIn("VERIFIED MEDICAL KNOWLEDGE BASE CONTEXT", context_str)
        self.assertIn("END OF VERIFIED MEDICAL CONTEXT", context_str)


class TestFlaskAPIWithRAG(unittest.TestCase):
    """Test Flask routes and API response structure with RAG."""

    @classmethod
    def setUpClass(cls):
        import app
        from models import User, db
        cls.flask_app = app.app
        cls.flask_app.config["TESTING"] = True
        cls.flask_app.config["WTF_CSRF_ENABLED"] = False
        cls.client = cls.flask_app.test_client()

        with cls.flask_app.app_context():
            db.create_all()
            test_user = User.query.filter_by(email="rag_test@example.com").first()
            if not test_user:
                test_user = User(name="RAG Test User", email="rag_test@example.com")
                test_user.set_password("SecurePassword123!")
                db.session.add(test_user)
                db.session.commit()

        # Log in test client
        cls.client.post("/login", json={"email": "rag_test@example.com", "password": "SecurePassword123!"})

    @patch("app.get_gemini_response")
    def test_chat_returns_response_and_sources(self, mock_gemini):
        mock_gemini.return_value = (
            "Normal blood pressure for most adults is defined as systolic less than 120 mm Hg and diastolic less than 80 mm Hg [hypertension:p0001].",
            [{"source_title": "Hypertension Factsheet", "source_url": "https://www.who.int/news-room/fact-sheets/detail/hypertension"}],
            {
                "is_emergency": False,
                "evidence": {"status": "sufficient"},
                "citation_validation": {"valid_ids": ["hypertension:p0001"]},
            },
        )
        payload = {"message": "What is normal blood pressure?"}
        res = self.client.post("/chat", json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("response", data)
        self.assertIn("sources", data)
        self.assertIsInstance(data["sources"], list)
        self.assertTrue(len(data["response"]) > 20)

    def test_chat_empty_input_rejected(self):
        res = self.client.post("/chat", json={"message": ""})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    def test_chat_oversized_input_rejected(self):
        res = self.client.post("/chat", json={"message": "X" * 2001})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    def test_reset_endpoint(self):
        res = self.client.post("/reset")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "cleared")


if __name__ == "__main__":
    unittest.main(verbosity=2)
