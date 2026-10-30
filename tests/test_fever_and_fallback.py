"""Tests for fever retrieval, educational fallback, and API error resilience."""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app
from rag.retriever import MedicalRetriever
from rag.evidence import parse_structured_response


class TestFeverAndFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.app.config["TESTING"] = True
        app.app.config["WTF_CSRF_ENABLED"] = False
        cls.client = app.app.test_client()

    def test_parse_structured_response_with_markdown_fences(self):
        fenced_json = """```json
{
  "answer": "Fever is an elevated body temperature.",
  "cited_passage_ids": ["fever_causes_and_management:p0001"],
  "evidence_sufficiency": "sufficient"
}
```"""
        parsed = parse_structured_response(fenced_json)
        self.assertTrue(parsed["structured_output_valid"])
        self.assertEqual(parsed["answer"], "Fever is an elevated body temperature.")
        self.assertEqual(parsed["cited_passage_ids"], ["fever_causes_and_management:p0001"])
        self.assertEqual(parsed["evidence_sufficiency"], "sufficient")

    def test_fever_retrieval_returns_relevant_chunks(self):
        if not app._retriever.is_ready():
            self.skipTest("Knowledge base index not loaded")
        chunks = app._retriever.retrieve("What causes a fever and when should I seek medical care?", top_k=3)
        self.assertTrue(len(chunks) > 0)
        source_titles = [c.get("source_title", "") for c in chunks]
        self.assertTrue(any("Fever" in t for t in source_titles))

    def test_api_status_endpoint(self):
        # Unauthenticated request must be rejected (Issue #9 in security audit)
        anon_res = self.client.get("/api/status")
        self.assertEqual(anon_res.status_code, 401)

        # Authenticated request succeeds and returns status
        with app.app.app_context():
            from models import db, User
            db.create_all()
            user = User.query.filter_by(email="status_tester@example.com").first()
            if not user:
                user = User(email="status_tester@example.com", name="Status Tester")
                user.set_password("Password123!")
                db.session.add(user)
                db.session.commit()
        self.client.post("/login", json={"email": "status_tester@example.com", "password": "Password123!"})
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model", data)
        self.assertIn("knowledge_chunks", data)
        self.assertTrue(data["knowledge_chunks"] > 0)

    @patch("app._client.models.generate_content")
    def test_all_models_failing_returns_infrastructure_error(self, mock_generate):
        mock_generate.side_effect = RuntimeError("Service Unavailable 503")
        response, sources, triage = app.get_gemini_response("General query")
        self.assertIn("temporarily unable to connect to the Google Gemini AI service", response)
    @patch("app._client.models.generate_content")
    def test_citation_sanitization_in_get_gemini_response(self, mock_generate):
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "answer": (
                "According to the document *Tension Headaches and Migraines: Causes, Symptoms, and Management*, "
                "tension headaches are very common [tension_headaches_and_migraines:p0002]. "
                "Additionally, mild dehydration can contribute to headaches [hydration_and_water_balance:p0005]."
            ),
            "cited_passage_ids": [
                "tension_headaches_and_migraines:p0002",
                "hydration_and_water_balance:p0005",
            ],
            "evidence_sufficiency": "sufficient"
        })
        mock_resp.usage_metadata = None
        mock_generate.return_value = mock_resp

        reply, sources, triage = app.get_gemini_response("Tell me about tension headaches and dehydration")

        # Crucial: raw internal passage IDs and metadata must never appear in final reply
        self.assertNotIn("p0002", reply)
        self.assertNotIn("p0005", reply)
        self.assertNotIn("[tension_headaches_and_migraines", reply)
        self.assertNotIn("[hydration_and_water_balance", reply)

        # Preceding attribution preserved without redundant citation token
        self.assertIn("According to the document *Tension Headaches and Migraines: Causes, Symptoms, and Management*", reply)

        # Unattributed source rendered cleanly as clickable markdown link or attribution
        self.assertIn("Water and Daily Hydration Requirements", reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
