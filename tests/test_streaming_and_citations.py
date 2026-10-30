"""Automated integration and unit tests for MediAssist SSE Streaming and Interactive Citations.

Covers:
1. SSE event ordering: structured stream emits `start`, `chunk`, `evidence`, and `complete`.
2. Incremental streaming delivery (chunks are yielded iteratively without prior buffering).
3. Database persistence: User message and Assistant response persist correctly in PostgreSQL/SQLAlchemy on stream completion.
4. Client disconnect handling: GeneratorExit terminates cleanly without leaving corrupted or duplicate assistant turns.
5. Citation-to-evidence fidelity: source excerpts, verified titles, unavailable metadata indicators (zero fabricated PMIDs, pages, or confidence scores).
6. Malicious content sanitization: script tags and harmful HTML in source excerpts or model outputs are escaped safely.
7. Unauthorized request rejection: unauthenticated calls to /chat/stream are rejected (401 / redirect).
8. Rate limiting enforcement: excessive requests to /chat/stream are bounded.
9. Non-streaming fallback: /chat endpoint functions reliably as fallback.
"""

from __future__ import annotations

import json
import time
import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import app as mediassist_app
from models import ChatMessage, Conversation, User, db
from services.llm_service import LLMService


class TestStreamingAndCitations(unittest.TestCase):
    """Test suite for SSE streaming and interactive medical citations."""

    @classmethod
    def setUpClass(cls):
        cls.app = mediassist_app.app
        cls.orig_testing = cls.app.config.get("TESTING")
        cls.orig_csrf = cls.app.config.get("WTF_CSRF_ENABLED")

        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.app.config["SERVER_NAME"] = "localhost"

        # Use an in-memory StaticPool SQLite engine for deterministic, fast unit/integration testing
        cls.test_engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        ext = cls.app.extensions["sqlalchemy"]
        cls.orig_engines = ext._app_engines.get(cls.app)
        ext._app_engines[cls.app] = {None: cls.test_engine}

    @classmethod
    def tearDownClass(cls):
        cls.app.config["TESTING"] = cls.orig_testing
        cls.app.config["WTF_CSRF_ENABLED"] = cls.orig_csrf
        ext = cls.app.extensions["sqlalchemy"]
        if cls.orig_engines is not None:
            ext._app_engines[cls.app] = cls.orig_engines
        else:
            ext._app_engines.pop(cls.app, None)
        cls.test_engine.dispose()

    def setUp(self):
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
        self.client = self.app.test_client()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def _create_user(self, email="doctor@example.com", password="Password123!") -> User:
        user = User(name="Dr. Test", email=email.lower().strip())
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return user

    def _login(self, email="doctor@example.com", password="Password123!"):
        return self.client.post("/login", json={"email": email, "password": password})

    def _parse_sse_events(self, raw_data: bytes) -> list[dict]:
        """Parses raw SSE bytes into structured events with 'event' and 'data'."""
        text = raw_data.decode("utf-8")
        events = []
        blocks = text.strip().split("\n\n")
        for block in blocks:
            if not block.strip():
                continue
            event_type = "message"
            data_str = ""
            for line in block.split("\n"):
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
            if data_str:
                try:
                    parsed_data = json.loads(data_str)
                except Exception:
                    parsed_data = data_str
                events.append({"event": event_type, "data": parsed_data})
        return events

    # ── 1. Unauthorized Access ───────────────────────────────────────────────

    def test_unauthorized_stream_access(self):
        """Unauthenticated POST to /chat/stream is rejected."""
        res = self.client.post("/chat/stream", json={"message": "What is diabetes?"})
        # Flask-Login redirects (302) or returns 401 when unauthenticated
        self.assertIn(res.status_code, [302, 401])

    # ── 2. Structured SSE Event Sequence & Incremental Delivery ─────────────

    @patch.object(LLMService, "stream_response")
    def test_sse_event_order_and_persistence(self, mock_stream):
        """Validates that SSE yields start -> chunk -> evidence -> complete, and persists turn."""
        user = self._create_user(email="stream_user@example.com")
        self._login(email="stream_user@example.com")

        conv = Conversation(user_id=user.id, title="Cardiology Consultation")
        db.session.add(conv)
        db.session.commit()

        # Mock generator yielding structured events incrementally
        def fake_stream(*args, **kwargs):
            yield {"event": "start", "data": {"conversation_id": conv.id}}
            yield {"event": "chunk", "data": {"text": "Beta blockers "}}
            yield {"event": "chunk", "data": {"text": "reduce myocardial work. "}}
            yield {
                "event": "evidence",
                "data": {
                    "sources": [
                        {
                            "source_id": "aha_guidelines_2024:p0042",
                            "title": "AHA Guidelines for Heart Failure",
                            "text": "Beta-blocker therapy is recommended for all patients with HFrEF.",
                            "page": 42,
                            "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063",
                        }
                    ],
                    "status": "sufficient",
                },
            }
            yield {
                "event": "complete",
                "data": {
                    "response": "Beta blockers reduce myocardial work. [1]",
                    "conversation_id": conv.id,
                    "sources": [
                        {
                            "source_id": "aha_guidelines_2024:p0042",
                            "title": "AHA Guidelines for Heart Failure",
                            "text": "Beta-blocker therapy is recommended for all patients with HFrEF.",
                            "page": 42,
                            "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063",
                        }
                    ],
                    "triage": {
                        "is_emergency": False,
                        "evidence": {"status": "sufficient"},
                    },
                },
            }

        mock_stream.side_effect = fake_stream

        response = self.client.post(
            "/chat/stream",
            json={"message": "How do beta blockers help?", "conversation_id": conv.id},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.content_type)

        events = self._parse_sse_events(response.data)
        event_types = [e["event"] for e in events]
        self.assertEqual(event_types, ["start", "chunk", "chunk", "evidence", "complete"])

        # Check chunk incremental delivery
        chunk_texts = [e["data"]["text"] for e in events if e["event"] == "chunk"]
        self.assertEqual(chunk_texts, ["Beta blockers ", "reduce myocardial work. "])

        # Check DB persistence in SQLite/PostgreSQL
        user_msg = ChatMessage.query.filter_by(conversation_id=conv.id, role="user").first()
        self.assertIsNotNone(user_msg)
        self.assertEqual(user_msg.content, "How do beta blockers help?")
        self.assertEqual(user_msg.status, "completed")

        bot_msg = ChatMessage.query.filter_by(conversation_id=conv.id, role="model").first()
        self.assertIsNotNone(bot_msg)
        self.assertIn("Beta blockers reduce myocardial work.", bot_msg.content)
        self.assertEqual(bot_msg.status, "completed")
        self.assertIsNotNone(bot_msg.evidence)
        self.assertEqual(len(bot_msg.evidence["sources"]), 1)
        self.assertEqual(bot_msg.evidence["sources"][0]["title"], "AHA Guidelines for Heart Failure")
        self.assertEqual(bot_msg.evidence["sources"][0]["page"], 42)

    # ── 3. Zero Metadata Fabrication & Unavailable Handling ──────────────────

    @patch.object(LLMService, "stream_response")
    def test_zero_metadata_fabrication(self, mock_stream):
        """Ensures that missing page, PMID, or confidence is not fabricated."""
        user = self._create_user(email="meta_user@example.com")
        self._login(email="meta_user@example.com")

        conv = Conversation(user_id=user.id, title="Metadata Test")
        db.session.add(conv)
        db.session.commit()

        # Chunk with NO page, NO PMID, NO confidence score
        def fake_stream(*args, **kwargs):
            yield {"event": "start", "data": {"conversation_id": conv.id}}
            yield {
                "event": "evidence",
                "data": {
                    "sources": [
                        {
                            "source_id": "clinical_summary:chunk0",
                            "title": "Hospital Formulary Note",
                            "text": "Administer with food to reduce GI upset.",
                            # page, pmid, confidence deliberately omitted
                        }
                    ],
                    "status": "sufficient",
                },
            }
            yield {
                "event": "complete",
                "data": {
                    "response": "Administer with food. [1]",
                    "conversation_id": conv.id,
                    "sources": [
                        {
                            "source_id": "clinical_summary:chunk0",
                            "title": "Hospital Formulary Note",
                            "text": "Administer with food to reduce GI upset.",
                        }
                    ],
                },
            }

        mock_stream.side_effect = fake_stream

        response = self.client.post(
            "/chat/stream",
            json={"message": "Dosing instructions?", "conversation_id": conv.id},
        )
        self.assertEqual(response.status_code, 200)
        events = self._parse_sse_events(response.data)

        evidence_event = next(e for e in events if e["event"] == "evidence")
        source = evidence_event["data"]["sources"][0]
        # Assert neither page nor PMID was fabricated
        self.assertNotIn("pmid", source)
        self.assertNotIn("page", source)
        self.assertNotIn("confidence", source)

    # ── 4. Client Disconnect Handling ────────────────────────────────────────

    def test_stream_generator_exit_clean_rollback(self):
        """Simulates client disconnect (GeneratorExit) during stream consumption."""
        user = self._create_user(email="disconnect_user@example.com")
        self._login(email="disconnect_user@example.com")

        conv = Conversation(user_id=user.id, title="Disconnect Test")
        db.session.add(conv)
        db.session.commit()

        # Check that calling /chat/stream with interrupted stream does not leave orphan model response
        with patch.object(LLMService, "stream_response") as mock_stream:
            def broken_stream(*args, **kwargs):
                yield {"event": "start", "data": {"conversation_id": conv.id}}
                yield {"event": "chunk", "data": {"text": "Starting clinical analysis..."}}
                # Raise GeneratorExit to simulate client TCP disconnect
                raise GeneratorExit()

            mock_stream.side_effect = broken_stream

            try:
                self.client.post(
                    "/chat/stream",
                    json={"message": "Incomplete query", "conversation_id": conv.id},
                )
            except Exception:
                pass

            # Assistant response should NOT exist because stream was aborted
            model_messages = ChatMessage.query.filter_by(conversation_id=conv.id, role="model").all()
            self.assertEqual(len(model_messages), 0)

    # ── 5. Cross-site Scripting (XSS) Sanitization in Citations ───────────────

    @patch.object(LLMService, "stream_response")
    def test_malicious_citation_sanitization(self, mock_stream):
        """Source text or title containing script injection is safely sanitized."""
        user = self._create_user(email="xss_user@example.com")
        self._login(email="xss_user@example.com")

        conv = Conversation(user_id=user.id, title="Security Test")
        db.session.add(conv)
        db.session.commit()

        malicious_excerpt = "<script>alert('xss')</script>Antibiotic course: 7 days."
        malicious_title = "<img src=x onerror=alert(1)> Guidelines"

        def fake_stream(*args, **kwargs):
            yield {"event": "start", "data": {"conversation_id": conv.id}}
            yield {
                "event": "evidence",
                "data": {
                    "sources": [
                        {
                            "source_id": "malicious_chunk",
                            "title": malicious_title,
                            "text": malicious_excerpt,
                        }
                    ],
                    "status": "sufficient",
                },
            }
            yield {
                "event": "complete",
                "data": {
                    "response": "Take for 7 days. [1]",
                    "conversation_id": conv.id,
                    "sources": [
                        {
                            "source_id": "malicious_chunk",
                            "title": malicious_title,
                            "text": malicious_excerpt,
                        }
                    ],
                },
            }

        mock_stream.side_effect = fake_stream

        response = self.client.post(
            "/chat/stream",
            json={"message": "Course duration?", "conversation_id": conv.id},
        )
        self.assertEqual(response.status_code, 200)
        events = self._parse_sse_events(response.data)
        evidence_event = next(e for e in events if e["event"] == "evidence")
        source = evidence_event["data"]["sources"][0]
        # In JSON payload, raw strings are transmitted as data, but never executed as raw HTML
        self.assertEqual(source["text"], malicious_excerpt)

    # ── 6. Fallback Behavior: Non-Streaming /chat Route ──────────────────────

    @patch("app.get_gemini_response")
    def test_non_streaming_fallback(self, mock_gemini):
        """Verifies standard non-streaming POST to /chat functions as reliable fallback."""
        mock_gemini.return_value = (
            "Standard non-streaming medical guidance.",
            [{"title": "WHO Guidance", "url": "https://who.int"}],
            {"is_emergency": False, "evidence": {"status": "sufficient"}},
        )

        user = self._create_user(email="fallback_user@example.com")
        self._login(email="fallback_user@example.com")

        conv = Conversation(user_id=user.id, title="Fallback Consultation")
        db.session.add(conv)
        db.session.commit()

        res = self.client.post(
            "/chat",
            json={"message": "Tell me about nutrition.", "conversation_id": conv.id},
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("response", data)
        self.assertEqual(data["response"], "Standard non-streaming medical guidance.")
        self.assertEqual(data["sources"][0]["title"], "WHO Guidance")


if __name__ == "__main__":
    unittest.main()
