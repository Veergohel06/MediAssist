"""Automated integration tests for MediAssist persistent chat history API.

Covers:
- Full user-message-to-assistant-response cycle with database turn persistence.
- Strict multi-tenant conversation ownership verification (404 on nonexistent, 403 on cross-user).
- Unsupplied conversation ID contract (creates or selects active conversation without duplicating on retry).
- Failed model request handling (zero fabricated assistant responses saved).
- Partial failure retry idempotency (user message not silently duplicated on retry).
- Bounded history context limits (MAX_HISTORY_TURNS) and strict cross-user history isolation.
- Source-reference and citation metadata persistence.
- Emergency symptom safety escalation and resilience during database persistence failures.
"""

from __future__ import annotations

import json
import time
import unittest
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import app as mediassist_app
from models import ChatMessage, Conversation, User, db, utc_now
from google.genai import errors as genai_errors


class TestChatHistoryAPI(unittest.TestCase):
    """Integration test suite for persistent conversation history with /chat API."""

    @classmethod
    def setUpClass(cls):
        cls.app = mediassist_app.app
        cls.orig_testing = cls.app.config.get("TESTING")
        cls.orig_csrf = cls.app.config.get("WTF_CSRF_ENABLED")

        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.app.config["SERVER_NAME"] = "localhost"

        # Isolated in-memory SQLite engine using StaticPool
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
        mediassist_app._ip_failed_logins.clear()
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
        self.client = self.app.test_client()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def _create_user(self, email="patient@example.com", name="Test Patient", password="Password123!") -> User:
        """Helper to create and persist a test user."""
        user = User(name=name, email=email.lower().strip())
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return user

    def _login(self, email="patient@example.com", password="Password123!", client=None):
        """Helper to log in a test client session."""
        c = client or self.client
        return c.post("/login", json={"email": email, "password": password})

    # ── 1. Full Chat Cycle & Turn Persistence ───────────────────────────────────

    @patch("app.get_gemini_response")
    def test_full_chat_turn_cycle(self, mock_gemini):
        """Full cycle: authenticated user sends message, AI answers, turns saved to DB."""
        mock_gemini.return_value = (
            "Hydration is essential for cognitive function and health.",
            [{"title": "Water Requirements", "url": "https://who.int/water"}],
            {
                "is_emergency": False,
                "evidence": {"status": "sufficient"},
                "citation_validation": {"valid_ids": ["water_needs:p0001"]},
            },
        )

        user = self._create_user(email="user_cycle@example.com")
        self._login(email="user_cycle@example.com")

        # Create explicit consultation conversation
        conv = Conversation(user_id=user.id, title="Water Query")
        db.session.add(conv)
        db.session.commit()
        orig_updated_at = conv.updated_at

        time.sleep(0.01)

        res = self.client.post(
            "/chat",
            json={"message": "How much water do I need daily?", "conversation_id": conv.id},
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Validate response payload structure
        self.assertIn("Hydration is essential", data["response"])
        self.assertEqual(data["conversation_id"], conv.id)
        self.assertIsNotNone(data["user_message_id"])
        self.assertIsNotNone(data["assistant_message_id"])
        self.assertIn("timestamp", data)
        self.assertEqual(len(data["sources"]), 1)

        # Validate database records
        db.session.refresh(conv)
        self.assertGreater(conv.updated_at, orig_updated_at)

        messages = conv.messages.all()
        self.assertEqual(len(messages), 2)

        user_turn = messages[0]
        self.assertEqual(user_turn.role, "user")
        self.assertEqual(user_turn.content, "How much water do I need daily?")
        self.assertEqual(user_turn.status, "completed")
        self.assertEqual(user_turn.id, data["user_message_id"])

        bot_turn = messages[1]
        self.assertEqual(bot_turn.role, "model")
        self.assertIn("Hydration is essential", bot_turn.content)
        self.assertEqual(bot_turn.status, "completed")
        self.assertEqual(bot_turn.id, data["assistant_message_id"])
        self.assertIn("sources", bot_turn.evidence)

    # ── 2. Conversation Ownership Verification ──────────────────────────────────

    def test_cross_user_conversation_ownership_enforcement(self):
        """Chat API rejects nonexistent conversation (404) and cross-user conversation (403)."""
        user_a = self._create_user(email="usera_owner@example.com")
        user_b = self._create_user(email="userb_owner@example.com")

        conv_a = Conversation(user_id=user_a.id, title="User A Confidential")
        db.session.add(conv_a)
        db.session.commit()

        # User B logs in
        self._login(email="userb_owner@example.com")

        # 1. Nonexistent conversation -> 404
        res_404 = self.client.post(
            "/chat",
            json={"message": "Hello", "conversation_id": "nonexistent-uuid-1234"},
        )
        self.assertEqual(res_404.status_code, 404)
        self.assertIn("not found", res_404.get_json()["error"].lower())

        # 2. User A's conversation -> 403 Forbidden
        res_403 = self.client.post(
            "/chat",
            json={"message": "I want to inject here", "conversation_id": conv_a.id},
        )
        self.assertEqual(res_403.status_code, 403)
        self.assertIn("access denied", res_403.get_json()["error"].lower())

        # Verify no messages were injected into User A's conversation
        self.assertEqual(conv_a.messages.count(), 0)

    # ── 3. Unsupplied Conversation ID Contract ──────────────────────────────────

    @patch("app.get_gemini_response")
    def test_unsupplied_conversation_id_creates_and_reuses_conversation(self, mock_gemini):
        """When no conversation_id is supplied, active conversation is selected/created without duplicating."""
        mock_gemini.return_value = ("General health advice.", [], {"is_emergency": False})

        user = self._create_user(email="no_conv@example.com")
        self._login(email="no_conv@example.com")

        # First call: no conversation exists -> creates a new one
        res1 = self.client.post("/chat", json={"message": "First consultation question"})
        self.assertEqual(res1.status_code, 200)
        conv_id_1 = res1.get_json()["conversation_id"]
        self.assertIsNotNone(conv_id_1)

        # Verify user has exactly 1 conversation
        self.assertEqual(user.conversations.count(), 1)

        # Second call without conversation_id -> reuses the active conversation
        res2 = self.client.post("/chat", json={"message": "Follow up question"})
        self.assertEqual(res2.status_code, 200)
        conv_id_2 = res2.get_json()["conversation_id"]

        # Must reuse the same conversation
        self.assertEqual(conv_id_1, conv_id_2)
        self.assertEqual(user.conversations.count(), 1)

        # Both turns exist in that single conversation
        conv = db.session.get(Conversation, conv_id_1)
        self.assertEqual(conv.messages.count(), 4)  # 2 user turns + 2 model turns

    # ── 4. Failed Model Requests & Zero Fabricated Responses ────────────────────

    @patch("app.get_gemini_response")
    def test_failed_model_request_does_not_save_fabricated_assistant_response(self, mock_gemini):
        """When AI service fails (e.g. 503 Service Unavailable), no fabricated assistant response is saved."""
        class MockAPIError(genai_errors.APIError):
            def __init__(self):
                super().__init__(503, "Google API temporary failure")

        mock_gemini.side_effect = MockAPIError()

        user = self._create_user(email="fail_test@example.com")
        self._login(email="fail_test@example.com")

        conv = Conversation(user_id=user.id, title="Failure Test")
        db.session.add(conv)
        db.session.commit()

        res = self.client.post(
            "/chat",
            json={"message": "Explain migraines", "conversation_id": conv.id},
        )
        self.assertEqual(res.status_code, 503)

        # Verify database state
        messages = conv.messages.all()
        # Exactly 1 message: the user message with status='error'
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].status, "error")

        # Zero assistant / model messages saved
        bot_msgs = conv.messages.filter_by(role="model").all()
        self.assertEqual(len(bot_msgs), 0)

    # ── 5. Retry Behavior & Idempotency ─────────────────────────────────────────

    @patch("app.get_gemini_response")
    def test_retry_after_failed_request_does_not_duplicate_user_message(self, mock_gemini):
        """Retrying an unanswered message reuses the user message record instead of duplicating it."""
        class MockAPIError(genai_errors.APIError):
            def __init__(self):
                super().__init__(503, "Service unavailable")

        # Attempt 1: Fails
        mock_gemini.side_effect = MockAPIError()

        user = self._create_user(email="retry_user@example.com")
        self._login(email="retry_user@example.com")

        conv = Conversation(user_id=user.id, title="Retry Test")
        db.session.add(conv)
        db.session.commit()

        res1 = self.client.post(
            "/chat",
            json={"message": "What is asthma?", "conversation_id": conv.id},
        )
        self.assertEqual(res1.status_code, 503)
        self.assertEqual(conv.messages.count(), 1)
        initial_user_msg_id = conv.messages.first().id

        # Attempt 2: User retries the exact same prompt
        mock_gemini.side_effect = None
        mock_gemini.return_value = (
            "Asthma is a chronic inflammatory airway condition.",
            [{"title": "Asthma Guide", "url": "https://who.int/asthma"}],
            {"is_emergency": False},
        )

        res2 = self.client.post(
            "/chat",
            json={"message": "What is asthma?", "conversation_id": conv.id},
        )
        self.assertEqual(res2.status_code, 200)

        # Database must now contain exactly 2 messages: 1 user message (reused) and 1 model response
        messages = conv.messages.all()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].id, initial_user_msg_id)  # Reused identical row
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].status, "completed")

        self.assertEqual(messages[1].role, "model")
        self.assertIn("Asthma is a chronic", messages[1].content)
        self.assertEqual(messages[1].status, "completed")

    # ── 6. Context Window Bounding & History Isolation ──────────────────────────

    @patch("app.get_gemini_response")
    def test_bounded_history_context_limits_and_isolation(self, mock_gemini):
        """History passed to model is bounded to MAX_HISTORY_TURNS and isolated across users."""
        captured_histories = []

        def capture_call(msg, history=None):
            captured_histories.append(list(history or []))
            return ("Response text", [], {"is_emergency": False})

        mock_gemini.side_effect = capture_call

        user_a = self._create_user(email="isolation_a@example.com")
        user_b = self._create_user(email="isolation_b@example.com")

        # User A has 5 turns in Conversation A
        conv_a = Conversation(user_id=user_a.id, title="User A History")
        db.session.add(conv_a)
        db.session.commit()
        for i in range(5):
            db.session.add(ChatMessage(conversation_id=conv_a.id, role="user", content=f"User A query {i}", status="completed"))
            db.session.add(ChatMessage(conversation_id=conv_a.id, role="model", content=f"User A reply {i}", status="completed"))
        db.session.commit()

        # User B logs in and has a multi-turn conversation (25 turns, exceeding MAX_HISTORY_TURNS=20)
        self._login(email="isolation_b@example.com")
        conv_b = Conversation(user_id=user_b.id, title="User B Long History")
        db.session.add(conv_b)
        db.session.commit()

        for i in range(25):
            db.session.add(ChatMessage(conversation_id=conv_b.id, role="user", content=f"User B query {i}", status="completed"))
            db.session.add(ChatMessage(conversation_id=conv_b.id, role="model", content=f"User B reply {i}", status="completed"))
        db.session.commit()

        # User B asks a new question
        res = self.client.post(
            "/chat",
            json={"message": "User B fresh question", "conversation_id": conv_b.id},
        )
        self.assertEqual(res.status_code, 200)

        self.assertEqual(len(captured_histories), 1)
        history_sent = captured_histories[0]

        # 1. Bounded: history must NOT exceed MAX_HISTORY_TURNS (20)
        self.assertLessEqual(len(history_sent), 20)

        # 2. Strict isolation: NO User A messages may exist in history
        history_texts = [turn["parts"][0]["text"] for turn in history_sent]
        for text in history_texts:
            self.assertNotIn("User A", text)
            self.assertIn("User B", text)

    # ── 7. Source References & Citation Metadata ────────────────────────────────

    @patch("app.get_gemini_response")
    def test_source_reference_metadata_persistence(self, mock_gemini):
        """Evidence and citations are persisted and retrieved via conversation GET endpoint."""
        sources = [
            {"title": "CDC Cold vs Flu", "url": "https://cdc.gov/flu", "source": "CDC"},
            {"title": "MedlinePlus Cold", "url": "https://medlineplus.gov/cold", "source": "NIH"},
        ]
        triage = {
            "is_emergency": False,
            "evidence": {"status": "sufficient"},
            "citation_validation": {"valid_ids": ["common_cold_vs_influenza:p0001"]},
        }
        mock_gemini.return_value = ("A cold develops gradually whereas the flu is abrupt.", sources, triage)

        user = self._create_user(email="sources_test@example.com")
        self._login(email="sources_test@example.com")

        res_chat = self.client.post("/chat", json={"message": "Cold vs flu symptoms"})
        self.assertEqual(res_chat.status_code, 200)
        conv_id = res_chat.get_json()["conversation_id"]

        # Fetch conversation details via API
        res_get = self.client.get(f"/api/conversations/{conv_id}")
        self.assertEqual(res_get.status_code, 200)
        conv_data = res_get.get_json()["conversation"]

        messages = conv_data["messages"]
        self.assertEqual(len(messages), 2)
        model_msg = messages[1]
        self.assertEqual(model_msg["role"], "model")
        self.assertEqual(len(model_msg["evidence"]["sources"]), 2)
        self.assertEqual(model_msg["evidence"]["sources"][0]["title"], "CDC Cold vs Flu")

    # ── 8. Emergency Safety Escalation & Database Resilience ───────────────────

    def test_emergency_escalation_bypasses_model_and_preserves_safety_on_db_error(self):
        """Urgent red flag query immediately returns emergency guidance even if DB persistence fails."""
        urgent_query = "Severe sudden chest pain radiating to left arm and jaw, sweating profusely, cannot breathe"

        user = self._create_user(email="emergency_patient@example.com")
        self._login(email="emergency_patient@example.com")

        # 1. Standard emergency query succeeds and delivers life-saving guidance
        res = self.client.post("/chat", json={"message": urgent_query})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["triage"]["is_emergency"])
        self.assertIn("EMERGENCY", data["response"])
        self.assertIn("112", data["response"])  # India emergency helpline guidance

        # 2. Simulate complete database commit failure during an emergency query
        with patch.object(db.session, "commit", side_effect=Exception("Database disk I/O failure")):
            res_db_fail = self.client.post("/chat", json={"message": urgent_query})
            # Must STILL return 200 and deliver emergency guidance to the patient!
            self.assertEqual(res_db_fail.status_code, 200)
            data_fail = res_db_fail.get_json()
            self.assertTrue(data_fail["triage"]["is_emergency"])
            self.assertIn("EMERGENCY", data_fail["response"])
            self.assertIn("112", data_fail["response"])


    # ── 9. Conversation Rename & Deletion Endpoints ────────────────────────────

    def test_rename_conversation_success(self):
        """User can rename their conversation using PATCH and PUT."""
        user = self._create_user(email="rename_owner@example.com")
        self._login(email="rename_owner@example.com")

        conv = Conversation(user_id=user.id, title="Old Title")
        db.session.add(conv)
        db.session.commit()

        # PATCH update
        res_patch = self.client.patch(
            f"/api/conversations/{conv.id}",
            json={"title": "Updated Consultation Title"},
        )
        self.assertEqual(res_patch.status_code, 200)
        self.assertEqual(res_patch.get_json()["conversation"]["title"], "Updated Consultation Title")

        # Verify DB
        db.session.refresh(conv)
        self.assertEqual(conv.title, "Updated Consultation Title")

        # PUT update
        res_put = self.client.put(
            f"/api/conversations/{conv.id}",
            json={"title": "PUT Consultation Title"},
        )
        self.assertEqual(res_put.status_code, 200)
        self.assertEqual(res_put.get_json()["conversation"]["title"], "PUT Consultation Title")

    def test_rename_conversation_validation_and_isolation(self):
        """Renaming rejects empty title, handles cross-user attempts and nonexistent IDs."""
        user1 = self._create_user(email="rename_user1@example.com")
        user2 = self._create_user(email="rename_user2@example.com")

        conv = Conversation(user_id=user1.id, title="Original Title")
        db.session.add(conv)
        db.session.commit()

        # 1. Unauthenticated rename
        res_anon = self.client.patch(f"/api/conversations/{conv.id}", json={"title": "Hacked"})
        self.assertEqual(res_anon.status_code, 401)

        # 2. Authenticated as user2 (cross-user access attempt)
        self._login(email="rename_user2@example.com")
        res_cross = self.client.patch(f"/api/conversations/{conv.id}", json={"title": "User 2 Title"})
        self.assertEqual(res_cross.status_code, 403)

        # 3. Authenticated as owner with empty title
        self._login(email="rename_user1@example.com")
        res_empty = self.client.patch(f"/api/conversations/{conv.id}", json={"title": "   "})
        self.assertEqual(res_empty.status_code, 400)

        # 4. Nonexistent conversation
        res_404 = self.client.patch("/api/conversations/nonexistent-id", json={"title": "New Title"})
        self.assertEqual(res_404.status_code, 404)

    def test_delete_conversation_success_and_isolation(self):
        """Deleting conversation cascade-removes its messages and blocks cross-user deletion."""
        user1 = self._create_user(email="delete_user1@example.com")
        user2 = self._create_user(email="delete_user2@example.com")

        conv = Conversation(user_id=user1.id, title="Consultation to Delete")
        db.session.add(conv)
        db.session.commit()

        msg = ChatMessage(conversation_id=conv.id, role="user", content="Secret message")
        db.session.add(msg)
        db.session.commit()

        # 1. Cross-user delete returns 403
        self._login(email="delete_user2@example.com")
        res_cross = self.client.delete(f"/api/conversations/{conv.id}")
        self.assertEqual(res_cross.status_code, 403)

        # 2. Owner delete returns 200
        self._login(email="delete_user1@example.com")
        res_del = self.client.delete(f"/api/conversations/{conv.id}")
        self.assertEqual(res_del.status_code, 200)
        self.assertEqual(res_del.get_json()["status"], "deleted")

        # Verify cascade deletion from DB
        self.assertIsNone(Conversation.query.filter_by(id=conv.id).first())
        self.assertIsNone(ChatMessage.query.filter_by(id=msg.id).first())

        # 3. Deleting nonexistent returns 404
        res_404 = self.client.delete(f"/api/conversations/{conv.id}")
        self.assertEqual(res_404.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)

