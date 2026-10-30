"""Automated tests for MediAssist persistent chat history database layer.

Covers:
- Conversation ownership enforcement and user foreign key relationships.
- ChatMessage foreign key relationships and CASCADE deletion behavior.
- Deterministic message ordering using created_at timestamp and secondary ID.
- Automatic conversation updated_at touch on message creation and updates.
- Role restriction and validation (user, assistant, model, system).
- Message content validation (rejection of empty/whitespace content).
- Source reference metadata validation, structure enforcement, and sensitive key scrubbing.
- Message status and error metadata handling.
- Multi-tier cascading deletion (User -> Conversation -> ChatMessages).
"""

from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone, timedelta

from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from models import (
    ChatMessage,
    Conversation,
    User,
    db,
    utc_now,
    SUPPORTED_ROLES,
    SUPPORTED_MESSAGE_STATUSES,
    sanitize_evidence_metadata,
)


class DatabaseHistoryTestCase(unittest.TestCase):
    """Test suite for persistent chat history models, constraints, and data integrity."""

    @classmethod
    def setUpClass(cls):
        cls.app = Flask("mediassist_test_db")
        cls.app.config["TESTING"] = True
        cls.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        cls.app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

        db.init_app(cls.app)

        with cls.app.app_context():
            db.create_all()

    def setUp(self):
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        db.session.rollback()
        # Clean all records for clean state between tests
        db.session.query(ChatMessage).delete()
        db.session.query(Conversation).delete()
        db.session.query(User).delete()
        db.session.commit()
        self.ctx.pop()

    def _create_user(self, email="patient@example.com", name="Test Patient") -> User:
        """Helper to create a persisted user."""
        user = User(email=email, name=name)
        user.set_password("SecurePassword123!")
        db.session.add(user)
        db.session.commit()
        return user

    # ── Conversation Ownership & Foreign Keys ───────────────────────────────────

    def test_conversation_requires_owner(self):
        """Conversations must have an owner; None or non-positive user_id raises ValueError."""
        with self.assertRaises(ValueError):
            Conversation(user_id=None, title="Orphaned Chat")

        with self.assertRaises(ValueError):
            Conversation(user_id=0, title="Zero User ID")

        with self.assertRaises(ValueError):
            Conversation(user_id=-5, title="Negative User ID")

    def test_conversation_nonexistent_foreign_key_rejected(self):
        """Foreign key constraint rejects conversation referencing a nonexistent user."""
        conv = Conversation(user_id=99999, title="Nonexistent User Chat")
        db.session.add(conv)
        with self.assertRaises(IntegrityError):
            db.session.commit()

    def test_conversation_safe_default_title(self):
        """Conversations provide safe default title when None, whitespace, or oversized."""
        user = self._create_user()

        conv_none = Conversation(user_id=user.id, title=None)
        conv_empty = Conversation(user_id=user.id, title="   ")
        long_title = "A" * 300
        conv_long = Conversation(user_id=user.id, title=long_title)

        db.session.add_all([conv_none, conv_empty, conv_long])
        db.session.commit()

        self.assertEqual(conv_none.title, "New Consultation")
        self.assertEqual(conv_empty.title, "New Consultation")
        self.assertEqual(len(conv_long.title), 200)

    # ── Message Role & Content Validation ───────────────────────────────────────

    def test_supported_roles_accepted(self):
        """All supported roles (user, assistant, model, system) are accepted and normalized."""
        user = self._create_user()
        conv = Conversation(user_id=user.id, title="Role Test")
        db.session.add(conv)
        db.session.commit()

        for role in ["user", "model", "assistant", "system", "USER", "Model"]:
            msg = ChatMessage(conversation_id=conv.id, role=role, content=f"Hello from {role}")
            db.session.add(msg)
            db.session.commit()
            self.assertEqual(msg.role, role.lower())
            self.assertIn(msg.role, SUPPORTED_ROLES)

    def test_invalid_role_rejected(self):
        """Invalid roles like 'admin', 'bot', 'doctor', or empty strings raise ValueError."""
        user = self._create_user()
        conv = Conversation(user_id=user.id, title="Invalid Role Test")
        db.session.add(conv)
        db.session.commit()

        for invalid_role in ["admin", "bot", "doctor", "", None, 123]:
            with self.subTest(invalid_role=invalid_role):
                with self.assertRaises(ValueError):
                    ChatMessage(conversation_id=conv.id, role=invalid_role, content="Test")

    def test_empty_content_rejected(self):
        """Empty or whitespace-only message content raises ValueError."""
        user = self._create_user()
        conv = Conversation(user_id=user.id)
        db.session.add(conv)
        db.session.commit()

        for bad_content in ["", "   ", "\n\t", None]:
            with self.subTest(bad_content=bad_content):
                with self.assertRaises(ValueError):
                    ChatMessage(conversation_id=conv.id, role="user", content=bad_content)

    # ── Deterministic Message Ordering ──────────────────────────────────────────

    def test_deterministic_message_ordering(self):
        """Messages within a conversation are deterministically ordered by created_at then ID."""
        user = self._create_user()
        conv = Conversation(user_id=user.id, title="Ordering Test")
        db.session.add(conv)
        db.session.commit()

        fixed_time = utc_now()

        # Insert 3 messages with the EXACT same timestamp but specific sequence
        m1 = ChatMessage(id=10, conversation_id=conv.id, role="user", content="Turn 10", created_at=fixed_time)
        m2 = ChatMessage(id=20, conversation_id=conv.id, role="model", content="Turn 20", created_at=fixed_time)
        m3 = ChatMessage(id=5, conversation_id=conv.id, role="user", content="Turn 5", created_at=fixed_time)

        db.session.add_all([m1, m2, m3])
        db.session.commit()

        # Query messages via conversation relationship
        retrieved_ids = [m.id for m in conv.messages.all()]
        self.assertEqual(retrieved_ids, [5, 10, 20])

        # Test chronological progression across different timestamps
        later_time = fixed_time + timedelta(seconds=10)
        m4 = ChatMessage(id=1, conversation_id=conv.id, role="user", content="Later Turn", created_at=later_time)
        db.session.add(m4)
        db.session.commit()

        retrieved_ids_after = [m.id for m in conv.messages.all()]
        # 5, 10, 20 had earlier timestamp, so id 1 appears at the end despite lower id
        self.assertEqual(retrieved_ids_after, [5, 10, 20, 1])

    # ── Conversation Timestamp Updates ──────────────────────────────────────────

    def test_conversation_updated_at_touches_on_new_message(self):
        """Adding a ChatMessage to a conversation updates conversation.updated_at."""
        user = self._create_user()
        conv = Conversation(user_id=user.id, title="Timestamp Test")
        db.session.add(conv)
        db.session.commit()

        original_updated_at = conv.updated_at
        time.sleep(0.02)

        msg = ChatMessage(conversation_id=conv.id, role="user", content="New query")
        db.session.add(msg)
        db.session.commit()

        db.session.refresh(conv)
        self.assertGreater(conv.updated_at, original_updated_at)

    def test_conversation_updated_at_touches_on_message_update(self):
        """Updating an existing message updates the parent conversation's updated_at."""
        user = self._create_user()
        conv = Conversation(user_id=user.id, title="Update Timestamp Test")
        db.session.add(conv)
        db.session.commit()

        msg = ChatMessage(conversation_id=conv.id, role="user", content="First text")
        db.session.add(msg)
        db.session.commit()
        db.session.refresh(conv)

        intermediate_updated_at = conv.updated_at
        time.sleep(0.02)

        msg.content = "Edited query text"
        db.session.commit()
        db.session.refresh(conv)

        self.assertGreater(conv.updated_at, intermediate_updated_at)

    # ── Source-Reference Metadata (Evidence) Validation & Scrubbing ─────────────

    def test_sanitize_evidence_preserves_valid_sources(self):
        """Valid source references and citation metadata are preserved."""
        valid_evidence = {
            "sources": [
                {
                    "title": "Hypertension Guidelines",
                    "url": "https://who.int/hypertension",
                    "snippet": "Normal blood pressure is under 120/80 mmHg.",
                    "score": 0.88,
                }
            ],
            "triage": {"is_emergency": False},
        }

        sanitized = sanitize_evidence_metadata(valid_evidence)
        self.assertEqual(sanitized["sources"][0]["title"], "Hypertension Guidelines")
        self.assertEqual(sanitized["triage"]["is_emergency"], False)

    def test_sanitize_evidence_scrubs_sensitive_keys(self):
        """Sensitive internal keys (reasoning, thought, api_key, passwords) are stripped."""
        dirty_evidence = {
            "sources": [{"title": "CDC Cold vs Flu", "url": "https://cdc.gov"}],
            "thought": "Internal thinking process that should not be visible",
            "reasoning": "Hidden chain of thought",
            "chain_of_thought": "More hidden details",
            "api_key": "secret-12345",
            "gemini_api_key": "AIzaSySecretApiKey12345678901234567890",
            "bearer_token": "token-abc",
        }

        sanitized = sanitize_evidence_metadata(dirty_evidence)
        self.assertIn("sources", sanitized)
        self.assertNotIn("thought", sanitized)
        self.assertNotIn("reasoning", sanitized)
        self.assertNotIn("chain_of_thought", sanitized)
        self.assertNotIn("api_key", sanitized)
        self.assertNotIn("gemini_api_key", sanitized)
        self.assertNotIn("bearer_token", sanitized)

    def test_sanitize_evidence_redacts_embedded_api_keys(self):
        """API key strings embedded in textual values are redacted."""
        evidence_with_leak = {
            "note": "Calling endpoint with AIzaSy0123456789012345678901234567890123 and sk-abcdef0123456789abcdef0123456789",
        }
        sanitized = sanitize_evidence_metadata(evidence_with_leak)
        self.assertNotIn("AIzaSy", sanitized["note"])
        self.assertIn("[REDACTED_API_KEY]", sanitized["note"])

    def test_malformed_evidence_rejected(self):
        """Evidence that is neither a dict, list, valid JSON string, nor None raises ValueError."""
        with self.assertRaises(ValueError):
            sanitize_evidence_metadata(12345)

        with self.assertRaises(ValueError):
            sanitize_evidence_metadata("not-valid-json")

        with self.assertRaises(ValueError):
            sanitize_evidence_metadata(object())

    # ── Message Status and Error Metadata ───────────────────────────────────────

    def test_message_status_defaults_and_validation(self):
        """Message status defaults to 'completed' and validates allowed states."""
        user = self._create_user()
        conv = Conversation(user_id=user.id)
        db.session.add(conv)
        db.session.commit()

        msg_default = ChatMessage(conversation_id=conv.id, role="user", content="Default status")
        db.session.add(msg_default)
        db.session.commit()
        self.assertEqual(msg_default.status, "completed")
        self.assertIsNone(msg_default.error_message)

        for status in SUPPORTED_MESSAGE_STATUSES:
            msg = ChatMessage(
                conversation_id=conv.id,
                role="assistant",
                content=f"Status {status}",
                status=status,
                error_message="Sample error" if status == "error" else None,
            )
            db.session.add(msg)
            db.session.commit()
            self.assertEqual(msg.status, status)
            if status == "error":
                self.assertEqual(msg.error_message, "Sample error")

        with self.assertRaises(ValueError):
            ChatMessage(conversation_id=conv.id, role="user", content="Hi", status="corrupted_status")

    # ── Cascade Deletion Behavior ───────────────────────────────────────────────

    def test_conversation_deletion_cascades_to_messages(self):
        """Deleting a conversation cascades and permanently deletes all its chat messages."""
        user = self._create_user()
        conv = Conversation(user_id=user.id, title="To Delete")
        db.session.add(conv)
        db.session.commit()

        m1 = ChatMessage(conversation_id=conv.id, role="user", content="Msg 1")
        m2 = ChatMessage(conversation_id=conv.id, role="assistant", content="Msg 2")
        db.session.add_all([m1, m2])
        db.session.commit()

        self.assertEqual(ChatMessage.query.filter_by(conversation_id=conv.id).count(), 2)

        db.session.delete(conv)
        db.session.commit()

        self.assertEqual(ChatMessage.query.filter_by(conversation_id=conv.id).count(), 0)

    def test_user_deletion_cascades_to_conversations_and_messages(self):
        """Deleting a user cascades and deletes all their conversations and all associated messages."""
        user = self._create_user(email="cascade@example.com")
        conv = Conversation(user_id=user.id, title="Cascade Conv")
        db.session.add(conv)
        db.session.commit()

        m = ChatMessage(conversation_id=conv.id, role="user", content="Cascade Msg")
        db.session.add(m)
        db.session.commit()

        db.session.delete(user)
        db.session.commit()

        self.assertIsNone(db.session.get(Conversation, conv.id))
        self.assertEqual(ChatMessage.query.filter_by(conversation_id=conv.id).count(), 0)

    # ── Serialization Integrity ─────────────────────────────────────────────────

    def test_conversation_and_message_serialization(self):
        """to_dict produces clean, complete dictionary representations."""
        user = self._create_user()
        conv = Conversation(user_id=user.id, title="Serialized Consultation")
        db.session.add(conv)
        db.session.commit()

        msg = ChatMessage(
            conversation_id=conv.id,
            role="model",
            content="Evidence-based response",
            evidence={"sources": [{"title": "WHO Factsheet"}]},
            status="completed",
        )
        db.session.add(msg)
        db.session.commit()

        msg_dict = msg.to_dict()
        self.assertEqual(msg_dict["id"], msg.id)
        self.assertEqual(msg_dict["conversation_id"], conv.id)
        self.assertEqual(msg_dict["role"], "model")
        self.assertEqual(msg_dict["content"], "Evidence-based response")
        self.assertEqual(msg_dict["status"], "completed")
        self.assertIn("sources", msg_dict["evidence"])

        conv_dict = conv.to_dict(include_messages=True)
        self.assertEqual(conv_dict["id"], conv.id)
        self.assertEqual(conv_dict["user_id"], user.id)
        self.assertEqual(conv_dict["title"], "Serialized Consultation")
        self.assertEqual(conv_dict["message_count"], 1)
        self.assertEqual(len(conv_dict["messages"]), 1)
        self.assertEqual(conv_dict["messages"][0]["id"], msg.id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
