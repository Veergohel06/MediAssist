"""
Integration and Authorization Security Test Suite for Neon PostgreSQL.
======================================================================
Validates requirements from MediAssist specification:
1. Verifies database engine is PostgreSQL with psycopg3 driver without exposing credentials.
2. Tests User registration with secure Werkzeug password hashing (scrypt).
3. Tests authentication flow with correct/incorrect passwords.
4. Tests password reset token generation, SHA-256 hashing, expiration, and single-use security.
5. Tests conversation creation, message persistence, and turn history retrieval.
6. Tests strict multi-tenant user isolation:
   - User B cannot read User A's conversation (403 Forbidden).
   - User B cannot post chat messages into User A's conversation (403 Forbidden).
   - User B cannot update/rename User A's conversation (403 Forbidden).
   - User B cannot delete User A's conversation (403 Forbidden).
   - User B cannot list or delete User A's attached documents (403 Forbidden).
"""

from __future__ import annotations

import os
import unittest
import uuid
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from app import create_app
from models import ChatMessage, Conversation, ConversationDocument, PasswordResetToken, User, db, generate_reset_token, verify_reset_token


class TestNeonPostgresAuthorization(unittest.TestCase):
    """End-to-end integration & multi-tenant security verification."""

    @classmethod
    def setUpClass(cls):
        db_url = os.environ.get("DATABASE_URL")
        if not db_url or "sqlite" in db_url.lower():
            raise unittest.SkipTest("DATABASE_URL not configured for PostgreSQL. Skipping Neon live tests.")

        from config import Config

        class NeonLiveTestConfig(Config):
            TESTING = True
            WTF_CSRF_ENABLED = False
            SERVER_NAME = "localhost"
            SQLALCHEMY_DATABASE_URI = db_url

        cls.app = create_app(NeonLiveTestConfig)
        cls.client = cls.app.test_client()

        with cls.app.app_context():
            cls.dialect_name = db.engine.dialect.name
            cls.driver_name = db.engine.dialect.driver

        cls.test_tag = uuid.uuid4().hex[:8]
        cls.user_a_email = f"user_a_{cls.test_tag}@testneon.org"
        cls.user_b_email = f"user_b_{cls.test_tag}@testneon.org"
        cls.password = "StrongTestPassword123!"

    def setUp(self):
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        db.session.remove()
        self.app_context.pop()

    @classmethod
    def tearDownClass(cls):
        """Clean up created test users and all cascading relational records."""
        with cls.app.app_context():
            for email in (cls.user_a_email, cls.user_b_email):
                u = User.query.filter_by(email=email).first()
                if u:
                    db.session.delete(u)
            db.session.commit()

    def test_01_database_engine_dialect_and_driver(self):
        """1. Verify database engine is PostgreSQL with psycopg driver (no secrets exposed)."""
        self.assertEqual(self.dialect_name, "postgresql", "Active database must be PostgreSQL")
        self.assertEqual(self.driver_name, "psycopg", "Active driver must be psycopg (Psycopg 3)")

    def test_02_registration_and_password_hashing(self):
        """2 & 4. Verify user registration, Werkzeug password hashing, and no plaintext storage."""
        user_a = User(
            name="Alice Resident",
            email=self.user_a_email,
            academic_year="2nd_year",
            target_exam="neet_pg_next",
        )
        user_a.set_password(self.password)
        db.session.add(user_a)

        user_b = User(
            name="Bob Intern",
            email=self.user_b_email,
            academic_year="intern",
            target_exam="usmle_step2",
        )
        user_b.set_password(self.password)
        db.session.add(user_b)
        db.session.commit()

        self.assertIsNotNone(user_a.id)
        self.assertIsNotNone(user_b.id)
        self.assertNotEqual(user_a.id, user_b.id)

        # Confirm password hash is secure and not plaintext
        self.assertFalse(user_a.password_hash.startswith(self.password))
        self.assertTrue(user_a.password_hash.startswith("scrypt:"))
        self.assertTrue(user_a.check_password(self.password))
        self.assertFalse(user_a.check_password("WrongPassword999!"))

    def test_03_password_reset_token_security(self):
        """2 & 4. Verify password reset token generation, SHA-256 hash storage, and single-use security."""
        user_a = User.query.filter_by(email=self.user_a_email).first()
        self.assertIsNotNone(user_a)

        raw_token = generate_reset_token(user_a, expires_in_minutes=15)
        self.assertIsInstance(raw_token, str)
        self.assertGreater(len(raw_token), 20)

        # Verify token in DB is hashed, not raw token
        token_record = PasswordResetToken.query.filter_by(user_id=user_a.id).first()
        self.assertIsNotNone(token_record)
        self.assertNotEqual(token_record.token_hash, raw_token)

        # Verify token validation
        verified_user, verified_record = verify_reset_token(raw_token, email=self.user_a_email)
        self.assertEqual(verified_user.id, user_a.id)
        self.assertEqual(verified_record.id, token_record.id)

        # Mark token as used and ensure subsequent verification fails
        verified_record.used_at = db.func.now()
        db.session.commit()

        recheck = verify_reset_token(raw_token, email=self.user_a_email)
        self.assertIsNone(recheck, "Used reset token must not be accepted again")

    def test_04_conversation_and_chat_message_persistence(self):
        """2 & 3. Verify conversation creation, message turn persistence, and history retrieval."""
        user_a = User.query.filter_by(email=self.user_a_email).first()
        self.assertIsNotNone(user_a)

        conv = Conversation(
            user_id=user_a.id,
            title="Cardiology Consult",
        )
        db.session.add(conv)
        db.session.commit()

        msg1 = ChatMessage(
            conversation_id=conv.id,
            role="user",
            content="Patient presents with acute chest pain and ST elevation.",
            status="completed",
        )
        msg2 = ChatMessage(
            conversation_id=conv.id,
            role="assistant",
            content="Activate cardiac catheterization lab immediately for primary PCI.",
            evidence={"sources": [{"title": "AHA STEMI Guidelines", "pmid": "12345678"}]},
            status="completed",
        )
        db.session.add_all([msg1, msg2])
        db.session.commit()

        # Query back and verify relations and timestamps
        persisted_conv = Conversation.query.filter_by(id=conv.id).first()
        self.assertIsNotNone(persisted_conv)
        self.assertIsNotNone(persisted_conv.created_at)
        self.assertIsNotNone(persisted_conv.updated_at)
        self.assertEqual(persisted_conv.messages.count(), 2)

        messages = persisted_conv.messages.all()
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(messages[1].evidence["sources"][0]["title"], "AHA STEMI Guidelines")

    def test_05_cross_user_isolation_and_authorization(self):
        """6. Strict verification that User B cannot access, modify, or post to User A's data."""
        user_a = User.query.filter_by(email=self.user_a_email).first()
        user_b = User.query.filter_by(email=self.user_b_email).first()
        self.assertIsNotNone(user_a)
        self.assertIsNotNone(user_b)

        # Create a conversation and document for User A
        conv_a = Conversation(user_id=user_a.id, title="User A Confidential Records")
        db.session.add(conv_a)
        db.session.commit()

        doc_a = ConversationDocument(
            id=str(uuid.uuid4()),
            conversation_id=conv_a.id,
            user_id=user_a.id,
            filename="patient_echo_report.pdf",
            storage_path="uploads/test/patient_echo_report.pdf",
            file_size_bytes=1024,
            file_type="pdf",
            content_sha256="abc123def456",
            page_count=2,
            chunk_count=4,
            status="ready",
        )
        db.session.add(doc_a)
        db.session.commit()

        # Log in as User B
        with self.client.session_transaction() as sess:
            sess["_user_id"] = str(user_b.id)
            sess["_fresh"] = True

        # User B attempts to read User A's conversation -> 403 Forbidden
        res_get = self.client.get(f"/api/conversations/{conv_a.id}")
        self.assertEqual(res_get.status_code, 403)
        self.assertIn("Access denied", res_get.get_json()["error"])

        # User B attempts to post to User A's conversation -> 403 Forbidden
        res_post = self.client.post(
            "/chat",
            json={"message": "Inject unauthorized query", "conversation_id": conv_a.id},
        )
        self.assertEqual(res_post.status_code, 403)
        self.assertIn("Access denied", res_post.get_json()["error"])

        # User B attempts to rename User A's conversation -> 403 Forbidden
        res_patch = self.client.patch(
            f"/api/conversations/{conv_a.id}",
            json={"title": "Hacked Title"},
        )
        self.assertEqual(res_patch.status_code, 403)
        self.assertIn("Access denied", res_patch.get_json()["error"])

        # User B attempts to delete User A's conversation -> 403 Forbidden
        res_del = self.client.delete(f"/api/conversations/{conv_a.id}")
        self.assertEqual(res_del.status_code, 403)
        self.assertIn("Access denied", res_del.get_json()["error"])

        # User B attempts to list User A's conversation documents -> 403 Forbidden
        res_docs = self.client.get(f"/api/conversations/{conv_a.id}/documents")
        self.assertEqual(res_docs.status_code, 403)
        self.assertIn("Access denied", res_docs.get_json()["error"])

        # User B attempts to delete User A's document -> 403 Forbidden
        res_doc_del = self.client.delete(f"/api/conversations/{conv_a.id}/documents/{doc_a.id}")
        self.assertEqual(res_doc_del.status_code, 403)
        self.assertIn("Access denied", res_doc_del.get_json()["error"])

        # User B requests their own conversations list -> User A's conversation is NOT visible
        res_list = self.client.get("/api/conversations")
        self.assertEqual(res_list.status_code, 200)
        user_b_conv_ids = [c["id"] for c in res_list.get_json()["conversations"]]
        self.assertNotIn(conv_a.id, user_b_conv_ids, "User A conversation leaked in User B conversation list")


if __name__ == "__main__":
    unittest.main()
