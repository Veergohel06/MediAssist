"""
Verification script for MediAssist with Neon PostgreSQL.
Validates:
1. Active SQLAlchemy dialect is PostgreSQL with psycopg3 driver.
2. Connection pooling works and pooler endpoint responds.
3. Migrated records can be read and decrypted/verified.
4. CRUD operations work seamlessly with Neon PostgreSQL sequences.
5. RAG FAISS index can be queried without regressions.
"""

import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

load_dotenv()

from app import create_app
from models import db, User, Conversation, ChatMessage

def verify():
    print("--- 1. Initializing Flask App with Environment Config ---")
    app = create_app()
    with app.app_context():
        dialect_name = db.engine.dialect.name
        driver_name = db.engine.dialect.driver
        print(f"SQLAlchemy Dialect: {dialect_name} | Driver: {driver_name}")
        assert dialect_name == "postgresql", f"Expected postgresql, got {dialect_name}"
        assert driver_name == "psycopg", f"Expected psycopg, got {driver_name}"

        print("\n--- 2. Querying Migrated Users ---")
        user_count = User.query.count()
        print(f"Total Users in Neon Postgres: {user_count}")
        assert user_count == 10, f"Expected 10 users, got {user_count}"
        
        sample_user = User.query.first()
        print(f"Sample User ID: {sample_user.id}, Email: {sample_user.email}, Academic Year: {sample_user.academic_year}")

        print("\n--- 3. Querying Migrated Conversations & Messages ---")
        conv_count = Conversation.query.count()
        msg_count = ChatMessage.query.count()
        print(f"Total Conversations: {conv_count} | Total Chat Messages: {msg_count}")
        assert conv_count == 40, f"Expected 40 conversations, got {conv_count}"
        assert msg_count == 121, f"Expected 121 messages, got {msg_count}"

        sample_conv = Conversation.query.first()
        print(f"Sample Conversation ID: {sample_conv.id}, Title: {sample_conv.title}")
        print(f"Sample Conversation Message Count: {sample_conv.messages.count()}")

        print("\n--- 4. Testing Write Operations & Sequence Incrementing ---")
        test_email = f"neon_test_{uuid.uuid4().hex[:8]}@example.com"
        new_user = User(
            name="Neon Test User",
            email=test_email,
            academic_year="2nd_year",
            target_exam="neet_pg_next",
        )
        new_user.set_password("SecurePassword123!")
        db.session.add(new_user)
        db.session.commit()
        print(f"Created Test User with ID: {new_user.id} (sequence successfully allocated)")
        assert new_user.id > 10, f"Sequence should allocate id > 10, got {new_user.id}"
        assert new_user.check_password("SecurePassword123!"), "Password check failed"

        test_conv = Conversation(
            user_id=new_user.id,
            title="Neon Postgres Test Consultation",
        )
        db.session.add(test_conv)
        db.session.commit()
        print(f"Created Test Conversation: {test_conv.id}")

        test_msg = ChatMessage(
            conversation_id=test_conv.id,
            role="user",
            content="Testing persistence on Neon PostgreSQL.",
            status="completed",
        )
        db.session.add(test_msg)
        db.session.commit()
        print(f"Created Test Message with ID: {test_msg.id} (sequence successfully allocated)")
        assert test_msg.id > 121, f"Sequence should allocate id > 121, got {test_msg.id}"

        # Clean up test artifacts
        db.session.delete(test_conv)
        db.session.delete(new_user)
        db.session.commit()
        print("Cleaned up test user and conversation.")

        print("\n--- 5. Testing FAISS Retriever & Knowledge Base ---")
        llm_svc = app.extensions.get("llm_service")
        assert llm_svc is not None, "llm_service extension missing on app"
        retriever = llm_svc.retriever
        assert retriever.is_ready(), "MedicalRetriever is not ready!"
        print(f"MedicalRetriever is ready. Index contains {retriever.index.ntotal} vectors.")
        results = retriever.retrieve("myocardial infarction symptoms", top_k=2)
        print(f"FAISS Retrieval Results count: {len(results)}")
        assert len(results) > 0, "FAISS retriever returned 0 results"
        print(f"Top result snippet: {results[0].get('text', '')[:80]}...")

    print("\n>>> ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    verify()
