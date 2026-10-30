"""Tests for LangChain integration: prompt building, document representation,
retriever adapter, text splitting, structured output validation, and privacy-safe error handling.
"""

import json
import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from rag.chunker import (
    load_and_chunk_langchain_documents,
    split_text_with_langchain,
)
from rag.evidence import (
    MEDICAL_RESPONSE_PARSER,
    MedicalResponseContract,
    parse_structured_response,
)
from rag.retriever import (
    LangChainMedicalRetriever,
    MedicalRetriever,
    chunk_to_document,
    document_to_chunk,
)
from services.langchain_service import (
    MBBSPromptBuilder,
    LangChainGeminiChatAdapter,
    sanitize_sensitive_content,
)
from services.llm_service import LLMService


class MockConfig:
    def __init__(self):
        self.GEMINI_MODEL = "gemini-3.5-flash-lite"
        self.GEMINI_API_KEY = "test-key-mock"
        self.RESPONSE_CACHE_TTL_SECONDS = 3600.0
        self.RESPONSE_CACHE_MAX_ENTRIES = 100
        self.RAG_TOP_K = 3
        self.RAG_SCORE_THRESHOLD = 0.58
        self.RAG_RETRIEVAL_MODE = "hybrid"
        self.RAG_RERANK_ENABLED = False
        self.MAX_HISTORY_TURNS = 20
        self.FAISS_INDEX_PATH = "knowledge/faiss_index.bin"
        self.METADATA_PATH = "knowledge/chunks_metadata.json"

    def get(self, key, default=None):
        return getattr(self, key, default)


class TestLangChainStructuredOutputValidation(unittest.TestCase):
    """Verify LangChain PydanticOutputParser on medical response contracts."""

    def test_valid_structured_json_parse(self):
        valid_json = json.dumps({
            "answer": "### Pathophysiology of Tetralogy of Fallot\nFour classic features...",
            "cited_passage_ids": ["pediatric_cardiology:p0012"],
            "evidence_sufficiency": "sufficient",
        })
        parsed = parse_structured_response(valid_json)
        self.assertTrue(parsed["structured_output_valid"])
        self.assertEqual(parsed["evidence_sufficiency"], "sufficient")
        self.assertEqual(parsed["cited_passage_ids"], ["pediatric_cardiology:p0012"])
        self.assertIn("Tetralogy of Fallot", parsed["answer"])

    def test_markdown_fenced_json_parse(self):
        raw = "```json\n" + json.dumps({
            "answer": "Myasthenia Gravis is an autoimmune neuromuscular junction disorder.",
            "cited_passage_ids": [],
            "evidence_sufficiency": "insufficient",
        }) + "\n```"
        parsed = parse_structured_response(raw)
        self.assertTrue(parsed["structured_output_valid"])
        self.assertEqual(parsed["evidence_sufficiency"], "insufficient")
        self.assertEqual(parsed["cited_passage_ids"], [])

    def test_malformed_unparseable_text_graceful_fallback(self):
        raw = "This is not JSON at all, just raw medical text from the model."
        parsed = parse_structured_response(raw)
        self.assertFalse(parsed["structured_output_valid"])
        self.assertEqual(parsed["answer"], raw)
        self.assertEqual(parsed["cited_passage_ids"], [])
        self.assertEqual(parsed["evidence_sufficiency"], "insufficient")

    def test_invalid_evidence_sufficiency_value_fallback(self):
        bad_json = json.dumps({
            "answer": "Test answer",
            "cited_passage_ids": [],
            "evidence_sufficiency": "invalid_status_enum",
        })
        parsed = parse_structured_response(bad_json)
        self.assertFalse(parsed["structured_output_valid"])


class TestLangChainDocumentAndRetriever(unittest.TestCase):
    """Verify Document conversion and BaseRetriever adapter."""

    def test_chunk_to_document_and_back(self):
        original_chunk = {
            "chunk_id": "cardio_1",
            "passage_id": "cardio:p0001",
            "source_title": "Harrison Cardiology",
            "source_url": "https://example.org/cardio",
            "retrieval_date": "2026-10-23",
            "section": "Ischemia",
            "text": "Myocardial ischemia results from coronary perfusion-demand mismatch.",
        }
        doc = chunk_to_document(original_chunk)
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.page_content, original_chunk["text"])
        self.assertEqual(doc.metadata["passage_id"], "cardio:p0001")
        self.assertEqual(doc.metadata["source_title"], "Harrison Cardiology")

        # Convert back
        recovered_chunk = document_to_chunk(doc)
        self.assertEqual(recovered_chunk["text"], original_chunk["text"])
        self.assertEqual(recovered_chunk["passage_id"], original_chunk["passage_id"])

    def test_retriever_as_langchain_retriever(self):
        retriever = MedicalRetriever(
            client=None,
            index_path="knowledge/faiss_index.bin",
            metadata_path="knowledge/chunks_metadata.json",
            retrieval_mode="keyword",
        )
        lc_retriever = retriever.as_langchain_retriever(top_k=2)
        self.assertIsInstance(lc_retriever, LangChainMedicalRetriever)
        self.assertEqual(lc_retriever.top_k, 2)

        # Retrieve documents
        docs = lc_retriever.invoke("tension headaches causes")
        self.assertIsInstance(docs, list)
        for d in docs:
            self.assertIsInstance(d, Document)
            self.assertTrue(len(d.page_content) > 0)


class TestLangChainTextSplitters(unittest.TestCase):
    """Verify document processing with langchain-text-splitters."""

    def test_split_text_with_langchain(self):
        text = (
            "## Acute Pancreatitis\n"
            "Acute pancreatitis is an acute inflammatory condition of the pancreas.\n\n"
            "### Etiology\n"
            "Gallstones and alcohol abuse account for over 70% of clinical cases."
        )
        metadata = {"title": "Pancreatitis Guide", "url": "https://example.org/panc"}
        chunks = split_text_with_langchain(text, metadata, "pancreatitis_01", chunk_size=80, chunk_overlap=15)
        self.assertTrue(len(chunks) >= 2)
        for chunk in chunks:
            self.assertIn("pancreatitis_01", chunk["chunk_id"])
            self.assertIn("passage_id", chunk)
            self.assertTrue(len(chunk["text"]) > 0)


class TestMBBSPromptBuilder(unittest.TestCase):
    """Verify LangChain MBBSPromptBuilder modular message assembly."""

    def test_prompt_builder_assembly(self):
        system_inst = "You are an MBBS pedagogical tutor."
        safety_dir = "Non-emergency academic query."
        user_q = "Explain the biochemical basis of diabetic ketoacidosis."
        history = [
            {"role": "user", "content": "Hello doctor."},
            {"role": "model", "content": "Welcome. What would you like to review today?"},
        ]

        messages = MBBSPromptBuilder.create_prompt_messages(
            system_instruction=system_inst,
            safety_directive=safety_dir,
            user_message=user_q,
            academic_year="2nd_year",
            target_exam="neet_pg_next",
            retrieved_documents=[],
            evidence_status={"status": "insufficient", "valid_source_count": 0},
            history=history,
        )

        self.assertIsInstance(messages[0], SystemMessage)
        self.assertEqual(messages[0].content, system_inst)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertEqual(messages[1].content, "Hello doctor.")
        self.assertIsInstance(messages[2], AIMessage)
        self.assertIsInstance(messages[3], HumanMessage)

        # Verify profile and safety inside final user prompt
        final_prompt = messages[3].content
        self.assertIn("STUDENT ACADEMIC PROFILE CONTEXT", final_prompt)
        self.assertIn("2nd Year MBBS", final_prompt)
        self.assertIn("NEET-PG / NExT", final_prompt)
        self.assertIn("diabetic ketoacidosis", final_prompt)


class TestLangChainSafeLoggingAndModelFailure(unittest.TestCase):
    """Verify sanitized error logging and robust model failure handling."""

    def test_sanitize_sensitive_content(self):
        sensitive_text = "Patient John Doe john@med.org phone 9876543210 complains of severe chest pain and vomiting."
        sanitized = sanitize_sensitive_content(sensitive_text)
        self.assertNotIn("john@med.org", sanitized)
        self.assertNotIn("9876543210", sanitized)
        self.assertIn("[EMAIL_REDACTED]", sanitized)
        self.assertIn("[PHONE_REDACTED]", sanitized)

    def test_model_failure_fallback_returns_graceful_message(self):
        config = MockConfig()
        service = LLMService(config)
        service.retriever = None

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API service quota exceeded")
        service.client = mock_client

        reply, sources, triage = service.get_response(
            user_message="Explain pathophysiology of Wilson disease",
            academic_year="1st_year",
        )

        self.assertIn("unable to connect", reply)
        self.assertIsInstance(sources, list)
        self.assertEqual(triage["usage"]["model"], "generation_failure_fallback")
        self.assertIn("API service quota exceeded", triage["infrastructure_error"])


class TestMultiTenantUserIsolationWithLangChain(unittest.TestCase):
    """Verify student profile and history isolation across different user contexts."""

    def test_isolated_prompts_for_different_academic_profiles(self):
        config = MockConfig()
        service = LLMService(config)
        service.retriever = None

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"answer": "Pedagogical explanation.", "cited_passage_ids": [], "evidence_sufficiency": "sufficient"}'
        mock_resp.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_resp
        service.client = mock_client

        # User 1: 1st Year USMLE Step 1
        service.get_response(
            user_message="Explain cellular respiration",
            user_id=101,
            conversation_id="conv-101",
            academic_year="1st_year",
            target_exam="usmle_step1",
        )
        call_1_kwargs = mock_client.models.generate_content.call_args.kwargs
        prompt_1 = call_1_kwargs["contents"][-1].parts[0].text
        self.assertIn("1st Year MBBS", prompt_1)
        self.assertIn("USMLE Step 1", prompt_1)

        # User 2: Final Year University Profs
        service.get_response(
            user_message="Explain cellular respiration",
            user_id=202,
            conversation_id="conv-202",
            academic_year="final_year",
            target_exam="university_profs",
        )
        call_2_kwargs = mock_client.models.generate_content.call_args.kwargs
        prompt_2 = call_2_kwargs["contents"][-1].parts[0].text
        self.assertIn("Final Year MBBS", prompt_2)
        self.assertIn("University Professional", prompt_2)
        self.assertNotIn("1st Year MBBS", prompt_2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
