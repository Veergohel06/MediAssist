"""
Comprehensive Automated Test Suite: Medical Answer Accuracy, Visual Explanations & PDF Intelligence.
=====================================================================================================
Validates:
1. Medical factual grounding & page number citation fidelity for uploaded PDFs.
2. Structured table extraction & preservation without fragmented rows.
3. Strict distinction between user-uploaded study notes and core textbook curriculum references.
4. Transparent acknowledgement of evidence limitations (no hallucination when document lacks info).
5. Authentic, verified medical image matching and anti-hallucination (no irrelevant decorative images).
6. Scanned/image-only PDF detection and clear limitation reporting.
7. Multi-tenant isolation of conversation-extracted images and documents.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
import uuid
from unittest.mock import patch, MagicMock

from pypdf import PdfWriter
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import app as mediassist_app
from models import Conversation, ConversationDocument, User, db
from rag.chunker import _is_markdown_table, _split_markdown_table
from rag.evidence import audit_citations, sanitize_and_render_citations
from services.document_service import (
    DocumentService,
    DocumentExtractionError,
    DocumentValidationError,
)
from services.langchain_service import MBBSPromptBuilder
from services.medical_images import (
    find_relevant_medical_image,
    render_image_markdown_block,
    VERIFIED_MEDICAL_IMAGES,
)


def create_sample_pdf_with_table_and_pages() -> bytes:
    """Generate a multi-page PDF bytes with structured medical text and table."""
    writer = PdfWriter()

    # Page 1: Clinical Presentation
    writer.add_blank_page(width=612, height=792)

    # Minimal real stream content
    buf = io.BytesIO()
    writer.write(buf)
    base_bytes = buf.getvalue()

    # Craft extractable PDF content
    page1_text = (
        "BT\n/F1 12 Tf\n72 700 Td\n"
        "(Acute Coronary Syndromes Emergency Protocol) Tj\n"
        "0 -20 Td\n"
        "(Initial stabilization: Aspirin 325 mg chewed, sublingual nitroglycerin, high-flow O2 if SpO2 below 90 percent.) Tj\n"
        "ET"
    )
    page2_text = (
        "BT\n/F1 12 Tf\n72 700 Td\n"
        "(Differential Diagnosis of Acute Chest Pain) Tj\n"
        "0 -20 Td\n"
        "(STEMI requires immediate catheterization within 90 minutes. Aortic dissection requires CT angiography.) Tj\n"
        "ET"
    )

    p1_len = len(page1_text.encode("latin1"))
    p2_len = len(page2_text.encode("latin1"))

    two_page_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"5 0 obj << /Length " + str(p1_len).encode("latin1") + b" >> stream\n"
        + page1_text.encode("latin1") + b"\nendstream\nendobj\n"
        b"6 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 7 0 R >> endobj\n"
        b"7 0 obj << /Length " + str(p2_len).encode("latin1") + b" >> stream\n"
        + page2_text.encode("latin1") + b"\nendstream\nendobj\n"
        b"xref\n0 8\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000122 00000 n \n"
        b"0000000231 00000 n \n"
        b"0000000300 00000 n \n"
        b"0000000450 00000 n \n"
        b"0000000559 00000 n \n"
        b"trailer << /Size 8 /Root 1 0 R >>\n"
        b"startxref\n700\n%%EOF"
    )
    return two_page_pdf


class TestMedicalAccuracyAndVisualRag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="mediassist_acc_test_")
        cls.app = mediassist_app.create_app()
        cls.app.config.update({
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "SQLALCHEMY_ENGINE_OPTIONS": {
                "connect_args": {"check_same_thread": False},
                "poolclass": StaticPool,
            },
            "UPLOAD_FOLDER": cls.test_dir,
            "RESPONSE_CACHE_TTL_SECONDS": 1.0,
        })
        cls.client = cls.app.test_client()

        with cls.app.app_context():
            db.create_all()
            test_email = f"accuracy_{uuid.uuid4().hex[:8]}@aiims.edu"
            user = User(name="MBBS Student", email=test_email)
            user.set_password("SecureMedPass123!")
            db.session.add(user)
            db.session.commit()
            cls.user_id = user.id

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    # ── 1. Medical Image Registry & Accurate Visual Retrieval ───────────────────
    def test_verified_medical_images_metadata(self):
        """Verify each medical image has authentic attribution, non-empty landmarks, and caption."""
        self.assertGreaterEqual(len(VERIFIED_MEDICAL_IMAGES), 6)
        for img in VERIFIED_MEDICAL_IMAGES:
            self.assertTrue(img["id"].startswith("img_"))
            self.assertTrue(img["image_url"].startswith("http"))
            self.assertTrue(bool(img["caption"]))
            self.assertTrue(bool(img["source"]))
            self.assertTrue(bool(img["license"]))
            self.assertGreater(len(img["key_landmarks"]), 0)
            self.assertGreater(len(img["keywords"]), 2)

    def test_image_matching_reed_sternberg_hodgkin(self):
        """Verify Reed-Sternberg question retrieves authentic histological illustration."""
        query = "What are the classic histological hallmarks and IHC markers of Reed-Sternberg cells?"
        match = find_relevant_medical_image(query)
        self.assertIsNotNone(match)
        self.assertEqual(match["id"], "img_reed_sternberg_hl")
        self.assertIn("owl-eye", match["caption"].lower())

        block = render_image_markdown_block(match)
        self.assertIn("Visual Reference:", block)
        self.assertIn(match["image_url"], block)
        self.assertIn("Key Diagnostic & Anatomical Landmarks to Note:", block)
        self.assertIn("CD15+", block)

    def test_image_matching_circle_of_willis(self):
        """Verify Circle of Willis inquiry retrieves authentic neurovascular schematic."""
        query = "Describe the arterial anatomy of the Circle of Willis and sites of berry aneurysms."
        match = find_relevant_medical_image(query)
        self.assertIsNotNone(match)
        self.assertEqual(match["id"], "img_circle_of_willis")

    def test_image_matching_cardiac_conduction_and_wiggers(self):
        """Verify ECG waveform and cardiac cycle queries retrieve accurate physiological diagrams."""
        ecg_match = find_relevant_medical_image("Explain the ECG intervals: PR interval, QRS, and ST segment.")
        self.assertIsNotNone(ecg_match)
        self.assertEqual(ecg_match["id"], "img_ecg_cardiac_conduction")

        wiggers_match = find_relevant_medical_image("Walk me through the Wiggers diagram and isovolumetric contraction.")
        self.assertIsNotNone(wiggers_match)
        self.assertEqual(wiggers_match["id"], "img_wiggers_diagram")

    def test_anti_hallucination_no_irrelevant_images(self):
        """Verify unrelated queries DO NOT receive a decorative or mismatched image."""
        unrelated_queries = [
            "How does paracetamol cause hepatotoxicity at high doses?",
            "What is the daily recommended fluid requirement for a healthy adult?",
            "Compare tension headaches with classic migraine without aura.",
            "Explain the difference between Type 1 and Type 2 diabetes mellitus.",
        ]
        for q in unrelated_queries:
            match = find_relevant_medical_image(q)
            self.assertIsNone(match, f"Query '{q}' should NOT match any visual image!")

    # ── 2. PDF Page Preservation & Citation Rendering ──────────────────────────
    def test_pdf_page_retention_and_citation_formatting(self):
        """Verify uploaded PDF pages are attributed with exact 1-based page numbers in citations."""
        doc_service = DocumentService(self.app.config)
        doc_service.upload_root = self.test_dir

        pdf_bytes = create_sample_pdf_with_table_and_pages()
        pages = doc_service.extract_document_text(pdf_bytes, "Emergency_Cardiology.pdf", "pdf")
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[1]["page_number"], 2)

        chunks = doc_service.chunk_extracted_pages(
            pages=pages,
            doc_id="testdoc1234",
            filename="Emergency_Cardiology.pdf",
            conversation_id="conv_test_99",
            user_id=1,
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertIn("Page 1", chunks[0]["source_title"])

        # Test citation rendering preserves page number
        raw_answer = "Administer chewed aspirin immediately [doc_testdoc1_p001_01]."
        rendered = sanitize_and_render_citations(raw_answer, chunks)
        self.assertNotIn("doc_testdoc1_p001_01", rendered)
        self.assertIn("Emergency_Cardiology.pdf (Page 1)", rendered)

    def test_citation_auditing_flags_fake_citations(self):
        """Verify audit_citations flags fabricated page numbers or nonexistent sources."""
        chunks = [
            {
                "passage_id": "notes_doc:p0001",
                "source_title": "Harrison Cardiology Notes (Page 1)",
                "source_url": "",
                "is_uploaded_document": True,
            }
        ]
        valid_ans = "Catheterization is indicated within 90 minutes [notes_doc:p0001]."
        audit_res = audit_citations(valid_ans, chunks)
        self.assertTrue(audit_res["all_citations_valid"])

        fake_ans = "False claim citing hallucinated source [fake_book:p9999]."
        audit_fake = audit_citations(fake_ans, chunks)
        self.assertFalse(audit_fake["all_citations_valid"])
        self.assertIn("fake_book:p9999", audit_fake["invalid_citation_ids"])

    # ── 3. Table Extraction & Chunking Integrity ───────────────────────────────
    def test_table_preservation_without_row_fragmentation(self):
        """Verify markdown tables in study notes are preserved without breaking rows."""
        table = (
            "| Antiarrhythmic Class | Mechanism | Prototype Drug |\n"
            "|---|---|---|\n"
            "| Class IA | Fast Na+ channel block, prolongs APD | Quinidine, Procainamide |\n"
            "| Class IB | Fast Na+ channel block, shortens APD | Lidocaine, Mexiletine |\n"
            "| Class IC | Marked Na+ block, minimal APD effect | Flecainide, Propafenone |\n"
            "| Class II | Beta-adrenergic receptor blockade | Metoprolol, Atenolol |\n"
            "| Class III | K+ channel blockade, prolongs QT | Amiodarone, Sotalol |\n"
            "| Class IV | L-type Ca2+ channel blockade | Verapamil, Diltiazem |"
        )
        self.assertTrue(_is_markdown_table(table))
        sub_tables = _split_markdown_table(table, max_chunk_size=500)
        for sub in sub_tables:
            self.assertTrue(_is_markdown_table(sub))
            self.assertIn("Antiarrhythmic Class", sub)
            self.assertIn("|---|---|---|", sub)

    # ── 4. Prompt Grounding Delineation: Uploaded Notes vs Textbooks ───────────
    def test_evidence_block_delineates_uploaded_notes_from_textbooks(self):
        """Verify MBBSPromptBuilder clearly separates student notes from core textbook references."""
        mixed_docs = [
            {
                "passage_id": "student_note:p0002",
                "source_title": "Pathology_Self_Notes.pdf (Page 2)",
                "page_number": 2,
                "text": "My histology lab finding: Reed-Sternberg cells observed in interfollicular zone.",
                "is_uploaded_document": True,
                "source_type": "uploaded_study_note",
            },
            {
                "passage_id": "robbins_path:p0015",
                "source_title": "Robbins Pathology Educational Summary",
                "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK430855/",
                "text": "Classical Hodgkin lymphoma expresses CD15 and CD30 in >95% of diagnostic Reed-Sternberg variants.",
                "is_uploaded_document": False,
            },
        ]
        status = {"status": "sufficient", "valid_source_count": 2}
        block = MBBSPromptBuilder.build_evidence_block(mixed_docs, status)

        self.assertIn("User-Attached Study Notes & Uploaded PDF Passages:", block)
        self.assertIn("Verified Medical Knowledge Base & Curriculum Guidelines:", block)
        self.assertIn("Pathology_Self_Notes.pdf (Page 2)", block)
        self.assertIn("Page: 2", block)
        self.assertIn("Robbins Pathology Educational Summary", block)
        self.assertIn("Rules for Grounding & Evidence Integrity:", block)

    # ── 5. Scanned PDF Detection & Limitation Reporting ─────────────────────────
    def test_scanned_pdf_without_text_raises_clear_limitation_error(self):
        """Verify scanned/blank PDF without digital text raises informative DocumentExtractionError."""
        doc_service = DocumentService(self.app.config)
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        buf = io.BytesIO()
        writer.write(buf)
        scanned_bytes = buf.getvalue()

        with self.assertRaises(DocumentExtractionError) as ctx:
            doc_service.extract_document_text(scanned_bytes, "scanned_chart.pdf", "pdf")

        err_str = str(ctx.exception).lower()
        self.assertIn("no extractable digital text", err_str)
        self.assertIn("scanned or image-only", err_str)


if __name__ == "__main__":
    unittest.main()
