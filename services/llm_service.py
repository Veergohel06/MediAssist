"""LLM & Clinical AI Orchestration Service.

Integrates Google Gemini AI via google-genai SDK with multi-tier model fallbacks,
safety red-flag pre-flight evaluation, RAG evidence augmentation, and collision-proof response caching.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors

from rag.evidence import (
    assess_evidence,
    audit_citations,
    parse_structured_response,
    sanitize_and_render_citations,
)
from rag.reranker import RerankerError, SentenceTransformerReranker, TokenOverlapReranker
from rag.retriever import MedicalRetriever
from safety.evaluator import SymptomSafetyEvaluator
from services.cache_service import ResponseCache
from services.langchain_service import MBBSPromptBuilder, LangChainGeminiChatAdapter

logger = logging.getLogger("mediassist.llm")

SYSTEM_INSTRUCTION = """
You are MediAssist, an advanced AI Academic Learning Mentor designed specifically for MBBS students.

Your role is to help undergraduate medical students master medicine by providing rigorous, undergraduate-level medical education aligned with Competency-Based Medical Education (CBME), university professional examinations, NEET-PG/NExT, and USMLE principles.

## Core Pedagogical Principles

1. **Rigorous Medical Accuracy & Terminology**: Use standard, precise medical terminology (e.g., etiology, etiopathogenesis, receptor kinetics, histological hallmarks, differential diagnosis, investigation of choice). Never downplay concepts to layperson language unless explicitly asked for a patient-counseling explanation.

2. **Mechanism-First Explanations**: Detail the biochemical, anatomical, physiological, and pathophysiological mechanisms underlying clinical signs, symptoms, lab findings, and pharmacotherapy. Explain the "why" behind the clinical presentation.

3. **CBME & Clinical Correlation**: Integrate pre-clinical, para-clinical, and clinical aspects (horizontal and vertical integration):
   - Correlate anatomical landmarks and physiological functions with pathology.
   - Explain mechanism of action (MOA), pharmacokinetics/dynamics, adverse effects, and drug interactions.
   - Differentiate "initial investigation" from "gold-standard / confirmatory investigation of choice".
   - Differentiate "first-line drug of choice" from alternatives and acute emergency stabilization protocols.

4. **Multi-Mode Academic Versatility**: Adapt seamlessly to the student's inquiry mode:
   - **Detailed Topic Explanation**: Provide comprehensive yet organized breakdowns (Overview, Etiopathogenesis, Clinical Features, Workup, Management, Complications).
   - **Quick High-Yield Revision**: Provide concise bullet points, high-yield triads/pentads, pathognomonic signs, and mnemonics.
   - **Differential Diagnosis & Comparison**: Present structured contrast tables highlighting differentiating features, epidemiology, and distinguishing lab/imaging findings.
   - **Case-Based Reasoning / Vignettes**: Walk through clinical reasoning systematically (pertinent positives/negatives, provisional vs. differential diagnosis, diagnostic algorithm, management hierarchy).
   - **Viva Voce & Exam Practice**: Deliver direct, examiner-focused points, classic "must-know" questions, and high-yield traps.

5. **Academic Level & Target Exam Adaptation**:
   When the student's academic level and target exam are provided in the prompt context:
   - **1st Year MBBS**: Emphasize Anatomy, Physiology, Biochemistry, embryological origins, and histological features.
   - **2nd Year MBBS**: Emphasize General & Systemic Pathology, Pharmacology (receptor targets, toxicities), Microbiology (morphology, culture, virulence), and Forensic Medicine.
   - **3rd Year MBBS / Part 1**: Emphasize Ophthalmology, ENT, Community Medicine (epidemiology, national health programs, biostatistics).
   - **Final Year MBBS / Intern**: Emphasize Clinical Medicine, General Surgery, OB/GYN, Pediatrics, diagnostic algorithms, emergency stabilization, and surgical anatomy/indications.
   - **University Profs Focus**: Emphasize structured long-answer format, definitions, classifications, labeled diagrams (textually described), and classic textbook citations.
   - **NEET-PG / NExT Focus**: Emphasize clinical vignettes, high-yield single-best answers, image/radiology interpretation descriptions, and modern clinical guideline updates.
   - **USMLE Step 1 / Step 2 Focus**: Emphasize two-step mechanistic reasoning, pathophysiology cascades, molecular genetics, and "next best step in management".

6. **Source Grounding & Evidence Integrity**:
   When verified medical reference documents or uploaded study notes are provided in the prompt:
   - Ground your factual statements in that context.
   - For user-uploaded notes and PDFs, cite the specific document name and page number (e.g., "(StudyNotes.pdf, p. 3)").
   - Clearly distinguish between facts sourced from the student's uploaded document and general textbook principles.
   - If the student inquires about an uploaded document (such as tables, summaries, or questions) and the uploaded document does not contain that information, explain the limitation clearly instead of guessing or inventing content.
   - In the "cited_passage_ids" list of your JSON response, include the exact passage identifiers (e.g., "document_name:p0001") that substantiate your claims.
   - In the "answer" field, write clean, authoritative medical prose. NEVER leak raw passage codes, chunk IDs, or internal tokens (like [doc:p0001] or [doc_1]) in the visible answer string.
   - Refer to sources naturally by title or topic when helpful.
   - Never invent or fabricate citations, book chapters, page numbers, or statistics.
   - If the knowledge base does not contain relevant information, acknowledge the gap clearly and provide an authoritative, well-established standard medical textbook explanation (e.g. from Robbins, Harrison, Guyton, or KD Tripathi principles) while noting that it is general curricular knowledge.
   - If evidence is conflicting, stale, or uncertain, state the diagnostic uncertainty or clinical controversy explicitly.

## Required Response Contract

Return ONLY a valid JSON object with exactly these keys:
{"answer": "...", "cited_passage_ids": ["doc_id:p0001"], "evidence_sufficiency": "sufficient|insufficient|conflicting|stale"}

- The "answer" must be formatted in clean, rich Markdown with clear headings (`###`), bold terminology, bullet lists, or tables where appropriate.
- Do not include markdown code block wrappers around the JSON unless strictly necessary.
- If no citations are present, return an empty array `[]` for "cited_passage_ids".
""".strip()


class LLMService:
    """Manages Gemini AI integration, RAG retrieval, and isolated caching."""

    def _get_config(self, key: str, default: Any = None) -> Any:
        if hasattr(self.config, "get"):
            val = self.config.get(key)
            if val is not None:
                return val
        return getattr(self.config, key, default)

    def __init__(self, app_config: Any):
        self.config = app_config
        self.gemini_model = self._get_config("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.api_key = self._get_config("GEMINI_API_KEY", "")

        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Real AI generation will fail unless mocked or configured.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

        self.langchain_adapter = LangChainGeminiChatAdapter(client=self.client, default_model=self.gemini_model)

        self.safety_evaluator = SymptomSafetyEvaluator()
        self.cache = ResponseCache(
            ttl_seconds=self._get_config("RESPONSE_CACHE_TTL_SECONDS", 86400.0),
            max_entries=self._get_config("RESPONSE_CACHE_MAX_ENTRIES", 1000),
        )

        self.retriever = self._init_retriever()
        from services.document_service import DocumentService
        self.document_service = DocumentService(self.config, client=self.client)
        self.system_instruction = SYSTEM_INSTRUCTION
        from services.workflow_service import InteractiveWorkflowEngine
        self.workflow_engine = InteractiveWorkflowEngine(self)

    def _init_retriever(self) -> MedicalRetriever:
        """Initialize FAISS vector retriever with hybrid BM25 support."""
        rerank_enabled = self._get_config("RAG_RERANK_ENABLED", False)
        reranker = None
        if rerank_enabled:
            reranker_type = self._get_config("RAG_RERANKER_TYPE", "token_overlap").lower()
            try:
                if reranker_type == "cross_encoder":
                    reranker = SentenceTransformerReranker(
                        model_name=self._get_config("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                        timeout_seconds=self._get_config("RAG_RERANK_TIMEOUT_SECONDS", 10.0),
                    )
                else:
                    reranker = TokenOverlapReranker()
            except RerankerError as exc:
                logger.warning("Reranker disabled during startup: %s", type(exc).__name__)
                rerank_enabled = False

        index_path = self._get_config("FAISS_INDEX_PATH", "knowledge/faiss_index.bin")
        metadata_path = self._get_config("METADATA_PATH", "knowledge/chunks_metadata.json")

        retriever = MedicalRetriever(
            client=self.client,
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_model=self._get_config("RAG_EMBEDDING_MODEL", "gemini-embedding-2"),
            score_threshold=self._get_config("RAG_SCORE_THRESHOLD", 0.58),
            semantic_weight=self._get_config("RAG_SEMANTIC_WEIGHT", 0.7),
            keyword_weight=self._get_config("RAG_KEYWORD_WEIGHT", 0.3),
            candidate_k=self._get_config("RAG_CANDIDATE_K", 12),
            semantic_candidate_k=self._get_config("RAG_SEMANTIC_CANDIDATE_K", 12),
            keyword_candidate_k=self._get_config("RAG_KEYWORD_CANDIDATE_K", 12),
            rrf_k=self._get_config("RAG_RRF_K", 60),
            rerank_enabled=rerank_enabled,
            reranker=reranker,
            rerank_candidate_k=self._get_config("RAG_RERANK_CANDIDATE_K", 20),
            retrieval_mode=self._get_config("RAG_RETRIEVAL_MODE", "hybrid"),
        )
        if retriever.is_ready():
            logger.info("MediAssist RAG: Knowledge base loaded (%d chunks).", retriever.index.ntotal)
        else:
            logger.warning("MediAssist RAG: Knowledge base index not found or empty.")
        return retriever

    @staticmethod
    def build_history_contents(history: Optional[List[dict]] = None, max_turns: int = 20) -> List[Any]:
        """Format history list into Google GenAI Content objects."""
        contents = []
        if not history:
            return contents

        for entry in history[-max_turns:]:
            role = entry.get("role", "user")
            parts = entry.get("parts", [])
            text = ""
            if parts and isinstance(parts, list):
                first_part = parts[0]
                if isinstance(first_part, dict):
                    text = first_part.get("text", "")
                elif hasattr(first_part, "text"):
                    text = getattr(first_part, "text", "")
            elif isinstance(entry.get("content"), str):
                text = entry["content"]

            contents.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part(text=text)],
                )
            )
        return contents

    def get_response(
        self,
        user_message: str,
        history: Optional[List[dict]] = None,
        user_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        academic_year: Optional[str] = None,
        target_exam: Optional[str] = None,
        selected_option: Optional[str] = None,
        question_id: Optional[str] = None,
    ) -> Tuple[str, List[dict], dict]:
        """Execute clinical RAG and AI generation with multi-tenant caching, safety triage,
        and student academic profile adaptation.

        Returns (reply_text, sources, triage_payload).
        """
        overall_start = time.perf_counter()

        # Resolve optional student academic profile context
        resolved_academic_year = academic_year
        resolved_target_exam = target_exam
        if user_id and (resolved_academic_year is None or resolved_target_exam is None):
            try:
                from models import User, db
                user_record = db.session.get(User, user_id)
                if user_record:
                    resolved_academic_year = resolved_academic_year or user_record.academic_year
                    resolved_target_exam = resolved_target_exam or user_record.target_exam
            except Exception as user_fetch_err:
                logger.debug("Could not resolve profile for user_id=%s: %s", user_id, user_fetch_err)

        # 1. Safety & Emergency Red Flag Evaluation
        safety_eval = self.safety_evaluator.evaluate(user_message)
        triage = dict(safety_eval["triage"])
        safety_directive = safety_eval["prompt_directive"]
        is_emergency = safety_eval["is_emergency"]

        # Emergency dispatch takes precedence over retrieval and generation
        if is_emergency:
            reply_text = safety_eval["triage"]["full_guidance"]
            triage["performance"] = {
                "retrieval_ms": 0.0,
                "generation_ms": 0.0,
                "total_ms": round((time.perf_counter() - overall_start) * 1000, 2),
            }
            triage["usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "model": "rule_based_emergency_dispatch",
            }
            return reply_text, [], triage

        # 2. Collision-Proof Cache Check
        retrieval_mode = self._get_config("RAG_RETRIEVAL_MODE", "hybrid")
        cached = self.cache.get(
            user_message=user_message,
            user_id=user_id,
            conversation_id=conversation_id,
            history=history,
            model_name=self.gemini_model,
            retrieval_mode=retrieval_mode,
        )
        if cached:
            cached_reply, cached_sources, cached_triage = cached
            triage_copy = copy.deepcopy(cached_triage)
            triage_copy["performance"] = {
                "retrieval_ms": 0.05,
                "generation_ms": 0.05,
                "total_ms": round((time.perf_counter() - overall_start) * 1000, 2),
            }
            triage_copy["cache_hit"] = True
            logger.info("Serving clinical query from isolated cache: '%s'", user_message[:40])
            return cached_reply, cached_sources, triage_copy

        # 3. Retrieve Medical Evidence
        base_chunks = []
        uploaded_chunks = []
        sources = []
        top_k = self._get_config("RAG_TOP_K", 3)
        retrieval_start = time.perf_counter()

        # 3a. Retrieve from base verified medical knowledge base
        if self.retriever and self.retriever.is_ready():
            try:
                base_chunks = self.retriever.retrieve(user_message, top_k=top_k)
            except Exception as exc:
                logger.warning("RAG retrieval encountered an error: %s", exc)

        # 3b. Retrieve from conversation-scoped uploaded study notes and PDFs
        if conversation_id and user_id and getattr(self, "document_service", None):
            try:
                uploaded_chunks = self.document_service.retrieve_for_conversation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    query=user_message,
                    top_k=2,
                )
            except Exception as doc_exc:
                logger.warning("Conversation document retrieval encountered an error: %s", doc_exc)

        # Prioritize conversation documents by prepending them to the evidence pool
        retrieved_chunks = uploaded_chunks + base_chunks

        if self.retriever:
            sources = self.retriever.extract_sources(retrieved_chunks)

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        evidence_status = assess_evidence(retrieved_chunks)

        # 4. Interactive LangGraph Workflow (Viva, MCQ, Clinical Cases, Tutoring)
        if getattr(self, "workflow_engine", None):
            try:
                wf_reply, wf_sources, wf_triage = self.workflow_engine.execute(
                    user_message=user_message,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    academic_year=resolved_academic_year,
                    target_exam=resolved_target_exam,
                    history=history,
                    retrieved_chunks=retrieved_chunks,
                    selected_option=selected_option,
                    question_id=question_id,
                )
                if wf_reply and "unable to connect" not in wf_reply.lower() and not wf_triage.get("infrastructure_error"):
                    # Check for verified authentic visual illustration
                    from services.medical_images import find_relevant_medical_image, render_image_markdown_block
                    matched_img = find_relevant_medical_image(user_message, wf_reply)
                    if matched_img and matched_img.get("image_url") and matched_img["image_url"] not in wf_reply:
                        wf_reply = wf_reply.rstrip() + "\n" + render_image_markdown_block(matched_img)
                        wf_triage["visual_reference"] = matched_img["title"]

                    # Preserve safety evaluation keys in wf_triage
                    for k, v in triage.items():
                        if k not in wf_triage:
                            wf_triage[k] = v
                    wf_triage["performance"] = {
                        "retrieval_ms": round(retrieval_ms, 2),
                        "generation_ms": round((time.perf_counter() - overall_start) * 1000 - retrieval_ms, 2),
                        "total_ms": round((time.perf_counter() - overall_start) * 1000, 2),
                    }
                    if "usage" not in wf_triage:
                        wf_triage["usage"] = {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "estimated_cost_usd": 0.0,
                            "model": wf_triage.get("active_model", "gemini-3.5-flash-lite"),
                        }
                    return wf_reply, wf_sources, wf_triage
            except Exception as wf_exc:
                logger.warning("Interactive workflow execution fell through to standard generator: %s", wf_exc)

        # 5. Build Context-Augmented Prompt with LangChain Prompt Builder
        max_turns = self._get_config("MAX_HISTORY_TURNS", 20)
        messages = MBBSPromptBuilder.create_prompt_messages(
            system_instruction=SYSTEM_INSTRUCTION,
            safety_directive=safety_directive,
            user_message=user_message,
            academic_year=resolved_academic_year,
            target_exam=resolved_target_exam,
            retrieved_documents=retrieved_chunks,
            evidence_status=evidence_status,
            history=history,
            max_history_turns=max_turns,
        )

        # 5. Generate Content with Fallback Models via LangChain Adapter
        gen_start = time.perf_counter()
        candidate_models = [self.gemini_model]
        for fallback in ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        response, active_model, last_gen_exc = self.langchain_adapter.execute_chat_turn(
            messages=messages,
            system_instruction=SYSTEM_INSTRUCTION,
            candidate_models=candidate_models,
            client=self.client,
            max_output_tokens=self._get_config("GEMINI_MAX_OUTPUT_TOKENS", 850),
            temperature=self._get_config("GEMINI_TEMPERATURE", 0.2),
        )

        resp_text = getattr(response, "text", None) if response is not None else None
        if response is None or not isinstance(resp_text, str) or not resp_text.strip():
            gen_ms = (time.perf_counter() - gen_start) * 1000
            if not self.api_key:
                reply_text = (
                    "Google Gemini API key is not configured. Please add your GEMINI_API_KEY to the .env file. "
                    "For urgent medical concerns, please consult a qualified healthcare provider or emergency services."
                )
            else:
                reply_text = (
                    "I am temporarily unable to connect to the Google Gemini AI service due to a technical or capacity issue. "
                    "Please try again shortly. For urgent medical concerns, please consult a qualified healthcare provider or emergency services."
                )
            triage["evidence"] = evidence_status
            triage["citation_validation"] = audit_citations(reply_text, retrieved_chunks)
            triage["structured_output"] = False
            triage["retrieved_evidence_count"] = len(retrieved_chunks)
            triage["infrastructure_error"] = str(last_gen_exc) if last_gen_exc else "No generation response"
            triage["performance"] = {
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(gen_ms, 2),
                "total_ms": round((time.perf_counter() - overall_start) * 1000, 2),
            }
            triage["usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "model": "generation_failure_fallback",
            }
            return reply_text, sources, triage

        gen_ms = (time.perf_counter() - gen_start) * 1000

        # 7. Parse Structured Evidence Response
        parsed = parse_structured_response(response.text)
        raw_reply = parsed["answer"]
        citation_validation = audit_citations(raw_reply, retrieved_chunks, parsed["cited_passage_ids"])
        clean_reply = sanitize_and_render_citations(raw_reply, retrieved_chunks)

        # 7b. Check for relevant verified medical educational illustration
        from services.medical_images import find_relevant_medical_image, render_image_markdown_block
        matched_img = find_relevant_medical_image(user_message, clean_reply)
        if matched_img and matched_img.get("image_url") and matched_img["image_url"] not in clean_reply:
            clean_reply = clean_reply.rstrip() + "\n" + render_image_markdown_block(matched_img)
            triage["visual_reference"] = matched_img["title"]

        # 8. Token & Cost Accounting
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            u = response.usage_metadata
            prompt_tokens = getattr(u, "prompt_token_count", 0) or 0
            completion_tokens = getattr(u, "candidates_token_count", 0) or 0
            total_tokens = getattr(u, "total_token_count", 0) or (prompt_tokens + completion_tokens)

        rate_multiplier = 0.5 if "lite" in active_model.lower() else 1.0
        estimated_cost = (prompt_tokens * 0.075 * rate_multiplier + completion_tokens * 0.30 * rate_multiplier) / 1_000_000.0

        triage["citation_validation"] = citation_validation
        triage["evidence"] = evidence_status
        triage["structured_output"] = parsed["structured_output_valid"]
        triage["model_evidence_sufficiency"] = parsed["evidence_sufficiency"]
        triage["retrieved_evidence_count"] = len(retrieved_chunks)
        triage["performance"] = {
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(gen_ms, 2),
            "total_ms": round((time.perf_counter() - overall_start) * 1000, 2),
        }
        triage["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 7),
            "model": active_model,
        }

        # 9. Store in Isolated Multi-Tenant Response Cache
        if clean_reply and not triage.get("infrastructure_error"):
            self.cache.set(
                user_message=user_message,
                reply=clean_reply,
                sources=sources,
                triage=triage,
                user_id=user_id,
                conversation_id=conversation_id,
                history=history,
                model_name=active_model,
                retrieval_mode=retrieval_mode,
            )

        return clean_reply, sources, triage

    def stream_response(
        self,
        user_message: str,
        history: Optional[List[dict]] = None,
        user_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        academic_year: Optional[str] = None,
        target_exam: Optional[str] = None,
        selected_option: Optional[str] = None,
        question_id: Optional[str] = None,
    ):
        """Stream response generator yielding structured SSE event dictionaries:
        - {"event": "start", "data": {...}}
        - {"event": "chunk", "data": {"text": "..."}}
        - {"event": "evidence", "data": {"sources": [...], "evidence": {...}}}
        - {"event": "complete", "data": {"response": "...", "sources": [...], "triage": {...}, ...}}
        - {"event": "error", "data": {"error": "..."}}
        """
        overall_start = time.perf_counter()
        triage: Dict[str, Any] = {
            "is_emergency": False,
            "severity": "EDUCATIONAL",
            "condition": None,
            "escalation_recommended": False,
            "action_advice": None,
            "red_flags": [],
            "cache_hit": False,
        }

        # 1. Urgent-Symptom Safety Evaluation
        safety_eval = self.safety_evaluator.evaluate(user_message)
        if safety_eval.get("is_emergency", False):
            reply_text = safety_eval["triage"]["full_guidance"]
            emergency_triage = dict(safety_eval["triage"])
            emergency_triage["performance"] = {
                "retrieval_ms": 0.0,
                "generation_ms": 0.0,
                "total_ms": round((time.perf_counter() - overall_start) * 1000, 2),
            }
            emergency_triage["usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "model": "rule_based_emergency_dispatch",
            }
            yield {
                "event": "start",
                "data": {
                    "is_emergency": True,
                    "condition": emergency_triage.get("condition"),
                    "cache_hit": False,
                    "sources_count": 0,
                },
            }
            yield {"event": "chunk", "data": {"text": reply_text}}
            yield {
                "event": "complete",
                "data": {
                    "response": reply_text,
                    "sources": [],
                    "triage": emergency_triage,
                    "evidence": {},
                    "citations": {},
                },
            }
            return

        safety_directive = safety_eval.get("system_prompt_directive", "")
        triage.update({
            "is_emergency": False,
            "severity": safety_eval["triage"].get("severity", "EDUCATIONAL"),
            "condition": safety_eval["triage"].get("condition"),
            "action_advice": safety_eval["triage"].get("action_advice"),
            "red_flags": safety_eval["triage"].get("red_flags", []),
        })

        # Academic profile resolution
        resolved_academic_year = academic_year or self._get_config("DEFAULT_ACADEMIC_YEAR", "final_year")
        resolved_target_exam = target_exam or self._get_config("DEFAULT_TARGET_EXAM", "university_profs")
        triage["academic_year"] = resolved_academic_year
        triage["target_exam"] = resolved_target_exam

        # 2. Collision-Proof Cache Check
        retrieval_mode = self._get_config("RAG_RETRIEVAL_MODE", "hybrid")
        cached = self.cache.get(
            user_message=user_message,
            user_id=user_id,
            conversation_id=conversation_id,
            history=history,
            model_name=self.gemini_model,
            retrieval_mode=retrieval_mode,
        )
        if cached:
            cached_reply, cached_sources, cached_triage = cached
            triage_copy = copy.deepcopy(cached_triage)
            triage_copy["performance"] = {
                "retrieval_ms": 0.05,
                "generation_ms": 0.05,
                "total_ms": round((time.perf_counter() - overall_start) * 1000, 2),
            }
            triage_copy["cache_hit"] = True
            logger.info("Serving clinical query from isolated cache (streaming): '%s'", user_message[:40])
            yield {
                "event": "start",
                "data": {
                    "is_emergency": False,
                    "cache_hit": True,
                    "sources_count": len(cached_sources),
                },
            }
            yield {
                "event": "evidence",
                "data": {
                    "sources": cached_sources,
                    "evidence": triage_copy.get("evidence", {}),
                    "citations": triage_copy.get("citation_validation", {}),
                },
            }
            yield {"event": "chunk", "data": {"text": cached_reply}}
            yield {
                "event": "complete",
                "data": {
                    "response": cached_reply,
                    "sources": cached_sources,
                    "triage": triage_copy,
                    "evidence": triage_copy.get("evidence", {}),
                    "citations": triage_copy.get("citation_validation", {}),
                },
            }
            return

        # 3. Retrieve Medical Evidence
        base_chunks = []
        uploaded_chunks = []
        sources = []
        top_k = self._get_config("RAG_TOP_K", 3)
        retrieval_start = time.perf_counter()

        if self.retriever and self.retriever.is_ready():
            try:
                base_chunks = self.retriever.retrieve(user_message, top_k=top_k)
            except Exception as exc:
                logger.warning("RAG retrieval encountered an error: %s", exc)

        if conversation_id and user_id and getattr(self, "document_service", None):
            try:
                uploaded_chunks = self.document_service.retrieve_for_conversation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    query=user_message,
                    top_k=2,
                )
            except Exception as doc_exc:
                logger.warning("Conversation document retrieval encountered an error: %s", doc_exc)

        retrieved_chunks = uploaded_chunks + base_chunks
        if self.retriever:
            sources = self.retriever.extract_sources(retrieved_chunks)

        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        evidence_status = assess_evidence(retrieved_chunks)

        yield {
            "event": "start",
            "data": {
                "is_emergency": False,
                "cache_hit": False,
                "sources_count": len(sources),
                "retrieved_count": len(retrieved_chunks),
            },
        }

        yield {
            "event": "evidence",
            "data": {
                "sources": sources,
                "evidence": evidence_status,
            },
        }

        # 4. Interactive LangGraph Workflow check
        if getattr(self, "workflow_engine", None) and (selected_option or question_id):
            try:
                wf_reply, wf_sources, wf_triage = self.workflow_engine.execute(
                    user_message=user_message,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    academic_year=resolved_academic_year,
                    target_exam=resolved_target_exam,
                    history=history,
                    retrieved_chunks=retrieved_chunks,
                    selected_option=selected_option,
                    question_id=question_id,
                )
                if wf_reply and "unable to connect" not in wf_reply.lower() and not wf_triage.get("infrastructure_error"):
                    from services.medical_images import find_relevant_medical_image, render_image_markdown_block
                    matched_img = find_relevant_medical_image(user_message, wf_reply)
                    if matched_img and matched_img.get("image_url") and matched_img["image_url"] not in wf_reply:
                        wf_reply = wf_reply.rstrip() + "\n" + render_image_markdown_block(matched_img)
                        wf_triage["visual_reference"] = matched_img["title"]

                    for k, v in triage.items():
                        if k not in wf_triage:
                            wf_triage[k] = v
                    wf_triage["performance"] = {
                        "retrieval_ms": round(retrieval_ms, 2),
                        "generation_ms": round((time.perf_counter() - overall_start) * 1000 - retrieval_ms, 2),
                        "total_ms": round((time.perf_counter() - overall_start) * 1000, 2),
                    }
                    yield {"event": "chunk", "data": {"text": wf_reply}}
                    yield {
                        "event": "complete",
                        "data": {
                            "response": wf_reply,
                            "sources": wf_sources or sources,
                            "triage": wf_triage,
                            "evidence": evidence_status,
                            "citations": wf_triage.get("citation_validation", {}),
                        },
                    }
                    return
            except Exception as wf_exc:
                logger.warning("Workflow execution fell through to streaming generator: %s", wf_exc)

        # 5. Build Messages
        max_turns = self._get_config("MAX_HISTORY_TURNS", 20)
        messages = MBBSPromptBuilder.create_prompt_messages(
            system_instruction=SYSTEM_INSTRUCTION,
            safety_directive=safety_directive,
            user_message=user_message,
            academic_year=resolved_academic_year,
            target_exam=resolved_target_exam,
            retrieved_documents=retrieved_chunks,
            evidence_status=evidence_status,
            history=history,
            max_history_turns=max_turns,
        )

        # 6. Stream content with fallback models
        gen_start = time.perf_counter()
        candidate_models = [self.gemini_model]
        for fallback in ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        accumulated_chunks: List[str] = []
        active_model = self.gemini_model
        last_gen_exc: Optional[Exception] = None

        try:
            for chunk_text, model_name in self.langchain_adapter.stream_chat_turn(
                messages=messages,
                system_instruction=SYSTEM_INSTRUCTION,
                candidate_models=candidate_models,
                client=self.client,
                max_output_tokens=self._get_config("GEMINI_MAX_OUTPUT_TOKENS", 850),
                temperature=self._get_config("GEMINI_TEMPERATURE", 0.2),
            ):
                active_model = model_name
                accumulated_chunks.append(chunk_text)
                yield {"event": "chunk", "data": {"text": chunk_text}}
        except Exception as exc:
            last_gen_exc = exc
            logger.warning("Streaming generation exception: %s", exc)

        gen_ms = (time.perf_counter() - gen_start) * 1000
        raw_reply = "".join(accumulated_chunks).strip()

        if not raw_reply:
            if not self.api_key:
                reply_text = (
                    "Google Gemini API key is not configured. Please add your GEMINI_API_KEY to the .env file. "
                    "For urgent medical concerns, please consult a qualified healthcare provider or emergency services."
                )
            else:
                reply_text = (
                    "I am temporarily unable to connect to the Google Gemini AI service due to a technical or capacity issue. "
                    "Please try again shortly. For urgent medical concerns, please consult a qualified healthcare provider or emergency services."
                )
            triage["evidence"] = evidence_status
            triage["citation_validation"] = audit_citations(reply_text, retrieved_chunks)
            triage["structured_output"] = False
            triage["retrieved_evidence_count"] = len(retrieved_chunks)
            triage["infrastructure_error"] = str(last_gen_exc) if last_gen_exc else "No streamed tokens"
            triage["performance"] = {
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(gen_ms, 2),
                "total_ms": round((time.perf_counter() - overall_start) * 1000, 2),
            }
            yield {"event": "chunk", "data": {"text": reply_text}}
            yield {
                "event": "complete",
                "data": {
                    "response": reply_text,
                    "sources": sources,
                    "triage": triage,
                    "evidence": evidence_status,
                    "citations": triage["citation_validation"],
                },
            }
            return

        # 7. Parse Structured Evidence & Audit Citations
        parsed = parse_structured_response(raw_reply)
        parsed_answer = parsed["answer"]
        citation_validation = audit_citations(parsed_answer, retrieved_chunks, parsed["cited_passage_ids"])
        clean_reply = sanitize_and_render_citations(parsed_answer, retrieved_chunks)

        # 7b. Verified medical illustration
        from services.medical_images import find_relevant_medical_image, render_image_markdown_block
        matched_img = find_relevant_medical_image(user_message, clean_reply)
        if matched_img and matched_img.get("image_url") and matched_img["image_url"] not in clean_reply:
            clean_reply = clean_reply.rstrip() + "\n" + render_image_markdown_block(matched_img)
            triage["visual_reference"] = matched_img["title"]

        triage["citation_validation"] = citation_validation
        triage["evidence"] = evidence_status
        triage["structured_output"] = parsed["structured_output_valid"]
        triage["model_evidence_sufficiency"] = parsed["evidence_sufficiency"]
        triage["retrieved_evidence_count"] = len(retrieved_chunks)
        triage["performance"] = {
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(gen_ms, 2),
            "total_ms": round((time.perf_counter() - overall_start) * 1000, 2),
        }
        triage["usage"] = {
            "prompt_tokens": len(user_message.split()) * 2,
            "completion_tokens": len(clean_reply.split()) * 2,
            "total_tokens": (len(user_message.split()) + len(clean_reply.split())) * 2,
            "estimated_cost_usd": 0.0001,
            "model": active_model,
        }

        # Store in Cache
        if clean_reply and not triage.get("infrastructure_error"):
            self.cache.set(
                user_message=user_message,
                reply=clean_reply,
                sources=sources,
                triage=triage,
                user_id=user_id,
                conversation_id=conversation_id,
                history=history,
                model_name=active_model,
                retrieval_mode=retrieval_mode,
            )

        yield {
            "event": "complete",
            "data": {
                "response": clean_reply,
                "sources": sources,
                "triage": triage,
                "evidence": evidence_status,
                "citations": citation_validation,
                "practice_mode": parsed.get("practice_mode", "general_tutoring"),
                "interactive_mcq": parsed.get("interactive_mcq"),
                "clinical_case": parsed.get("clinical_case"),
            },
        }

