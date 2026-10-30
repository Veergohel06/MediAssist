"""Interactive MBBS Exam Practice & Clinical Reasoning Workflow Service.
====================================================================
Powers the conversational exam practice and clinical reasoning engine for MediAssist
using LangGraph state graphs.

Supports:
1. Conversational Viva Voce:
   - Recognizes "quiz me", "simulate a viva", "oral exam".
   - Asks one question at a time.
   - Evaluates student answers: correct points, omissions, and misconceptions.
   - Adapts difficulty to student academic year and target examination.
   - Allows natural exit to general tutoring.

2. Interactive MCQs:
   - Generates clinical single-best-answer questions with 4 plausible distractors.
   - Validates question structure, options, and defensible answer keys.
   - Provides accessible interactive buttons rendered in chat bubbles.
   - Prevents duplicate submissions and duplicate scoring.
   - Supports ordinary text-based answering fallback.
   - Labels questions as practice questions (not official exam items).

3. Progressive Clinical Case Discussions:
   - Educational simulation with progressive disclosure:
     Stage 1: Patient presentation & vitals.
     Stage 2: Physical examination findings.
     Stage 3: Diagnostic investigations and lab results.
     Stage 4: Structured clinical debrief & evidence citations.
   - Enforces clinical supervision disclaimer on all simulated decisions.

4. LangGraph Graph Architecture:
   - StateGraph routing across emergency triage, general tutoring, viva, MCQ, and case discussions.
   - Preserves conversation ID, persistent history, and safety check integration.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict

from langgraph.graph import StateGraph, START, END

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from rag.evidence import (
    assess_evidence,
    audit_citations,
    parse_structured_response,
    sanitize_and_render_citations,
)
from safety.evaluator import SymptomSafetyEvaluator
from services.langchain_service import MBBSPromptBuilder, LangChainGeminiChatAdapter

logger = logging.getLogger("mediassist.workflow")

# Trigger patterns for interactive modes
VIVA_PATTERNS = [
    re.compile(r"\b(?:quiz me|simulate a viva|start viva|viva on|oral exam|test my knowledge|ask me questions|viva voce|grill me)\b", re.IGNORECASE),
]

MCQ_PATTERNS = [
    re.compile(r"\b(?:give me an? mcq|give me mcqs?|practice questions?|neet pg mcq|multiple choice|ask me an? mcq|generate an? mcq|single best answer|generate\s+(?:a\s+)?(?:single\s+best\s+answer\s+)?(?:practice\s+)?questions?)\b", re.IGNORECASE),
]

CASE_PATTERNS = [
    re.compile(r"\b(?:present\s+(?:a\s+)?(?:fictional\s+)?(?:patient\s+)?case|clinical\s+case|case\s+study|discuss\s+(?:a\s+)?case|patient\s+scenario|clinical\s+vignette|simulation\s+case)\b", re.IGNORECASE),
]

EXIT_PATTERNS = [
    re.compile(r"\b(?:quit practice|exit practice|stop practice|stop viva|exit viva|stop quiz|end viva|stop case|exit case|new topic|let'?s switch|back to tutor|stop questions?|enough questions?|explain this instead)\b", re.IGNORECASE),
]

OPTION_LETTER_PATTERN = re.compile(r"^(?:option\s+)?([A-Da-d])(?:[:.)\s]|$)", re.IGNORECASE)


def classify_interactive_mode(text: str) -> str:
    """Classify user intent into an interactive mode: 'viva_voce', 'mcq_practice', 'clinical_case', or 'general_tutoring'."""
    text_s = (text or "").strip()
    for pat in EXIT_PATTERNS:
        if pat.search(text_s):
            return "general_tutoring"
    for pat in MCQ_PATTERNS:
        if pat.search(text_s):
            return "mcq_practice"
    for pat in VIVA_PATTERNS:
        if pat.search(text_s):
            return "viva_voce"
    for pat in CASE_PATTERNS:
        if pat.search(text_s):
            return "clinical_case"
    return "general_tutoring"


def is_natural_exit(text: str) -> bool:
    """Check if the student is requesting to exit an interactive exam/case session."""
    text_s = (text or "").strip()
    return any(pat.search(text_s) for pat in EXIT_PATTERNS)


def extract_option_selection(text: str) -> Optional[str]:
    """Extract an MCQ option letter (A, B, C, D) from user text, supporting 'Option B: ...', 'B', etc."""
    text_s = (text or "").strip()
    if text_s.upper() in {"A", "B", "C", "D"}:
        return text_s.upper()
    match = OPTION_LETTER_PATTERN.match(text_s)
    if match:
        return match.group(1).upper()
    loose_match = re.search(r"\b(?:choose|select|answer is|option)\s+([A-Da-d])\b", text_s, re.IGNORECASE)
    if loose_match:
        return loose_match.group(1).upper()
    return None


class ChatWorkflowState(TypedDict):
    """LangGraph State representation for the clinical consultation workflow."""
    user_message: str
    conversation_id: Optional[str]
    user_id: Optional[int]
    academic_year: Optional[str]
    target_exam: Optional[str]
    history: List[Dict[str, Any]]
    selected_option: Optional[str]
    question_id: Optional[str]
    workflow_mode: str  # "general_tutoring" | "viva_voce" | "mcq_practice" | "mcq_answer" | "clinical_case" | "emergency_triage"
    safety_directive: str
    safety_result: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    evidence_status: Dict[str, Any]
    system_instruction: str
    response_text: str
    sources: List[Dict[str, Any]]
    triage: Dict[str, Any]
    interactive_mcq: Optional[Dict[str, Any]]
    clinical_case: Optional[Dict[str, Any]]
    active_model: str
    error: Optional[str]


def detect_previous_mcq(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Inspect recent conversation history to locate the most recent interactive MCQ."""
    if not history:
        return None
    for entry in reversed(history[-6:]):
        evidence = entry.get("evidence") or {}
        if isinstance(evidence, dict) and evidence.get("interactive_mcq"):
            return evidence["interactive_mcq"]
    return None


def detect_previous_case(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Inspect recent conversation history to locate an ongoing clinical case discussion."""
    if not history:
        return None
    for entry in reversed(history[-6:]):
        evidence = entry.get("evidence") or {}
        if isinstance(evidence, dict) and evidence.get("clinical_case"):
            return evidence["clinical_case"]
        # Also check content cues for ongoing case
        content = str(entry.get("content") or "")
        if "Clinical Case Presentation" in content or "Educational Clinical Case" in content:
            if "Debrief & Clinical Pearls" not in content:
                return {"stage": "in_progress"}
    return None


def detect_previous_viva(history: List[Dict[str, Any]]) -> bool:
    """Inspect recent conversation history to determine if a viva session is active."""
    if not history:
        return False
    for entry in reversed(history[-4:]):
        content = str(entry.get("content") or "")
        evidence = entry.get("evidence") or {}
        if isinstance(evidence, dict) and evidence.get("practice_mode") == "viva_voce":
            return True
        if "MBBS Viva Voce Examination" in content or "Question 1" in content or "Next Question" in content:
            return True
    return False


class InteractiveWorkflowEngine:
    """LangGraph-powered stateful conversational engine for MediAssist."""

    def __init__(self, llm_service: Any):
        self.llm_service = llm_service
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct and compile the LangGraph workflow StateGraph."""
        builder = StateGraph(ChatWorkflowState)

        # Nodes
        builder.add_node("classify_intent", self._node_classify_intent)
        builder.add_node("emergency_node", self._node_emergency)
        builder.add_node("viva_node", self._node_viva)
        builder.add_node("mcq_node", self._node_mcq)
        builder.add_node("clinical_case_node", self._node_clinical_case)
        builder.add_node("tutoring_node", self._node_tutoring)
        builder.add_node("audit_and_persist", self._node_audit_and_persist)

        # Edges from START to classification
        builder.add_edge(START, "classify_intent")

        # Conditional branching from classify_intent
        builder.add_conditional_edges(
            "classify_intent",
            self._route_intent,
            {
                "emergency_triage": "emergency_node",
                "viva_voce": "viva_node",
                "mcq_practice": "mcq_node",
                "mcq_answer": "mcq_node",
                "clinical_case": "clinical_case_node",
                "general_tutoring": "tutoring_node",
            },
        )

        # Convergence to audit_and_persist
        builder.add_edge("emergency_node", END)
        builder.add_edge("viva_node", "audit_and_persist")
        builder.add_edge("mcq_node", "audit_and_persist")
        builder.add_edge("clinical_case_node", "audit_and_persist")
        builder.add_edge("tutoring_node", "audit_and_persist")
        builder.add_edge("audit_and_persist", END)

        return builder.compile()

    def _route_intent(self, state: ChatWorkflowState) -> str:
        """Route to appropriate conversational node based on resolved mode."""
        return state.get("workflow_mode", "general_tutoring")

    # ── Node 1: Intent Classification & Routing ────────────────────────────────
    def _node_classify_intent(self, state: ChatWorkflowState) -> Dict[str, Any]:
        user_msg = state["user_message"].strip()
        history = state.get("history") or []
        explicit_option = state.get("selected_option")

        # 1. Safety Red Flag Check (Mandatory First-Priority)
        safety_evaluator = getattr(self.llm_service, "safety_evaluator", None)
        if safety_evaluator:
            safety_res = safety_evaluator.evaluate(user_msg)
        else:
            safety_res = {"is_emergency": False, "urgency": "ROUTINE", "clinical_topic": "general"}

        state["safety_result"] = safety_res
        if safety_res.get("is_emergency"):
            return {
                "workflow_mode": "emergency_triage",
                "safety_result": safety_res,
            }

        # 2. Check for MCQ Option Submission (Button click or text fallback)
        active_mcq = detect_previous_mcq(history)
        if explicit_option and explicit_option.upper() in {"A", "B", "C", "D"}:
            return {
                "workflow_mode": "mcq_answer",
                "selected_option": explicit_option.upper(),
                "interactive_mcq": active_mcq,
            }

        if active_mcq and not active_mcq.get("answered"):
            chosen = extract_option_selection(user_msg)
            if chosen:
                return {
                    "workflow_mode": "mcq_answer",
                    "selected_option": chosen,
                    "interactive_mcq": active_mcq,
                }

        # 3. Check for Natural Exit back to Tutoring
        for pat in EXIT_PATTERNS:
            if pat.search(user_msg):
                return {"workflow_mode": "general_tutoring"}

        # 4. Check for MCQ Generation Request
        for pat in MCQ_PATTERNS:
            if pat.search(user_msg):
                return {"workflow_mode": "mcq_practice"}

        # 5. Check for Viva Voce Request
        for pat in VIVA_PATTERNS:
            if pat.search(user_msg):
                return {"workflow_mode": "viva_voce"}

        # 6. Check for Clinical Case Request
        for pat in CASE_PATTERNS:
            if pat.search(user_msg):
                return {"workflow_mode": "clinical_case"}

        # 7. Check Continuity with Ongoing Session
        if detect_previous_viva(history):
            return {"workflow_mode": "viva_voce"}

        if detect_previous_case(history):
            return {"workflow_mode": "clinical_case"}

        return {"workflow_mode": "general_tutoring"}

    # ── Node 2: Emergency Immediate Response ────────────────────────────────────
    def _node_emergency(self, state: ChatWorkflowState) -> Dict[str, Any]:
        safety_res = state.get("safety_result") or {}
        triage = safety_res.get("triage") or safety_res
        condition = triage.get("condition") or "Acute Medical Emergency"
        advisory = triage.get("advisory") or "Immediate medical attention required."

        emergency_reply = (
            "### 🚨 Immediate Medical Alert\n\n"
            f"**Your inquiry describes potential emergency symptoms ({condition}):** {advisory}\n\n"
            "1. **Call 112 / 108 (India) or your local emergency number immediately.**\n"
            "2. **Do not attempt to self-treat or drive yourself.**\n"
            "3. If accompanied, alert someone nearby right away.\n\n"
            "*(MediAssist provides educational guidance for students and cannot manage real-time medical emergencies.)*"
        )
        return {
            "response_text": emergency_reply,
            "sources": [],
            "triage": triage,
            "interactive_mcq": None,
            "clinical_case": None,
        }

    # ── Node 3: Conversational Viva Voce ────────────────────────────────────────
    def _node_viva(self, state: ChatWorkflowState) -> Dict[str, Any]:
        user_msg = state["user_message"]
        history = state.get("history") or []
        academic_year = state.get("academic_year")
        target_exam = state.get("target_exam")
        retrieved_chunks = state.get("retrieved_chunks") or []
        evidence_status = state.get("evidence_status") or {"status": "insufficient"}

        # Determine if this is question 1 or answering a previous question
        is_followup = detect_previous_viva(history) and not any(pat.search(user_msg) for pat in VIVA_PATTERNS)

        viva_prompt_directive = (
            "--- MBBS VIVA VOCE EXAMINATION MODE ---\n"
            "You are an experienced MBBS University Professor and Viva Voce Examiner.\n"
            "Your objective is to conduct a realistic, rigorous, and supportive oral viva.\n"
            "Strict Rules for this Viva Mode:\n"
            "1. Ask ONE question at a time. Never ask multiple separate questions in a single turn.\n"
            "2. If this is the start of the viva, greet the student crisply, state the topic, and ask Question 1.\n"
            "3. If the student has just answered a viva question, evaluate their response systematically:\n"
            "   - **Strengths**: Acknowledge key accurate pathophysiological or clinical points they mentioned.\n"
            "   - **Omissions**: Highlight high-yield points, mechanisms, or diagnostic criteria they missed.\n"
            "   - **Misconceptions**: Clarify any inaccuracies gently and precisely.\n"
            "   - **Next Question**: Ask the next question, stepping up difficulty according to their academic profile.\n"
            "4. Adapt questions to the student's profile: "
            f"Academic Stage={academic_year or 'General MBBS'}, Target Exam={target_exam or 'University Profs'}.\n"
            "5. If evidence from the verified knowledge base is available, ground your questions and answers in it.\n"
            "6. Always keep your response focused and examiner-like. Return response adhering to the JSON schema."
        )

        resp_text, sources, active_model, mcq_obj, case_obj = self._call_model(
            system_directive=viva_prompt_directive,
            user_msg=user_msg,
            state=state,
            practice_mode="viva_voce",
        )

        return {
            "response_text": resp_text,
            "sources": sources,
            "active_model": active_model,
            "interactive_mcq": None,
            "clinical_case": None,
        }

    # ── Node 4: Interactive MCQs (Generation & Evaluation) ──────────────────────
    def _node_mcq(self, state: ChatWorkflowState) -> Dict[str, Any]:
        mode = state.get("workflow_mode")
        user_msg = state["user_message"]
        selected_option = state.get("selected_option")
        active_mcq = state.get("interactive_mcq") or detect_previous_mcq(state.get("history") or [])

        # CASE A: Answering an existing MCQ
        if mode == "mcq_answer" or selected_option:
            if not active_mcq:
                # Fallback if no active MCQ found
                return self._node_tutoring(state)

            question_id = active_mcq.get("question_id", "mcq_active")
            correct_key = (active_mcq.get("correct_key") or active_mcq.get("correct_option") or "A").upper()
            extracted = extract_option_selection(user_msg)
            chosen_key = (selected_option or extracted or user_msg[:1]).upper()

            # Duplicate Submission Guard: Check if already answered
            if active_mcq.get("answered"):
                prior_ans = active_mcq.get("user_answer", "recorded")
                already_answered_msg = (
                    f"### ℹ️ Question Already Answered\n\n"
                    f"You have already submitted an answer for this practice question (Selected: **Option {prior_ans}**).\n\n"
                    f"To preserve scoring integrity, duplicate submissions are not scored. "
                    f"The correct answer remains **Option {correct_key}**.\n\n"
                    f"Would you like another practice question or would you like to review the concept in detail?"
                )
                return {
                    "response_text": already_answered_msg,
                    "sources": [],
                    "interactive_mcq": active_mcq,
                    "clinical_case": None,
                }

            is_correct = (chosen_key == correct_key)
            status_badge = "✅ **Correct! Well done.**" if is_correct else f"❌ **Incorrect. The correct answer is Option {correct_key}.**"

            explanation = active_mcq.get("explanation") or "Standard undergraduate clinical guideline."
            opt_map = {opt.get("key", ""): opt.get("text", "") for opt in active_mcq.get("options", [])}

            eval_reply = (
                f"### Practice Question Evaluation\n\n"
                f"{status_badge}\n\n"
                f"**Your Choice**: Option {chosen_key}: {opt_map.get(chosen_key, chosen_key)}\n\n"
                f"**Correct Answer**: Option {correct_key}: {opt_map.get(correct_key, correct_key)}\n\n"
                f"#### Pathophysiological & Clinical Rationale:\n"
                f"{explanation}\n\n"
                f"---\n"
                f"*(MBBS Practice Item · For Educational Revision Only)*"
            )

            # Mark active MCQ as answered
            updated_mcq = dict(active_mcq)
            updated_mcq["answered"] = True
            updated_mcq["user_answer"] = chosen_key
            updated_mcq["is_correct"] = is_correct

            return {
                "response_text": eval_reply,
                "sources": [],
                "interactive_mcq": updated_mcq,
                "clinical_case": None,
            }

        # CASE B: Generating a new MCQ
        mcq_generation_directive = (
            "--- MBBS INTERACTIVE MCQ GENERATION MODE ---\n"
            "You are an expert medical item writer for undergraduate MBBS, NEET-PG/NExT, and USMLE exams.\n"
            "Generate exactly ONE high-yield clinical single-best-answer multiple choice question.\n\n"
            "Requirements for the Question:\n"
            "1. Stem: Present a realistic clinical vignette with patient age, sex, setting, chief complaint, relevant physical findings, or lab values.\n"
            "2. Options: Provide exactly 4 options labeled A, B, C, D. All 4 distractors must be clinically plausible differentials or pharmacologic alternatives.\n"
            "3. Answer Key: Explicitly designate ONE defensible correct option (e.g., 'B').\n"
            "4. Rationale: Provide a comprehensive explanation detailing why the correct option is the best step, why each of the 3 distractors is suboptimal, and cite relevant medical sources.\n"
            "5. Label clearly at the top: '[MBBS Practice Question · For Educational Revision Only — Not an Official Examination Item]'.\n"
            "6. In your JSON response, return the structured question in 'interactive_mcq' with keys: 'question_id', 'stem', 'options' (list of {key, text}), 'correct_key', 'explanation', 'answered': false."
        )

        resp_text, sources, active_model, mcq_obj, case_obj = self._call_model(
            system_directive=mcq_generation_directive,
            user_msg=user_msg,
            state=state,
            practice_mode="mcq_practice",
        )

        # Fallback structured MCQ object if model omitted JSON field
        if not mcq_obj:
            mcq_obj = self._extract_mcq_from_markdown(resp_text)

        return {
            "response_text": resp_text,
            "sources": sources,
            "active_model": active_model,
            "interactive_mcq": mcq_obj,
            "clinical_case": None,
        }

    # ── Node 5: Progressive Clinical Case Discussions ──────────────────────────
    def _node_clinical_case(self, state: ChatWorkflowState) -> Dict[str, Any]:
        user_msg = state["user_message"]
        history = state.get("history") or []
        academic_year = state.get("academic_year")
        target_exam = state.get("target_exam")

        # Determine current stage from history
        prev_case = detect_previous_case(history)
        current_stage = 1
        if prev_case:
            stage_val = prev_case.get("current_stage") or prev_case.get("stage") or 1
            if isinstance(stage_val, int):
                current_stage = min(stage_val + 1, 4)
            elif "in_progress" in str(stage_val):
                current_stage = 2

        # Check if user requested debrief immediately
        if any(term in user_msg.lower() for term in ["debrief", "reveal diagnosis", "final answer", "summary"]):
            current_stage = 4

        case_directive = (
            "--- MBBS PROGRESSIVE CLINICAL CASE DISCUSSION MODE ---\n"
            "You are a clinical mentor facilitating an interactive clinical reasoning case discussion for MBBS students.\n"
            f"CURRENT PROGRESSION STAGE: {current_stage} of 4.\n\n"
            "Stage Guidelines:\n"
            "- **Stage 1 (Presentation & History)**: Present a fictional educational case. Provide patient age, sex, setting (OPD/Emergency), chief complaint, HPI, past history, and initial vital signs. Ask student for their initial top 3 differential diagnoses.\n"
            "- **Stage 2 (Physical Examination)**: Provide focused physical exam findings by system (General, CVS, RS, Abdomen, Neuro). Ask student which investigations they would order to rule in/out the differentials.\n"
            "- **Stage 3 (Investigations & Labs)**: Reveal lab results, ECG, imaging, or biomarkers. Ask student for their definitive diagnosis and acute management plan.\n"
            "- **Stage 4 (Structured Debrief)**: Provide the final diagnosis, differential comparison, pathophysiological review, gold-standard guidelines, high-yield exam pearls, and source citations.\n\n"
            "Mandatory Safety Rule for All Stages:\n"
            "Always include the supervisory disclaimer: '⚠️ Educational clinical simulation only. All real-patient clinical decisions and prescriptions require direct evaluation by licensed attending physicians.'\n"
            "Return JSON response adhering to schema with 'clinical_case' dict containing: 'current_stage': int, 'stage': int, 'is_debrief': bool."
        )

        resp_text, sources, active_model, mcq_obj, case_obj = self._call_model(
            system_directive=case_directive,
            user_msg=user_msg,
            state=state,
            practice_mode="clinical_case",
        )

        if not case_obj:
            case_obj = {"stage": current_stage, "current_stage": current_stage, "is_debrief": (current_stage == 4)}
        else:
            if "current_stage" not in case_obj and "stage" in case_obj:
                case_obj["current_stage"] = case_obj["stage"]
            elif "stage" not in case_obj and "current_stage" in case_obj:
                case_obj["stage"] = case_obj["current_stage"]

        return {
            "response_text": resp_text,
            "sources": sources,
            "active_model": active_model,
            "interactive_mcq": None,
            "clinical_case": case_obj,
        }

    # ── Node 6: General MBBS Tutoring (Default Fallback) ────────────────────────
    def _node_tutoring(self, state: ChatWorkflowState) -> Dict[str, Any]:
        user_msg = state["user_message"]
        tutoring_directive = (
            "--- MBBS GENERAL ACADEMIC TUTORING MODE ---\n"
            "Provide an undergraduate MBBS medical explanation integrating basic sciences, pathophysiology, "
            "clinical manifestations, diagnostic choice, and management principles according to CBME standards."
        )

        resp_text, sources, active_model, mcq_obj, case_obj = self._call_model(
            system_directive=tutoring_directive,
            user_msg=user_msg,
            state=state,
            practice_mode="general_tutoring",
        )

        return {
            "response_text": resp_text,
            "sources": sources,
            "active_model": active_model,
            "interactive_mcq": None,
            "clinical_case": None,
        }

    # ── Node 7: Evidence Audit, Citation Sanitization & Persistence ─────────────
    def _node_audit_and_persist(self, state: ChatWorkflowState) -> Dict[str, Any]:
        resp_text = state.get("response_text", "")
        retrieved_chunks = state.get("retrieved_chunks") or []
        safety_res = state.get("safety_result") or {}
        safety_triage = safety_res.get("triage") or safety_res
        triage = dict(state.get("triage") or {})
        for k, v in safety_triage.items():
            if k not in triage:
                triage[k] = v

        # 1. Audit citations and clean internal tokens
        audit_res = audit_citations(resp_text, retrieved_chunks)
        clean_reply = sanitize_and_render_citations(resp_text, retrieved_chunks)

        triage["citation_validation"] = audit_res
        triage["evidence"] = state.get("evidence_status") or assess_evidence(retrieved_chunks)
        mode = state.get("workflow_mode", "general_tutoring")
        triage["practice_mode"] = "mcq_practice" if mode in ("mcq_practice", "mcq_answer") else mode

        # 2. Attach interactive metadata to triage for client consumption
        if state.get("interactive_mcq"):
            triage["interactive_mcq"] = state["interactive_mcq"]
        if state.get("clinical_case"):
            triage["clinical_case"] = state["clinical_case"]

        # 3. Cache response if non-empty
        if clean_reply and hasattr(self.llm_service, "cache") and self.llm_service.cache:
            self.llm_service.cache.set(
                user_message=state["user_message"],
                reply=clean_reply,
                sources=state.get("sources") or [],
                triage=triage,
                user_id=state.get("user_id"),
                conversation_id=state.get("conversation_id"),
                history=state.get("history"),
                model_name=state.get("active_model", "gemini-3.5-flash-lite"),
            )

        return {
            "response_text": clean_reply,
            "triage": triage,
            "interactive_mcq": state.get("interactive_mcq"),
            "clinical_case": state.get("clinical_case"),
        }

    # ── Helper: Call Model via LangChain Adapter ───────────────────────────────
    def _call_model(
        self,
        system_directive: str,
        user_msg: str,
        state: ChatWorkflowState,
        practice_mode: str,
    ) -> Tuple[str, List[Dict[str, Any]], str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Invoke Gemini through the LangChain adapter with context and structured parsing."""
        llm = self.llm_service
        academic_year = state.get("academic_year")
        target_exam = state.get("target_exam")
        history = state.get("history") or []
        retrieved_chunks = state.get("retrieved_chunks") or []
        evidence_status = state.get("evidence_status") or assess_evidence(retrieved_chunks)

        sources = []
        if hasattr(llm, "retriever") and llm.retriever:
            sources = llm.retriever.extract_sources(retrieved_chunks)

        # Assemble prompt messages
        max_turns = llm._get_config("MAX_HISTORY_TURNS", 20) if hasattr(llm, "_get_config") else 20
        combined_instruction = f"{llm.system_instruction}\n\n{system_directive}" if hasattr(llm, "system_instruction") else system_directive

        messages = MBBSPromptBuilder.create_prompt_messages(
            system_instruction=combined_instruction,
            safety_directive="",
            user_message=user_msg,
            academic_year=academic_year,
            target_exam=target_exam,
            retrieved_documents=retrieved_chunks,
            evidence_status=evidence_status,
            history=history,
            max_history_turns=max_turns,
        )

        candidate_models = [getattr(llm, "gemini_model", "gemini-3.5-flash-lite"), "gemini-3.5-flash-lite", "gemini-flash-lite-latest"]

        if hasattr(llm, "langchain_adapter") and llm.langchain_adapter:
            response, active_model, last_exc = llm.langchain_adapter.execute_chat_turn(
                messages=messages,
                system_instruction=combined_instruction,
                candidate_models=candidate_models,
                client=getattr(llm, "client", None),
                max_output_tokens=llm._get_config("GEMINI_MAX_OUTPUT_TOKENS", 900) if hasattr(llm, "_get_config") else 900,
                temperature=0.3,
            )
        else:
            response, active_model, last_exc = None, "fallback", None

        resp_text = ""
        if isinstance(response, dict):
            resp_text = json.dumps(response)
        elif hasattr(response, "text"):
            resp_text = getattr(response, "text", "") or ""
        elif isinstance(response, str):
            resp_text = response
        elif response is not None:
            resp_text = str(response)

        if not resp_text:
            return (
                "MediAssist is temporarily unable to connect to the Google Gemini AI service. Please try again shortly.",
                sources,
                active_model or "fallback",
                None,
                None,
            )

        parsed = parse_structured_response(resp_text)
        answer = parsed.get("answer") or resp_text
        mcq_payload = parsed.get("interactive_mcq")
        case_payload = parsed.get("clinical_case")

        if isinstance(response, dict):
            answer = response.get("answer") or answer
            mcq_payload = response.get("interactive_mcq") or mcq_payload
            case_payload = response.get("clinical_case") or case_payload

        return answer, sources, active_model or "gemini-3.5-flash-lite", mcq_payload, case_payload

    # ── Helper: Extract MCQ Structure from Markdown Text Fallback ───────────────
    def _extract_mcq_from_markdown(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse options A, B, C, D from generated text when model output omits JSON block."""
        opt_matches = re.findall(r"(?m)^(?:\*\*)?(?:Option\s+)?([A-D])(?:\*\*)?[:.)]\s+(.+)$", text)
        if len(opt_matches) >= 4:
            options = [{"key": m[0].upper(), "text": m[1].strip()} for m in opt_matches[:4]]
            correct_match = re.search(r"(?i)(?:correct(?:\s+answer)?|answer key)[\s:*]+(?:Option\s+)?([A-D])", text)
            correct_key = correct_match.group(1).upper() if correct_match else "A"
            return {
                "question_id": f"mcq_{uuid.uuid4().hex[:8]}",
                "stem": text.split("\n\n")[0],
                "options": options,
                "correct_key": correct_key,
                "explanation": "See detailed breakdown in clinical response above.",
                "answered": False,
                "user_answer": None,
            }
        return None

    # ── Primary Public Interface ───────────────────────────────────────────────
    def execute(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[int] = None,
        academic_year: Optional[str] = None,
        target_exam: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        selected_option: Optional[str] = None,
        question_id: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Execute the LangGraph clinical consultation workflow.

        Returns (reply_text, sources, triage_dict).
        """
        initial_state: ChatWorkflowState = {
            "user_message": user_message,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "academic_year": academic_year,
            "target_exam": target_exam,
            "history": history or [],
            "selected_option": selected_option,
            "question_id": question_id,
            "workflow_mode": "general_tutoring",
            "safety_directive": "",
            "safety_result": {},
            "retrieved_chunks": retrieved_chunks or [],
            "evidence_status": assess_evidence(retrieved_chunks or []),
            "system_instruction": getattr(self.llm_service, "system_instruction", ""),
            "response_text": "",
            "sources": [],
            "triage": {},
            "interactive_mcq": None,
            "clinical_case": None,
            "active_model": "gemini-3.5-flash-lite",
            "error": None,
        }

        final_state = self.graph.invoke(initial_state)

        triage_out = dict(final_state.get("triage", {}) or {})
        if final_state.get("interactive_mcq") and "interactive_mcq" not in triage_out:
            triage_out["interactive_mcq"] = final_state["interactive_mcq"]
        if final_state.get("clinical_case") and "clinical_case" not in triage_out:
            triage_out["clinical_case"] = final_state["clinical_case"]
        if "practice_mode" not in triage_out:
            mode = final_state.get("workflow_mode", "general_tutoring")
            triage_out["practice_mode"] = "mcq_practice" if mode in ("mcq_practice", "mcq_answer") else mode

        return (
            final_state.get("response_text", ""),
            final_state.get("sources", []),
            triage_out,
        )
