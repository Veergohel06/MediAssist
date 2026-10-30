"""Automated test suite for Phase 6: Interactive MBBS Exam Practice & Clinical Reasoning.

Validates:
1. Conversational Viva:
   - Recognition of viva requests ("quiz me", "simulate a viva", "start viva on shock").
   - One question at a time sequencing.
   - Waiting for answer before feedback.
   - Structured feedback on correct points, omissions, and misconceptions.
   - Academic profile adaptation (1st year, final year, NEET-PG/NExT).
2. Interactive MCQs:
   - Single-best-answer question generation with clinical vignette and defensible key.
   - Practice question disclaimer label.
   - Option submission via chat flow and dedicated endpoint.
   - Duplicate submission prevention and no duplicate scoring.
   - Explanations, feedback, and verified citations.
   - Text-based fallback answering (e.g., "Option B" or "B").
3. Clinical Case Discussions:
   - Progressive disclosure across clinical stages (presentation, exam, labs, debrief).
   - Evidence-based differential diagnosis.
   - Clinical supervision disclaimer.
4. LangGraph Workflow Routing & Safety:
   - Natural exit/transition back to general tutoring.
   - Safety pre-flight escalation overrides exam modes for real-patient emergencies.
   - Turn-by-turn state persistence in DB.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import app as mediassist_app
from models import ChatMessage, Conversation, User, db
from services.workflow_service import (
    InteractiveWorkflowEngine,
    classify_interactive_mode,
    is_natural_exit,
    extract_option_selection,
)


class TestInteractiveExamAndReasoning(unittest.TestCase):
    """Integration and unit tests for Phase 6 interactive MBBS exam practice."""

    @classmethod
    def setUpClass(cls):
        cls.app = mediassist_app.app
        cls.orig_testing = cls.app.config.get("TESTING")
        cls.orig_csrf = cls.app.config.get("WTF_CSRF_ENABLED")

        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.app.config["SERVER_NAME"] = "localhost"

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

    def _create_user(self, email="student@medschool.edu", name="MBBS Candidate", password="Password123!", academic_year="final_year", target_exam="neet_pg_next") -> User:
        user = User(
            name=name,
            email=email.lower().strip(),
            academic_year=academic_year,
            target_exam=target_exam,
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return user

    def _login(self, email="student@medschool.edu", password="Password123!"):
        return self.client.post(
            "/login",
            data={"email": email, "password": password},
            follow_redirects=True,
        )

    # ── 1. Intent Classification & Natural Transition Unit Tests ───────────────

    def test_classify_interactive_mode_viva(self):
        self.assertEqual(classify_interactive_mode("simulate a viva on shock"), "viva_voce")
        self.assertEqual(classify_interactive_mode("quiz me on acute abdomen viva"), "viva_voce")
        self.assertEqual(classify_interactive_mode("can you conduct a viva voce examination?"), "viva_voce")

    def test_classify_interactive_mode_mcq(self):
        self.assertEqual(classify_interactive_mode("give me an MCQ on cardiology"), "mcq_practice")
        self.assertEqual(classify_interactive_mode("quiz me with a practice question"), "mcq_practice")
        self.assertEqual(classify_interactive_mode("generate a single best answer question on asthma"), "mcq_practice")

    def test_classify_interactive_mode_clinical_case(self):
        self.assertEqual(classify_interactive_mode("let's do a clinical case discussion on chest pain"), "clinical_case")
        self.assertEqual(classify_interactive_mode("present a fictional patient case for differential reasoning"), "clinical_case")

    def test_classify_interactive_mode_tutoring_default(self):
        self.assertEqual(classify_interactive_mode("explain the mechanism of Lisinopril in heart failure"), "general_tutoring")
        self.assertEqual(classify_interactive_mode("what are the histological layers of the adrenal cortex?"), "general_tutoring")

    def test_natural_exit_detection(self):
        self.assertTrue(is_natural_exit("stop viva"))
        self.assertTrue(is_natural_exit("let's exit case mode and return to tutoring"))
        self.assertTrue(is_natural_exit("new topic please"))
        self.assertTrue(is_natural_exit("quit practice"))
        self.assertFalse(is_natural_exit("The correct answer is Option B because Lisinopril is an ACE inhibitor"))

    def test_extract_option_selection(self):
        self.assertEqual(extract_option_selection("A"), "A")
        self.assertEqual(extract_option_selection("Option B: Metoprolol"), "B")
        self.assertEqual(extract_option_selection("I choose C."), "C")
        self.assertEqual(extract_option_selection("d"), "D")
        self.assertIsNone(extract_option_selection("I don't know the exact answer, maybe calcium channel blockers?"))

    # ── 2. Conversational Viva Flow Tests ──────────────────────────────────────

    @patch("services.langchain_service.LangChainGeminiChatAdapter.execute_chat_turn")
    def test_viva_initiates_with_single_question_and_profile_adaptation(self, mock_chat_turn):
        user = self._create_user(academic_year="2nd_year", target_exam="university_profs")
        self._login(user.email)

        mock_viva_output = {
            "answer": "### Viva Voce: Acute Septic Shock\n\n**Examiner Question 1:** A 45-year-old patient presents with fever, tachycardia, and hypotension refractory to initial fluid boluses. What is the fundamental hemodynamic definition of septic shock, and which vascular receptor mechanisms are impaired?",
            "cited_passage_ids": ["surgery_acute_abdomen:p0008"],
            "evidence_sufficiency": "sufficient",
            "practice_mode": "viva_voce",
        }
        mock_chat_turn.return_value = (
            mock_viva_output,
            "```json\n" + json.dumps(mock_viva_output) + "\n```",
            250,
        )

        res = self.client.post("/chat", json={"message": "Simulate a viva voce on Acute Shock"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["practice_mode"], "viva_voce")
        self.assertIn("Examiner Question 1", data["response"])

        # Verify prompt received 2nd year profile context
        call_kwargs = mock_chat_turn.call_args.kwargs
        sys_inst = call_kwargs.get("system_instruction", "")
        self.assertIn("2nd_year", sys_inst)
        self.assertIn("university_profs", sys_inst)

    @patch("services.langchain_service.LangChainGeminiChatAdapter.execute_chat_turn")
    def test_viva_evaluates_student_answer_with_strengths_omissions_misconceptions(self, mock_chat_turn):
        user = self._create_user()
        self._login(user.email)

        # Setup conversation with initial viva question in history
        conv = Conversation(user_id=user.id, title="Septic Shock Viva")
        db.session.add(conv)
        db.session.commit()

        initial_viva_msg = ChatMessage(
            conversation_id=conv.id,
            role="model",
            content="### Viva Voce: Question 1\nWhat are the initial resuscitation fluid choices in septic shock according to Surviving Sepsis guidelines?",
            evidence={"practice_mode": "viva_voce", "viva_stage": 1},
        )
        db.session.add(initial_viva_msg)
        db.session.commit()

        mock_feedback_output = {
            "answer": "### Examiner Evaluation\n\n- **Correct Points:** Accurately identified balanced crystalloids (Plasmalyte/Ringer's Lactate) at 30 mL/kg.\n- **Omissions:** Omitted the target mean arterial pressure (MAP >= 65 mmHg) and timing within the first 3 hours.\n- **Misconceptions Addressed:** Normal saline in large volumes is not preferred due to hyperchloremic metabolic acidosis risk.\n\n**Examiner Question 2:** If MAP remains below 65 mmHg after 30 mL/kg fluid resuscitation, which first-line vasopressor would you initiate and why?",
            "cited_passage_ids": ["surgery_acute_abdomen:p0008"],
            "evidence_sufficiency": "sufficient",
            "practice_mode": "viva_voce",
        }
        mock_chat_turn.return_value = (
            mock_feedback_output,
            "```json\n" + json.dumps(mock_feedback_output) + "\n```",
            280,
        )

        student_answer = "We give Ringer's lactate or normal saline as a 30 ml/kg bolus immediately."
        res = self.client.post("/chat", json={"message": student_answer, "conversation_id": conv.id})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["practice_mode"], "viva_voce")
        self.assertIn("Correct Points", data["response"])
        self.assertIn("Omissions", data["response"])
        self.assertIn("Misconceptions Addressed", data["response"])
        self.assertIn("Examiner Question 2", data["response"])

    # ── 3. Interactive MCQ Generation & Scoring Tests ─────────────────────────

    @patch("services.langchain_service.LangChainGeminiChatAdapter.execute_chat_turn")
    def test_mcq_generation_with_defensible_key_and_practice_label(self, mock_chat_turn):
        user = self._create_user(academic_year="final_year", target_exam="neet_pg_next")
        self._login(user.email)

        mock_mcq_output = {
            "answer": "A 58-year-old male with chronic heart failure (reduced ejection fraction 30%) presents with persistent cough and peripheral edema. He is already taking Carvedilol and Spironolactone. Which medication is the most appropriate next guideline-directed therapy to reduce mortality?",
            "cited_passage_ids": ["medicine_cardiovascular:p0005"],
            "evidence_sufficiency": "sufficient",
            "practice_mode": "mcq_practice",
            "interactive_mcq": {
                "question_id": "mcq_cv_001",
                "stem": "Which medication is the most appropriate next guideline-directed therapy to reduce mortality in HFrEF?",
                "options": [
                    {"key": "A", "text": "Diltiazem"},
                    {"key": "B", "text": "Sacubitril / Valsartan (ARNI)"},
                    {"key": "C", "text": "Digoxin monotherapy"},
                    {"key": "D", "text": "Amlodipine"}
                ],
                "correct_option": "B",
                "subject": "Medicine / Cardiology",
                "competency": "IM8.11 Guideline-Directed Medical Therapy in Heart Failure",
                "disclaimer": "MBBS Practice Question (Not Official Exam Content)",
                "answered": False,
            }
        }
        mock_chat_turn.return_value = (
            mock_mcq_output,
            "```json\n" + json.dumps(mock_mcq_output) + "\n```",
            300,
        )

        res = self.client.post("/chat", json={"message": "Give me an MCQ on heart failure therapy"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["practice_mode"], "mcq_practice")
        self.assertIn("interactive_mcq", data)
        mcq = data["interactive_mcq"]
        self.assertEqual(mcq["question_id"], "mcq_cv_001")
        self.assertEqual(len(mcq["options"]), 4)
        self.assertEqual(mcq["correct_option"], "B")
        self.assertIn("Not Official Exam Content", mcq["disclaimer"])

    @patch("services.langchain_service.LangChainGeminiChatAdapter.execute_chat_turn")
    def test_mcq_option_submission_via_chat_flow_and_score(self, mock_chat_turn):
        user = self._create_user()
        self._login(user.email)

        conv = Conversation(user_id=user.id, title="Cardiology MCQ Session")
        db.session.add(conv)
        db.session.commit()

        # Seed conversation with active MCQ
        mcq_payload = {
            "question_id": "mcq_cv_001",
            "stem": "Which drug is guideline-directed first-line therapy?",
            "options": [
                {"key": "A", "text": "Diltiazem"},
                {"key": "B", "text": "Sacubitril / Valsartan (ARNI)"},
                {"key": "C", "text": "Digoxin"},
                {"key": "D", "text": "Amlodipine"}
            ],
            "correct_option": "B",
            "explanation": "ARNI (Sacubitril/Valsartan) reduces all-cause mortality and heart failure hospitalization compared to ACE inhibitors alone.",
            "subject": "Cardiology",
            "competency": "IM8.11",
            "disclaimer": "MBBS Practice Question (Not Official Exam Content)",
            "answered": False,
        }
        bot_msg = ChatMessage(
            conversation_id=conv.id,
            role="model",
            content="A 58-year-old male with HFrEF requires next GDMT step.",
            evidence={"interactive_mcq": mcq_payload, "practice_mode": "mcq_practice"},
        )
        db.session.add(bot_msg)
        db.session.commit()

        # Student submits Option B
        res = self.client.post("/chat", json={
            "message": "Option B: Sacubitril / Valsartan",
            "selected_option": "B",
            "question_id": "mcq_cv_001",
            "conversation_id": conv.id
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["practice_mode"], "mcq_practice")
        self.assertIn("Correct", data["response"])
        self.assertIn("Sacubitril / Valsartan", data["response"])
        self.assertTrue(data["interactive_mcq"]["answered"])
        self.assertEqual(data["interactive_mcq"]["user_answer"], "B")

    @patch("services.langchain_service.LangChainGeminiChatAdapter.execute_chat_turn")
    def test_mcq_option_submission_text_fallback(self, mock_chat_turn):
        user = self._create_user()
        self._login(user.email)

        conv = Conversation(user_id=user.id, title="Cardiology MCQ Session Text Fallback")
        db.session.add(conv)
        db.session.commit()

        mcq_payload = {
            "question_id": "mcq_cv_002",
            "stem": "Which drug causes dry cough?",
            "options": [
                {"key": "A", "text": "Lisinopril"},
                {"key": "B", "text": "Losartan"},
                {"key": "C", "text": "Amlodipine"},
                {"key": "D", "text": "Atenolol"}
            ],
            "correct_option": "A",
            "explanation": "Lisinopril is an ACE inhibitor that leads to bradykinin accumulation causing dry cough.",
            "disclaimer": "MBBS Practice Question (Not Official Exam Content)",
            "answered": False,
        }
        bot_msg = ChatMessage(
            conversation_id=conv.id,
            role="model",
            content="Which antihypertensive drug causes dry cough?",
            evidence={"interactive_mcq": mcq_payload, "practice_mode": "mcq_practice"},
        )
        db.session.add(bot_msg)
        db.session.commit()

        # Student types just "A" without extraPayload
        res = self.client.post("/chat", json={
            "message": "Option A: Lisinopril",
            "conversation_id": conv.id
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["practice_mode"], "mcq_practice")
        self.assertIn("Correct", data["response"])
        self.assertTrue(data["interactive_mcq"]["answered"])
        self.assertEqual(data["interactive_mcq"]["user_answer"], "A")

    def test_duplicate_mcq_submission_prevention_on_dedicated_endpoint(self):
        user = self._create_user()
        self._login(user.email)

        conv = Conversation(user_id=user.id, title="Dedicate Endpoint MCQ")
        db.session.add(conv)
        db.session.commit()

        mcq_payload = {
            "question_id": "mcq_cv_003",
            "stem": "Identify the first-line medication for STEMI antiplatelet therapy.",
            "options": [
                {"key": "A", "text": "Aspirin 300 mg chewable"},
                {"key": "B", "text": "Warfarin 5 mg"},
                {"key": "C", "text": "Dipyridamole"},
                {"key": "D", "text": "Heparin only"}
            ],
            "correct_option": "A",
            "explanation": "Chewable non-enteric coated aspirin is immediately administered upon STEMI suspicion.",
            "disclaimer": "MBBS Practice Question (Not Official Exam Content)",
            "answered": False,
        }
        bot_msg = ChatMessage(
            conversation_id=conv.id,
            role="model",
            content="STEMI antiplatelet question.",
            evidence={"interactive_mcq": mcq_payload, "practice_mode": "mcq_practice"},
        )
        db.session.add(bot_msg)
        db.session.commit()

        # 1st Submission: should score successfully
        res1 = self.client.post(f"/api/conversations/{conv.id}/mcq/answer", json={
            "question_id": "mcq_cv_003",
            "selected_option": "A"
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.get_json()
        self.assertTrue(data1["is_correct"])
        self.assertFalse(data1.get("already_answered", False))

        # 2nd Submission (Duplicate): must be prevented from re-scoring
        res2 = self.client.post(f"/api/conversations/{conv.id}/mcq/answer", json={
            "question_id": "mcq_cv_003",
            "selected_option": "A"
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.get_json()
        self.assertTrue(data2.get("already_answered"))
        self.assertIn("already recorded", data2.get("message", "").lower())

    # ── 4. Clinical Case Discussion Progressive Disclosure Tests ──────────────

    @patch("services.langchain_service.LangChainGeminiChatAdapter.execute_chat_turn")
    def test_clinical_case_stage_one_presentation_with_supervisory_disclaimer(self, mock_chat_turn):
        user = self._create_user(academic_year="final_year")
        self._login(user.email)

        mock_case_output = {
            "answer": "### Clinical Case Discussion · Stage 1: Initial Presentation\n\n**Patient Profile:** A 42-year-old female presents to the triage unit with severe, boring epigastric pain of 8 hours duration that radiates straight through to her back. She has had two episodes of non-bilious vomiting.\n\n**Vital Signs:**\n- BP: 100/60 mmHg, HR: 112 bpm (regular), RR: 22 /min, Temp: 38.1°C, SpO2: 96% on room air.\n\n**Clinical Reasoning Task:** What are your top 3 differential diagnoses based on this presentation, and what initial bedside examination signs would you check?\n\n> *Note: Educational simulation for MBBS students. All actual patient management decisions must be made under qualified clinical supervision.*",
            "cited_passage_ids": ["surgery_acute_abdomen:p0002"],
            "evidence_sufficiency": "sufficient",
            "practice_mode": "clinical_case",
            "clinical_case": {
                "case_id": "case_panc_001",
                "case_title": "Severe Acute Epigastric Pain in a 42-Year-Old Female",
                "current_stage": 1,
                "disclaimer": "Educational simulation only. Actual clinical management requires qualified medical supervision.",
            }
        }
        mock_chat_turn.return_value = (
            mock_case_output,
            "```json\n" + json.dumps(mock_case_output) + "\n```",
            320,
        )

        res = self.client.post("/chat", json={"message": "Let's do a clinical case discussion on epigastric pain"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["practice_mode"], "clinical_case")
        self.assertIn("Stage 1", data["response"])
        self.assertIn("Vital Signs", data["response"])
        self.assertIn("clinical supervision", data["response"].lower())
        self.assertIn("clinical_case", data)
        self.assertEqual(data["clinical_case"]["current_stage"], 1)

    @patch("services.langchain_service.LangChainGeminiChatAdapter.execute_chat_turn")
    def test_clinical_case_progressive_advance_to_physical_exam_and_labs(self, mock_chat_turn):
        user = self._create_user()
        self._login(user.email)

        conv = Conversation(user_id=user.id, title="Epigastric Pain Case")
        db.session.add(conv)
        db.session.commit()

        initial_case_msg = ChatMessage(
            conversation_id=conv.id,
            role="model",
            content="Stage 1: 42yo female with severe epigastric pain radiating to back.",
            evidence={
                "clinical_case": {
                    "case_id": "case_panc_001",
                    "current_stage": 1,
                },
                "practice_mode": "clinical_case",
            },
        )
        db.session.add(initial_case_msg)
        db.session.commit()

        mock_stage2_output = {
            "answer": "### Clinical Case Discussion · Stage 2: Physical Examination Findings\n\n- Abdomen: Marked epigastric tenderness with voluntary guarding; no rigidity. Absent bowel sounds.\n- Cullen's sign and Grey Turner's signs are absent at this point.\n\n**Next Step Reasoning:** Which urgent diagnostic investigations and laboratory tests would you order to confirm your primary differential diagnosis?",
            "cited_passage_ids": ["surgery_acute_abdomen:p0002"],
            "evidence_sufficiency": "sufficient",
            "practice_mode": "clinical_case",
            "clinical_case": {
                "case_id": "case_panc_001",
                "current_stage": 2,
            }
        }
        mock_chat_turn.return_value = (
            mock_stage2_output,
            "```json\n" + json.dumps(mock_stage2_output) + "\n```",
            260,
        )

        res = self.client.post("/chat", json={
            "message": "My primary differential is Acute Pancreatitis, followed by Perforated Peptic Ulcer. I would examine for abdominal guarding and Grey Turner sign.",
            "conversation_id": conv.id
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["practice_mode"], "clinical_case")
        self.assertIn("Stage 2", data["response"])
        self.assertEqual(data["clinical_case"]["current_stage"], 2)

    # ── 5. Natural Transition & Safety Escalation Tests ────────────────────────

    @patch("services.langchain_service.LangChainGeminiChatAdapter.execute_chat_turn")
    def test_natural_exit_switches_back_to_general_tutoring(self, mock_chat_turn):
        user = self._create_user()
        self._login(user.email)

        conv = Conversation(user_id=user.id, title="Viva with Natural Exit")
        db.session.add(conv)
        db.session.commit()

        viva_msg = ChatMessage(
            conversation_id=conv.id,
            role="model",
            content="Examiner Question 3: Detail the SIRS criteria.",
            evidence={"practice_mode": "viva_voce", "viva_stage": 3},
        )
        db.session.add(viva_msg)
        db.session.commit()

        mock_tutoring_output = {
            "answer": "We've exited viva practice mode. Here is the concise comparison of typical vs atypical antipsychotics...",
            "cited_passage_ids": ["pharmacology_autonomic:p0012"],
            "evidence_sufficiency": "sufficient",
            "practice_mode": "general_tutoring",
        }
        mock_chat_turn.return_value = (
            mock_tutoring_output,
            "```json\n" + json.dumps(mock_tutoring_output) + "\n```",
            220,
        )

        res = self.client.post("/chat", json={
            "message": "Stop viva. Let's return to ordinary tutoring and compare typical vs atypical antipsychotics.",
            "conversation_id": conv.id
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["practice_mode"], "general_tutoring")
        self.assertIn("exited viva practice mode", data["response"])

    @patch("services.llm_service.LLMService.get_response")
    def test_safety_escalation_takes_priority_over_interactive_modes(self, mock_llm_response):
        """First-person emergency symptoms must trigger the red-flag emergency card even if student asks for a quiz."""
        user = self._create_user()
        self._login(user.email)

        # Student mentions acute emergency symptoms happening to them
        emergency_msg = "I have sudden severe crushing chest pain radiating to my left arm and I cannot breathe. Quiz me on ECG signs."
        res = self.client.post("/chat", json={"message": emergency_msg})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Must trigger emergency triage and warning
        triage = data.get("triage", {})
        self.assertTrue(triage.get("is_emergency"))
        self.assertEqual(triage.get("severity"), "EMERGENCY")
        self.assertIn("112", data.get("response"))
        # Model should NOT be invoked for exam quiz when active patient emergency exists
        mock_llm_response.assert_not_called()

    # ── 6. Conversation Persistence Tests ─────────────────────────────────────

    def test_interactive_mcq_state_persisted_and_retrievable(self):
        user = self._create_user()
        self._login(user.email)

        conv = Conversation(user_id=user.id, title="Persisted MCQ Session")
        db.session.add(conv)
        db.session.commit()

        mcq_data = {
            "question_id": "mcq_persisted_001",
            "stem": "Which antibiotic is first line for uncomplicated cystitis?",
            "options": [
                {"key": "A", "text": "Nitrofurantoin"},
                {"key": "B", "text": "Ciprofloxacin"},
                {"key": "C", "text": "Amikacin"},
                {"key": "D", "text": "Vancomycin"}
            ],
            "correct_option": "A",
            "answered": True,
            "user_answer": "A",
            "disclaimer": "MBBS Practice Question (Not Official Exam Content)",
        }
        msg = ChatMessage(
            conversation_id=conv.id,
            role="model",
            content="Here is your practice question.",
            evidence={"interactive_mcq": mcq_data, "practice_mode": "mcq_practice"},
        )
        db.session.add(msg)
        db.session.commit()

        # Retrieve conversation via API
        res = self.client.get(f"/api/conversations/{conv.id}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        messages = data["conversation"]["messages"]
        self.assertEqual(len(messages), 1)
        persisted_evidence = messages[0]["evidence"]
        self.assertIn("interactive_mcq", persisted_evidence)
        self.assertTrue(persisted_evidence["interactive_mcq"]["answered"])
        self.assertEqual(persisted_evidence["interactive_mcq"]["user_answer"], "A")


if __name__ == "__main__":
    unittest.main()
