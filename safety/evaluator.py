"""
Symptom Safety Evaluator Module
===============================
Analyzes user messages for clinical emergencies and active symptom descriptions,
injects safety directives into the AI prompt, and generates structured triage data.

Critical Safety Invariant:
"A missing keyword match is NEVER proof that a symptom is safe or benign."
"""

import re
import logging
from typing import Dict, Any, Optional
from .red_flags import match_red_flags, classify_safety_context
from .emergency_contacts import INDIA_EMERGENCY_CONTACTS, get_emergency_guidance_text

logger = logging.getLogger("mediassist.safety.evaluator")

# Common symptom indicators to distinguish general health questions from personal symptom descriptions
SYMPTOM_INDICATORS = [
    r"\b(i\s+have|i'?m\s+having|i\s+feel|i\s+felt|my\s+.*\s+(hurts|aches|is\s+sore))\b",
    r"\b(symptom|symptoms|pain|ache|aching|hurt|hurting|fever|cough|nausea|vomit|dizzy|dizziness)\b",
    r"\b(headache|migraine|chest\s+pain|stomach\s+ache|belly\s+pain|throat\s+pain|sore\s+throat)\b",
    r"\b(tired|fatigue|exhausted|weakness|rash|swelling|cramp|cramps|chills|congestion|diarrhea)\b",
    r"\b(bleeding|short\s+of\s+breath|breathless|palpitation|palpitations|numbness|tingling)\b",
]


class SymptomSafetyEvaluator:
    """
    Evaluates incoming user text for safety risks, emergency warning signs,
    and active symptom descriptions with dual-mode awareness for MBBS students.
    """

    @staticmethod
    def is_describing_symptoms(text: str) -> bool:
        """
        Check if the user is likely describing personal symptoms rather than
        asking an abstract or general question.
        """
        clean_text = text.lower().strip()
        for pattern in SYMPTOM_INDICATORS:
            if re.search(pattern, clean_text):
                return True
        return False

    def evaluate(self, user_message: str) -> Dict[str, Any]:
        """
        Run the complete safety evaluation:
        1. Check deterministic clinician-reviewed emergency red flags.
        2. Classify intent: first-person, third-person, mixed, or academic discussion.
        3. Detect active symptom descriptions.
        4. Formulate safety directives and structured triage payload.
        """
        red_flag_match = match_red_flags(user_message)
        intent, is_emergency = classify_safety_context(user_message, red_flag_match)
        has_symptoms = self.is_describing_symptoms(user_message)

        if is_emergency and red_flag_match:
            # ── ACUTE CLINICAL EMERGENCY SCENARIO ────────────────────────────
            logger.warning(
                "SAFETY ALERT: Emergency red flag triggered (%s: %s, intent=%s)",
                red_flag_match["rule_id"], red_flag_match["condition"], intent
            )

            guidance_text = get_emergency_guidance_text("India", intent=intent)
            advisory = (
                f"🚨 URGENT MEDICAL WARNING: Symptoms described suggest a potential {red_flag_match['condition']}. "
                "Please call 112 or 108 (India) or your local emergency number immediately. Do not attempt to self-treat."
            )

            prompt_directive = (
                "--- CRITICAL SAFETY DIRECTIVE: MEDICAL EMERGENCY DETECTED ---\n"
                f"The user has described symptoms matching an acute emergency condition: {red_flag_match['condition']} "
                f"(Category: {red_flag_match['category']}, Intent: {intent}).\n"
                "MANDATORY INSTRUCTIONS:\n"
                "1. Immediately and prominently advise calling emergency medical services without delay "
                "(Dial 112 or 108 in India; 911/999/112 internationally).\n"
                "2. Explicitly urge them NOT to drive themselves or delay clinical care.\n"
                "3. Emphasize that MediAssist cannot provide emergency intervention, dispatch an ambulance, or acute medical diagnosis.\n"
                "4. Do NOT suggest home remedies, waiting periods, or unverified medications.\n"
                "--- END CRITICAL SAFETY DIRECTIVE ---\n"
            )

            triage_payload = {
                "is_emergency": True,
                "severity": "EMERGENCY",
                "intent": intent,
                "condition": red_flag_match["condition"],
                "category": red_flag_match["category"],
                "advisory": advisory,
                "emergency_contacts": INDIA_EMERGENCY_CONTACTS,
                "full_guidance": guidance_text,
            }

            return {
                "triage": triage_payload,
                "prompt_directive": prompt_directive,
                "is_emergency": True,
            }

        elif red_flag_match and intent == "academic_emergency":
            # ── ACADEMIC EMERGENCY MEDICINE DISCUSSION SCENARIO ──────────────
            logger.info(
                "Safety check: Academic emergency medicine discussion detected (%s: %s).",
                red_flag_match["rule_id"], red_flag_match["condition"]
            )

            prompt_directive = (
                "--- ACADEMIC EMERGENCY MEDICINE DIRECTIVE ---\n"
                f"The user is an MBBS student inquiring academically about an acute emergency topic: {red_flag_match['condition']} "
                f"({red_flag_match['category']}).\n"
                "MANDATORY INSTRUCTIONS:\n"
                "1. Provide a rigorous, academic MBBS-level explanation covering pathophysiological mechanisms, immediate emergency stabilization (ABCDE), diagnostic criteria (e.g. ECG changes, biomarkers, imaging), and evidence-based management protocols (e.g. ACLS/ATLS, first-line pharmacotherapy, urgent interventional procedures).\n"
                "2. Frame the response for medical education and exam preparation (CBME competencies, NEET-PG/NEXT, USMLE).\n"
                "3. Do not provide personalized medical advice or address the user as an active patient.\n"
                "--- END ACADEMIC EMERGENCY MEDICINE DIRECTIVE ---\n"
            )

            triage_payload = {
                "is_emergency": False,
                "severity": "ACADEMIC_DISCUSSION",
                "intent": "academic_emergency",
                "condition": red_flag_match["condition"],
                "category": red_flag_match["category"],
                "advisory": "Academic discussion of acute emergency condition. For educational and clinical training purposes only.",
                "emergency_contacts": [],
                "full_guidance": "",
            }

            return {
                "triage": triage_payload,
                "prompt_directive": prompt_directive,
                "is_emergency": False,
            }

        elif has_symptoms:
            # ── NON-EMERGENCY SYMPTOM SCENARIO ───────────────────────────────
            logger.info("Safety check: Non-emergency symptom inquiry detected.")

            prompt_directive = (
                "--- SYMPTOM GUIDANCE SAFETY PROTOCOL ---\n"
                "The user is describing personal health symptoms.\n"
                "IMPORTANT SAFETY RULES:\n"
                "1. Do NOT interpret the lack of a critical red flag as proof that the symptom is safe or benign.\n"
                "2. Clarifying Follow-ups: If the user has not provided sufficient context, ask gentle, structured questions about:\n"
                "   - DURATION: 'How long have you been experiencing this?'\n"
                "   - SEVERITY: 'How severe or uncomfortable does it feel on a scale of 1 to 10?'\n"
                "   - ASSOCIATED SIGNS: 'Are there any accompanying symptoms, such as fever, dizziness, or localized pain?'\n"
                "3. Educational Insight: Provide general medical information explaining what typically causes such symptoms.\n"
                "4. Next Steps: Recommend a routine consultation with a healthcare professional, and remind them to seek prompt or urgent evaluation if symptoms worsen, feel severe, or persist.\n"
                "5. Strict Limits: Do NOT provide a definitive diagnosis or recommend drug dosages.\n"
                "--- END SYMPTOM GUIDANCE SAFETY PROTOCOL ---\n"
            )

            triage_payload = {
                "is_emergency": False,
                "severity": "ROUTINE_GUIDANCE",
                "condition": "General Symptom Inquiry",
                "category": "Symptom Assessment",
                "advisory": "Educational guidance only. Always consult a healthcare provider for diagnosis or persistent symptoms.",
                "emergency_contacts": [],
                "full_guidance": "",
            }

            return {
                "triage": triage_payload,
                "prompt_directive": prompt_directive,
                "is_emergency": False,
            }

        else:
            # ── GENERAL HEALTH QUESTION ──────────────────────────────────────
            prompt_directive = (
                "--- GENERAL HEALTH EDUCATION DIRECTIVE ---\n"
                "The user is asking a general health/wellness question. Provide clear, empathetic, and evidence-based information. "
                "Remind the user that MediAssist is an educational assistant.\n"
                "--- END DIRECTIVE ---\n"
            )

            triage_payload = {
                "is_emergency": False,
                "severity": "GENERAL",
                "condition": "General Health Education",
                "category": "General Education",
                "advisory": "",
                "emergency_contacts": [],
                "full_guidance": "",
            }

            return {
                "triage": triage_payload,
                "prompt_directive": prompt_directive,
                "is_emergency": False,
            }
