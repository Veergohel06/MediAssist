"""
MediAssist Safety & Triage Module
=================================
Clinician-reviewed deterministic red flag emergency rules, symptom safety evaluation,
and emergency contact dispatch.
"""

from .red_flags import RED_FLAG_RULES, match_red_flags
from .emergency_contacts import INDIA_EMERGENCY_CONTACTS, GLOBAL_EMERGENCY_REFERENCES, get_emergency_guidance_text
from .evaluator import SymptomSafetyEvaluator

__all__ = [
    "RED_FLAG_RULES",
    "match_red_flags",
    "INDIA_EMERGENCY_CONTACTS",
    "GLOBAL_EMERGENCY_REFERENCES",
    "get_emergency_guidance_text",
    "SymptomSafetyEvaluator",
]
