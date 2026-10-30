"""Deterministic tests for MediAssist symptom guidance and emergency safety."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from safety.evaluator import SymptomSafetyEvaluator
from safety.red_flags import match_red_flags


class TestSymptomSafetyEvaluator(unittest.TestCase):
    """Check safety behavior without calling Gemini or the RAG service."""

    def setUp(self):
        self.evaluator = SymptomSafetyEvaluator()

    def test_common_symptom_requests_structured_follow_ups(self):
        result = self.evaluator.evaluate("I have had a headache for two days")

        self.assertFalse(result["is_emergency"])
        self.assertEqual(result["triage"]["severity"], "ROUTINE_GUIDANCE")
        self.assertIn("DURATION", result["prompt_directive"])
        self.assertIn("SEVERITY", result["prompt_directive"])
        self.assertIn("ASSOCIATED SIGNS", result["prompt_directive"])
        self.assertIn("lack of a critical red flag", result["prompt_directive"])

    def test_ambiguous_symptom_stays_uncertain(self):
        result = self.evaluator.evaluate("I feel weak and dizzy sometimes")

        self.assertFalse(result["is_emergency"])
        self.assertEqual(result["triage"]["severity"], "ROUTINE_GUIDANCE")
        self.assertIn("consult", result["triage"]["advisory"].lower())

    def test_general_unsupported_question_remains_available(self):
        result = self.evaluator.evaluate("How does a binary search tree work?")

        self.assertFalse(result["is_emergency"])
        self.assertEqual(result["triage"]["severity"], "GENERAL")
        self.assertEqual(result["triage"]["emergency_contacts"], [])

    def test_emergency_warning_contains_india_guidance(self):
        result = self.evaluator.evaluate("I cannot breathe and my lips are blue")

        self.assertTrue(result["is_emergency"])
        self.assertEqual(result["triage"]["severity"], "EMERGENCY")
        self.assertIn("112", result["triage"]["full_guidance"])
        self.assertIn("108", result["triage"]["full_guidance"])
        self.assertIn("Do not wait", result["triage"]["full_guidance"])

    def test_missing_red_flag_match_is_not_a_safety_claim(self):
        result = self.evaluator.evaluate("I have an unusual symptom that is hard to describe")

        self.assertFalse(result["is_emergency"])
        self.assertIn("lack of a critical red flag", result["prompt_directive"])
        self.assertIn("urgent", result["prompt_directive"].lower())

    def test_red_flag_rules_are_deterministic(self):
        match = match_red_flags("sudden weakness on my right side and slurred speech")

        self.assertIsNotNone(match)
        self.assertEqual(match["rule_id"], "STROKE_01")

    def test_emergency_response_does_not_call_gemini(self):
        try:
            import app
        except SystemExit:
            self.skipTest("GEMINI_API_KEY is not configured")

        with patch.object(app._client.models, "generate_content") as generate_content:
            response, sources, triage = app.get_gemini_response(
                "I cannot breathe and my lips are blue"
            )

        generate_content.assert_not_called()
        self.assertEqual(sources, [])
        self.assertTrue(triage["is_emergency"])
        self.assertIn("112", response)


    def test_academic_emergency_inquiry_does_not_trigger_emergency_blockade(self):
        result = self.evaluator.evaluate("Explain the emergency management protocol for acute myocardial infarction")

        self.assertFalse(result["is_emergency"])
        self.assertEqual(result["triage"]["severity"], "ACADEMIC_DISCUSSION")
        self.assertEqual(result["triage"]["intent"], "academic_emergency")
        self.assertIn("ACADEMIC EMERGENCY MEDICINE DIRECTIVE", result["prompt_directive"])
        self.assertIn("pathophysiological mechanisms", result["prompt_directive"])

    def test_clinical_case_vignette_does_not_trigger_emergency_blockade(self):
        result = self.evaluator.evaluate(
            "A 55-year-old male presents with crushing substernal chest pain radiating to left arm. "
            "What is the initial investigation and gold-standard treatment?"
        )

        self.assertFalse(result["is_emergency"])
        self.assertEqual(result["triage"]["severity"], "ACADEMIC_DISCUSSION")
        self.assertEqual(result["triage"]["intent"], "academic_emergency")

    def test_third_person_real_emergency_triggers_advisory(self):
        result = self.evaluator.evaluate("My father collapsed, is gasping for breath and his lips are blue")

        self.assertTrue(result["is_emergency"])
        self.assertEqual(result["triage"]["severity"], "EMERGENCY")
        self.assertEqual(result["triage"]["intent"], "third_person_emergency")
        self.assertIn("ASSISTING SOMEONE IN DISTRESS", result["triage"]["full_guidance"])
        self.assertIn("112", result["triage"]["full_guidance"])
        self.assertIn("108", result["triage"]["full_guidance"])

    def test_mixed_intent_symptoms_with_academic_question_triggers_emergency_advisory(self):
        result = self.evaluator.evaluate(
            "I have severe crushing chest pain right now, can you explain the pathophysiology of STEMI?"
        )

        self.assertTrue(result["is_emergency"])
        self.assertEqual(result["triage"]["severity"], "EMERGENCY")
        self.assertEqual(result["triage"]["intent"], "mixed_intent")
        self.assertIn("ACUTE DISTRESS DETECTED", result["triage"]["full_guidance"])
        self.assertIn("112", result["triage"]["full_guidance"])

    def test_academic_emergency_allows_gemini_generation(self):
        try:
            import app
        except SystemExit:
            self.skipTest("GEMINI_API_KEY is not configured")

        with patch.object(app._client.models, "generate_content") as generate_content:
            from unittest.mock import MagicMock
            mock_resp = MagicMock()
            mock_resp.text = '{"answer": "STEMI emergency protocol involves aspirin, ticagrelor, and immediate PCI within 90 minutes.", "cited_passage_ids": [], "evidence_sufficiency": "sufficient"}'
            mock_resp.usage_metadata = None
            generate_content.return_value = mock_resp

            response, sources, triage = app.get_gemini_response(
                "Explain the emergency management protocol for acute myocardial infarction"
            )

        generate_content.assert_called_once()
        self.assertFalse(triage["is_emergency"])
        self.assertEqual(triage["severity"], "ACADEMIC_DISCUSSION")
        self.assertIn("aspirin", response.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)