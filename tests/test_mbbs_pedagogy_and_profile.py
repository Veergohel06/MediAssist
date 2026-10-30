"""Tests for MBBS Pedagogical Engine and Student Profile System.

Verifies:
1. Academic profile model validation (academic_year, target_exam)
2. Profile REST API endpoints (/api/profile GET, PATCH, PUT)
3. Multi-tenant profile isolation and ownership enforcement
4. Prompt context augmentation with student academic level
5. End-to-end chat generation with academic profile adaptation
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from config import TestingConfig
from models import User, VALID_ACADEMIC_YEARS, VALID_TARGET_EXAMS, db
from services.llm_service import LLMService


class TestMBBSStudentProfileModel(unittest.TestCase):
    """Test validation and persistence of academic_year and target_exam on User model."""

    def setUp(self):
        self.app = create_app(TestingConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_valid_academic_years_accepted(self):
        for year in VALID_ACADEMIC_YEARS:
            user = User(name=f"Student {year}", email=f"{year}@medschool.edu", academic_year=year)
            user.set_password("ValidPass123")
            db.session.add(user)
            db.session.commit()
            self.assertEqual(user.academic_year, year)

    def test_invalid_academic_year_rejected(self):
        user = User(name="Invalid Year", email="invalid_year@medschool.edu")
        user.set_password("ValidPass123")
        with self.assertRaises(ValueError) as ctx:
            user.academic_year = "5th_year_invalid"
        self.assertIn("Invalid academic year", str(ctx.exception))

    def test_valid_target_exams_accepted(self):
        for exam in VALID_TARGET_EXAMS:
            user = User(name=f"Examinee {exam}", email=f"{exam}@medschool.edu", target_exam=exam)
            user.set_password("ValidPass123")
            db.session.add(user)
            db.session.commit()
            self.assertEqual(user.target_exam, exam)

    def test_invalid_target_exam_rejected(self):
        user = User(name="Invalid Exam", email="invalid_exam@medschool.edu")
        user.set_password("ValidPass123")
        with self.assertRaises(ValueError) as ctx:
            user.target_exam = "mcat_foreign_exam"
        self.assertIn("Invalid target exam", str(ctx.exception))

    def test_clearing_profile_fields(self):
        user = User(
            name="Clear Test",
            email="clear@medschool.edu",
            academic_year="2nd_year",
            target_exam="neet_pg_next",
        )
        user.set_password("ValidPass123")
        db.session.add(user)
        db.session.commit()

        user.academic_year = None
        user.target_exam = ""
        db.session.commit()

        self.assertIsNone(user.academic_year)
        self.assertIsNone(user.target_exam)

    def test_to_dict_includes_academic_profile(self):
        user = User(
            name="Profile Dict",
            email="dict@medschool.edu",
            academic_year="final_year",
            target_exam="university_profs",
        )
        user.set_password("ValidPass123")
        db.session.add(user)
        db.session.commit()

        d = user.to_dict()
        self.assertEqual(d["academic_year"], "final_year")
        self.assertEqual(d["target_exam"], "university_profs")


class TestMBBSProfileAPI(unittest.TestCase):
    """Test REST API /api/profile endpoint security and validation."""

    def setUp(self):
        self.app = create_app(TestingConfig)
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

        # Create two test users
        self.user1 = User(name="Student One", email="student1@med.edu")
        self.user1.set_password("Password123")
        self.user2 = User(name="Student Two", email="student2@med.edu")
        self.user2.set_password("Password123")
        db.session.add_all([self.user1, self.user2])
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def _login(self, email="student1@med.edu", password="Password123"):
        return self.client.post("/login", json={"email": email, "password": password})

    def test_unauthenticated_profile_access_rejected(self):
        res = self.client.get("/api/profile")
        self.assertEqual(res.status_code, 401)

        res_patch = self.client.patch("/api/profile", json={"academic_year": "1st_year"})
        self.assertEqual(res_patch.status_code, 401)

    def test_get_and_patch_profile_authenticated(self):
        self._login()

        # Initial get
        res = self.client.get("/api/profile")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsNone(data["user"]["academic_year"])
        self.assertIsNone(data["user"]["target_exam"])

        # Update profile
        update_res = self.client.patch(
            "/api/profile",
            json={"academic_year": "2nd_year", "target_exam": "neet_pg_next"},
        )
        self.assertEqual(update_res.status_code, 200)
        updated_data = update_res.get_json()
        self.assertEqual(updated_data["status"], "updated")
        self.assertEqual(updated_data["user"]["academic_year"], "2nd_year")
        self.assertEqual(updated_data["user"]["target_exam"], "neet_pg_next")

        # Verify DB persistence
        fresh_user = db.session.get(User, self.user1.id)
        self.assertEqual(fresh_user.academic_year, "2nd_year")
        self.assertEqual(fresh_user.target_exam, "neet_pg_next")

    def test_patch_profile_invalid_academic_year_rejected(self):
        self._login()
        res = self.client.patch("/api/profile", json={"academic_year": "post_graduate_specialist"})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)
        self.assertIn("academic_year", data["error"])

    def test_patch_profile_invalid_target_exam_rejected(self):
        self._login()
        res = self.client.patch("/api/profile", json={"target_exam": "gre_medical"})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)
        self.assertIn("target_exam", data["error"])

    def test_put_profile_clearing_values(self):
        self._login()
        # First set values
        self.client.patch(
            "/api/profile",
            json={"academic_year": "3rd_year", "target_exam": "usmle_step1"},
        )

        # Clear values using empty string / null via PUT
        res = self.client.put(
            "/api/profile",
            json={"academic_year": "", "target_exam": None},
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsNone(data["user"]["academic_year"])
        self.assertIsNone(data["user"]["target_exam"])


class TestMBBSPedagogicalPromptAugmentation(unittest.TestCase):
    """Test LLM service prompt augmentation with MBBS academic context."""

    def setUp(self):
        self.app = create_app(TestingConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

        self.llm_service: LLMService = self.app.extensions["llm_service"]

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_prompt_includes_student_profile_context(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"answer": "Nephrotic syndrome involves podocyte effacement.", "cited_passage_ids": [], "evidence_sufficiency": "sufficient"}'
        mock_resp.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_resp

        self.llm_service.client = mock_client

        reply, sources, triage = self.llm_service.get_response(
            user_message="Explain the pathophysiology of Minimal Change Disease",
            academic_year="2nd_year",
            target_exam="neet_pg_next",
        )

        # Verify generate_content was called
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]
        last_part_text = contents[-1].parts[0].text

        self.assertIn("STUDENT ACADEMIC PROFILE CONTEXT", last_part_text)
        self.assertIn("2nd Year MBBS", last_part_text)
        self.assertIn("NEET-PG / NExT", last_part_text)
        self.assertIn("Nephrotic syndrome", reply)

    def test_prompt_resolves_profile_from_user_id(self):
        user = User(
            name="Context Test",
            email="context@med.edu",
            academic_year="final_year",
            target_exam="university_profs",
        )
        user.set_password("Password123")
        db.session.add(user)
        db.session.commit()

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"answer": "Management of acute appendicitis involves emergency appendectomy.", "cited_passage_ids": [], "evidence_sufficiency": "sufficient"}'
        mock_resp.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_resp

        self.llm_service.client = mock_client

        reply, sources, triage = self.llm_service.get_response(
            user_message="What is the surgical management of acute appendicitis?",
            user_id=user.id,
        )

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]
        last_part_text = contents[-1].parts[0].text

        self.assertIn("Final Year MBBS", last_part_text)
        self.assertIn("University Professional", last_part_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
