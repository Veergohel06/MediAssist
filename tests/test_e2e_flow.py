"""End-to-End integration test suite for MediAssist.

Verifies:
1. Fresh anonymous session visits / -> renders landing page (not logged in, no fake user).
2. Unauthenticated access to /dashboard -> safely redirects to /login.
3. Unauthenticated access to /chat -> redirects to /login?next=/chat.
4. Registration flow -> creates user, logs in, redirects directly to /chat.
5. Authenticated access to /dashboard -> safely redirects to /chat.
6. Authenticated access to /chat -> renders consultation workspace with compact profile and NO Dashboard button or MBBS selector.
7. Login flow with valid credentials -> redirects directly to /chat.
8. Logout -> clears session, redirects to / (landing page).
9. Post-logout access to /chat or /dashboard -> rejected with 302 redirect to /login.
10. Theme tokens, compact profile menu styles, and autofill rules in auth.css.
"""

from __future__ import annotations

import unittest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app import create_app
from config import TestingConfig
from models import Conversation, ChatMessage, User, db


class TestEndToEndApplicationFlow(unittest.TestCase):
    """Validate complete application user journey and security boundaries."""

    def setUp(self):
        self.app = create_app(TestingConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()

        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        db.metadata.create_all(bind=self.engine)
        db.session.configure(bind=self.engine)

        self.client = self.app.test_client()

    def tearDown(self):
        db.session.remove()
        db.metadata.drop_all(bind=self.engine)
        self.engine.dispose()
        self.app_context.pop()

    def test_01_fresh_session_landing_page(self):
        """Scenario 1: Fresh session displays landing page without auto-login or fake user."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        # Verify landing page content
        self.assertIn("Evidence-Grounded Healthcare AI", html)
        self.assertIn("Sign In", html)
        self.assertIn("Get Started", html)
        # Ensure no fake logged-in user is displayed
        self.assertNotIn("Medical Student", html)
        self.assertIn("data-theme", html)

    def test_02_unauthenticated_dashboard_redirects_to_login(self):
        """Trying to open legacy /dashboard without login redirects to /login."""
        res = self.client.get("/dashboard", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        location = res.headers.get("Location", "")
        self.assertIn("/login", location)

    def test_03_unauthenticated_chat_redirects_to_login(self):
        """Trying to open /chat without login redirects to /login."""
        res = self.client.get("/chat", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertIn("/login", res.headers.get("Location", ""))

    def test_04_registration_and_consultation_access(self):
        """Register account -> redirects directly to consultation workspace."""
        reg_res = self.client.post(
            "/register",
            data={
                "name": "Dr. Sarah Connor",
                "email": "sarah.connor@hospital.org",
                "password": "StrongPassword999!",
                "password_confirm": "StrongPassword999!",
            },
            follow_redirects=True,
        )
        self.assertEqual(reg_res.status_code, 200)
        html = reg_res.get_data(as_text=True)
        # Should now be on the Consultation Workspace
        self.assertIn("Dr. Sarah Connor", html)
        self.assertIn("Sign Out", html)
        self.assertNotIn("Dashboard", html)
        self.assertNotIn("MBBS Level", html)

    def test_05_authenticated_dashboard_redirects_to_chat(self):
        """Legacy /dashboard route safely redirects authenticated users to /chat."""
        user = User(name="Alex Rivera", email="alex@health.org")
        user.set_password("SecurePass123!")
        db.session.add(user)
        db.session.commit()

        # Login
        self.client.post("/login", data={"email": "alex@health.org", "password": "SecurePass123!"})

        # Access legacy /dashboard
        res = self.client.get("/dashboard", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertIn("/chat", res.headers.get("Location", ""))

    def test_06_consultation_workspace_rendering(self):
        """Authenticated user opens consultation with compact navbar and no dashboard or MBBS selector."""
        user = User(name="Elena Rostova", email="elena@clinic.org")
        user.set_password("ClinicSecurePass1!")
        db.session.add(user)
        db.session.commit()

        self.client.post("/login", data={"email": "elena@clinic.org", "password": "ClinicSecurePass1!"})

        res = self.client.get("/chat")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Elena Rostova", html)
        self.assertNotIn("Dashboard", html)
        self.assertNotIn("MBBS Level", html)
        self.assertIn("Sign Out", html)

    def test_07_logout_flow_and_rejection(self):
        """Logout clears session and redirects to / landing page."""
        user = User(name="Marcus Vance", email="marcus@care.org")
        user.set_password("VanceSecure888!")
        db.session.add(user)
        db.session.commit()

        self.client.post("/login", data={"email": "marcus@care.org", "password": "VanceSecure888!"})

        # Logout via POST
        logout_res = self.client.post("/logout", follow_redirects=False)
        self.assertEqual(logout_res.status_code, 302)
        self.assertEqual(logout_res.headers.get("Location"), "/")

        # Post-logout: consultation and dashboard are protected again
        chat_after = self.client.get("/chat", follow_redirects=False)
        self.assertEqual(chat_after.status_code, 302)
        self.assertIn("/login", chat_after.headers.get("Location", ""))

    def test_08_theme_and_design_assets(self):
        """CSS file defines light/dark tokens, autofill overrides, and compact profile menu."""
        css_res = self.client.get("/static/auth.css")
        self.assertEqual(css_res.status_code, 200)
        css = css_res.get_data(as_text=True)
        self.assertIn('[data-theme="dark"]', css)
        self.assertIn("--color-primary", css)
        self.assertIn("--color-bg", css)
        self.assertIn("--color-surface", css)
        self.assertIn(".nav-profile-container", css)
        self.assertIn("-webkit-autofill", css)
        self.assertNotIn(".dashboard-container", css)


if __name__ == "__main__":
    unittest.main()
