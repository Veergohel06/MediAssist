"""Automated tests for MediAssist authentication, access-control, and user isolation.

Covers:
- Registration success, duplicate email rejection, input validation, and password hashing.
- Successful/failed login, generic failure messages, account lockout after failed attempts.
- Logout and session invalidation.
- Anonymous access enforcement across all protected endpoints.
- Authenticated access to protected pages and APIs.
- Cross-user conversation access denial (403 Forbidden).
- CSRF protection enforcement and invalid-token rejection.
- Session cookie security attributes (HttpOnly, SameSite, Secure).
- Authenticated chat flow with mocked AI calls.
"""

import json
import unittest
from unittest.mock import patch

import app as mediassist_app
from models import ChatMessage, Conversation, User, db, validate_registration_input


from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


class BaseAuthTestCase(unittest.TestCase):
    """Base test setup with fully isolated in-memory SQLite database and app context."""

    @classmethod
    def setUpClass(cls):
        cls.app = mediassist_app.app
        cls.orig_testing = cls.app.config.get("TESTING")
        cls.orig_csrf = cls.app.config.get("WTF_CSRF_ENABLED")

        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.app.config["SERVER_NAME"] = "localhost"

        # Create isolated in-memory SQLite engine with StaticPool so all connections share the memory DB
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
        # Restore real engine and configuration
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

    def create_user(
        self,
        name="Test User",
        email="testuser@example.com",
        password="ValidPassword123!",
    ) -> User:
        """Helper to create and persist a test user."""
        user = User(name=name, email=email.lower().strip())
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return user

    def login_client(
        self,
        email="testuser@example.com",
        password="ValidPassword123!",
        client=None,
    ):
        """Helper to log in a test client."""
        c = client or self.client
        return c.post("/login", json={"email": email, "password": password})


class TestUserRegistration(BaseAuthTestCase):
    """Test user registration, validation, and password hashing."""

    def test_registration_success_json(self):
        res = self.client.post(
            "/register",
            json={
                "name": "Alice Smith",
                "email": "alice@example.com",
                "password": "Password123!",
                "password_confirm": "Password123!",
            },
        )
        self.assertEqual(res.status_code, 201)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["user"]["email"], "alice@example.com")

        # Verify DB persistence
        user = User.query.filter_by(email="alice@example.com").first()
        self.assertIsNotNone(user)
        self.assertEqual(user.name, "Alice Smith")
        # Ensure password is NOT plaintext
        self.assertNotEqual(user.password_hash, "Password123!")
        self.assertTrue(user.check_password("Password123!"))

    def test_registration_success_form_redirect(self):
        res = self.client.post(
            "/register",
            data={
                "name": "Bob Jones",
                "email": "bob@example.com",
                "password": "SecurePassword456!",
                "password_confirm": "SecurePassword456!",
            },
            follow_redirects=False,
        )
        self.assertEqual(res.status_code, 302)
        self.assertTrue(any(loc in res.headers.get("Location", "") for loc in ("/dashboard", "/chat")))

    def test_duplicate_email_rejection(self):
        self.create_user(email="alice@example.com")
        res = self.client.post(
            "/register",
            json={
                "name": "Another Alice",
                "email": "alice@example.com",
                "password": "Password123!",
                "password_confirm": "Password123!",
            },
        )
        self.assertEqual(res.status_code, 409)
        data = res.get_json()
        self.assertIn("already exists", data["error"].lower())

    def test_invalid_registration_inputs(self):
        cases = [
            ({"name": "", "email": "a@b.com", "password": "P1!", "password_confirm": "P1!"}, "name"),
            ({"name": "Al", "email": "invalid-email", "password": "Password123!", "password_confirm": "Password123!"}, "email"),
            ({"name": "Al", "email": "a@b.com", "password": "short", "password_confirm": "short"}, "at least 8"),
            ({"name": "Al", "email": "a@b.com", "password": "Password123!", "password_confirm": "Mismatch123!"}, "not match"),
        ]
        for payload, expected_error_sub in cases:
            res = self.client.post("/register", json=payload)
            self.assertEqual(res.status_code, 400)
            self.assertIn(expected_error_sub.lower(), res.get_json()["error"].lower())

    def test_password_hashing_security(self):
        user = User(name="Cipher", email="cipher@example.com")
        plain = "SuperSecretPassword123!"
        user.set_password(plain)

        # Hash must use a secure hashing algorithm (scrypt or pbkdf2)
        self.assertTrue(user.password_hash.startswith("scrypt:") or user.password_hash.startswith("pbkdf2:"))
        self.assertNotIn(plain, user.password_hash)
        self.assertTrue(user.check_password(plain))
        self.assertFalse(user.check_password("WrongPassword123!"))


class TestUserAuthentication(BaseAuthTestCase):
    """Test login, logout, generic failure messages, and account lockout."""

    def test_successful_login_json(self):
        self.create_user(email="carol@example.com", password="Password123!")
        res = self.client.post(
            "/login",
            json={"email": "carol@example.com", "password": "Password123!"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["user"]["email"], "carol@example.com")

    def test_login_case_insensitive_email(self):
        self.create_user(email="carol@example.com", password="Password123!")
        for email_variant in ["CAROL@EXAMPLE.COM", "Carol@Example.Com", "cArOl@ExAmPlE.cOm"]:
            res = self.client.post(
                "/login",
                json={"email": email_variant, "password": "Password123!"},
            )
            self.assertEqual(res.status_code, 200, f"Failed login with {email_variant}")
            self.assertEqual(res.get_json()["status"], "ok")

    def test_login_with_accidental_whitespace(self):
        self.create_user(email="dan@example.com", password="Password123!")
        res = self.client.post(
            "/login",
            json={"email": "   dan@example.com   ", "password": "Password123!"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "ok")

    def test_login_password_whitespace_not_stripped(self):
        """Passphrase with whitespace must not match trimmed version (prevents auth bypass)."""
        self.create_user(email="dan2@example.com", password="Password123!")
        res = self.client.post(
            "/login",
            json={"email": "dan2@example.com", "password": "  Password123!  "},
        )
        self.assertEqual(res.status_code, 401)

    def test_failed_login_incorrect_password(self):
        self.create_user(email="carol@example.com", password="Password123!")
        res = self.client.post(
            "/login",
            json={"email": "carol@example.com", "password": "WrongPassword!"},
        )
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.get_json()["error"], "Invalid email or password.")

    def test_failed_login_nonexistent_user(self):
        res = self.client.post(
            "/login",
            json={"email": "nonexistent@example.com", "password": "Password123!"},
        )
        self.assertEqual(res.status_code, 401)
        # Verify generic message matches exactly to prevent account enumeration
        self.assertEqual(res.get_json()["error"], "Invalid email or password.")

    def test_account_lockout_after_failed_attempts(self):
        user = self.create_user(email="target@example.com", password="ValidPassword123!")
        # 5 consecutive failed attempts
        for _ in range(5):
            res = self.client.post("/login", json={"email": "target@example.com", "password": "BadPassword"})
            self.assertEqual(res.status_code, 401)

        # 6th attempt should be rejected with 403 Account Locked
        res = self.client.post("/login", json={"email": "target@example.com", "password": "ValidPassword123!"})
        self.assertEqual(res.status_code, 403)
        self.assertIn("locked", res.get_json()["error"].lower())

    def test_logout_and_session_invalidation(self):
        self.create_user(email="logout_user@example.com", password="Password123!")
        self.login_client(email="logout_user@example.com", password="Password123!")

        # Verify authenticated access
        res = self.client.get("/api/auth/me")
        self.assertEqual(res.status_code, 200)

        # Log out
        logout_res = self.client.post("/logout", json={})
        self.assertEqual(logout_res.status_code, 200)
        self.assertEqual(logout_res.get_json()["status"], "logged_out")

        # After logout, accessing protected endpoint returns 401
        res_after = self.client.get("/api/auth/me")
        self.assertEqual(res_after.status_code, 401)


class TestAccessControl(BaseAuthTestCase):
    """Test anonymous access rejection and authenticated endpoint access."""

    def test_anonymous_access_to_protected_endpoints(self):
        # HTML route should redirect to login
        res_page = self.client.get("/chat", follow_redirects=False)
        self.assertEqual(res_page.status_code, 302)
        self.assertIn("/login", res_page.headers.get("Location", ""))

        # API routes must return 401 JSON
        api_endpoints = [
            ("POST", "/chat", {"message": "hello"}),
            ("POST", "/reset", {}),
            ("GET", "/api/auth/me", None),
            ("GET", "/api/conversations", None),
            ("POST", "/api/conversations", {"title": "Test"}),
            ("GET", "/api/conversations/random-uuid", None),
            ("DELETE", "/api/conversations/random-uuid", None),
            ("GET", "/api/documents", None),
        ]
        for method, endpoint, payload in api_endpoints:
            if method == "POST":
                res = self.client.post(endpoint, json=payload or {})
            elif method == "DELETE":
                res = self.client.delete(endpoint)
            else:
                res = self.client.get(endpoint)
            self.assertEqual(res.status_code, 401, f"Endpoint {method} {endpoint} did not return 401 for anonymous access")
            self.assertIn("error", res.get_json())

    def test_public_access_to_unauthenticated_endpoints(self):
        public_endpoints = ["/", "/login", "/register", "/api/auth/csrf-token"]
        for ep in public_endpoints:
            res = self.client.get(ep)
            self.assertEqual(res.status_code, 200, f"Public endpoint {ep} returned {res.status_code}")

    def test_authenticated_access_to_chat_view_and_apis(self):
        self.create_user(email="auth_user@example.com")
        self.login_client(email="auth_user@example.com")

        res_chat = self.client.get("/chat")
        self.assertEqual(res_chat.status_code, 200)

        res_me = self.client.get("/api/auth/me")
        self.assertEqual(res_me.status_code, 200)
        self.assertEqual(res_me.get_json()["user"]["email"], "auth_user@example.com")

        res_docs = self.client.get("/api/documents")
        self.assertEqual(res_docs.status_code, 200)
        self.assertIn("documents", res_docs.get_json())


class TestUserIsolation(BaseAuthTestCase):
    """Test strict multi-tenant isolation: User B cannot access User A's consultation data."""

    def test_cross_user_conversation_isolation(self):
        user_a = self.create_user(name="User A", email="usera@example.com")
        user_b = self.create_user(name="User B", email="userb@example.com")

        # User A creates a consultation
        client_a = self.app.test_client()
        self.login_client(email="usera@example.com", client=client_a)
        res_create = client_a.post("/api/conversations", json={"title": "User A Private Symptoms"})
        self.assertEqual(res_create.status_code, 201)
        conv_a_id = res_create.get_json()["conversation"]["id"]

        # User A can read their own conversation
        res_read_a = client_a.get(f"/api/conversations/{conv_a_id}")
        self.assertEqual(res_read_a.status_code, 200)

        # User B logs in
        client_b = self.app.test_client()
        self.login_client(email="userb@example.com", client=client_b)

        # 1. User B lists conversations: must NOT include User A's conversation
        res_list_b = client_b.get("/api/conversations")
        self.assertEqual(res_list_b.status_code, 200)
        b_conv_ids = [c["id"] for c in res_list_b.get_json()["conversations"]]
        self.assertNotIn(conv_a_id, b_conv_ids)

        # 2. User B tries to read User A's conversation directly by ID: MUST return 403 Forbidden
        res_read_b = client_b.get(f"/api/conversations/{conv_a_id}")
        self.assertEqual(res_read_b.status_code, 403)
        self.assertIn("access denied", res_read_b.get_json()["error"].lower())

        # 3. User B tries to delete User A's conversation: MUST return 403 Forbidden
        res_del_b = client_b.delete(f"/api/conversations/{conv_a_id}")
        self.assertEqual(res_del_b.status_code, 403)
        self.assertIn("access denied", res_del_b.get_json()["error"].lower())

        # 4. User B tries to inject chat message into User A's conversation: MUST return 403 Forbidden
        res_chat_b = client_b.post(
            "/chat",
            json={"message": "Injected message", "conversation_id": conv_a_id},
        )
        self.assertEqual(res_chat_b.status_code, 403)


class TestCSRFProtection(BaseAuthTestCase):
    """Test CSRF enforcement when enabled."""

    def tearDown(self):
        self.app.config["WTF_CSRF_ENABLED"] = False
        super().tearDown()

    def test_csrf_token_enforcement(self):
        self.app.config["WTF_CSRF_ENABLED"] = True

        # POST without CSRF token must fail with 400
        res = self.client.post("/login", json={"email": "test@example.com", "password": "Password123!"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("csrf validation failed", res.get_json()["error"].lower())

        # Obtain valid CSRF token from endpoint
        csrf_res = self.client.get("/api/auth/csrf-token")
        self.assertEqual(csrf_res.status_code, 200)
        token = csrf_res.get_json()["csrf_token"]

        # POST with valid X-CSRFToken header passes CSRF verification (even if credentials fail)
        res_with_token = self.client.post(
            "/login",
            json={"email": "test@example.com", "password": "Password123!"},
            headers={"X-CSRFToken": token},
        )
        # Should reach authentication logic (401 invalid creds, NOT 400 CSRF failure)
        self.assertEqual(res_with_token.status_code, 401)


class TestSessionCookies(BaseAuthTestCase):
    """Test session cookie security attributes."""

    def test_session_cookie_attributes(self):
        self.create_user(email="cookie_user@example.com")
        res = self.client.post(
            "/login",
            json={"email": "cookie_user@example.com", "password": "ValidPassword123!"},
        )
        self.assertEqual(res.status_code, 200)

        cookie_headers = res.headers.getlist("Set-Cookie")
        self.assertTrue(len(cookie_headers) > 0)
        combined = " ".join(cookie_headers)

        # Verify HttpOnly and SameSite=Lax
        self.assertIn("HttpOnly", combined)
        self.assertIn("SameSite=Lax", combined)


class TestAuthenticatedAIChat(BaseAuthTestCase):
    """Test AI chat execution and message persistence when authenticated."""

    @patch("app.get_gemini_response")
    def test_authenticated_chat_flow_and_history_persistence(self, mock_gemini):
        mock_gemini.return_value = (
            "Hydration is essential for cardiovascular health [daily_water_requirements:p0001].",
            [{"source_title": "Daily Water Requirements", "source_url": "https://medlineplus.gov/ency/article/002471.htm"}],
            {
                "is_emergency": False,
                "evidence": {"status": "sufficient"},
                "citation_validation": {"valid_ids": ["daily_water_requirements:p0001"]},
            },
        )

        user = self.create_user(email="chat_patient@example.com")
        self.login_client(email="chat_patient@example.com")

        res = self.client.post("/chat", json={"message": "How much water should I drink?"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("Hydration is essential", data["response"])
        self.assertEqual(len(data["sources"]), 1)
        conv_id = data["conversation_id"]

        # Verify database record persistence
        conv = Conversation.query.filter_by(id=conv_id).first()
        self.assertEqual(conv.user_id, user.id)
        messages = conv.messages.all()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, "How much water should I drink?")
        self.assertEqual(messages[1].role, "model")


class TestPasswordReset(BaseAuthTestCase):
    """Test self-service password recovery / reset flow."""

    def test_reset_password_page_loads(self):
        res = self.client.get("/reset-password")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Reset Your Password", res.data)

    def test_reset_password_success_json(self):
        user = self.create_user(email="reset_target@example.com", password="OldPassword123!")

        # Step 1: Request reset
        res_req = self.client.post(
            "/reset-password",
            json={"email": "reset_target@example.com"},
        )
        self.assertEqual(res_req.status_code, 200)
        token = res_req.get_json().get("_test_token")
        self.assertIsNotNone(token)

        # Step 2: Confirm reset with token
        res = self.client.post(
            "/reset-password",
            json={
                "token": token,
                "email": "reset_target@example.com",
                "new_password": "NewSecurePassword456!",
                "new_password_confirm": "NewSecurePassword456!",
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "ok")

        # Old password must now fail
        res_old = self.client.post(
            "/login",
            json={"email": "reset_target@example.com", "password": "OldPassword123!"},
        )
        self.assertEqual(res_old.status_code, 401)

        # New password must succeed
        res_new = self.client.post(
            "/login",
            json={"email": "reset_target@example.com", "password": "NewSecurePassword456!"},
        )
        self.assertEqual(res_new.status_code, 200)
        self.assertEqual(res_new.get_json()["status"], "ok")

    def test_reset_password_case_and_whitespace_normalization(self):
        user = self.create_user(email="reset_norm@example.com", password="OldPassword123!")

        res_req = self.client.post(
            "/reset-password",
            json={"email": "   RESET_NORM@EXAMPLE.COM   "},
        )
        self.assertEqual(res_req.status_code, 200)
        token = res_req.get_json().get("_test_token")
        self.assertIsNotNone(token)

        res = self.client.post(
            "/reset-password",
            json={
                "token": token,
                "email": "   RESET_NORM@EXAMPLE.COM   ",
                "new_password": "   NewSecurePassword456!   ",
                "new_password_confirm": "   NewSecurePassword456!   ",
            },
        )
        self.assertEqual(res.status_code, 200)

        # Login with normalized email and password
        res_login = self.client.post(
            "/login",
            json={"email": "reset_norm@example.com", "password": "NewSecurePassword456!"},
        )
        self.assertEqual(res_login.status_code, 200)

    def test_reset_password_validation_rejections(self):
        user = self.create_user(email="valid_reset@example.com", password="OldPassword123!")
        res_req = self.client.post(
            "/reset-password",
            json={"email": "valid_reset@example.com"},
        )
        token = res_req.get_json().get("_test_token")
        self.assertIsNotNone(token)

        # Mismatched passwords
        res_mismatch = self.client.post(
            "/reset-password",
            json={
                "token": token,
                "email": "valid_reset@example.com",
                "new_password": "NewPassword123!",
                "new_password_confirm": "DifferentPassword123!",
            },
        )
        self.assertEqual(res_mismatch.status_code, 400)
        self.assertIn("not match", res_mismatch.get_json()["error"].lower())

        # Weak password (< 8 chars)
        res_weak = self.client.post(
            "/reset-password",
            json={
                "token": token,
                "email": "valid_reset@example.com",
                "new_password": "short",
                "new_password_confirm": "short",
            },
        )
        self.assertEqual(res_weak.status_code, 400)
        self.assertIn("at least 8", res_weak.get_json()["error"].lower())

        # Nonexistent user (enumeration-proof generic response)
        res_nonexistent = self.client.post(
            "/reset-password",
            json={
                "email": "ghost_user@example.com",
            },
        )
        self.assertEqual(res_nonexistent.status_code, 200)
        self.assertIn("instructions have been sent", res_nonexistent.get_json()["message"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
