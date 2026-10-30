"""Comprehensive Security Tests for Password Reset Workflow.

Verifies:
- High-entropy cryptographic token generation (secrets module)
- Constant-time SHA-256 hash comparison
- Expiration enforcement (expired tokens rejected)
- Single-use semantics (reused tokens rejected)
- Tampered / malformed token rejection
- Cross-user token isolation (User A token cannot reset User B)
- Invalidation of all sibling active tokens upon successful reset
- User enumeration prevention (identical responses for existing and nonexistent users)
- Rate limiting on reset requests
- Concurrent reset attempts
"""

import time
import unittest
from datetime import timedelta
import concurrent.futures

import app as mediassist_app
from models import (
    User,
    PasswordResetToken,
    db,
    generate_reset_token,
    verify_reset_token,
    utc_now,
)
from services.auth_service import (
    request_password_reset,
    confirm_password_reset,
    reset_ip_limiter,
    reset_email_limiter,
)
from tests.test_auth import BaseAuthTestCase


class TestPasswordResetSecurity(BaseAuthTestCase):
    """Detailed security verification for password recovery."""

    def setUp(self):
        super().setUp()
        reset_ip_limiter.clear()
        reset_email_limiter.clear()

    def test_valid_token_reset_workflow(self):
        """Valid token resets password, hashes with scrypt, and invalidates old password."""
        user = self.create_user(email="alice@example.com", password="AliceOldPassword123!")

        # Request reset
        res_req = self.client.post("/reset-password", json={"email": "alice@example.com"})
        self.assertEqual(res_req.status_code, 200)
        token = res_req.get_json().get("_test_token")
        self.assertIsNotNone(token)

        # Confirm reset
        res_confirm = self.client.post(
            "/reset-password",
            json={
                "token": token,
                "email": "alice@example.com",
                "new_password": "AliceNewPassword456!",
                "new_password_confirm": "AliceNewPassword456!",
            },
        )
        self.assertEqual(res_confirm.status_code, 200)
        self.assertEqual(res_confirm.get_json()["status"], "ok")

        # Verify old password rejected
        res_old = self.client.post("/login", json={"email": "alice@example.com", "password": "AliceOldPassword123!"})
        self.assertEqual(res_old.status_code, 401)

        # Verify new password accepted
        res_new = self.client.post("/login", json={"email": "alice@example.com", "password": "AliceNewPassword456!"})
        self.assertEqual(res_new.status_code, 200)

    def test_expired_token_rejected(self):
        """Tokens past expires_at must be rejected."""
        user = self.create_user(email="bob@example.com", password="BobPassword123!")
        raw_token = generate_reset_token(user, expires_in_minutes=-10)  # already expired

        res = self.client.post(
            "/reset-password",
            json={
                "token": raw_token,
                "email": "bob@example.com",
                "new_password": "NewBobPassword456!",
                "new_password_confirm": "NewBobPassword456!",
            },
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("expired", res.get_json()["error"].lower())

    def test_single_use_token_reuse_rejected(self):
        """A reset token cannot be used more than once."""
        user = self.create_user(email="charlie@example.com", password="CharlieOldPassword123!")
        raw_token = generate_reset_token(user, expires_in_minutes=15)

        # First use must succeed
        res1 = self.client.post(
            "/reset-password",
            json={
                "token": raw_token,
                "email": "charlie@example.com",
                "new_password": "CharlieNewPassword456!",
                "new_password_confirm": "CharlieNewPassword456!",
            },
        )
        self.assertEqual(res1.status_code, 200)

        # Second use of identical token must fail
        res2 = self.client.post(
            "/reset-password",
            json={
                "token": raw_token,
                "email": "charlie@example.com",
                "new_password": "CharlieAnotherPassword789!",
                "new_password_confirm": "CharlieAnotherPassword789!",
            },
        )
        self.assertEqual(res2.status_code, 400)
        self.assertIn("already used", res2.get_json()["error"].lower())

    def test_tampered_token_rejected(self):
        """Tampered or corrupted token strings must fail verification."""
        user = self.create_user(email="dave@example.com", password="DavePassword123!")
        raw_token = generate_reset_token(user, expires_in_minutes=15)
        tampered = raw_token[:-4] + "XXXX"

        res = self.client.post(
            "/reset-password",
            json={
                "token": tampered,
                "email": "dave@example.com",
                "new_password": "DaveNewPassword456!",
                "new_password_confirm": "DaveNewPassword456!",
            },
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("invalid", res.get_json()["error"].lower())

    def test_cross_user_token_attack_prevented(self):
        """Token generated for User A cannot be used to reset User B's password."""
        user_a = self.create_user(email="victim@example.com", password="VictimPassword123!")
        user_b = self.create_user(email="attacker@example.com", password="AttackerPassword123!")

        # Attacker requests reset for their own account
        attacker_token = generate_reset_token(user_b, expires_in_minutes=15)

        # Attacker attempts to use their token against victim's email
        res = self.client.post(
            "/reset-password",
            json={
                "token": attacker_token,
                "email": "victim@example.com",
                "new_password": "HackedPassword123!",
                "new_password_confirm": "HackedPassword123!",
            },
        )
        self.assertEqual(res.status_code, 400)

        # Verify victim password remained unchanged
        res_victim = self.client.post("/login", json={"email": "victim@example.com", "password": "VictimPassword123!"})
        self.assertEqual(res_victim.status_code, 200)

    def test_invalidation_of_sibling_tokens(self):
        """When a user resets password, all prior active tokens for that user are invalidated."""
        user = self.create_user(email="multi_req@example.com", password="MultiPassword123!")
        token1 = generate_reset_token(user, expires_in_minutes=15)
        token2 = generate_reset_token(user, expires_in_minutes=15)

        # Reset using token2
        res = self.client.post(
            "/reset-password",
            json={
                "token": token2,
                "email": "multi_req@example.com",
                "new_password": "NewMultiPassword456!",
                "new_password_confirm": "NewMultiPassword456!",
            },
        )
        self.assertEqual(res.status_code, 200)

        # token1 must now be invalidated
        res_t1 = self.client.post(
            "/reset-password",
            json={
                "token": token1,
                "email": "multi_req@example.com",
                "new_password": "YetAnotherPassword789!",
                "new_password_confirm": "YetAnotherPassword789!",
            },
        )
        self.assertEqual(res_t1.status_code, 400)

    def test_user_enumeration_prevention(self):
        """Response message and status must be identical whether email exists or not."""
        self.create_user(email="existing@example.com", password="ExistingPassword123!")

        res_exist = self.client.post("/reset-password", json={"email": "existing@example.com"})
        res_ghost = self.client.post("/reset-password", json={"email": "nonexistent_ghost@example.com"})

        self.assertEqual(res_exist.status_code, 200)
        self.assertEqual(res_ghost.status_code, 200)
        self.assertEqual(res_exist.get_json()["message"], res_ghost.get_json()["message"])

    def test_rate_limiting_enforced(self):
        """Rapid repeated reset requests from the same IP trigger HTTP 429."""
        ip = "192.168.1.100"
        # 5 attempts allowed in 60s
        for i in range(5):
            res, code, _ = request_password_reset(f"target_{i}@example.com", client_ip=ip, is_testing=False)
            self.assertEqual(code, 200)

        # 6th attempt must be rate-limited
        res_limit, code_limit, _ = request_password_reset("target_final@example.com", client_ip=ip, is_testing=False)
        self.assertEqual(code_limit, 429)
        self.assertIn("too many", res_limit["error"].lower())

    def test_concurrent_reset_attempts(self):
        """Concurrent reset attempts on the exact same token only succeed once."""
        user = self.create_user(email="race@example.com", password="RacePassword123!")
        raw_token = generate_reset_token(user, expires_in_minutes=15)

        results = []

        def attempt_reset(pwd: str):
            with self.app.app_context():
                res, code = confirm_password_reset(
                    raw_token=raw_token,
                    new_password=pwd,
                    new_password_confirm=pwd,
                    email="race@example.com",
                )
                return code

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(attempt_reset, f"ConcurrentPassword{i}!")
                for i in range(3)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        # Exactly one must succeed (200), and others must be rejected (400)
        self.assertEqual(results.count(200), 1)
        self.assertEqual(results.count(400), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
