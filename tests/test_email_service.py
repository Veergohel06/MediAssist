"""Unit and Security Tests for Transactional Email Delivery (Password Reset).

Validates:
- HTML and plaintext email rendering with token, expiration, and security warnings.
- Test outbox recording and suppression modes.
- End-to-end integration with auth_service.request_password_reset.
- Nonexistent email requests do not dispatch emails or leak user presence.
- Graceful SMTP network failure handling and error isolation.
"""

import unittest
from unittest.mock import patch, MagicMock
import smtplib

from services.email_service import EmailService, email_service
from services.auth_service import (
    request_password_reset,
    confirm_password_reset,
    reset_ip_limiter,
    reset_email_limiter,
)
from tests.test_auth import BaseAuthTestCase


class TestEmailService(unittest.TestCase):
    """Direct unit tests for EmailService methods."""

    def setUp(self):
        self.service = EmailService({
            "MAIL_SERVER": "mock",
            "MAIL_DEFAULT_SENDER": "MediAssist Test <test@mediassist.health>",
            "APP_BASE_URL": "https://mediassist.example.com",
            "MAIL_SUPPRESS_SEND": True,
        })
        self.service.clear_outbox()

    def test_password_reset_email_rendering(self):
        """Verifies plaintext and HTML templates have required security notices and links."""
        token = "test-crypto-token-xyz-123456"
        plain, html = self.service._render_password_reset_content(
            token=token,
            recipient_name="Dr. Mehta",
            base_url="https://app.mediassist.health",
        )

        expected_url = f"https://app.mediassist.health/reset-password?token={token}"

        # Verify plaintext content
        self.assertIn("Hello Dr. Mehta,", plain)
        self.assertIn(expected_url, plain)
        self.assertIn("15 minutes", plain)
        self.assertIn("If you did not request a password reset, you can safely ignore this email.", plain)

        # Verify HTML content
        self.assertIn("Dr. Mehta", html)
        self.assertIn(expected_url, html)
        self.assertIn("15 minutes", html)
        self.assertIn("Reset Password", html)

    def test_send_email_records_to_outbox(self):
        """Suppressed/mock email dispatch logs message in outbox without network sockets."""
        success = self.service.send_email(
            to_email="patient@example.com",
            subject="Important Health Notice",
            text_body="Here is your health update.",
            sync=True,
        )
        self.assertTrue(success)
        outbox = self.service.get_outbox()
        self.assertEqual(len(outbox), 1)
        self.assertEqual(outbox[0]["to"], "patient@example.com")
        self.assertEqual(outbox[0]["subject"], "Important Health Notice")
        self.assertEqual(outbox[0]["text"], "Here is your health update.")

    def test_empty_email_rejected(self):
        """Sending to an empty address returns False immediately."""
        self.assertFalse(self.service.send_email("", "Subject", "Body", sync=True))
        self.assertEqual(len(self.service.get_outbox()), 0)

    def test_smtp_network_failure_resilience(self):
        """SMTP exceptions are caught gracefully and do not crash the application."""
        live_service = EmailService({
            "MAIL_SERVER": "smtp.live-provider.com",
            "MAIL_PORT": 587,
            "MAIL_SUPPRESS_SEND": False,
        })

        with patch("smtplib.SMTP", side_effect=smtplib.SMTPConnectError(421, b"Connection refused")):
            success = live_service.send_email(
                to_email="test@example.com",
                subject="Test Subject",
                text_body="Test Body",
                sync=True,
            )
            self.assertFalse(success)


class TestAuthPasswordResetEmailIntegration(BaseAuthTestCase):
    """End-to-end integration tests between request_password_reset and EmailService."""

    def setUp(self):
        super().setUp()
        reset_ip_limiter.clear()
        reset_email_limiter.clear()
        email_service.clear_outbox()
        email_service.update_config({"MAIL_SUPPRESS_SEND": True, "MAIL_SERVER": "mock"})

    def test_password_reset_delivers_email_to_registered_user(self):
        """Requesting password reset for an existing user dispatches email with valid token."""
        user = self.create_user(email="registered@patient.org", password="ValidPass123!")

        res_data, code, token = request_password_reset(
            email="registered@patient.org",
            client_ip="10.0.0.1",
            is_testing=True,
        )
        self.assertEqual(code, 200)
        self.assertIsNotNone(token)

        outbox = email_service.get_outbox()
        self.assertEqual(len(outbox), 1)
        sent = outbox[0]
        self.assertEqual(sent["to"], "registered@patient.org")
        self.assertIn("Reset Your MediAssist Password", sent["subject"])
        self.assertIn(token, sent["text"])
        self.assertIn(token, sent["html"])

        # Verify that the token delivered in the email can be used to reset password
        confirm_res, confirm_code = confirm_password_reset(
            raw_token=token,
            new_password="NewSecurePass456!",
            new_password_confirm="NewSecurePass456!",
            email="registered@patient.org",
        )
        self.assertEqual(confirm_code, 200)

    def test_password_reset_does_not_dispatch_email_for_unregistered_user(self):
        """Requesting reset for nonexistent user returns identical 200 message but sends no email."""
        res_data, code, token = request_password_reset(
            email="nonexistent_ghost@patient.org",
            client_ip="10.0.0.2",
            is_testing=True,
        )
        self.assertEqual(code, 200)
        self.assertIsNone(token)

        outbox = email_service.get_outbox()
        self.assertEqual(len(outbox), 0)


if __name__ == "__main__":
    unittest.main()
