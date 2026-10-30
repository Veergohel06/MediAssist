"""Transactional Email Delivery Service for MediAssist.

Provides secure, robust, and asynchronous email delivery for account workflows,
specifically password recovery links and system notifications.

Supports:
- SMTP over TLS / SSL (SendGrid, Mailgun, AWS SES, Postmark, generic SMTP).
- In-memory test outbox for verification in automated test suites without live network calls.
- Console / dev logger fallback when SMTP credentials are not yet configured.
- Asynchronous non-blocking dispatch via ThreadPoolExecutor.
- Responsive HTML email templates with plain-text fallback.
"""

from __future__ import annotations

import logging
import smtplib
import threading
from concurrent.futures import ThreadPoolExecutor
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mediassist.email")


class EmailService:
    """Manages transactional email dispatch and templating."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or {}
        self.outbox: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="email-dispatch-")

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update active configuration."""
        self.config.update(config)

    def clear_outbox(self) -> None:
        """Clear test outbox."""
        with self._lock:
            self.outbox.clear()

    def get_outbox(self) -> List[Dict[str, Any]]:
        """Return a snapshot copy of the outbox."""
        with self._lock:
            return list(self.outbox)

    def _render_password_reset_content(
        self,
        token: str,
        recipient_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> tuple[str, str]:
        """Generate plaintext and HTML bodies for password reset."""
        host = (base_url or self.config.get("APP_BASE_URL", "")).rstrip("/")
        reset_link = f"{host}/reset-password?token={token}" if host else f"/reset-password?token={token}"
        greeting_name = (recipient_name or "").strip()
        salutation = f"Hello {greeting_name}," if greeting_name else "Hello,"

        plain_text = (
            f"{salutation}\n\n"
            f"We received a request to reset the password for your MediAssist account.\n\n"
            f"To choose a new password, open the following link in your browser:\n"
            f"{reset_link}\n\n"
            f"This link is valid for 15 minutes and can only be used once.\n\n"
            f"If you did not request a password reset, you can safely ignore this email.\n"
            f"Your account remains secure and no changes have been made.\n\n"
            f"— The MediAssist Team"
        )

        html_body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reset Your MediAssist Password</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #1e293b;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #f8fafc; padding: 32px 16px;">
    <tr>
      <td align="center">
        <table role="presentation" width="100%" style="max-width: 540px; background-color: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
          <!-- Header -->
          <tr>
            <td style="padding: 28px 32px 20px; background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); color: #ffffff;">
              <h1 style="margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.3px;">MediAssist</h1>
              <p style="margin: 4px 0 0; font-size: 13px; opacity: 0.9;">Clinical AI Health Assistant</p>
            </td>
          </tr>
          <!-- Body Content -->
          <tr>
            <td style="padding: 32px;">
              <h2 style="margin: 0 0 16px; font-size: 18px; font-weight: 600; color: #0f172a;">Password Recovery Request</h2>
              <p style="margin: 0 0 16px; font-size: 15px; line-height: 1.5; color: #334155;">{salutation}</p>
              <p style="margin: 0 0 24px; font-size: 15px; line-height: 1.5; color: #334155;">
                We received a request to reset your MediAssist password. Click the button below to choose a new, secure password:
              </p>
              <div style="text-align: center; margin: 28px 0;">
                <a href="{reset_link}" style="display: inline-block; background-color: #0284c7; color: #ffffff; text-decoration: none; font-size: 15px; font-weight: 600; padding: 12px 28px; border-radius: 8px; box-shadow: 0 2px 4px rgba(2, 132, 199, 0.3);">Reset Password</a>
              </div>
              <p style="margin: 24px 0 8px; font-size: 13px; color: #64748b; line-height: 1.5;">
                Or copy and paste this URL into your browser:<br>
                <a href="{reset_link}" style="color: #0284c7; word-break: break-all; font-size: 12px;">{reset_link}</a>
              </p>
              <div style="margin-top: 24px; padding: 14px 16px; background-color: #f1f5f9; border-radius: 8px; border-left: 4px solid #0284c7;">
                <p style="margin: 0; font-size: 13px; color: #475569; line-height: 1.4;">
                  <strong>Note:</strong> This link will expire in <strong>15 minutes</strong> for your security.
                </p>
              </div>
              <p style="margin: 24px 0 0; font-size: 13px; color: #64748b; line-height: 1.5;">
                If you did not request this password reset, no action is needed. Your account remains completely secure.
              </p>
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="padding: 16px 32px 24px; border-top: 1px solid #f1f5f9; text-align: center;">
              <p style="margin: 0; font-size: 12px; color: #94a3b8;">
                &copy; MediAssist &bull; Evidence-Grounded Health AI
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""
        return plain_text, html_body

    def send_email(
        self,
        to_email: str,
        subject: str,
        text_body: str,
        html_body: Optional[str] = None,
        sender: Optional[str] = None,
        sync: bool = False,
    ) -> bool:
        """Send an email.

        If sync is True, sends in the current thread (useful for testing).
        If sync is False, dispatches asynchronously to avoid blocking user response.
        """
        clean_to = to_email.strip().lower()
        if not clean_to:
            logger.warning("Attempted to send email to empty address.")
            return False

        if sync:
            return self._execute_send(clean_to, subject, text_body, html_body, sender)

        self.executor.submit(self._execute_send, clean_to, subject, text_body, html_body, sender)
        return True

    def _execute_send(
        self,
        to_email: str,
        subject: str,
        text_body: str,
        html_body: Optional[str] = None,
        sender: Optional[str] = None,
    ) -> bool:
        """Internal synchronous mail transport worker."""
        sender_address = (
            sender
            or self.config.get("MAIL_DEFAULT_SENDER")
            or "MediAssist <noreply@mediassist.health>"
        )

        message_record = {
            "to": to_email,
            "sender": sender_address,
            "subject": subject,
            "text": text_body,
            "html": html_body,
        }

        # Check suppression or mock environment
        suppress_send = self.config.get("MAIL_SUPPRESS_SEND", False)
        server_host = self.config.get("MAIL_SERVER", "").strip()

        if suppress_send or not server_host or server_host in ("mock", "console", "test", "localhost"):
            with self._lock:
                self.outbox.append(message_record)
            logger.info(
                "Transactional email queued to outbox (suppressed/console): to='%s', subject='%s'",
                to_email,
                subject,
            )
            return True

        # Live SMTP delivery
        try:
            port = int(self.config.get("MAIL_PORT", 587))
            use_tls = bool(self.config.get("MAIL_USE_TLS", True))
            use_ssl = bool(self.config.get("MAIL_USE_SSL", False))
            username = self.config.get("MAIL_USERNAME", "")
            password = self.config.get("MAIL_PASSWORD", "")
            timeout = float(self.config.get("MAIL_TIMEOUT_SECONDS", 10.0))

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = sender_address
            msg["To"] = to_email

            part_text = MIMEText(text_body, "plain", "utf-8")
            msg.attach(part_text)
            if html_body:
                part_html = MIMEText(html_body, "html", "utf-8")
                msg.attach(part_html)

            if use_ssl:
                server = smtplib.SMTP_SSL(server_host, port, timeout=timeout)
            else:
                server = smtplib.SMTP(server_host, port, timeout=timeout)

            with server:
                if use_tls and not use_ssl:
                    server.starttls()
                if username and password:
                    server.login(username, password)
                server.sendmail(sender_address, [to_email], msg.as_string())

            with self._lock:
                self.outbox.append(message_record)
            logger.info("Email delivered via SMTP to '%s' (subject='%s')", to_email, subject)
            return True

        except Exception as err:
            logger.error("Failed to send transactional email to '%s': %s", to_email, err)
            return False

    def send_password_reset_email(
        self,
        to_email: str,
        token: str,
        recipient_name: Optional[str] = None,
        base_url: Optional[str] = None,
        sync: bool = False,
    ) -> bool:
        """Construct and send password recovery email."""
        subject = "Reset Your MediAssist Password"
        text_body, html_body = self._render_password_reset_content(
            token=token,
            recipient_name=recipient_name,
            base_url=base_url,
        )
        return self.send_email(
            to_email=to_email,
            subject=subject,
            text_body=text_body,
            html_body=html_body,
            sync=sync,
        )


email_service = EmailService()
