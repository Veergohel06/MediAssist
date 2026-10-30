"""Authentication & Password Security Services.

Handles rate-limited login tracking, cryptographically secure password reset
workflows with single-use token verification, and constant-time token comparison.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Tuple

from flask import session

_reset_mutex = threading.Lock()
from models import (
    User,
    PasswordResetToken,
    db,
    generate_reset_token,
    verify_reset_token,
    invalidate_user_reset_tokens,
    validate_password_strength,
)

logger = logging.getLogger("mediassist.auth")


# Import Redis-backed shared rate limiters
# These are shared across all Gunicorn workers via Redis.
from services.rate_limiter import (
    RateLimiter,
    login_ip_limiter,
    reset_ip_limiter,
    reset_email_limiter,
)

# Backward-compatible alias so that any code importing SlidingWindowRateLimiter
# from auth_service still works (e.g. older test files).
SlidingWindowRateLimiter = RateLimiter


def request_password_reset(
    email: str,
    client_ip: str,
    is_testing: bool = False,
) -> Tuple[dict, int, Optional[str]]:
    """Initiate a password reset request.

    Enforces rate limits, validates format, generates a high-entropy token,
    hashes and stores it, and returns an enumeration-safe response.
    Returns (response_dict, status_code, generated_token_for_tests_only).
    """
    clean_email = email.strip().lower()

    # IP rate limit
    if not reset_ip_limiter.check(client_ip, is_testing=is_testing):
        logger.warning("Password reset rate limit exceeded for IP %s", client_ip)
        return {
            "error": "Too many password reset requests from this network. Please wait a few minutes before trying again."
        }, 429, None

    # Email rate limit
    if clean_email and not reset_email_limiter.check(clean_email, is_testing=is_testing):
        logger.warning("Password reset rate limit exceeded for email: %s", clean_email)
        return {
            "error": "Too many reset attempts for this email address. Please check your inbox or wait 15 minutes."
        }, 429, None

    reset_ip_limiter.record(client_ip)
    if clean_email:
        reset_email_limiter.record(clean_email)

    raw_token = None
    user = User.find_by_email(clean_email) if clean_email else None
    if user:
        raw_token = generate_reset_token(user)
        logger.info("Password reset token generated for user id=%s (email=%s)", user.id, user.email)
        try:
            from services.email_service import email_service
            email_service.send_password_reset_email(
                to_email=user.email,
                token=raw_token,
                recipient_name=getattr(user, "name", None),
                sync=is_testing,
            )
        except Exception as mail_err:
            logger.error("Failed to dispatch password reset email for %s: %s", user.email, mail_err)
    else:
        logger.info("Password reset requested for nonexistent or empty email: '%s'", clean_email)

    # Universal, enumeration-proof message
    safe_response = {
        "status": "ok",
        "message": "If that email address is registered, your password reset instructions have been sent.",
    }
    return safe_response, 200, raw_token


def confirm_password_reset(
    raw_token: str,
    new_password: str,
    new_password_confirm: str,
    email: Optional[str] = None,
) -> Tuple[dict, int]:
    """Verify a reset token and set a new password.

    Ensures token validity, single-use invalidation, password complexity,
    and resets failed attempts.
    """
    clean_token = (raw_token or "").strip()
    if not clean_token:
        return {"error": "Password reset token is required. Please check your reset link."}, 400

    new_pwd = (new_password or "").strip()
    confirm_pwd = (new_password_confirm or "").strip()

    if not new_pwd:
        return {"error": "Please enter a new password."}, 400

    if new_pwd != confirm_pwd:
        return {"error": "Passwords do not match."}, 400

    is_strong, pwd_err = validate_password_strength(new_pwd)
    if not is_strong:
        return {"error": pwd_err or "Password does not meet complexity requirements."}, 400

    with _reset_mutex:
        # Verify token
        try:
            verification = verify_reset_token(clean_token, email=email)
        except Exception as exc:
            db.session.rollback()
            logger.warning("Database error during reset token verification: %s", exc)
            return {
                "error": "Invalid, expired, or already used reset token. Please request a new password reset."
            }, 400

        if not verification:
            logger.warning("Password reset failed: Invalid, expired, or used token.")
            return {
                "error": "Invalid, expired, or already used reset token. Please request a new password reset."
            }, 400

        user, token_record = verification
        user_id = getattr(user, "id", None)
        user_email = getattr(user, "email", "")

        try:
            from models import utc_now
            # Atomically mark current token as used only if it's still unused
            rows_updated = PasswordResetToken.query.filter(
                PasswordResetToken.id == token_record.id,
                PasswordResetToken.used_at.is_(None),
            ).update({"used_at": utc_now()}, synchronize_session=False)
            db.session.commit()

            if rows_updated == 0:
                return {
                    "error": "Invalid, expired, or already used reset token. Please request a new password reset."
                }, 400

            # Invalidate any other active tokens for this user
            invalidate_user_reset_tokens(user_id)

            # Update password
            user.set_password(new_pwd)
            user.reset_failed_logins()

            db.session.commit()

            from flask import has_request_context
            if has_request_context():
                session.clear()

            logger.info("Password reset successfully completed for user id=%s (email=%s)", user_id, user_email)
            return {
                "status": "ok",
                "message": "Your password has been updated successfully. Please sign in with your new password.",
            }, 200
        except Exception as exc:
            db.session.rollback()
            logger.warning("Database error or concurrency conflict during password reset for user id=%s: %s", user_id, exc)
            return {"error": "Invalid, expired, or already used reset token. Please request a new password reset."}, 400
