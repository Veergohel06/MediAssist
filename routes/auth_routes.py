"""Authentication & Session Management Routes.

Provides login, registration, password recovery via secure single-use tokens,
logout, and session identity endpoints.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user
from flask_wtf.csrf import generate_csrf

from models import User, db, validate_registration_input
from services.auth_service import (
    confirm_password_reset,
    login_ip_limiter,
    request_password_reset,
)

logger = logging.getLogger("mediassist.auth")
auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Handle user authentication for standard browser forms and JSON clients."""
    if request.method == "GET":
        if current_user.is_authenticated:
            return redirect(url_for("chat.chat_view"))
        return render_template("login.html", next_url=request.args.get("next", ""))

    client_ip = request.remote_addr or "127.0.0.1"
    is_testing = current_app.config.get("TESTING", False)

    if not login_ip_limiter.check(client_ip, is_testing=is_testing):
        logger.warning("Auth failure: Too many login attempts from IP %s", client_ip)
        msg = "Too many login attempts. Please wait a minute before trying again."
        if request.is_json:
            return jsonify({"error": msg}), 429
        flash(msg, "danger")
        return render_template("login.html"), 429

    data = request.get_json(silent=True) or request.form
    raw_email = data.get("email") or ""
    email = raw_email.strip().lower()
    password = data.get("password") or ""
    remember = str(data.get("remember", "")).lower() in ("true", "1", "on")

    if not email or not password:
        msg = "Please enter both email and password."
        if request.is_json:
            return jsonify({"error": msg}), 400
        flash(msg, "danger")
        return render_template("login.html"), 400

    user = User.find_by_email(email)

    if user and user.is_locked():
        logger.warning("Auth attempt on locked account (id=%s, email=%s, ip=%s)", user.id, email, client_ip)
        msg = "Your account is temporarily locked due to excessive failed attempts. Please try again in 5 minutes."
        if request.is_json:
            return jsonify({"error": msg}), 403
        flash(msg, "danger")
        return render_template("login.html"), 403

    if not user or not user.check_password(password):
        login_ip_limiter.record(client_ip)
        if user:
            user.record_failed_login(lockout_minutes=5, max_attempts=5)
            db.session.commit()

        logger.info("Auth failure: Invalid credentials for '%s' from IP %s", email, client_ip)
        generic_err = "Invalid email or password."
        if request.is_json:
            return jsonify({"error": generic_err}), 401
        flash(generic_err, "danger")
        return render_template("login.html"), 401

    # Login succeeded
    user.reset_failed_logins()
    db.session.commit()
    login_user(user, remember=remember)
    login_ip_limiter.reset(client_ip)

    logger.info("User logged in successfully (id=%s, email=%s, ip=%s).", user.id, user.email, client_ip)

    next_page = request.args.get("next") or data.get("next")
    # Validate next_page is a safe internal path (no external redirect)
    if next_page:
        parsed = urlparse(next_page)
        if (
            parsed.netloc
            or parsed.scheme
            or not next_page.startswith("/")
            or next_page.startswith("//")
            or "\\" in next_page
            or next_page in ("/dashboard", "/dashboard/")
        ):
            next_page = url_for("chat.chat_view")
    else:
        next_page = url_for("chat.chat_view")

    if request.is_json:
        return jsonify({
            "status": "ok",
            "message": "Login successful.",
            "user": user.to_dict(),
            "redirect_url": next_page,
        }), 200

    flash(f"Welcome back, {user.name}!", "success")
    return redirect(next_page)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """Handle new user registration with input validation and secure hashing."""
    if request.method == "GET":
        if current_user.is_authenticated:
            return redirect(url_for("chat.chat_view"))
        return render_template("register.html")

    data = request.get_json(silent=True) or request.form
    name = (data.get("name") or "").strip()
    raw_email = data.get("email") or ""
    email = raw_email.strip().lower()
    password = data.get("password") or ""
    password_confirm = data.get("password_confirm") or ""

    is_valid, err_msg = validate_registration_input(
        name=name,
        email=email,
        password=password,
        password_confirm=password_confirm,
    )
    if not is_valid:
        if request.is_json:
            return jsonify({"error": err_msg}), 400
        flash(err_msg, "danger")
        return render_template("register.html"), 400

    existing_user = User.find_by_email(email)
    if existing_user:
        logger.info("Registration rejected: Duplicate email '%s'.", email)
        msg = "An account with that email address already exists. Please sign in."
        if request.is_json:
            return jsonify({"error": msg}), 409
        flash(msg, "warning")
        return render_template("register.html"), 409

    new_user = User(name=name, email=email)
    new_user.set_password(password)

    try:
        db.session.add(new_user)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        logger.error("Database error during registration: %s", exc)
        msg = "Registration failed due to a server error. Please try again."
        if request.is_json:
            return jsonify({"error": msg}), 500
        flash(msg, "danger")
        return render_template("register.html"), 500

    logger.info("New user registered successfully (id=%s, email=%s).", new_user.id, new_user.email)
    login_user(new_user, remember=False)

    next_page = request.args.get("next") or data.get("next") or url_for("chat.chat_view")
    if next_page:
        parsed = urlparse(next_page)
        if (
            parsed.netloc
            or parsed.scheme
            or not next_page.startswith("/")
            or next_page.startswith("//")
            or "\\" in next_page
            or next_page in ("/dashboard", "/dashboard/")
        ):
            next_page = url_for("chat.chat_view")
    else:
        next_page = url_for("chat.chat_view")

    if request.is_json:
        return jsonify({
            "status": "ok",
            "message": "Account created successfully.",
            "user": new_user.to_dict(),
            "redirect_url": next_page,
        }), 201

    flash("Account created successfully! Welcome to MediAssist.", "success")
    return redirect(next_page)


@auth_bp.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    """Secure password reset endpoint supporting two-step token workflow.

    Step 1 (Request reset): User provides email -> returns enumeration-safe status.
    Step 2 (Confirm reset): User provides token + new password -> verifies token and updates password.
    """
    if request.method == "GET":
        if current_user.is_authenticated:
            return redirect(url_for("chat.chat_view"))
        token = request.args.get("token", "").strip()
        email = request.args.get("email", "").strip()
        return render_template("reset_password.html", token=token, email=email)

    data = request.get_json(silent=True) or request.form
    client_ip = request.remote_addr or "127.0.0.1"
    is_testing = current_app.config.get("TESTING", False)

    raw_token = (data.get("token") or request.args.get("token") or "").strip()
    raw_email = (data.get("email") or "").strip().lower()
    new_password = (data.get("new_password") or data.get("password") or "").strip()
    new_password_confirm = (data.get("new_password_confirm") or data.get("password_confirm") or "").strip()

    # Step 2: Confirmation with token
    if raw_token:
        res_data, status_code = confirm_password_reset(
            raw_token=raw_token,
            new_password=new_password,
            new_password_confirm=new_password_confirm,
            email=raw_email or None,
        )
        if status_code == 200:
            if request.is_json:
                return jsonify(res_data), 200
            flash(res_data["message"], "success")
            return redirect(url_for("auth.login"))
        else:
            if request.is_json:
                return jsonify(res_data), status_code
            flash(res_data.get("error", "Password reset failed."), "danger")
            return render_template("reset_password.html", token=raw_token, email=raw_email), status_code

    # Step 1: Request reset instructions (no token supplied)
    if not raw_email:
        msg = "Please enter your registered email address."
        if request.is_json:
            return jsonify({"error": msg}), 400
        flash(msg, "danger")
        return render_template("reset_password.html"), 400

    res_data, status_code, test_token = request_password_reset(
        email=raw_email,
        client_ip=client_ip,
        is_testing=is_testing,
    )
    if status_code != 200:
        if request.is_json:
            return jsonify(res_data), status_code
        flash(res_data.get("error", "Too many requests."), "danger")
        return render_template("reset_password.html"), status_code

    # If in test mode, attach test_token to JSON for automated verification
    if is_testing and test_token and request.is_json:
        res_data["_test_token"] = test_token

    if request.is_json:
        return jsonify(res_data), 200

    flash(res_data["message"], "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/logout", methods=["POST", "GET"])
@login_required
def logout():
    """Securely terminate session."""
    user_id = current_user.id if current_user else "unknown"
    logout_user()
    session.clear()
    logger.info("User logged out (id=%s).", user_id)

    if request.is_json:
        return jsonify({"status": "logged_out", "message": "Successfully logged out."}), 200

    flash("You have been signed out.", "info")
    return redirect(url_for("core.index"))


@auth_bp.route("/api/auth/me", methods=["GET"])
@login_required
def api_me():
    """Return profile details for currently authenticated user."""
    return jsonify({"user": current_user.to_dict()}), 200


@auth_bp.route("/api/auth/csrf-token", methods=["GET"])
def api_csrf_token():
    """Return fresh CSRF token."""
    token = generate_csrf()
    return jsonify({"csrf_token": token}), 200
