"""MediAssist – Clinical AI & Medical RAG Application.
====================================================
Modular Flask application factory initializing security extensions, database
persistence, route blueprints, and Google Gemini clinical AI services.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, request, url_for
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFError, CSRFProtect
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from config import Config, DevelopmentConfig, ProductionConfig, TestingConfig
from models import User, db
from routes import auth_bp, chat_bp, core_bp
from services.llm_service import LLMService

# Load environment configuration
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mediassist")

# Shared extensions
migrate = Migrate(render_as_batch=True)
csrf = CSRFProtect()
login_manager = LoginManager()


def create_app(config_class: Optional[type] = None) -> Flask:
    """Application factory for MediAssist."""
    app = Flask(__name__)

    # Determine configuration class
    if config_class is None:
        env = os.environ.get("FLASK_ENV", "").lower()
        if env == "production":
            config_class = ProductionConfig
        elif env == "testing":
            config_class = TestingConfig
        else:
            config_class = DevelopmentConfig

    app.config.from_object(config_class)

    # Validate production configuration on any non-development/testing environment
    if not app.config.get("DEBUG") and not app.config.get("TESTING"):
        secret = app.config.get("SECRET_KEY", "")
        if secret == "mediassist-dev-secret-key-replace-in-production" or len(secret) < 24:
            raise SystemExit(
                "ERROR: SECRET_KEY environment variable must be set to a strong random value "
                "(min 24 chars) before running in non-development mode."
            )
        db_uri = app.config.get("SQLALCHEMY_DATABASE_URI")
        if not db_uri or not (str(db_uri).startswith("postgresql://") or str(db_uri).startswith("postgres://") or str(db_uri).startswith("postgresql+")):
            raise SystemExit(
                "ERROR: DATABASE_URL must be configured with a valid PostgreSQL connection string "
                "in production/non-development mode. Silent fallback to SQLite is prohibited."
            )

    # Ensure SQLite configurations strip unsupported pooling arguments
    configured_db_uri = str(app.config.get("SQLALCHEMY_DATABASE_URI") or "")
    if configured_db_uri.startswith("sqlite") and "pool_size" in app.config.get("SQLALCHEMY_ENGINE_OPTIONS", {}):
        sanitized_opts = dict(app.config.get("SQLALCHEMY_ENGINE_OPTIONS", {}))
        sanitized_opts.pop("pool_size", None)
        sanitized_opts.pop("max_overflow", None)
        sanitized_opts.pop("connect_args", None)
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = sanitized_opts

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)
    login_manager.init_app(app)

    # Configure Flask-Login
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please sign in to access MediAssist."
    login_manager.login_message_category = "warning"

    @login_manager.user_loader
    def load_user(user_id: str) -> User | None:
        try:
            return db.session.get(User, int(user_id))
        except OperationalError as exc:
            logger.warning("Database operational error in load_user, rolling back and retrying: %s", exc)
            db.session.rollback()
            try:
                return db.session.get(User, int(user_id))
            except Exception as retry_exc:
                logger.error("Failed to load user on retry: %s", retry_exc)
                db.session.rollback()
                return None
        except (ValueError, TypeError):
            return None
        except SQLAlchemyError as exc:
            logger.error("Database error in load_user: %s", exc)
            db.session.rollback()
            return None

    @login_manager.unauthorized_handler
    def handle_unauthorized():
        if (
            request.path.startswith("/api/")
            or (request.method == "POST" and request.path in ("/chat", "/chat/stream", "/reset"))
            or request.is_json
            or request.headers.get("Accept") == "application/json"
        ):
            return jsonify({"error": "Authentication required. Please log in."}), 401
        return redirect(url_for("auth.login", next=request.path))

    @app.errorhandler(OperationalError)
    def handle_operational_error(exc):
        logger.error("Database OperationalError encountered: %s", exc)
        db.session.rollback()
        if (
            request.path.startswith("/api/")
            or (request.method == "POST" and request.path in ("/chat", "/chat/stream", "/reset"))
            or request.is_json
            or request.headers.get("Accept") == "application/json"
        ):
            return jsonify({"error": "Database connection was interrupted. Please retry."}), 503
        flash("A momentary database connection issue occurred. Please try again.", "warning")
        return redirect(request.referrer or url_for("core.landing"))

    @app.errorhandler(CSRFError)
    def handle_csrf_error(exc):
        reason = getattr(exc, "description", "CSRF token missing or invalid")
        logger.warning("CSRF validation failed: %s", reason)
        if (
            request.path.startswith("/api/")
            or request.path in ("/chat", "/chat/stream", "/reset")
            or request.is_json
            or request.headers.get("Accept") == "application/json"
        ):
            return jsonify({"error": f"CSRF validation failed: {reason}"}), 400
        flash("Session security token expired or invalid. Please try again.", "danger")
        return redirect(request.referrer or url_for("auth.login"))

    # Initialize clinical AI and RAG services
    llm_service = LLMService(app.config)
    app.extensions["llm_service"] = llm_service

    # Initialize transactional email service
    from services.email_service import email_service
    email_service.update_config(app.config)
    app.extensions["email_service"] = email_service

    # Register modular blueprints
    app.register_blueprint(core_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)

    # Endpoint alias mapping for url_for compatibility across blueprints
    endpoint_aliases = {
        "index": "core.index",
        "dashboard": "core.dashboard",
        "chat_view": "chat.chat_view",
        "login": "auth.login",
        "register": "auth.register",
        "reset_password": "auth.reset_password",
        "logout": "auth.logout",
    }

    def handle_url_build_error(error: Exception, endpoint: str, values: dict[str, Any]) -> str | None:
        bare = endpoint.split(".")[-1]
        target = endpoint_aliases.get(bare)
        if target and target != endpoint:
            try:
                return url_for(target, **values)
            except Exception:
                pass
        return None

    app.url_build_error_handlers.append(handle_url_build_error)

    # Template alias helpers for url_for compatibility
    app.view_functions.setdefault("login", app.view_functions.get("auth.login"))
    app.view_functions.setdefault("register", app.view_functions.get("auth.register"))
    app.view_functions.setdefault("reset_password", app.view_functions.get("auth.reset_password"))
    app.view_functions.setdefault("logout", app.view_functions.get("auth.logout"))
    app.view_functions.setdefault("chat_view", app.view_functions.get("chat.chat_view"))
    app.view_functions.setdefault("dashboard", app.view_functions.get("core.dashboard"))
    app.view_functions.setdefault("index", app.view_functions.get("core.index"))

    # For SQLite (local dev & testing), auto-create tables if not present.
    # For PostgreSQL (Neon/production), schema is managed strictly via Flask-Migrate migrations.
    with app.app_context():
        if db.engine.dialect.name == "sqlite":
            db.create_all()
            try:
                from sqlalchemy import inspect, text
                inspector = inspect(db.engine)
                if "users" in inspector.get_table_names():
                    cols = {c["name"] for c in inspector.get_columns("users")}
                    with db.engine.begin() as conn:
                        if "academic_year" not in cols:
                            conn.execute(text("ALTER TABLE users ADD COLUMN academic_year VARCHAR(20)"))
                        if "target_exam" not in cols:
                            conn.execute(text("ALTER TABLE users ADD COLUMN target_exam VARCHAR(30)"))
            except Exception as schema_err:
                logger.debug("Schema column check: %s", schema_err)

    return app


# Module-level instance for test runners, WSGI servers, and migrations
app = create_app()

# Backward-compatibility exports for existing unit tests and scripts
from services.auth_service import login_ip_limiter
_ip_failed_logins = login_ip_limiter._records

_client = app.extensions["llm_service"].client

GEMINI_MODEL = app.config.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
_safety_evaluator = app.extensions["llm_service"].safety_evaluator
_retriever = app.extensions["llm_service"].retriever
_index_path = getattr(_retriever, "index_path", "knowledge/faiss_index.bin")
_metadata_path = getattr(_retriever, "metadata_path", "knowledge/chunks_metadata.json")
_conversation_history: list[dict] = []


def get_gemini_response(
    user_message: str,
    history: list[dict] | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[str, list[dict], dict]:
    """Backward-compatible wrapper for LLM generation service."""
    service: LLMService = app.extensions["llm_service"]
    user_id = kwargs.get("user_id")
    conversation_id = kwargs.get("conversation_id")
    academic_year = kwargs.get("academic_year")
    target_exam = kwargs.get("target_exam")
    selected_option = kwargs.get("selected_option")
    question_id = kwargs.get("question_id") or kwargs.get("mcq_id")
    return service.get_response(
        user_message=user_message,
        history=history,
        user_id=user_id,
        conversation_id=conversation_id,
        academic_year=academic_year,
        target_exam=target_exam,
        selected_option=selected_option,
        question_id=question_id,
    )


if __name__ == "__main__":
    is_debug = app.config.get("DEBUG", False)
    host = "0.0.0.0" if not is_debug else "127.0.0.1"
    port = int(os.environ.get("PORT", 5000))

    print("=" * 55)
    print(f"  MediAssist (Gemini AI) -> http://{host}:{port}")
    print(f"  Model : {GEMINI_MODEL}")
    print(f"  Debug : {is_debug}")
    print("  Press Ctrl+C to stop.")
    print("=" * 55)
    app.run(host=host, port=port, debug=is_debug)
