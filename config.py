"""Application Configuration for MediAssist.

Provides environment-based configuration classes for development, testing,
and production deployments.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load local environment variables
load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class Config:
    """Base configuration with safe clinical defaults."""

    # Application Paths
    BASE_DIR = BASE_DIR
    KNOWLEDGE_DIR = BASE_DIR / "knowledge"
    FAISS_INDEX_PATH = str(KNOWLEDGE_DIR / "faiss_index.bin")
    METADATA_PATH = str(KNOWLEDGE_DIR / "chunks_metadata.json")
    UPLOAD_FOLDER = str(BASE_DIR / "instance" / "uploads")

    # Document & Upload Limits
    MAX_UPLOAD_FILE_SIZE_BYTES = _env_int("MAX_UPLOAD_FILE_SIZE_BYTES", 15 * 1024 * 1024)  # 15 MB
    MAX_DOCUMENT_PAGES = _env_int("MAX_DOCUMENT_PAGES", 50)
    MAX_DOCUMENTS_PER_CONVERSATION = _env_int("MAX_DOCUMENTS_PER_CONVERSATION", 5)
    DOCUMENT_EXTRACTION_TIMEOUT_SECONDS = _env_float("DOCUMENT_EXTRACTION_TIMEOUT_SECONDS", 30.0)
    ALLOWED_EXTENSIONS = frozenset({"pdf", "txt", "md"})
    ASYNC_DOCUMENT_PROCESSING = _env_bool("ASYNC_DOCUMENT_PROCESSING", True)

    # Security & Sessions
    SECRET_KEY = os.environ.get("SECRET_KEY", "mediassist-dev-secret-key-replace-in-production").strip()
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = _env_bool("SESSION_COOKIE_SECURE", False)
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_SECURE = SESSION_COOKIE_SECURE
    WTF_CSRF_ENABLED = _env_bool("WTF_CSRF_ENABLED", True)
    WTF_CSRF_CHECK_DEFAULT = True

    # Database
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        SQLALCHEMY_DATABASE_URI = db_url
        SQLALCHEMY_ENGINE_OPTIONS = {
            "pool_pre_ping": True,
            "pool_recycle": _env_int("DB_POOL_RECYCLE_SECONDS", 280),
            "pool_timeout": _env_int("DB_POOL_TIMEOUT_SECONDS", 30),
            "pool_size": _env_int("DB_POOL_SIZE", 10),
            "max_overflow": _env_int("DB_MAX_OVERFLOW", 15),
            "connect_args": {
                "connect_timeout": 10,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        }
    else:
        instance_dir = BASE_DIR / "instance"
        instance_dir.mkdir(exist_ok=True)
        db_path = str(instance_dir / "mediassist.db").replace("\\", "/")
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_path}"
        SQLALCHEMY_ENGINE_OPTIONS = {
            "pool_pre_ping": True,
            "pool_recycle": _env_int("DB_POOL_RECYCLE_SECONDS", 280),
        }

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Message & History Limits
    MAX_USER_MSG_LEN = 2_000
    MAX_HISTORY_TURNS = 20
    MAX_STORED_MSG_LEN = 4_000

    # AI Model Settings
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    GEMINI_TEMPERATURE = _env_float("GEMINI_TEMPERATURE", 0.2)
    GEMINI_MAX_OUTPUT_TOKENS = _env_int("GEMINI_MAX_OUTPUT_TOKENS", 850)

    # RAG Settings
    RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "gemini-embedding-2").strip()
    RAG_SCORE_THRESHOLD = _env_float("RAG_SCORE_THRESHOLD", 0.58)
    RAG_SEMANTIC_WEIGHT = _env_float("RAG_SEMANTIC_WEIGHT", 0.7)
    RAG_KEYWORD_WEIGHT = _env_float("RAG_KEYWORD_WEIGHT", 0.3)
    RAG_TOP_K = _env_int("RAG_TOP_K", 3)
    RAG_CANDIDATE_K = _env_int("RAG_CANDIDATE_K", 12)
    RAG_SEMANTIC_CANDIDATE_K = _env_int("RAG_SEMANTIC_CANDIDATE_K", 12)
    RAG_KEYWORD_CANDIDATE_K = _env_int("RAG_KEYWORD_CANDIDATE_K", 12)
    RAG_RRF_K = _env_int("RAG_RRF_K", 60)
    RAG_RETRIEVAL_MODE = os.environ.get("RAG_RETRIEVAL_MODE", "hybrid").strip().lower()
    RAG_RERANK_ENABLED = _env_bool("RAG_RERANK_ENABLED", False)
    RAG_RERANKER_TYPE = os.environ.get("RAG_RERANKER_TYPE", "token_overlap").strip().lower()
    RAG_RERANKER_MODEL = os.environ.get("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
    RAG_RERANK_CANDIDATE_K = _env_int("RAG_RERANK_CANDIDATE_K", 20)
    RAG_RERANK_TIMEOUT_SECONDS = _env_float("RAG_RERANK_TIMEOUT_SECONDS", 10.0)

    # Cache Settings
    RESPONSE_CACHE_TTL_SECONDS = _env_float("RESPONSE_CACHE_TTL_SECONDS", 86400.0)
    RESPONSE_CACHE_MAX_ENTRIES = _env_int("RESPONSE_CACHE_MAX_ENTRIES", 1000)

    # Rate Limiting
    MAX_IP_ATTEMPTS_PER_MIN = 10
    MAX_PASSWORD_RESET_PER_IP_MIN = 5
    MAX_PASSWORD_RESET_PER_EMAIL_15MIN = 3

    # Email / Transactional Delivery Settings
    MAIL_SERVER = os.environ.get("MAIL_SERVER", "").strip()
    MAIL_PORT = _env_int("MAIL_PORT", 587)
    MAIL_USE_TLS = _env_bool("MAIL_USE_TLS", True)
    MAIL_USE_SSL = _env_bool("MAIL_USE_SSL", False)
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "").strip()
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "").strip()
    MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER", "MediAssist <noreply@mediassist.health>").strip()
    MAIL_SUPPRESS_SEND = _env_bool("MAIL_SUPPRESS_SEND", False)
    MAIL_TIMEOUT_SECONDS = _env_float("MAIL_TIMEOUT_SECONDS", 10.0)
    APP_BASE_URL = os.environ.get("APP_BASE_URL", "").strip()


class DevelopmentConfig(Config):
    """Development settings."""
    DEBUG = True


class TestingConfig(Config):
    """Testing settings with isolated memory database and disabled CSRF."""
    TESTING = True
    DEBUG = False
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
    }
    SERVER_NAME = "localhost"
    ASYNC_DOCUMENT_PROCESSING = False
    MAIL_SUPPRESS_SEND = True
    MAIL_SERVER = "mock"


class ProductionConfig(Config):
    """Production settings enforcing strict security."""
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True

    # In production, require PostgreSQL via DATABASE_URL; do not silently inherit SQLite
    _prod_db = os.environ.get("DATABASE_URL", "").strip()
    if _prod_db:
        if _prod_db.startswith("postgres://"):
            _prod_db = _prod_db.replace("postgres://", "postgresql://", 1)
        SQLALCHEMY_DATABASE_URI = _prod_db
        SQLALCHEMY_ENGINE_OPTIONS = {
            "pool_pre_ping": True,
            "pool_recycle": _env_int("DB_POOL_RECYCLE_SECONDS", 280),
            "pool_timeout": _env_int("DB_POOL_TIMEOUT_SECONDS", 30),
            "pool_size": _env_int("DB_POOL_SIZE", 10),
            "max_overflow": _env_int("DB_MAX_OVERFLOW", 15),
            "connect_args": {
                "connect_timeout": 10,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        }
    else:
        SQLALCHEMY_DATABASE_URI = None
        SQLALCHEMY_ENGINE_OPTIONS = {}
