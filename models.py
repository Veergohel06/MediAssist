"""Database models and user authentication entities for MediAssist.

Provides persistent User accounts, secure password hashing via Werkzeug,
failed login tracking/lockout, and strictly isolated multi-tenant Conversation
and ChatMessage history.
"""

from __future__ import annotations

import json
import re
import uuid
import secrets
import hashlib
import hmac
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple, Set

from email_validator import EmailNotValidError, validate_email
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event, Index, CheckConstraint
from sqlalchemy.engine import Engine
from sqlalchemy.orm import validates
from sqlalchemy.orm.attributes import set_committed_value
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


def utc_now() -> datetime:
    """Return naive UTC timestamp to ensure consistent comparisons across databases."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# SQLite Foreign Key Pragma Enforcement
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    """Ensure SQLite enforces foreign key constraints and cascade actions on every connection."""
    if type(dbapi_connection).__module__ == "sqlite3":
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# Supported role and status definitions for persistent chat history
SUPPORTED_ROLES: frozenset[str] = frozenset({"user", "assistant", "model", "system"})
SUPPORTED_MESSAGE_STATUSES: frozenset[str] = frozenset({"completed", "streaming", "pending", "error"})

SENSITIVE_METADATA_KEYS: frozenset[str] = frozenset({
    "api_key",
    "apikey",
    "gemini_api_key",
    "secret",
    "password",
    "auth",
    "authorization",
    "bearer",
    "token",
    "access_token",
    "private_key",
    "thought",
    "reasoning",
    "chain_of_thought",
    "hidden_reasoning",
})

API_KEY_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"sk-[0-9A-Za-z]{20,}"),
]


def sanitize_evidence_metadata(data: Any) -> Optional[Any]:
    """Recursively clean source/evidence metadata before persistence.

    - Strips internal reasoning, chain-of-thought, or hidden thoughts.
    - Strips API keys, passwords, bearer tokens, or secrets.
    - Redacts any API key strings found in string values.
    - Ensures the resulting payload is strictly JSON-serializable.
    """
    if data is None:
        return None

    # Allow valid JSON string input by deserializing
    if isinstance(data, str):
        trimmed = data.strip()
        if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
            try:
                data = json.loads(trimmed)
            except Exception:
                raise ValueError("Malformed source metadata: invalid JSON string.")
        else:
            raise ValueError("Evidence metadata must be a JSON object, list, or valid JSON string.")

    if not isinstance(data, (dict, list)):
        raise ValueError(f"Evidence metadata must be a dictionary or list, got {type(data).__name__}.")

    def _clean(val: Any) -> Any:
        if isinstance(val, dict):
            cleaned = {}
            for k, v in val.items():
                k_str = str(k)
                k_norm = k_str.lower().replace("-", "_").strip()
                if any(sens in k_norm for sens in SENSITIVE_METADATA_KEYS):
                    continue
                cleaned[k_str] = _clean(v)
            return cleaned
        elif isinstance(val, list):
            return [_clean(elem) for elem in val]
        elif isinstance(val, str):
            res = val
            for pat in API_KEY_PATTERNS:
                res = pat.sub("[REDACTED_API_KEY]", res)
            return res
        elif isinstance(val, (int, float, bool)) or val is None:
            return val
        else:
            return str(val)

    cleaned = _clean(data)
    # Ensure JSON serializable
    json.dumps(cleaned)
    return cleaned


# Supported student profile settings for MBBS learning assistant
VALID_ACADEMIC_YEARS: frozenset[str] = frozenset({
    "1st_year",
    "2nd_year",
    "3rd_year",
    "final_year",
    "intern",
})
VALID_TARGET_EXAMS: frozenset[str] = frozenset({
    "university_profs",
    "neet_pg_next",
    "usmle_step1",
    "usmle_step2",
})


class User(UserMixin, db.Model):
    """User account model with secure password hashing and rate-limited lockout."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False)
    failed_login_attempts = db.Column(db.Integer, default=0, nullable=False)
    locked_until = db.Column(db.DateTime, nullable=True)

    # Optional student profile attributes for MBBS academic adaptation
    academic_year = db.Column(db.String(20), nullable=True)
    target_exam = db.Column(db.String(30), nullable=True)

    conversations = db.relationship(
        "Conversation",
        backref="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
        order_by="Conversation.updated_at.desc()",
    )
    reset_tokens = db.relationship(
        "PasswordResetToken",
        backref="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
        order_by="PasswordResetToken.created_at.desc()",
    )

    @classmethod
    def find_by_email(cls, email: Optional[str]) -> Optional[User]:
        """Look up user by email case-insensitively with leading/trailing whitespace removed."""
        if not email or not str(email).strip():
            return None
        clean_email = str(email).strip().lower()
        return cls.query.filter(db.func.lower(db.func.trim(cls.email)) == clean_email).first()

    def set_password(self, password: str) -> None:
        """Hash and store password using Werkzeug's secure hashing (scrypt)."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify the password against the stored hash in constant time."""
        if not self.password_hash or not password:
            return False

        # Guard against truncated or corrupted hashes
        if len(self.password_hash) < 30 or "$" not in self.password_hash:
            return False

        return check_password_hash(self.password_hash, password)

    def is_locked(self) -> bool:
        """Check if the user is currently locked out due to excessive failed attempts."""
        if self.locked_until and self.locked_until > utc_now():
            return True
        return False

    def record_failed_login(self, lockout_minutes: int = 5, max_attempts: int = 5) -> None:
        """Increment failed attempts and lock out user if threshold is reached."""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= max_attempts:
            self.locked_until = utc_now() + timedelta(minutes=lockout_minutes)

    def reset_failed_logins(self) -> None:
        """Reset failed login count and lockout status upon successful authentication."""
        self.failed_login_attempts = 0
        self.locked_until = None

    @validates("academic_year")
    def validate_academic_year(self, key: str, value: Any) -> Optional[str]:
        """Validate optional academic year for MBBS student profile."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        val = str(value).strip().lower()
        if val not in VALID_ACADEMIC_YEARS:
            raise ValueError(
                f"Invalid academic year '{value}'. Supported values are: {sorted(VALID_ACADEMIC_YEARS)}"
            )
        return val

    @validates("target_exam")
    def validate_target_exam(self, key: str, value: Any) -> Optional[str]:
        """Validate optional target examination for MBBS student profile."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        val = str(value).strip().lower()
        if val not in VALID_TARGET_EXAMS:
            raise ValueError(
                f"Invalid target exam '{value}'. Supported values are: {sorted(VALID_TARGET_EXAMS)}"
            )
        return val

    def to_dict(self) -> Dict[str, Any]:
        """Return safe, non-sensitive representation of the user."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "academic_year": self.academic_year,
            "target_exam": self.target_exam,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return f"<User id={self.id} email='{self.email}'>"


class Conversation(db.Model):
    """User-isolated consultation conversation container."""

    __tablename__ = "conversations"
    __table_args__ = (
        db.Index("ix_conversations_user_updated", "user_id", "updated_at"),
    )

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title = db.Column(db.String(200), default="New Consultation", nullable=False)
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
        index=True,
    )

    messages = db.relationship(
        "ChatMessage",
        backref="conversation",
        cascade="all, delete-orphan",
        lazy="dynamic",
        order_by="[ChatMessage.created_at.asc(), ChatMessage.id.asc()]",
    )
    documents = db.relationship(
        "ConversationDocument",
        backref="conversation",
        cascade="all, delete-orphan",
        lazy="dynamic",
        order_by="ConversationDocument.created_at.asc()",
    )

    @validates("user_id")
    def validate_user_id(self, key: str, value: Any) -> int:
        """Ensure conversation strictly has a valid owner."""
        if value is None:
            raise ValueError("Conversation must have an owner (user_id cannot be None).")
        try:
            uid = int(value)
            if uid <= 0:
                raise ValueError
            return uid
        except (ValueError, TypeError):
            raise ValueError("Conversation user_id must be a positive integer.")

    @validates("title")
    def validate_title(self, key: str, value: Any) -> str:
        """Ensure safe default title and bounded length."""
        if not value or not str(value).strip():
            return "New Consultation"
        return str(value).strip()[:200]

    def touch(self) -> None:
        """Update updated_at timestamp to current UTC time."""
        self.updated_at = utc_now()

    def to_dict(self, include_messages: bool = False, include_documents: bool = False) -> Dict[str, Any]:
        """Serialize conversation details with optional message thread and attachments."""
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": self.messages.count(),
            "document_count": self.documents.count() if hasattr(self, "documents") else 0,
        }
        if include_messages:
            data["messages"] = [m.to_dict() for m in self.messages.all()]
        if include_documents and hasattr(self, "documents"):
            data["documents"] = [d.to_dict() for d in self.documents.all()]
        return data

    def __repr__(self) -> str:
        return f"<Conversation id='{self.id}' user_id={self.user_id}>"


class ConversationDocument(db.Model):
    """User-uploaded study note or medical PDF attached to a consultation session."""

    __tablename__ = "conversation_documents"
    __table_args__ = (
        db.Index("ix_conv_docs_conv_created", "conversation_id", "created_at"),
        db.Index("ix_conv_docs_user_created", "user_id", "created_at"),
    )

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = db.Column(
        db.String(36),
        db.ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename = db.Column(db.String(255), nullable=False)
    storage_path = db.Column(db.String(512), nullable=False)
    file_size_bytes = db.Column(db.Integer, nullable=False, default=0)
    file_type = db.Column(db.String(20), nullable=False)  # 'pdf' | 'text'
    content_sha256 = db.Column(db.String(64), nullable=False)
    page_count = db.Column(db.Integer, nullable=False, default=1)
    chunk_count = db.Column(db.Integer, nullable=False, default=0)
    status = db.Column(db.String(20), default="ready", nullable=False)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False)

    @validates("filename")
    def validate_filename(self, key: str, value: Any) -> str:
        if not value or not str(value).strip():
            return "untitled_document"
        return str(value).strip()[:255]

    def to_dict(self) -> Dict[str, Any]:
        """Return safe, non-sensitive client representation without exposing internal server paths."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "filename": self.filename,
            "file_size_bytes": self.file_size_bytes,
            "file_type": self.file_type,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return f"<ConversationDocument id='{self.id}' name='{self.filename}' conv='{self.conversation_id}'>"


class ChatMessage(db.Model):
    """Individual message record in a consultation conversation."""

    __tablename__ = "chat_messages"
    __table_args__ = (
        db.CheckConstraint(
            "role IN ('user', 'assistant', 'model', 'system')",
            name="ck_chat_messages_role",
        ),
        db.Index("ix_chat_messages_conv_created", "conversation_id", "created_at", "id"),
    )

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(
        db.String(36),
        db.ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = db.Column(db.String(20), nullable=False)  # 'user' | 'model' | 'assistant' | 'system'
    content = db.Column(db.Text, nullable=False)
    evidence = db.Column(db.JSON, nullable=True)  # RAG sources and citation metadata
    status = db.Column(db.String(20), default="completed", server_default="completed", nullable=False)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False, index=True)

    @validates("conversation_id")
    def validate_conversation_id(self, key: str, value: Any) -> str:
        if not value or not str(value).strip():
            raise ValueError("ChatMessage must belong to a conversation (conversation_id required).")
        return str(value).strip()

    @validates("role")
    def validate_role(self, key: str, value: Any) -> str:
        if not value or not isinstance(value, str):
            raise ValueError("Message role must be a non-empty string.")
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_ROLES:
            raise ValueError(
                f"Invalid role '{value}'. Supported roles are: {sorted(SUPPORTED_ROLES)}"
            )
        return normalized

    @validates("content")
    def validate_content(self, key: str, value: Any) -> str:
        if value is None:
            raise ValueError("Message content cannot be None.")
        if not isinstance(value, str):
            value = str(value)
        if not value.strip():
            raise ValueError("Message content cannot be empty.")
        return value

    @validates("evidence")
    def validate_evidence(self, key: str, value: Any) -> Optional[Any]:
        return sanitize_evidence_metadata(value)

    @validates("status")
    def validate_status(self, key: str, value: Any) -> str:
        if value is None:
            return "completed"
        normalized = str(value).strip().lower()
        if normalized not in SUPPORTED_MESSAGE_STATUSES:
            raise ValueError(
                f"Invalid message status '{value}'. Supported statuses: {sorted(SUPPORTED_MESSAGE_STATUSES)}"
            )
        return normalized

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message record."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "evidence": self.evidence,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return f"<ChatMessage id={self.id} role='{self.role}' conv='{self.conversation_id}'>"


@event.listens_for(ChatMessage, "after_insert")
def _touch_conversation_on_message_insert(mapper: Any, connection: Any, target: ChatMessage) -> None:
    """Automatically update parent conversation's updated_at timestamp when a message is inserted."""
    if target.conversation_id:
        now = utc_now()
        connection.execute(
            db.update(Conversation)
            .where(Conversation.id == target.conversation_id)
            .values(updated_at=now)
        )
        if getattr(target, "conversation", None) is not None:
            set_committed_value(target.conversation, "updated_at", now)


@event.listens_for(ChatMessage, "after_update")
def _touch_conversation_on_message_update(mapper: Any, connection: Any, target: ChatMessage) -> None:
    """Automatically update parent conversation's updated_at timestamp when a message is updated."""
    if target.conversation_id:
        now = utc_now()
        connection.execute(
            db.update(Conversation)
            .where(Conversation.id == target.conversation_id)
            .values(updated_at=now)
        )
        if getattr(target, "conversation", None) is not None:
            set_committed_value(target.conversation, "updated_at", now)


def validate_password_strength(password: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Validate password length and complexity.

    Returns (is_valid, error_message).
    """
    if not password:
        return False, "Please enter a password."

    if len(password) < 8:
        return False, "Password must be at least 8 characters long."

    if len(password) > 128:
        return False, "Password cannot exceed 128 characters."

    if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        return False, "Password must contain at least one letter and one number."

    return True, None


def validate_registration_input(
    name: Optional[str],
    email: Optional[str],
    password: Optional[str],
    password_confirm: Optional[str],
) -> Tuple[bool, Optional[str]]:
    """Validate user registration inputs.

    Returns (is_valid, error_message).
    """
    if not name or not name.strip():
        return False, "Please enter your full name."

    cleaned_name = name.strip()
    if len(cleaned_name) < 2 or len(cleaned_name) > 100:
        return False, "Name must be between 2 and 100 characters."

    if not email or not email.strip():
        return False, "Please enter your email address."

    try:
        validated = validate_email(email.strip(), check_deliverability=False)
        _ = validated.normalized
    except EmailNotValidError as err:
        return False, f"Invalid email format: {str(err)}"

    pwd_valid, pwd_err = validate_password_strength(password)
    if not pwd_valid:
        return False, pwd_err

    if password != password_confirm:
        return False, "Passwords do not match."

    return True, None


class PasswordResetToken(db.Model):
    """Cryptographically hashed, time-limited password reset tokens."""

    __tablename__ = "password_reset_tokens"
    __table_args__ = (
        db.Index("ix_pwd_reset_user_expires", "user_id", "expires_at"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)
    used_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False)

    def is_valid(self) -> bool:
        """Check if the token has not been used and has not expired."""
        if self.used_at is not None:
            return False
        if self.expires_at <= utc_now():
            return False
        return True

    def __repr__(self) -> str:
        return f"<PasswordResetToken id={self.id} user_id={self.user_id}>"


TOKEN_TTL_MINUTES = 15


def generate_reset_token(user: User, expires_in_minutes: int = TOKEN_TTL_MINUTES) -> str:
    """Generate a high-entropy reset token, store its SHA-256 hash in DB, and return the raw token."""
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    expires_at = utc_now() + timedelta(minutes=expires_in_minutes)

    reset_entry = PasswordResetToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    db.session.add(reset_entry)
    db.session.commit()
    return raw_token


def verify_reset_token(raw_token: str, email: Optional[str] = None) -> Optional[Tuple[User, PasswordResetToken]]:
    """Verify that a raw reset token is valid, unexpired, and unused.
    Optionally verifies that it belongs to the given email if supplied.
    Returns (User, PasswordResetToken) or None.
    """
    if not raw_token or not isinstance(raw_token, str):
        return None
    clean_token = raw_token.strip()
    if len(clean_token) < 20:
        return None

    candidate_hash = hashlib.sha256(clean_token.encode("utf-8")).hexdigest()
    record = PasswordResetToken.query.filter_by(token_hash=candidate_hash).first()
    if not record:
        return None

    # Constant-time comparison of the hash
    if not hmac.compare_digest(record.token_hash, candidate_hash):
        return None

    if not record.is_valid():
        return None

    user = db.session.get(User, record.user_id)
    if not user:
        return None

    if email:
        clean_email = email.strip().lower()
        if user.email.lower() != clean_email:
            return None

    return user, record


def invalidate_user_reset_tokens(user_id: int) -> None:
    """Mark all active reset tokens for a given user as used."""
    PasswordResetToken.query.filter(
        PasswordResetToken.user_id == user_id,
        PasswordResetToken.used_at.is_(None),
    ).update({"used_at": utc_now()}, synchronize_session=False)
    db.session.commit()

