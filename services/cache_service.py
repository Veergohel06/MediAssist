"""Response Cache Service with Multi-Tenant Isolation & Conversational Context Binding.

Ensures that cached LLM responses never leak across different users,
different consultations, or different turns of the same conversation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mediassist.cache")

CACHE_VERSION = "v2"
DEFAULT_CACHE_TTL = 86400.0  # 24 hours
DEFAULT_MAX_ENTRIES = 1000


class ResponseCache:
    """Thread-safe, multi-tenant clinical response cache.

    Binds cached entries to:
    - User ID (tenant isolation)
    - Conversation ID (session/thread isolation)
    - Conversation History Hash (multi-turn context isolation)
    - AI Model Name & Retrieval Mode
    - Normalized User Message
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_CACHE_TTL,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        persistence_path: Optional[str] = None,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.persistence_path = persistence_path
        self._lock = threading.RLock()
        # key -> (timestamp, reply, sources, triage, user_id, conversation_id)
        self._cache: Dict[str, Tuple[float, str, List[dict], dict, Optional[int], Optional[str]]] = {}
        # Inverted index: conversation_id -> set of keys (for instant cleanup on reset/delete)
        self._conv_keys: Dict[str, set[str]] = {}
        # Inverted index: user_id -> set of keys
        self._user_keys: Dict[int, set[str]] = {}

    @staticmethod
    def compute_history_hash(history: Optional[List[dict]]) -> str:
        """Compute deterministic SHA-256 hash of previous turns in the conversation.

        If history is empty or None, returns 'empty'.
        Otherwise hashes role and text of all prior completed messages.
        """
        if not history:
            return "empty"

        canonical_parts = []
        for entry in history:
            role = str(entry.get("role", "")).strip().lower()
            parts = entry.get("parts", [])
            text = ""
            if parts and isinstance(parts, list):
                first_part = parts[0]
                if isinstance(first_part, dict):
                    text = first_part.get("text", "")
                elif hasattr(first_part, "text"):
                    text = getattr(first_part, "text", "")
            elif isinstance(entry.get("content"), str):
                text = entry["content"]
            canonical_parts.append(f"{role}:{text.strip()}")

        raw = "|".join(canonical_parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def generate_cache_key(
        cls,
        user_message: str,
        user_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        history: Optional[List[dict]] = None,
        model_name: str = "default",
        retrieval_mode: str = "hybrid",
    ) -> str:
        """Construct a collision-proof, deterministic cache key.

        Includes user identity, consultation thread, conversation history hash,
        model parameters, and normalized question.
        """
        norm_query = " ".join(user_message.strip().lower().split())
        history_hash = cls.compute_history_hash(history)
        uid_str = str(user_id) if user_id is not None else "anon"
        cid_str = str(conversation_id) if conversation_id is not None else "standalone"

        raw_key = (
            f"{CACHE_VERSION}:"
            f"u={uid_str}:"
            f"c={cid_str}:"
            f"h={history_hash}:"
            f"m={model_name.strip().lower()}:"
            f"r={retrieval_mode.strip().lower()}:"
            f"q={norm_query}"
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(
        self,
        user_message: str,
        user_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        history: Optional[List[dict]] = None,
        model_name: str = "default",
        retrieval_mode: str = "hybrid",
    ) -> Optional[Tuple[str, List[dict], dict]]:
        """Retrieve a cached response matching the exact user, conversation, history, and query.

        Returns (reply, sources, triage) or None on cache miss / expiration.
        """
        key = self.generate_cache_key(
            user_message=user_message,
            user_id=user_id,
            conversation_id=conversation_id,
            history=history,
            model_name=model_name,
            retrieval_mode=retrieval_mode,
        )

        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None

            created_time, reply, sources, triage, entry_uid, entry_cid = entry

            # Check expiration
            if (time.time() - created_time) > self.ttl_seconds:
                self._remove_key_locked(key)
                return None

            # Verify tenant isolation invariants
            if entry_uid is not None and user_id is not None and entry_uid != user_id:
                logger.error("SECURITY ALERT: Cache key collision detected between users %s and %s", entry_uid, user_id)
                self._remove_key_locked(key)
                return None

            if entry_cid is not None and conversation_id is not None and entry_cid != conversation_id:
                logger.error("SECURITY ALERT: Cache key collision detected between conversations %s and %s", entry_cid, conversation_id)
                self._remove_key_locked(key)
                return None

            return reply, copy.deepcopy(sources), copy.deepcopy(triage)

    def set(
        self,
        user_message: str,
        reply: str,
        sources: List[dict],
        triage: dict,
        user_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        history: Optional[List[dict]] = None,
        model_name: str = "default",
        retrieval_mode: str = "hybrid",
    ) -> str:
        """Store a response under the fully isolated composite key."""
        # Never cache emergency dispatch responses
        if triage.get("is_emergency") or triage.get("urgency") == "EMERGENCY":
            return ""

        key = self.generate_cache_key(
            user_message=user_message,
            user_id=user_id,
            conversation_id=conversation_id,
            history=history,
            model_name=model_name,
            retrieval_mode=retrieval_mode,
        )

        with self._lock:
            # Enforce max entries via LRU-style eviction (first inserted item)
            if len(self._cache) >= self.max_entries and key not in self._cache:
                oldest_key = next(iter(self._cache))
                self._remove_key_locked(oldest_key)

            self._cache[key] = (
                time.time(),
                reply,
                copy.deepcopy(sources),
                copy.deepcopy(triage),
                user_id,
                conversation_id,
            )

            if conversation_id:
                self._conv_keys.setdefault(str(conversation_id), set()).add(key)
            if user_id is not None:
                self._user_keys.setdefault(user_id, set()).add(key)

        return key

    def invalidate_conversation(self, conversation_id: str) -> int:
        """Purge all cached responses for a specific consultation thread."""
        cid_str = str(conversation_id)
        count = 0
        with self._lock:
            keys = self._conv_keys.pop(cid_str, set())
            for k in keys:
                if k in self._cache:
                    self._remove_key_locked(k)
                    count += 1
        return count

    def invalidate_user(self, user_id: int) -> int:
        """Purge all cached responses belonging to a specific user."""
        count = 0
        with self._lock:
            keys = self._user_keys.pop(user_id, set())
            for k in keys:
                if k in self._cache:
                    self._remove_key_locked(k)
                    count += 1
        return count

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._conv_keys.clear()
            self._user_keys.clear()

    def _remove_key_locked(self, key: str) -> None:
        """Remove a key and clean up inverted indices (must be called with lock held)."""
        entry = self._cache.pop(key, None)
        if entry:
            _, _, _, _, uid, cid = entry
            if cid and str(cid) in self._conv_keys:
                self._conv_keys[str(cid)].discard(key)
                if not self._conv_keys[str(cid)]:
                    self._conv_keys.pop(str(cid), None)
            if uid is not None and uid in self._user_keys:
                self._user_keys[uid].discard(key)
                if not self._user_keys[uid]:
                    self._user_keys.pop(uid, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
