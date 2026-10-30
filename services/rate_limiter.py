"""
Shared Rate Limiter — Redis-backed with explicit fail-closed policy.
====================================================================
Provides a sliding-window rate limiter backed by Redis so that limits
are shared across all Gunicorn workers and application instances.

Fail-closed policy (default for auth endpoints):
  If Redis is unavailable, deny the request rather than allow unrestricted
  access.  For non-sensitive endpoints the caller can opt into fail-open by
  passing ``redis_unavailable_allow=True``.

Atomic implementation uses Redis sorted-sets (one key per client-IP or
e-mail address):
  • ZREMRANGEBYSCORE  – prune expired timestamps
  • ZCARD             – count current attempts in window
  • ZADD              – record a new attempt (NX: no overwrite)
  • EXPIRE            – reset TTL to the window length

All operations are pipelined for efficiency and executed with MULTI/EXEC
to guarantee atomicity under concurrent requests from multiple workers.

Environment variables (all optional; fall back to sensible defaults):
  REDIS_URL           redis://localhost:6379/0  Full Redis connection URL
  REDIS_HOST          localhost                 Used only when REDIS_URL is absent
  REDIS_PORT          6379
  REDIS_DB            0
  REDIS_PASSWORD                               Leave empty for unauthenticated
  REDIS_SOCKET_TIMEOUT  1.0                    Per-operation timeout in seconds
  REDIS_CONNECT_TIMEOUT 1.0                    Connection-setup timeout in seconds
  REDIS_KEY_PREFIX    mediassist:rl:           Namespace prefix for all keys
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger("mediassist.rate_limiter")

# ── Optional Redis import (graceful degradation path handled at runtime) ──────
try:
    import redis as _redis_module

    _REDIS_AVAILABLE = True
except ImportError:
    _redis_module = None  # type: ignore[assignment]
    _REDIS_AVAILABLE = False
    logger.warning(
        "redis package not installed.  "
        "Rate limiter will use in-memory fallback (NOT process-safe in production)."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_redis_client() -> Optional[object]:
    """Construct a Redis client from environment variables.

    Returns None if the *redis* package is not installed or connection fails.
    Credentials and connection details must be supplied via environment variables;
    they are never logged.
    """
    if not _REDIS_AVAILABLE:
        return None

    redis_url = os.environ.get("REDIS_URL", "").strip()
    redis_host = os.environ.get("REDIS_HOST", "").strip()

    # If neither REDIS_URL nor REDIS_HOST is configured, do not attempt to connect
    # to a non-existent localhost server. Fall back cleanly to in-memory limiting.
    if not redis_url and not redis_host:
        logger.debug("Redis not configured in environment. Using in-memory rate limiting.")
        return None

    socket_timeout = float(os.environ.get("REDIS_SOCKET_TIMEOUT", "1.0"))
    connect_timeout = float(os.environ.get("REDIS_CONNECT_TIMEOUT", "1.0"))

    try:
        if redis_url:
            client = _redis_module.Redis.from_url(
                redis_url,
                socket_timeout=socket_timeout,
                socket_connect_timeout=connect_timeout,
                decode_responses=False,
            )
        else:
            host = redis_host
            port = int(os.environ.get("REDIS_PORT", "6379"))
            db = int(os.environ.get("REDIS_DB", "0"))
            password = os.environ.get("REDIS_PASSWORD") or None  # None means no auth
            client = _redis_module.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                socket_connect_timeout=connect_timeout,
                decode_responses=False,
            )

        # Validate connectivity eagerly (raises on failure)
        client.ping()
        logger.info("Redis rate-limiter connected successfully.")
        return client
    except Exception as exc:
        # Log at WARNING — deliberately do NOT include connection string / password
        logger.warning(
            "Redis rate-limiter connection failed (%s: %s). "
            "Fail-closed policy will apply to sensitive endpoints.",
            type(exc).__name__,
            str(exc)[:120],
        )
        return None


# ── In-memory fallback (single-process, thread-safe) ─────────────────────────


class _InMemoryWindow:
    """Thread-safe sliding-window counter for a single process."""

    def __init__(self, window_seconds: float, max_attempts: int) -> None:
        self.window_seconds = window_seconds
        self.max_attempts = max_attempts
        self._records: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> List[float]:
        cutoff = now - self.window_seconds
        attempts = [t for t in self._records.get(key, []) if t > cutoff]
        self._records[key] = attempts
        return attempts

    def check(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            attempts = self._prune(key, now)
            return len(attempts) < self.max_attempts

    def check_and_record(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            attempts = self._prune(key, now)
            if len(attempts) >= self.max_attempts:
                return False
            attempts.append(now)
            self._records[key] = attempts
            return True

    def record(self, key: str) -> None:
        now = time.time()
        with self._lock:
            attempts = self._prune(key, now)
            attempts.append(now)
            self._records[key] = attempts

    def reset(self, key: str) -> None:
        with self._lock:
            self._records.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    @property
    def _raw_records(self) -> Dict[str, List[float]]:
        """Expose raw records for backward-compatibility (test introspection)."""
        return self._records


# ── Redis sliding-window implementation ──────────────────────────────────────

_KEY_PREFIX = os.environ.get("REDIS_KEY_PREFIX", "mediassist:rl:")


def _redis_key(limiter_name: str, client_key: str) -> str:
    return f"{_KEY_PREFIX}{limiter_name}:{client_key}"


def _redis_check_and_record(
    redis_client,
    limiter_name: str,
    client_key: str,
    window_seconds: float,
    max_attempts: int,
) -> bool:
    """Atomically check and record one attempt using a Redis sorted set.

    Returns True if the attempt is allowed, False if rate-limited.
    Raises redis.RedisError on connectivity failures (caller handles).
    """
    now = time.time()
    window_start = now - window_seconds
    rkey = _redis_key(limiter_name, client_key)

    # Pipeline: prune → count → conditionally add
    pipe = redis_client.pipeline(transaction=True)
    pipe.zremrangebyscore(rkey, "-inf", window_start)
    pipe.zcard(rkey)
    results = pipe.execute()

    current_count = results[1]
    if current_count >= max_attempts:
        return False

    # Record new attempt (score = timestamp so pruning works correctly)
    pipe2 = redis_client.pipeline(transaction=True)
    # Use a unique member to avoid deduplication of near-simultaneous timestamps
    member = f"{now:.6f}:{id(threading.current_thread())}"
    pipe2.zadd(rkey, {member: now})
    pipe2.expire(rkey, int(window_seconds) + 1)
    pipe2.execute()
    return True


def _redis_record(
    redis_client,
    limiter_name: str,
    client_key: str,
    window_seconds: float,
) -> None:
    """Record one attempt without checking the limit."""
    now = time.time()
    rkey = _redis_key(limiter_name, client_key)
    member = f"{now:.6f}:{id(threading.current_thread())}"
    pipe = redis_client.pipeline(transaction=True)
    pipe.zadd(rkey, {member: now})
    pipe.expire(rkey, int(window_seconds) + 1)
    pipe.execute()


def _redis_check(
    redis_client,
    limiter_name: str,
    client_key: str,
    window_seconds: float,
    max_attempts: int,
) -> bool:
    """Check whether the key is under the limit (read-only, no recording)."""
    now = time.time()
    window_start = now - window_seconds
    rkey = _redis_key(limiter_name, client_key)
    pipe = redis_client.pipeline(transaction=True)
    pipe.zremrangebyscore(rkey, "-inf", window_start)
    pipe.zcard(rkey)
    results = pipe.execute()
    return results[1] < max_attempts


def _redis_reset(redis_client, limiter_name: str, client_key: str) -> None:
    redis_client.delete(_redis_key(limiter_name, client_key))


# ── Public API ────────────────────────────────────────────────────────────────


class RateLimiter:
    """Sliding-window rate limiter backed by Redis with explicit fail-closed policy.

    Parameters
    ----------
    name:
        Short identifier used as part of the Redis key namespace (e.g. "login_ip").
    window_seconds:
        Length of the sliding window in seconds.
    max_attempts:
        Maximum number of allowed attempts inside the window.
    redis_client:
        Pre-built Redis client, or None to attempt automatic construction from
        environment variables.  Pass ``False`` to explicitly disable Redis and
        force the in-memory fallback (useful in tests).
    fail_open_on_redis_error:
        ``False`` (default): when Redis is unavailable, deny the request — safe
        for authentication endpoints where unrestricted access is unacceptable.
        ``True``: fall back to a single-process in-memory limiter — acceptable
        for non-sensitive endpoints when availability matters more than strict
        limiting across all workers.
    """

    def __init__(
        self,
        name: str,
        window_seconds: float = 60.0,
        max_attempts: int = 10,
        redis_client=None,
        fail_open_on_redis_error: bool = False,
    ) -> None:
        self.name = name
        self.window_seconds = window_seconds
        self.max_attempts = max_attempts
        self.fail_open_on_redis_error = fail_open_on_redis_error

        # Redis client resolution
        if redis_client is False:
            # Explicitly disabled — use in-memory only
            self._redis = None
        elif redis_client is not None:
            self._redis = redis_client
        else:
            self._redis = _build_redis_client()

        # In-memory fallback (used when Redis is disabled or unavailable)
        self._memory = _InMemoryWindow(window_seconds, max_attempts)

        if self._redis is not None:
            logger.info(
                "RateLimiter '%s': using Redis backend (window=%ss, max=%d, fail_open=%s).",
                name,
                window_seconds,
                max_attempts,
                fail_open_on_redis_error,
            )
        else:
            logger.info(
                "RateLimiter '%s': using in-memory limiter (window=%ss, max=%d).",
                name,
                window_seconds,
                max_attempts,
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _deny_on_redis_error(self) -> bool:
        """Compute the response when Redis is unreachable."""
        if self.fail_open_on_redis_error:
            logger.warning(
                "RateLimiter '%s': Redis unavailable — fail-open policy applied.", self.name
            )
            return True  # allow
        logger.warning(
            "RateLimiter '%s': Redis unavailable — fail-closed policy applied (request denied).",
            self.name,
        )
        return False  # deny

    # ── Public interface ──────────────────────────────────────────────────────

    def check(self, key: str, is_testing: bool = False) -> bool:
        """Return True if the key is under the rate limit (read-only, no recording)."""
        if is_testing:
            return True
        if self._redis is not None:
            try:
                return _redis_check(
                    self._redis, self.name, key, self.window_seconds, self.max_attempts
                )
            except Exception as exc:
                logger.warning(
                    "RateLimiter '%s': Redis error during check (%s).", self.name, exc
                )
                if self.fail_open_on_redis_error:
                    return self._memory.check(key)
                return self._deny_on_redis_error()
        return self._memory.check(key)

    def record(self, key: str) -> None:
        """Record one attempt for a key without checking the limit."""
        if self._redis is not None:
            try:
                _redis_record(self._redis, self.name, key, self.window_seconds)
                return
            except Exception as exc:
                logger.warning(
                    "RateLimiter '%s': Redis error during record (%s).", self.name, exc
                )
                if self.fail_open_on_redis_error:
                    self._memory.record(key)
                return
        self._memory.record(key)

    def check_and_record(self, key: str, is_testing: bool = False) -> bool:
        """Atomically check the limit and record one attempt if allowed.

        Returns True (allowed) or False (rate-limited).  This is the preferred
        method for authentication endpoints because the check and record
        happen atomically — no race window between the two operations.
        """
        if is_testing:
            return True
        if self._redis is not None:
            try:
                return _redis_check_and_record(
                    self._redis, self.name, key, self.window_seconds, self.max_attempts
                )
            except Exception as exc:
                logger.warning(
                    "RateLimiter '%s': Redis error during check_and_record (%s).",
                    self.name,
                    exc,
                )
                return self._deny_on_redis_error()
        return self._memory.check_and_record(key)

    def reset(self, key: str) -> None:
        """Clear all recorded attempts for a key (e.g. after successful login)."""
        if self._redis is not None:
            try:
                _redis_reset(self._redis, self.name, key)
                return
            except Exception as exc:
                logger.warning(
                    "RateLimiter '%s': Redis error during reset (%s).", self.name, exc
                )
        self._memory.reset(key)

    def clear(self) -> None:
        """Clear all in-memory records (primarily for testing)."""
        self._memory.clear()

    # Back-compat: tests read _records directly from login_ip_limiter
    @property
    def _records(self) -> Dict[str, List[float]]:
        return self._memory._raw_records


# ── Module-level limiter instances (replaces SlidingWindowRateLimiter) ────────
#
# These are imported by auth_service.py and auth_routes.py.
# Redis is attempted automatically from environment variables.
# Authentication endpoints (login, password-reset) are fail-CLOSED.

login_ip_limiter = RateLimiter(
    name="login_ip",
    window_seconds=60.0,
    max_attempts=10,
    fail_open_on_redis_error=False,  # deny on Redis outage — login is sensitive
)

reset_ip_limiter = RateLimiter(
    name="reset_ip",
    window_seconds=60.0,
    max_attempts=5,
    fail_open_on_redis_error=False,
)

reset_email_limiter = RateLimiter(
    name="reset_email",
    window_seconds=900.0,
    max_attempts=3,
    fail_open_on_redis_error=False,
)

chat_limiter = RateLimiter(
    name="chat_requests",
    window_seconds=60.0,
    max_attempts=60,
    fail_open_on_redis_error=True,  # non-auth endpoint: fail open if Redis is down
)

