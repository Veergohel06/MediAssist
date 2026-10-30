"""
Pytest configuration for MediAssist test suite.
Ensures that unit tests run with TestingConfig (isolated in-memory SQLite)
by default, while dedicated integration tests can opt in to Neon PostgreSQL.
"""

import os
import pytest

# Ensure testing mode is activated for unit test discovery and isolation
os.environ.setdefault("FLASK_ENV", "testing")
