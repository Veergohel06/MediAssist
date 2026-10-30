"""Comprehensive Regression Tests for Response Cache Multi-Tenant Isolation.

Verifies:
- Cache key separation across different users (User A vs User B)
- Cache key separation across different conversations for the same user
- Cache key separation across multiple turns within the same conversation (history hash)
- Changing conversation context produces cache misses
- Proper TTL expiration and eviction
- Invalidation on conversation delete or reset
- Exclusion of emergency triage responses from caching
- Thread-safe concurrent access
"""

import time
import unittest
import concurrent.futures

from services.cache_service import ResponseCache, CACHE_VERSION


class TestResponseCacheIsolation(unittest.TestCase):
    """Test suite for clinical response cache multi-tenant and multi-turn isolation."""

    def setUp(self):
        self.cache = ResponseCache(ttl_seconds=3600.0, max_entries=50)

    def test_cross_user_isolation_identical_query(self):
        """User A asking 'What dosage should I take?' must NEVER return User B's cached response."""
        prompt = "What dosage should I take?"
        sources = [{"source_title": "Pediatric Care", "source_url": "https://example.com/peds"}]
        triage_a = {"urgency": "ROUTINE", "clinical_topic": "pediatric"}
        triage_b = {"urgency": "ROUTINE", "clinical_topic": "geriatric"}

        # Store response for User 1 (pediatric context)
        self.cache.set(
            user_message=prompt,
            reply="Pediatric guidance: consult doctor for infant dosage.",
            sources=sources,
            triage=triage_a,
            user_id=1,
            conversation_id="conv-user1",
        )

        # Query cache for User 2 with identical prompt
        cached_user2 = self.cache.get(
            user_message=prompt,
            user_id=2,
            conversation_id="conv-user2",
        )
        self.assertIsNone(cached_user2, "User 2 must NOT receive User 1's cached response!")

        # Query cache for User 1 must hit
        cached_user1 = self.cache.get(
            user_message=prompt,
            user_id=1,
            conversation_id="conv-user1",
        )
        self.assertIsNotNone(cached_user1)
        self.assertIn("Pediatric guidance", cached_user1[0])

    def test_multi_turn_history_hash_isolation(self):
        """Identical follow-up query ('What about children?') must generate distinct cache keys depending on prior conversation context."""
        followup_q = "What about children?"

        # Conversation A context: discussing headaches
        history_headache = [
            {"role": "user", "parts": [{"text": "I have a throbbing headache."}]},
            {"role": "model", "parts": [{"text": "Headaches can be caused by tension or migraines."}]},
        ]

        # Conversation B context: discussing blood pressure
        history_hypertension = [
            {"role": "user", "parts": [{"text": "My blood pressure reading was 150/95."}]},
            {"role": "model", "parts": [{"text": "That indicates stage 1 hypertension."}]},
        ]

        key_headache = ResponseCache.generate_cache_key(
            user_message=followup_q,
            user_id=10,
            conversation_id="conv-1",
            history=history_headache,
        )

        key_hypertension = ResponseCache.generate_cache_key(
            user_message=followup_q,
            user_id=10,
            conversation_id="conv-2",
            history=history_hypertension,
        )

        key_standalone = ResponseCache.generate_cache_key(
            user_message=followup_q,
            user_id=10,
            conversation_id="conv-3",
            history=None,
        )

        # All three must have different keys
        self.assertNotEqual(key_headache, key_hypertension)
        self.assertNotEqual(key_headache, key_standalone)
        self.assertNotEqual(key_hypertension, key_standalone)

    def test_cross_conversation_isolation_same_user(self):
        """Same user asking the same question in two different consultations must have separate cache scopes."""
        prompt = "How much water should I drink?"
        self.cache.set(
            user_message=prompt,
            reply="Drink 2-3 liters daily.",
            sources=[],
            triage={"urgency": "ROUTINE"},
            user_id=5,
            conversation_id="conv-hydration-1",
        )

        hit_conv1 = self.cache.get(prompt, user_id=5, conversation_id="conv-hydration-1")
        self.assertIsNotNone(hit_conv1)

        miss_conv2 = self.cache.get(prompt, user_id=5, conversation_id="conv-hydration-2")
        self.assertIsNone(miss_conv2)

    def test_conversation_invalidation(self):
        """Deleting or clearing a conversation invalidates its cached responses immediately."""
        conv_id = "conv-to-delete"
        self.cache.set(
            user_message="Explain fever",
            reply="Fever is an elevated body temperature.",
            sources=[],
            triage={"urgency": "ROUTINE"},
            user_id=8,
            conversation_id=conv_id,
        )
        self.assertEqual(len(self.cache), 1)

        # Invalidate specific conversation
        purged = self.cache.invalidate_conversation(conv_id)
        self.assertEqual(purged, 1)
        self.assertEqual(len(self.cache), 0)

        # Subsequent lookup must miss
        self.assertIsNone(self.cache.get("Explain fever", user_id=8, conversation_id=conv_id))

    def test_ttl_expiration(self):
        """Entries past TTL must expire and be evicted."""
        short_cache = ResponseCache(ttl_seconds=0.01)
        short_cache.set(
            user_message="Short lived",
            reply="Going away soon",
            sources=[],
            triage={"urgency": "ROUTINE"},
            user_id=1,
            conversation_id="c1",
        )
        time.sleep(0.02)
        res = short_cache.get("Short lived", user_id=1, conversation_id="c1")
        self.assertIsNone(res)

    def test_emergency_not_cached(self):
        """Emergency red-flag guidance must NEVER be cached."""
        saved_key = self.cache.set(
            user_message="Severe chest pain radiating to left arm",
            reply="Call emergency services immediately.",
            sources=[],
            triage={"urgency": "EMERGENCY", "is_emergency": True},
            user_id=1,
            conversation_id="c1",
        )
        self.assertEqual(saved_key, "")
        self.assertEqual(len(self.cache), 0)

    def test_concurrent_read_write_safety(self):
        """Thread safety under concurrent reads and writes."""
        def worker(i: int):
            msg = f"Question {i % 5}"
            self.cache.set(
                user_message=msg,
                reply=f"Answer {i}",
                sources=[],
                triage={"urgency": "ROUTINE"},
                user_id=i,
                conversation_id=f"conv-{i}",
            )
            return self.cache.get(msg, user_id=i, conversation_id=f"conv-{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(50)]
            for f in concurrent.futures.as_completed(futures):
                res = f.result()
                self.assertIsNotNone(res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
