"""
governance/cache.py - Normalized Query Response Cache
Track: Banking & FinTech (Cred)
Task 16: Sub-millisecond response caching keyed by normalized query strings.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, Optional, Tuple, Any
import time
from agents.schemas import AgentResponseSchema


class NormalizedQueryCache:
    """
    In-memory normalized query cache.
    Keyed by normalized query string: query.strip().lower()
    Bypasses vector retrieval and CrewAI orchestration upon cache hits.
    """

    def __init__(self):
        self._cache: Dict[str, AgentResponseSchema] = {}
        self.hit_count = 0
        self.miss_count = 0

    def normalize_key(self, query: str) -> str:
        """Normalizes query text: strips whitespace, lowercases, and compresses spacing."""
        return " ".join(query.strip().lower().split())

    def get(self, query: str) -> Optional[AgentResponseSchema]:
        """Looks up cached response. Returns None on cache miss."""
        key = self.normalize_key(query)
        res = self._cache.get(key)
        if res is not None:
            self.hit_count += 1
            return res
        self.miss_count += 1
        return None

    def set(self, query: str, response: AgentResponseSchema):
        """Stores response in cache under normalized key."""
        key = self.normalize_key(query)
        self._cache[key] = response

    def clear(self):
        """Empties cache store."""
        self._cache.clear()
        self.hit_count = 0
        self.miss_count = 0

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._cache),
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_ratio": (
                round(self.hit_count / (self.hit_count + self.miss_count), 2)
                if (self.hit_count + self.miss_count) > 0
                else 0.0
            ),
        }


# Singleton query cache instance
QUERY_CACHE = NormalizedQueryCache()


def get_query_cache() -> NormalizedQueryCache:
    return QUERY_CACHE


if __name__ == "__main__":
    cache = get_query_cache()
    mock_resp = AgentResponseSchema(
        query="What is the EMI rule?",
        intent="Policy inquiry",
        direct_answer="EMIs are calculated on a reducing balance basis.",
        citations=["doc_02_emi_rules"],
    )

    # First turn: Cache miss
    miss_val = cache.get("What is the EMI rule?")
    print("Initial cache lookup (Miss):", miss_val)
    assert miss_val is None

    # Store
    cache.set("What is the EMI rule?", mock_resp)

    # Second turn with whitespace and case variations: Cache hit
    hit_val = cache.get("   what IS the  emi RULE?  ")
    print("Normalized cache lookup (Hit):", hit_val is not None)
    assert hit_val is not None
    assert hit_val.direct_answer == mock_resp.direct_answer
    print("Cache statistics:", cache.stats())
