# app/local_cache.py
import os
import time
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

# TTL (seconds) and max cache size from env
_CACHE_TTL_SECONDS = int(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", "300"))
_CACHE_MAX_SIZE = int(os.getenv("EMBEDDING_CACHE_MAX_SIZE", "512"))


class RecentMatchCache:
    """
    Small in-process cache of recently matched persons.

    Key: person_id
    Value: {
        "emb_norm": np.ndarray,  # normalized embedding
        "metadata": dict,
        "expires_at": float      # unix timestamp
    }

    Used as a fast path before hitting the DB.
    """

    def __init__(self, ttl_seconds: int, max_size: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._store: Dict[str, Dict[str, Any]] = {}

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys: List[str] = [
            pid for pid, entry in self._store.items()
            if entry["expires_at"] <= now
        ]
        for pid in expired_keys:
            del self._store[pid]

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is too big."""
        if len(self._store) <= self.max_size:
            return

        items = sorted(
            self._store.items(),
            key=lambda kv: kv[1]["expires_at"]
        )
        overflow = len(self._store) - self.max_size
        for i in range(overflow):
            pid, _ = items[i]
            self._store.pop(pid, None)

    def put(self, person_id: str, embedding, metadata: Dict[str, Any]) -> None:
        """
        Insert or update a person in the cache.
        We store a normalized copy of the embedding.
        """
        if self.ttl_seconds <= 0:
            return  # cache disabled

        emb = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb)) + 1e-8
        emb_norm = emb / norm

        self._store[person_id] = {
            "emb_norm": emb_norm,
            "metadata": metadata or {},
            "expires_at": time.time() + self.ttl_seconds,
        }

        self._evict_if_needed()

    def best_match(
        self,
        query_embedding,
        threshold: float
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], float]:
        """
        Try to find the best match among cached persons.

        Returns:
            (person_id, metadata, score)

        If no cached person has score >= threshold,
        returns (None, None, best_score).

        Score here is cosine similarity in [-1, 1].
        """

        if not self._store or self.ttl_seconds <= 0:
            return None, None, 0.0

        self._evict_expired()
        if not self._store:
            return None, None, 0.0

        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm_val = float(np.linalg.norm(q)) + 1e-8
        q_norm = q / q_norm_val

        best_pid: Optional[str] = None
        best_meta: Optional[Dict[str, Any]] = None
        best_score: float = -1.0

        print(f"[cache] scanning {len(self._store)} cached persons")
        for pid, entry in self._store.items():
            emb_norm = entry["emb_norm"]
            score = float(np.dot(emb_norm, q_norm))
            if score > best_score:
                best_score = score
                best_pid = pid
                best_meta = entry["metadata"]

        if best_pid is None or best_score < threshold:
            return None, None, best_score

        return best_pid, best_meta, best_score

    def clear(self) -> None:
        self._store.clear()


# Global cache instance used in the router
recent_match_cache = RecentMatchCache(
    ttl_seconds=_CACHE_TTL_SECONDS,
    max_size=_CACHE_MAX_SIZE,
)
