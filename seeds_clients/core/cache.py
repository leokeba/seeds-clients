"""Cache manager for storing raw API responses."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

from diskcache import Cache

from seeds_clients.core.exceptions import CacheError


class CacheManager:
    """Manages persistent caching of raw API responses.

    Key features:
    - Stores complete raw responses (not transformed data)
    - Content-based cache keys
    - TTL-based expiration
    - Thread-safe operations
    """

    def __init__(
        self,
        cache_dir: Path | str,
        ttl_hours: float | None = 24,
        size_limit: int = 10 * 1024 * 1024 * 1024,  # 10 GB default
    ) -> None:
        """Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours (None = no expiration)
            size_limit: Maximum cache size in bytes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600 if ttl_hours else None

        try:
            self.cache = Cache(
                str(self.cache_dir),
                size_limit=size_limit,
                eviction_policy="least-recently-used",
            )
        except Exception as e:
            raise CacheError(f"Failed to initialize cache: {e}") from e

    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve raw API response from cache.

        Args:
            key: Cache key

        Returns:
            Raw API response dict if found, None otherwise.
            Expiration is handled automatically by diskcache's TTL mechanism.
        """
        try:
            cached = self.cache.get(key)
            if cached is None:
                return None

            # Type assertion: we know cached is a dict with our structure
            cached_entry = cast(dict[str, Any], cached)
            return cast(dict[str, Any], cached_entry["raw_response"])
        except Exception as e:
            raise CacheError(f"Failed to read from cache: {e}") from e

    def set(
        self,
        key: str,
        raw_response: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store raw API response in cache.

        Args:
            key: Cache key
            raw_response: Complete raw API response
            metadata: Optional metadata to store with response
        """
        try:
            cache_entry = {
                "raw_response": raw_response,
                "cached_at": time.time(),
                "metadata": metadata or {},
            }

            # Set with TTL if configured
            if self.ttl_seconds:
                self.cache.set(key, cache_entry, expire=self.ttl_seconds)
            else:
                self.cache.set(key, cache_entry)
        except Exception as e:
            raise CacheError(f"Failed to write to cache: {e}") from e

    def delete(self, key: str) -> bool:
        """Delete an entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted, False if not found
        """
        try:
            result: bool = self.cache.delete(key)
            return result
        except Exception as e:
            raise CacheError(f"Failed to delete from cache: {e}") from e

    def clear(self) -> None:
        """Clear all entries from cache."""
        try:
            self.cache.clear()
        except Exception as e:
            raise CacheError(f"Failed to clear cache: {e}") from e

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (size, count, hit rate, etc.)
        """
        try:
            volume = self.cache.volume()
            cache_len = self.cache.__len__()
            return {
                "size_bytes": volume,
                "size_mb": volume / (1024 * 1024),
                "count": cache_len,
                "cache_dir": str(self.cache_dir),
                "ttl_hours": self.ttl_seconds / 3600 if self.ttl_seconds else None,
            }
        except Exception as e:
            raise CacheError(f"Failed to get cache stats: {e}") from e

    @staticmethod
    def generate_key(data: dict[str, Any]) -> str:
        """Generate a cache key from request parameters.

        Creates a deterministic hash from the request data.

        Args:
            data: Dictionary containing request parameters

        Returns:
            Hexadecimal cache key
        """
        # Sort keys to ensure deterministic hashing
        normalized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def close(self) -> None:
        """Close the cache."""
        if getattr(self, "_closed", False):
            return

        try:
            self.cache.close()
        except Exception as e:
            raise CacheError(f"Failed to close cache: {e}") from e
        finally:
            self._closed = True

    def __enter__(self) -> "CacheManager":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Ensure cache resources are released when garbage collected."""
        try:
            self.close()
        except Exception:
            # Avoid raising during interpreter shutdown
            pass
