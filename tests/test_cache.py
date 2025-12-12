"""Tests for cache manager."""

import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import pytest
from PIL import Image

from seeds_clients.core.base_client import BaseClient
from seeds_clients.core.cache import CacheManager
from seeds_clients.core.types import Message, Response, Usage


class TestCacheManager:
    """Tests for CacheManager."""

    @pytest.fixture
    def cache_dir(self) -> Generator[Path, None, None]:
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache(self, cache_dir: Path) -> CacheManager:
        """Create a cache manager instance."""
        return CacheManager(cache_dir, ttl_hours=1)

    def test_cache_initialization(self, cache_dir: Path) -> None:
        """Test cache initialization creates directory."""
        cache = CacheManager(cache_dir)
        assert cache_dir.exists()
        cache.close()

    def test_set_and_get(self, cache: CacheManager) -> None:
        """Test setting and getting cache entries."""
        key = "test_key"
        data = {"model": "gpt-4.1", "content": "Hello!"}

        cache.set(key, data)
        retrieved = cache.get(key)

        assert retrieved == data

    def test_get_nonexistent(self, cache: CacheManager) -> None:
        """Test getting a non-existent key returns None."""
        assert cache.get("nonexistent") is None

    def test_delete(self, cache: CacheManager) -> None:
        """Test deleting cache entries."""
        key = "test_key"
        data = {"model": "gpt-4.1"}

        cache.set(key, data)
        assert cache.get(key) is not None

        cache.delete(key)
        assert cache.get(key) is None

    def test_clear(self, cache: CacheManager) -> None:
        """Test clearing all cache entries."""
        cache.set("key1", {"data": "1"})
        cache.set("key2", {"data": "2"})

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_ttl_expiration(self, cache_dir: Path) -> None:
        """Test that entries expire after TTL."""
        # Create cache with 1 second TTL
        cache = CacheManager(cache_dir, ttl_hours=1 / 3600)

        key = "test_key"
        data = {"model": "gpt-4.1"}

        cache.set(key, data)
        assert cache.get(key) is not None

        # Wait for expiration
        time.sleep(1.5)

        assert cache.get(key) is None
        cache.close()

    def test_no_ttl(self, cache_dir: Path) -> None:
        """Test cache with no TTL doesn't expire."""
        cache = CacheManager(cache_dir, ttl_hours=None)

        key = "test_key"
        data = {"model": "gpt-4.1"}

        cache.set(key, data)
        time.sleep(0.1)

        # Should still be available
        assert cache.get(key) is not None
        cache.close()

    def test_stats(self, cache: CacheManager) -> None:
        """Test getting cache statistics."""
        cache.set("key1", {"data": "1"})
        cache.set("key2", {"data": "2"})

        stats = cache.stats()

        assert stats["count"] == 2
        assert stats["size_bytes"] > 0
        assert "cache_dir" in stats
        assert stats["ttl_hours"] == 1

    def test_generate_key(self) -> None:
        """Test cache key generation is deterministic."""
        data1 = {"model": "gpt-4.1", "messages": [{"role": "user", "content": "Hello"}]}
        data2 = {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4.1"}

        key1 = CacheManager.generate_key(data1)
        key2 = CacheManager.generate_key(data2)

        # Same data in different order should produce same key
        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex digest

    def test_context_manager(self, cache_dir: Path) -> None:
        """Test cache can be used as context manager."""
        with CacheManager(cache_dir) as cache:
            cache.set("key", {"data": "value"})
            assert cache.get("key") is not None

    def test_metadata(self, cache: CacheManager) -> None:
        """Test storing metadata with cache entries."""
        key = "test_key"
        data = {"model": "gpt-4.1"}
        metadata = {"provider": "openai", "duration": 1.5}

        cache.set(key, data, metadata=metadata)
        retrieved = cache.get(key)

        # Metadata is stored but not returned with data
        assert retrieved == data


class _DummyClient(BaseClient):
    """Minimal BaseClient implementation for TTL testing."""

    def _setup_tracking(self) -> None:  # pragma: no cover - not used
        return None

    def _call_api(
        self, messages: list[Message], **kwargs: object
    ) -> dict[str, object]:  # pragma: no cover - not used
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    async def _acall_api(
        self, messages: list[Message], **kwargs: object
    ) -> dict[str, object]:  # pragma: no cover - not used
        return self._call_api(messages, **kwargs)

    def _parse_response(self, raw: dict[str, object]) -> Response:
        usage = Usage(prompt_tokens=1, completion_tokens=1)
        return Response(content="ok", usage=usage, model="dummy", raw=raw)

    def _get_provider_name(self) -> str:
        return "dummy"


def test_base_client_uses_ttl_override(tmp_path: Path) -> None:
    """Ensure BaseClient forwards ttl_hours to the cache manager."""
    client = _DummyClient(
        model="dummy",
        api_key="k",
        cache_dir=tmp_path,
        ttl_hours=1.5,
        enable_tracking=False,
    )

    assert client.cache is not None
    assert client.cache.ttl_seconds == pytest.approx(1.5 * 3600)

    # Cleanup to avoid resource warnings
    client.cache.close()


def test_cache_key_is_deterministic_for_pil_images(tmp_path: Path) -> None:
    """Cache keys should stay stable for identical PIL image content."""
    client = _DummyClient(
        model="dummy",
        api_key="k",
        cache_dir=tmp_path,
        ttl_hours=1.5,
        enable_tracking=False,
    )

    img1 = Image.new("RGB", (4, 4), color="red")
    img2 = Image.new("RGB", (4, 4), color="red")

    messages1 = [Message(role="user", content=[{"type": "image", "source": img1}])]
    messages2 = [Message(role="user", content=[{"type": "image", "source": img2}])]

    key1 = client._compute_cache_key(messages1, {})
    key2 = client._compute_cache_key(messages2, {})

    assert key1 == key2

    if client.cache:
        client.cache.close()
