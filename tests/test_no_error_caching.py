"""Tests to verify that errors are not cached."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest

from seeds_clients import Message, OpenAIClient
from seeds_clients.core.exceptions import ProviderError


class TestErrorCaching:
    """Tests for error caching behavior."""

    def test_error_not_cached(self) -> None:
        """Test that API errors are not cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            client = OpenAIClient(api_key="test-key", cache_dir=cache_dir)

            # Mock API to raise an error
            with patch.object(client._http_client, "post") as mock_post:
                # Simulate a connection error
                mock_post.side_effect = httpx.RequestError("Connection failed", request=Mock())

                messages = [Message(role="user", content="Hello")]

                # First call - should raise ProviderError
                with pytest.raises(ProviderError):
                    client.generate(messages, use_cache=True)

                # Verify cache is empty
                # We need to compute the cache key to check
                cache_key = client._compute_cache_key(messages, {})
                assert client.cache.get(cache_key) is None

                # Verify nothing was written to cache directory (except maybe metadata files from init)
                # The cache implementation might create some files on init, but let's check the key specifically

                # Now make it succeed
                mock_post.side_effect = None
                mock_post.return_value = Mock(
                    json=Mock(return_value={
                        "id": "chatcmpl-123",
                        "object": "chat.completion",
                        "created": 1677652288,
                        "model": "gpt-4.1",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "Hello!"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }),
                    raise_for_status=Mock(),
                )

                # Second call - should succeed and cache
                response = client.generate(messages, use_cache=True)
                assert response.content == "Hello!"

                # Verify it is now cached
                assert client.cache.get(cache_key) is not None
