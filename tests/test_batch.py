"""Tests for batch processing functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seeds_clients.core.batch import BatchResult
from seeds_clients.core.types import Message, Response, TrackingData, Usage


def make_response(content: str, idx: int = 0) -> Response:
    """Helper to create Response objects with all required fields."""
    return Response(
        content=content,
        usage=Usage(prompt_tokens=10, completion_tokens=20),
        model="gpt-4.1",
        raw={"choices": [{"message": {"content": content}}]},
        tracking=TrackingData(
            cost_usd=0.01,
            energy_kwh=0.001,
            gwp_kgco2eq=0.0001,
            prompt_tokens=10,
            completion_tokens=20,
            provider="openai",
            model="gpt-4.1",
            tracking_method="ecologits",
            duration_seconds=0.5,
        ),
    )


def make_simple_response(content: str) -> Response:
    """Helper to create Response objects without tracking."""
    return Response(
        content=content,
        usage=Usage(prompt_tokens=10, completion_tokens=20),
        model="gpt-4.1",
        raw={"choices": [{"message": {"content": content}}]},
    )


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_default_values(self):
        """Test BatchResult default values."""
        result = BatchResult()

        assert result.responses == []
        assert result.errors == []
        assert result.total_prompt_tokens == 0
        assert result.total_completion_tokens == 0
        assert result.total_cost_usd == 0.0
        assert result.total_energy_kwh == 0.0
        assert result.total_gwp_kgco2eq == 0.0
        assert result.total_duration_seconds == 0.0

    def test_successful_count(self):
        """Test successful_count property."""
        result = BatchResult()
        result.responses = [MagicMock(), MagicMock(), MagicMock()]

        assert result.successful_count == 3

    def test_failed_count(self):
        """Test failed_count property."""
        result = BatchResult()
        result.errors = [(0, Exception("Error 1")), (2, Exception("Error 2"))]

        assert result.failed_count == 2

    def test_total_count(self):
        """Test total_count property."""
        result = BatchResult()
        result.responses = [MagicMock(), MagicMock()]
        result.errors = [(1, Exception("Error"))]

        assert result.total_count == 3

    def test_total_tokens(self):
        """Test total_tokens property."""
        result = BatchResult()
        result.total_prompt_tokens = 100
        result.total_completion_tokens = 50

        assert result.total_tokens == 150

    def test_repr(self):
        """Test string representation."""
        result = BatchResult()
        result.responses = [MagicMock(), MagicMock()]
        result.errors = [(1, Exception("Error"))]
        result.total_cost_usd = 0.0123
        result.total_gwp_kgco2eq = 0.000456

        repr_str = repr(result)

        assert "successful=2" in repr_str
        assert "failed=1" in repr_str
        assert "cost=$0.0123" in repr_str
        assert "carbon=" in repr_str


class TestBatchGenerate:
    """Tests for batch_generate method in BaseClient."""

    @pytest.mark.asyncio
    async def test_batch_generate_basic(self):
        """Test basic batch generation."""
        from seeds_clients.core.base_client import BaseClient

        client = MagicMock()
        client.cache = None

        # Create mock responses
        call_idx = [0]

        async def mock_agenerate(messages, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            return make_response(f"Response {idx}", idx)

        client.agenerate = mock_agenerate

        # Test batch_generate
        messages_list = [[Message(role="user", content=f"Question {i}")] for i in range(3)]

        result = await BaseClient.batch_generate(client, messages_list, max_concurrent=2)

        assert result.successful_count == 3
        assert result.failed_count == 0
        assert result.total_prompt_tokens == 30
        assert result.total_completion_tokens == 60
        assert result.total_cost_usd == pytest.approx(0.03, rel=0.01)

    @pytest.mark.asyncio
    async def test_batch_generate_with_errors(self):
        """Test batch generation with some failures."""
        from seeds_clients.core.base_client import BaseClient

        client = MagicMock()
        client.cache = None

        call_count = [0]

        async def mock_agenerate(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("API Error")
            return make_response(f"Response {call_count[0]}")

        client.agenerate = mock_agenerate

        messages_list = [[Message(role="user", content=f"Question {i}")] for i in range(3)]

        result = await BaseClient.batch_generate(client, messages_list)

        assert result.successful_count == 2
        assert result.failed_count == 1
        assert len(result.errors) == 1
        # The error should be at index 1 (second request)
        assert result.errors[0][0] == 1

    @pytest.mark.asyncio
    async def test_batch_generate_with_progress_callback(self):
        """Test batch generation with progress callback."""
        from seeds_clients.core.base_client import BaseClient

        client = MagicMock()
        client.cache = None

        async def mock_agenerate(messages, **kwargs):
            await asyncio.sleep(0.01)  # Small delay
            return make_response("Response")

        client.agenerate = mock_agenerate

        progress_calls = []

        def on_progress(completed, total, result):
            progress_calls.append((completed, total))

        messages_list = [[Message(role="user", content=f"Question {i}")] for i in range(3)]

        await BaseClient.batch_generate(client, messages_list, on_progress=on_progress)

        assert len(progress_calls) == 3
        # Progress should track completed vs total
        for completed, total in progress_calls:
            assert total == 3
            assert 1 <= completed <= 3


class TestBatchGenerateIter:
    """Tests for batch_generate_iter method."""

    @pytest.mark.asyncio
    async def test_batch_generate_iter_basic(self):
        """Test basic async iteration over batch results."""
        from seeds_clients.core.base_client import BaseClient

        client = MagicMock()
        client.cache = None

        call_idx = [0]

        async def mock_agenerate(messages, **kwargs):
            await asyncio.sleep(0.01)
            idx = call_idx[0]
            call_idx[0] += 1
            return make_simple_response(f"Response {idx}")

        client.agenerate = mock_agenerate

        messages_list = [[Message(role="user", content=f"Question {i}")] for i in range(3)]

        results = []
        async for idx, result in BaseClient.batch_generate_iter(
            client, messages_list, max_concurrent=2
        ):
            results.append((idx, result))

        assert len(results) == 3
        # All results should be Response objects
        for idx, result in results:
            assert isinstance(result, Response)
            assert 0 <= idx <= 2

    @pytest.mark.asyncio
    async def test_batch_generate_iter_with_errors(self):
        """Test async iteration with some failures."""
        from seeds_clients.core.base_client import BaseClient

        client = MagicMock()
        client.cache = None

        call_count = [0]

        async def mock_agenerate(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("API Error")
            return make_simple_response(f"Response {call_count[0]}")

        client.agenerate = mock_agenerate

        messages_list = [[Message(role="user", content=f"Question {i}")] for i in range(3)]

        successes = []
        failures = []

        async for idx, result in BaseClient.batch_generate_iter(client, messages_list):
            if isinstance(result, Exception):
                failures.append((idx, result))
            else:
                successes.append((idx, result))

        assert len(successes) == 2
        assert len(failures) == 1


class TestAsyncGenerate:
    """Tests for agenerate method."""

    @pytest.mark.asyncio
    async def test_agenerate_basic(self):
        """Test basic async generation."""
        from seeds_clients.core.base_client import BaseClient

        client = MagicMock()
        client.cache = None
        client.enable_tracking = False
        client.tracker = None

        async def mock_acall_api(messages, **kwargs):
            return {
                "choices": [{"message": {"content": "Hello!"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10},
            }

        def mock_parse_response(raw):
            return Response(
                content=raw["choices"][0]["message"]["content"],
                usage=Usage(
                    prompt_tokens=raw["usage"]["prompt_tokens"],
                    completion_tokens=raw["usage"]["completion_tokens"],
                ),
                model="gpt-4.1",
                raw=raw,
            )

        def mock_compute_cache_key(messages, kwargs):
            return "test_key"

        client._acall_api = mock_acall_api
        client._parse_response = mock_parse_response
        client._compute_cache_key = mock_compute_cache_key

        messages = [Message(role="user", content="Hi")]
        response = await BaseClient.agenerate(client, messages)

        assert response.content == "Hello!"
        assert response.usage.prompt_tokens == 5
        assert response.usage.completion_tokens == 10

    @pytest.mark.asyncio
    async def test_agenerate_uses_cache(self):
        """Test that agenerate uses cache when available."""
        from seeds_clients.core.base_client import BaseClient

        client = MagicMock()
        client.enable_tracking = False
        client.tracker = None

        cached_response = {
            "choices": [{"message": {"content": "Cached!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }

        client.cache.get.return_value = cached_response

        def mock_parse_response(raw):
            return Response(
                content=raw["choices"][0]["message"]["content"],
                usage=Usage(
                    prompt_tokens=raw["usage"]["prompt_tokens"],
                    completion_tokens=raw["usage"]["completion_tokens"],
                ),
                model="gpt-4.1",
                raw=raw,
            )

        def mock_compute_cache_key(messages, kwargs):
            return "test_key"

        client._parse_response = mock_parse_response
        client._compute_cache_key = mock_compute_cache_key

        messages = [Message(role="user", content="Hi")]
        response = await BaseClient.agenerate(client, messages, use_cache=True)

        assert response.content == "Cached!"
        assert response.cached is True


class TestAsyncContextManager:
    """Tests for async context manager support."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager entry and exit."""
        from seeds_clients.core.base_client import BaseClient

        client = MagicMock(spec=BaseClient)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        async with client as c:
            assert c is client

        client.__aenter__.assert_called_once()
        client.__aexit__.assert_called_once()
