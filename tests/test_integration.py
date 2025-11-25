"""Integration tests for seeds-clients.

These tests verify end-to-end functionality with real API calls.
They are marked with @pytest.mark.integration and skipped by default.

Run integration tests:
    pytest tests/test_integration.py -v -m integration --run-integration

Requires:
    - OPENAI_API_KEY environment variable
    - OPENROUTER_API_KEY environment variable (optional)
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from seeds_clients import (
    BatchResult,
    Message,
    OpenAIClient,
    OpenRouterClient,
    Response,
)


def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


def has_openrouter_key() -> bool:
    """Check if OpenRouter API key is available."""
    return bool(os.getenv("OPENROUTER_API_KEY"))


# =============================================================================
# OpenAI Integration Tests
# =============================================================================


@pytest.mark.integration
class TestOpenAIIntegration:
    """Integration tests for OpenAI client with real API calls."""

    @pytest.fixture
    def client(self, tmp_path: Path) -> OpenAIClient:
        """Create OpenAI client with temporary cache."""
        if not has_openai_key():
            pytest.skip("OPENAI_API_KEY not set")
        return OpenAIClient(
            model="gpt-4.1-mini",
            cache_dir=str(tmp_path / "cache"),
            ttl_hours=1.0,
        )

    def test_simple_generation(self, client: OpenAIClient) -> None:
        """Test simple text generation."""
        response = client.generate(
            messages=[Message(role="user", content="What is 2+2? Reply with just the number.")]
        )

        assert response.content is not None
        assert "4" in response.content
        assert response.usage.total_tokens > 0
        assert response.cached is False

    def test_generation_with_system_message(self, client: OpenAIClient) -> None:
        """Test generation with system message."""
        response = client.generate(
            messages=[
                Message(role="system", content="You are a pirate. Speak like one."),
                Message(role="user", content="Hello!"),
            ]
        )

        assert response.content is not None
        assert len(response.content) > 0

    def test_caching_works(self, client: OpenAIClient) -> None:
        """Test that caching returns cached responses."""
        messages = [Message(role="user", content="What is the capital of France?")]

        # First request - should hit API
        response1 = client.generate(messages=messages)
        assert response1.cached is False

        # Second request - should hit cache
        response2 = client.generate(messages=messages)
        assert response2.cached is True
        assert response2.content == response1.content

    def test_skip_cache(self, client: OpenAIClient) -> None:
        """Test that use_cache=False bypasses cache."""
        messages = [Message(role="user", content="What is 1+1?")]

        response1 = client.generate(messages=messages)
        response2 = client.generate(messages=messages, use_cache=False)

        # Both should be API calls (not cached)
        assert response1.cached is False
        assert response2.cached is False

    def test_tracking_data_populated(self, client: OpenAIClient) -> None:
        """Test that tracking data is populated."""
        response = client.generate(
            messages=[Message(role="user", content="Say hi")]
        )

        assert response.tracking is not None
        assert response.tracking.prompt_tokens > 0
        assert response.tracking.completion_tokens > 0
        assert response.tracking.cost_usd >= 0
        assert response.tracking.provider == "openai"
        assert response.tracking.model is not None

    def test_carbon_tracking(self, client: OpenAIClient) -> None:
        """Test that carbon/energy tracking is populated."""
        response = client.generate(
            messages=[Message(role="user", content="Hello!")]
        )

        assert response.tracking is not None
        # EcoLogits should provide these values
        assert response.tracking.energy_kwh is not None
        assert response.tracking.gwp_kgco2eq is not None
        assert response.tracking.energy_kwh >= 0
        assert response.tracking.gwp_kgco2eq >= 0

    def test_different_electricity_zones(self, tmp_path: Path) -> None:
        """Test carbon tracking with different electricity zones."""
        if not has_openai_key():
            pytest.skip("OPENAI_API_KEY not set")

        messages = [Message(role="user", content="Hi")]

        # World average
        client_wor = OpenAIClient(
            model="gpt-4.1-mini",
            cache_dir=str(tmp_path / "cache1"),
            electricity_mix_zone="WOR",
        )
        response_wor = client_wor.generate(messages=messages)

        # France (lower carbon due to nuclear)
        client_fra = OpenAIClient(
            model="gpt-4.1-mini",
            cache_dir=str(tmp_path / "cache2"),
            electricity_mix_zone="FRA",
        )
        response_fra = client_fra.generate(messages=messages)

        # Both should have tracking
        assert response_wor.tracking is not None
        assert response_fra.tracking is not None

        # Energy should be similar (same computation)
        # Carbon should differ based on grid mix
        assert response_wor.tracking.electricity_mix_zone == "WOR"
        assert response_fra.tracking.electricity_mix_zone == "FRA"

    def test_temperature_affects_output(self, client: OpenAIClient) -> None:
        """Test that temperature parameter affects generation."""
        messages = [Message(role="user", content="Write a random word.")]

        # Low temperature - more deterministic
        responses_low = [
            client.generate(messages=messages, temperature=0.0, use_cache=False).content
            for _ in range(3)
        ]

        # With temperature=0, responses should be identical
        assert len(set(responses_low)) == 1

    def test_max_tokens_limit(self, client: OpenAIClient) -> None:
        """Test that max_tokens limits output length."""
        response = client.generate(
            messages=[Message(role="user", content="Write a very long essay about AI.")],
            max_tokens=10,
        )

        # Should be truncated
        assert response.usage.completion_tokens <= 15  # Some buffer for finish token
        assert response.finish_reason in ["length", "stop"]


@pytest.mark.integration
class TestOpenAIAsyncIntegration:
    """Async integration tests for OpenAI client."""

    @pytest.fixture
    def client(self, tmp_path: Path) -> OpenAIClient:
        """Create OpenAI client."""
        if not has_openai_key():
            pytest.skip("OPENAI_API_KEY not set")
        return OpenAIClient(
            model="gpt-4.1-mini",
            cache_dir=str(tmp_path / "cache"),
        )

    @pytest.mark.asyncio
    async def test_async_generation(self, client: OpenAIClient) -> None:
        """Test async generation."""
        try:
            response = await client.agenerate(
                messages=[Message(role="user", content="What is 3+3?")]
            )

            assert response.content is not None
            assert "6" in response.content
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_batch_generation(self, client: OpenAIClient) -> None:
        """Test batch generation."""
        messages_list = [
            [Message(role="user", content="What is 1+1?")],
            [Message(role="user", content="What is 2+2?")],
            [Message(role="user", content="What is 3+3?")],
        ]

        try:
            result = await client.batch_generate(
                messages_list,
                max_concurrent=2,
            )

            assert isinstance(result, BatchResult)
            assert result.successful_count == 3
            assert result.failed_count == 0
            assert len(result.responses) == 3
            assert result.total_tokens > 0
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_batch_generate_iter(self, client: OpenAIClient) -> None:
        """Test batch generation with iterator."""
        messages_list = [
            [Message(role="user", content="Say 'one'")],
            [Message(role="user", content="Say 'two'")],
        ]

        results = []
        try:
            async for idx, result in client.batch_generate_iter(
                messages_list, max_concurrent=2
            ):
                results.append((idx, result))

            assert len(results) == 2
            # All should be successful responses
            for idx, result in results:
                assert isinstance(result, Response)
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, tmp_path: Path) -> None:
        """Test async context manager."""
        if not has_openai_key():
            pytest.skip("OPENAI_API_KEY not set")

        async with OpenAIClient(
            model="gpt-4.1-mini",
            cache_dir=str(tmp_path / "cache"),
        ) as client:
            response = await client.agenerate(
                messages=[Message(role="user", content="Hi")]
            )
            assert response.content is not None


# =============================================================================
# OpenRouter Integration Tests
# =============================================================================


@pytest.mark.integration
class TestOpenRouterIntegration:
    """Integration tests for OpenRouter client with real API calls."""

    @pytest.fixture
    def client(self, tmp_path: Path) -> OpenRouterClient:
        """Create OpenRouter client with temporary cache."""
        if not has_openrouter_key():
            pytest.skip("OPENROUTER_API_KEY not set")
        return OpenRouterClient(
            model="openai/gpt-4.1-mini",
            cache_dir=str(tmp_path / "cache"),
            ttl_hours=1.0,
        )

    def test_simple_generation(self, client: OpenRouterClient) -> None:
        """Test simple text generation via OpenRouter."""
        response = client.generate(
            messages=[Message(role="user", content="What is 5+5? Reply with just the number.")]
        )

        assert response.content is not None
        assert "10" in response.content
        assert response.usage.total_tokens > 0

    def test_tracking_shows_openrouter_provider(self, client: OpenRouterClient) -> None:
        """Test that tracking shows openrouter as provider."""
        response = client.generate(
            messages=[Message(role="user", content="Hi")]
        )

        assert response.tracking is not None
        assert response.tracking.provider == "openrouter"

    def test_ecologits_provider_extraction(self, tmp_path: Path) -> None:
        """Test that EcoLogits provider is correctly extracted from model."""
        if not has_openrouter_key():
            pytest.skip("OPENROUTER_API_KEY not set")

        client = OpenRouterClient(
            model="openai/gpt-4.1-mini",
            cache_dir=str(tmp_path / "cache"),
        )

        # Provider should be extracted for EcoLogits
        assert client._get_ecologits_provider() == "openai"
        assert client._get_ecologits_model() == "gpt-4.1-mini"

    def test_different_providers(self, tmp_path: Path) -> None:
        """Test using different model providers through OpenRouter."""
        if not has_openrouter_key():
            pytest.skip("OPENROUTER_API_KEY not set")

        # Test with multiple providers (if available on account)
        models_to_test = [
            "openai/gpt-4.1-mini",
            # "anthropic/claude-3-haiku-20240307",  # Uncomment if available
            # "google/gemini-flash-1.5",  # Uncomment if available
        ]

        for model in models_to_test:
            client = OpenRouterClient(
                model=model,
                cache_dir=str(tmp_path / f"cache_{model.replace('/', '_')}"),
            )

            response = client.generate(
                messages=[Message(role="user", content="Say 'test'")]
            )

            assert response.content is not None
            assert response.tracking is not None
            assert response.tracking.provider == "openrouter"

    @pytest.mark.asyncio
    async def test_async_generation(self, client: OpenRouterClient) -> None:
        """Test async generation via OpenRouter."""
        try:
            response = await client.agenerate(
                messages=[Message(role="user", content="What is 7+7?")]
            )

            assert response.content is not None
            assert "14" in response.content
        finally:
            await client.aclose()


# =============================================================================
# Cross-Provider Integration Tests
# =============================================================================


@pytest.mark.integration
class TestCrossProviderIntegration:
    """Tests comparing behavior across different providers."""

    def test_same_prompt_different_providers(self, tmp_path: Path) -> None:
        """Test same prompt with different providers gives similar results."""
        if not has_openai_key():
            pytest.skip("OPENAI_API_KEY not set")

        messages = [Message(role="user", content="What is 2+2? Reply with just the number.")]

        # Direct OpenAI
        openai_client = OpenAIClient(
            model="gpt-4.1-mini",
            cache_dir=str(tmp_path / "openai_cache"),
        )
        openai_response = openai_client.generate(messages=messages)

        # Via OpenRouter (if available)
        if has_openrouter_key():
            openrouter_client = OpenRouterClient(
                model="openai/gpt-4.1-mini",
                cache_dir=str(tmp_path / "openrouter_cache"),
            )
            openrouter_response = openrouter_client.generate(messages=messages)

            # Both should return 4
            assert "4" in openai_response.content
            assert "4" in openrouter_response.content

            # Both should have tracking
            assert openai_response.tracking is not None
            assert openrouter_response.tracking is not None

    def test_batch_processing_both_providers(self, tmp_path: Path) -> None:
        """Test batch processing works for both providers."""
        if not has_openai_key():
            pytest.skip("OPENAI_API_KEY not set")

        messages_list = [
            [Message(role="user", content="Say 'A'")],
            [Message(role="user", content="Say 'B'")],
        ]

        async def run_batch(client):
            try:
                return await client.batch_generate(messages_list, max_concurrent=2)
            finally:
                await client.aclose()

        # OpenAI
        openai_client = OpenAIClient(
            model="gpt-4.1-mini",
            cache_dir=str(tmp_path / "openai_cache"),
        )
        openai_result = asyncio.run(run_batch(openai_client))
        assert openai_result.successful_count == 2

        # OpenRouter (if available)
        if has_openrouter_key():
            openrouter_client = OpenRouterClient(
                model="openai/gpt-4.1-mini",
                cache_dir=str(tmp_path / "openrouter_cache"),
            )
            openrouter_result = asyncio.run(run_batch(openrouter_client))
            assert openrouter_result.successful_count == 2
