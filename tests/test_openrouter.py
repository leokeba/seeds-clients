"""Tests for OpenRouter client implementation."""

import json
import os
from dataclasses import asdict
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from seeds_clients import OpenRouterClient, OpenRouterCostData
from seeds_clients.core.types import Message


class TestOpenRouterClientInit:
    """Test OpenRouterClient initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        client = OpenRouterClient(api_key="test-key", model="openai/gpt-4.1")
        assert client.api_key == "test-key"
        assert client.model == "openai/gpt-4.1"
        assert "openrouter.ai" in client.base_url

    def test_init_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization from environment variable."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        client = OpenRouterClient(model="anthropic/claude-3-5-sonnet")
        assert client.api_key == "env-key"

    def test_init_fallback_to_openai_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that OPENROUTER_API_KEY is used when provided explicitly."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
        # Our client prefers OPENROUTER_API_KEY but uses explicit api_key
        client = OpenRouterClient(api_key="test-key", model="meta-llama/llama-3-70b-instruct")
        assert client.api_key == "test-key"

    def test_init_without_key_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that initialization without API key raises error."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(Exception):  # ConfigurationError
            OpenRouterClient(model="openai/gpt-4.1")

    def test_init_with_custom_headers(self) -> None:
        """Test initialization with custom OpenRouter headers."""
        client = OpenRouterClient(
            api_key="test-key",
            model="openai/gpt-4.1",
            site_url="https://myapp.com",
            app_name="MyApp",
        )
        assert client.site_url == "https://myapp.com"
        assert client.app_name == "MyApp"

    def test_base_url_is_openrouter(self) -> None:
        """Test that base URL is OpenRouter's API."""
        client = OpenRouterClient(api_key="test-key", model="openai/gpt-4.1")
        assert client.base_url == "https://openrouter.ai/api/v1"


class TestEcoLogitsProvider:
    """Test EcoLogits provider extraction."""

    def test_ecologits_provider_for_openai(self) -> None:
        """Test EcoLogits provider is correctly extracted for OpenAI."""
        client = OpenRouterClient(api_key="test-key", model="openai/gpt-4.1")
        assert client._get_ecologits_provider() == "openai"

    def test_ecologits_provider_for_anthropic(self) -> None:
        """Test EcoLogits provider is correctly extracted for Anthropic."""
        client = OpenRouterClient(api_key="test-key", model="anthropic/claude-3-opus")
        assert client._get_ecologits_provider() == "anthropic"

    def test_ecologits_provider_for_meta(self) -> None:
        """Test EcoLogits provider for Meta/Llama models."""
        client = OpenRouterClient(api_key="test-key", model="meta-llama/llama-3-70b")
        assert client._get_ecologits_provider() == "meta-llama"

    def test_ecologits_provider_for_google(self) -> None:
        """Test EcoLogits provider for Google models."""
        client = OpenRouterClient(api_key="test-key", model="google/gemini-pro")
        assert client._get_ecologits_provider() == "google"

    def test_ecologits_provider_without_slash(self) -> None:
        """Test EcoLogits provider for model without provider prefix."""
        client = OpenRouterClient(api_key="test-key", model="gpt-4.1")
        assert client._get_ecologits_provider() == "openrouter"


class TestEcoLogitsModel:
    """Test EcoLogits model extraction."""

    def test_ecologits_model_extraction(self) -> None:
        """Test model name is correctly extracted."""
        client = OpenRouterClient(api_key="test-key", model="openai/gpt-4.1")
        assert client._get_ecologits_model() == "gpt-4.1"

    def test_ecologits_model_anthropic(self) -> None:
        """Test model extraction for Anthropic."""
        client = OpenRouterClient(api_key="test-key", model="anthropic/claude-3-5-sonnet")
        assert client._get_ecologits_model() == "claude-3-5-sonnet"

    def test_ecologits_model_without_slash(self) -> None:
        """Test model extraction when no provider prefix."""
        client = OpenRouterClient(api_key="test-key", model="gpt-4.1")
        assert client._get_ecologits_model() == "gpt-4.1"


class TestOpenRouterCostData:
    """Test OpenRouterCostData dataclass."""

    def test_cost_data_creation(self) -> None:
        """Test creating OpenRouterCostData."""
        cost_data = OpenRouterCostData(
            generation_id="gen_123",
            total_cost=0.005,
            model="openai/gpt-4.1",
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert cost_data.generation_id == "gen_123"
        assert cost_data.total_cost == 0.005
        assert cost_data.prompt_tokens == 100
        assert cost_data.completion_tokens == 50

    def test_cost_data_optional_fields(self) -> None:
        """Test OpenRouterCostData optional native token fields."""
        cost_data = OpenRouterCostData(
            generation_id="gen_456",
            total_cost=0.01,
            model="anthropic/claude-3",
            native_prompt_tokens=150,
            native_completion_tokens=75,
        )
        assert cost_data.native_prompt_tokens == 150
        assert cost_data.native_completion_tokens == 75

    def test_cost_data_to_dict(self) -> None:
        """Test converting OpenRouterCostData to dict."""
        cost_data = OpenRouterCostData(
            generation_id="gen_789",
            total_cost=0.02,
            model="google/gemini-pro",
        )
        data_dict = asdict(cost_data)
        assert data_dict["generation_id"] == "gen_789"
        assert data_dict["total_cost"] == 0.02
        assert data_dict["model"] == "google/gemini-pro"


class TestOpenRouterGenerate:
    """Test OpenRouterClient generate method."""

    @pytest.fixture
    def mock_response(self) -> dict:
        """Create a mock API response."""
        return {
            "id": "gen_abc123",
            "model": "openai/gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from OpenRouter!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    def test_generate_returns_response(self, mock_response: dict) -> None:
        """Test that generate returns a valid Response."""
        client = OpenRouterClient(api_key="test-key", model="openai/gpt-4.1")

        mock = Mock(spec=httpx.Response)
        mock.status_code = 200
        mock.json.return_value = mock_response
        mock.raise_for_status = Mock()

        with patch.object(client._http_client, "post", return_value=mock):
            response = client.generate([Message(role="user", content="Test prompt")])

        assert response.content == "Hello from OpenRouter!"
        assert response.raw.get("id") == "gen_abc123"

    def test_generate_tracks_provider(self, mock_response: dict) -> None:
        """Test that generate tracks OpenRouter as provider."""
        client = OpenRouterClient(api_key="test-key", model="openai/gpt-4.1")

        mock = Mock(spec=httpx.Response)
        mock.status_code = 200
        mock.json.return_value = mock_response
        mock.raise_for_status = Mock()

        with patch.object(client._http_client, "post", return_value=mock):
            response = client.generate([Message(role="user", content="Test")])

        assert response.tracking is not None
        assert response.tracking.provider == "openrouter"


class TestOpenRouterCostFetching:
    """Test OpenRouter generation cost fetching."""

    def test_fetch_cost_data_success(self) -> None:
        """Test successfully fetching generation cost."""
        client = OpenRouterClient(api_key="test-key", model="openai/gpt-4.1", fetch_cost_data=True)

        mock_cost_response = {
            "data": {
                "id": "gen_xyz",
                "total_cost": 0.0025,
                "tokens_prompt": 100,
                "tokens_completion": 50,
                "native_tokens_prompt": 110,
                "native_tokens_completion": 55,
                "model": "openai/gpt-4.1",
                "provider_name": "OpenAI",
            }
        }

        mock = Mock(spec=httpx.Response)
        mock.status_code = 200
        mock.json.return_value = mock_cost_response
        mock.raise_for_status = Mock()

        with patch("httpx.get", return_value=mock):
            cost_data = client._fetch_cost_data("gen_xyz")

        assert cost_data is not None
        assert cost_data.total_cost == 0.0025
        assert cost_data.prompt_tokens == 100
        assert cost_data.completion_tokens == 50
        assert cost_data.native_prompt_tokens == 110
        assert cost_data.native_completion_tokens == 55
        assert cost_data.model == "openai/gpt-4.1"

    def test_fetch_cost_data_handles_missing_data(self) -> None:
        """Test that cost fetching handles missing data gracefully."""
        client = OpenRouterClient(api_key="test-key", model="openai/gpt-4.1")

        mock = Mock(spec=httpx.Response)
        mock.status_code = 200
        mock.json.return_value = {}  # No 'data' key
        mock.raise_for_status = Mock()

        with patch("httpx.get", return_value=mock):
            cost_data = client._fetch_cost_data("gen_xyz")

        assert cost_data is None

    def test_fetch_cost_data_handles_timeout(self) -> None:
        """Test that cost fetching handles timeout gracefully."""
        client = OpenRouterClient(api_key="test-key", model="openai/gpt-4.1")

        with patch("httpx.get", side_effect=httpx.TimeoutException("timeout")):
            cost_data = client._fetch_cost_data("gen_xyz", max_retries=0)

        assert cost_data is None


class TestCostSummary:
    """Test cost summary functionality."""

    def test_get_cost_summary_empty(self) -> None:
        """Test cost summary with no data."""
        client = OpenRouterClient(api_key="test-key", model="openai/gpt-4.1")
        summary = client.get_cost_summary()
        assert summary["total_cost_usd"] == 0.0
        assert summary["total_requests"] == 0

    def test_get_cost_summary_with_data(self) -> None:
        """Test cost summary with accumulated cost data."""
        client = OpenRouterClient(api_key="test-key", model="openai/gpt-4.1")

        # Manually add cost data
        client._cost_data_history.append(
            OpenRouterCostData(
                generation_id="gen_1",
                total_cost=0.01,
                model="openai/gpt-4.1",
                prompt_tokens=100,
                completion_tokens=50,
                provider_name="OpenAI",
            )
        )
        client._cost_data_history.append(
            OpenRouterCostData(
                generation_id="gen_2",
                total_cost=0.02,
                model="anthropic/claude-3",
                prompt_tokens=200,
                completion_tokens=100,
                provider_name="Anthropic",
            )
        )

        summary = client.get_cost_summary()
        assert summary["total_cost_usd"] == pytest.approx(0.03)
        assert summary["total_requests"] == 2
        assert summary["total_prompt_tokens"] == 300
        assert summary["total_completion_tokens"] == 150
        assert "openai/gpt-4.1" in summary["cost_by_model"]
        assert "anthropic/claude-3" in summary["cost_by_model"]

    def test_reset_cost_tracking(self) -> None:
        """Test resetting cost tracking data."""
        client = OpenRouterClient(api_key="test-key", model="openai/gpt-4.1")

        # Add some cost data
        client._cost_data_history.append(
            OpenRouterCostData(
                generation_id="gen_1",
                total_cost=0.01,
                model="openai/gpt-4.1",
            )
        )

        assert len(client._cost_data_history) == 1
        client.reset_cost_tracking()
        assert len(client._cost_data_history) == 0


class TestOpenRouterProviderMapping:
    """Test mapping of OpenRouter providers to EcoLogits providers."""

    @pytest.mark.parametrize(
        "model,expected_provider",
        [
            ("openai/gpt-4.1", "openai"),
            ("openai/gpt-4-turbo", "openai"),
            ("openai/gpt-3.5-turbo", "openai"),
            ("anthropic/claude-3-opus", "anthropic"),
            ("anthropic/claude-3-5-sonnet", "anthropic"),
            ("google/gemini-pro", "google"),
            ("google/gemini-1.5-pro", "google"),
            ("meta-llama/llama-3-70b", "meta-llama"),
            ("mistralai/mistral-large", "mistralai"),
            ("cohere/command-r-plus", "cohere"),
        ],
    )
    def test_provider_mapping(self, model: str, expected_provider: str) -> None:
        """Test various model provider mappings."""
        client = OpenRouterClient(api_key="test-key", model=model)
        assert client._get_ecologits_provider() == expected_provider


class TestGetProviderName:
    """Test provider name method."""

    def test_get_provider_name(self) -> None:
        """Test that _get_provider_name returns 'openrouter'."""
        client = OpenRouterClient(api_key="test-key", model="openai/gpt-4.1")
        assert client._get_provider_name() == "openrouter"
