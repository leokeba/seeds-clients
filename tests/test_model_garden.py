"""Tests for Model Garden client."""

import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from seeds_clients.core.exceptions import ConfigurationError, ProviderError
from seeds_clients.core.types import Message
from seeds_clients.providers.model_garden import ModelGardenClient


class TestModelGardenClientInit:
    """Tests for ModelGardenClient initialization."""

    def test_init_raises_without_base_url(self, monkeypatch):
        """Test client raises ConfigurationError without base_url."""
        # Clear any existing env var
        monkeypatch.delenv("MODEL_GARDEN_BASE_URL", raising=False)

        with pytest.raises(ConfigurationError) as exc_info:
            ModelGardenClient()

        assert "base url required" in str(exc_info.value).lower()

    def test_init_custom_values(self):
        """Test client initialization with custom values."""
        client = ModelGardenClient(
            base_url="http://custom-server:9000/v1",
            api_key="my-api-key",
            model="Qwen/Qwen2.5-3B-Instruct",
            temperature=0.7,
            max_tokens=1000,
        )

        assert client.base_url == "http://custom-server:9000/v1"
        assert client.model == "Qwen/Qwen2.5-3B-Instruct"
        assert client.api_key == "my-api-key"
        assert client.temperature == 0.7
        assert client.max_tokens == 1000

    def test_init_from_env_var(self, monkeypatch):
        """Test client initialization from environment variable."""
        monkeypatch.setenv("MODEL_GARDEN_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("MODEL_GARDEN_API_KEY", "env-api-key")

        client = ModelGardenClient()

        assert client.api_key == "env-api-key"

    def test_init_base_url_from_env_var(self, monkeypatch):
        """Test client initialization with base URL from environment variable."""
        monkeypatch.setenv("MODEL_GARDEN_BASE_URL", "http://env-server:8000/v1")

        client = ModelGardenClient()

        assert client.base_url == "http://env-server:8000/v1"

    def test_init_default_api_key(self, monkeypatch):
        """Test client uses 'not-needed' as default API key."""
        monkeypatch.setenv("MODEL_GARDEN_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.delenv("MODEL_GARDEN_API_KEY", raising=False)

        client = ModelGardenClient()

        assert client.api_key == "not-needed"
        assert client.model == "default"
        assert client.temperature == 1.0

    def test_provider_name(self, monkeypatch):
        """Test that provider name is correct."""
        monkeypatch.setenv("MODEL_GARDEN_BASE_URL", "http://localhost:8000/v1")
        client = ModelGardenClient()

        assert client._get_provider_name() == "model_garden"
        assert client._get_ecologits_provider() == "model_garden"


class TestModelGardenClientParseResponse:
    """Tests for response parsing with CodeCarbon data."""

    @pytest.fixture
    def client(self):
        """Create a Model Garden client instance."""
        client = ModelGardenClient(base_url="http://localhost:8000/v1")
        yield client
        client.close()

    @pytest.fixture
    def sample_response_with_codecarbon(self):
        """Create a sample API response with x_carbon_trace."""
        return {
            "id": "chatcmpl-123",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
            "x_carbon_trace": {
                "emissions_g_co2": 0.00123,  # grams
                "energy_consumed_wh": 0.0456,  # watt-hours
                "cpu_energy_wh": 0.0100,
                "gpu_energy_wh": 0.0300,
                "ram_energy_wh": 0.0056,
                "cpu_power_watts": 45.0,
                "gpu_power_watts": 150.0,
                "ram_power_watts": 12.0,
                "duration_seconds": 1.5,
                "measured": True,
                "tracking_active": True,
                "session_total_kg_co2": 0.0005,  # kilograms (0.5g)
                "session_requests": 10,
                "session_tokens": 1800,
            },
        }

    @pytest.fixture
    def sample_response_without_codecarbon(self):
        """Create a sample API response without x_carbon_trace."""
        return {
            "id": "chatcmpl-456",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hi there!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        }

    def test_parse_response_with_codecarbon(self, client, sample_response_with_codecarbon):
        """Test parsing response with CodeCarbon data."""
        response = client._parse_response(sample_response_with_codecarbon)

        assert response.content == "Hello! How can I help you today?"
        assert response.model == "Qwen/Qwen2.5-3B-Instruct"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 8
        assert response.finish_reason == "stop"
        assert response.response_id == "chatcmpl-123"

        # Check tracking data
        tracking = response.tracking
        assert tracking is not None
        assert tracking.provider == "model_garden"
        assert tracking.tracking_method == "codecarbon"

        # Energy and emissions (converted from Wh/gCO2 to kWh/kgCO2)
        assert tracking.energy_kwh == pytest.approx(0.0000456, rel=1e-3)
        assert tracking.gwp_kgco2eq == pytest.approx(0.00000123, rel=1e-3)

        # Hardware measurements
        assert tracking.cpu_power_watts == 45.0
        assert tracking.gpu_power_watts == 150.0
        assert tracking.ram_power_watts == 12.0

        # Hardware energy (converted from Wh to kWh)
        assert tracking.cpu_energy_kwh == pytest.approx(0.00001, rel=1e-3)
        assert tracking.gpu_energy_kwh == pytest.approx(0.00003, rel=1e-3)
        assert tracking.ram_energy_kwh == pytest.approx(0.0000056, rel=1e-3)

        # Duration
        assert tracking.duration_seconds == 1.5

    def test_parse_response_without_codecarbon(self, client, sample_response_without_codecarbon):
        """Test parsing response without CodeCarbon data."""
        response = client._parse_response(sample_response_without_codecarbon)

        assert response.content == "Hi there!"
        assert response.usage.prompt_tokens == 5
        assert response.usage.completion_tokens == 3

        # Check tracking data exists but has no carbon info
        tracking = response.tracking
        assert tracking is not None
        assert tracking.provider == "model_garden"
        assert tracking.tracking_method == "none"
        assert tracking.energy_kwh == 0.0
        assert tracking.gwp_kgco2eq == 0.0

        # Hardware measurements should be None
        assert tracking.cpu_power_watts is None
        assert tracking.gpu_power_watts is None
        assert tracking.ram_power_watts is None

    def test_parse_response_stores_codecarbon_metadata(
        self, client, sample_response_with_codecarbon
    ):
        """Test that CodeCarbon metadata is stored in raw response."""
        response = client._parse_response(sample_response_with_codecarbon)

        assert "_codecarbon_metrics" in response.raw
        metrics = response.raw["_codecarbon_metrics"]

        assert metrics["measured"] is True
        assert metrics["tracking_active"] is True
        assert metrics["session_total_kg_co2"] == pytest.approx(
            0.0005, rel=1e-3
        )  # 0.5g -> 0.0005kg
        assert metrics["session_requests"] == 10
        assert metrics["session_tokens"] == 1800

    def test_parse_response_invalid_format(self, client):
        """Test parsing response with invalid format raises error."""
        invalid_response = {"invalid": "data"}

        with pytest.raises(ProviderError) as exc_info:
            client._parse_response(invalid_response)

        assert "Invalid response format" in str(exc_info.value)
        assert "model_garden" in str(exc_info.value)


class TestModelGardenClientAPICall:
    """Tests for API call methods."""

    @pytest.fixture
    def client(self):
        """Create a Model Garden client instance."""
        client = ModelGardenClient(
            base_url="http://localhost:8000/v1",
            model="test-model",
            max_tokens=100,
            temperature=0.5,
        )
        yield client
        client.close()

    @pytest.fixture
    def sample_api_response(self):
        """Sample API response for mocking."""
        return {
            "id": "chatcmpl-test",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test response",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
            "x_carbon_trace": {
                "emissions_g_co2": 0.001,
                "energy_consumed_wh": 0.01,
                "measured": True,
                "tracking_active": True,
            },
        }

    def test_call_api_builds_correct_payload(self, client, sample_api_response):
        """Test that API call builds the correct request payload."""
        mock_response = Mock()
        mock_response.json.return_value = sample_api_response
        mock_response.raise_for_status = Mock()

        with patch.object(client._http_client, "post", return_value=mock_response) as mock_post:
            messages = [Message(role="user", content="Hello")]
            client._call_api(messages)

            # Verify the call was made with correct payload
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args

            payload = call_kwargs.kwargs["json"]
            assert payload["model"] == "test-model"
            assert payload["max_tokens"] == 100
            assert payload["temperature"] == 0.5
            assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    def test_call_api_handles_http_error(self, client):
        """Test that API call handles HTTP errors correctly."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"detail": "Internal server error"}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=Mock(), response=mock_response
        )

        with patch.object(client._http_client, "post", return_value=mock_response):
            messages = [Message(role="user", content="Hello")]

            with pytest.raises(ProviderError) as exc_info:
                client._call_api(messages)

            assert "Model Garden API error" in str(exc_info.value)
            assert "model_garden" in str(exc_info.value)

    def test_call_api_handles_connection_error(self, client):
        """Test that API call handles connection errors correctly."""
        with patch.object(
            client._http_client,
            "post",
            side_effect=httpx.RequestError("Connection refused", request=Mock()),
        ):
            messages = [Message(role="user", content="Hello")]

            with pytest.raises(ProviderError) as exc_info:
                client._call_api(messages)

            assert "Request failed" in str(exc_info.value)
            assert "Model Garden running" in str(exc_info.value)

    def test_call_api_adds_duration(self, client, sample_api_response):
        """Test that API call adds duration to response."""
        mock_response = Mock()
        mock_response.json.return_value = sample_api_response.copy()
        mock_response.raise_for_status = Mock()

        with patch.object(client._http_client, "post", return_value=mock_response):
            messages = [Message(role="user", content="Hello")]
            result = client._call_api(messages)

            assert "_duration_seconds" in result
            assert result["_duration_seconds"] >= 0


class TestModelGardenStructuredOutputs:
    """Tests for structured output handling."""

    class SampleModel(BaseModel):
        field: str

    def test_forwards_response_format_when_supported(self):
        """Ensure response_format is sent to the API when supported."""
        client = ModelGardenClient(
            base_url="http://localhost:8000/v1",
            model="test-model",
            supports_response_format=True,
        )

        api_response = {
            "id": "chatcmpl-schema",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"field":"value"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        }

        mock_response = Mock()
        mock_response.json.return_value = api_response
        mock_response.raise_for_status = Mock()

        with patch.object(client._http_client, "post", return_value=mock_response) as mock_post:
            response = client.generate(
                messages=[Message(role="user", content="Hello")],
                response_format=self.SampleModel,
            )

            payload = mock_post.call_args.kwargs["json"]
            assert payload["model"] == "test-model"
            assert payload["response_format"]["type"] == "json_schema"
            assert payload["response_format"]["json_schema"]["name"] == "SampleModel"

            assert response.parsed is not None
            assert response.parsed.field == "value"

    def test_falls_back_to_prompt_when_not_supported(self):
        """Ensure prompt-based enforcement is used when disabled."""
        client = ModelGardenClient(
            base_url="http://localhost:8000/v1",
            model="test-model",
            supports_response_format=False,
        )

        api_response = {
            "id": "chatcmpl-prompt",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"field":"value"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        }

        mock_response = Mock()
        mock_response.json.return_value = api_response
        mock_response.raise_for_status = Mock()

        with patch.object(client._http_client, "post", return_value=mock_response) as mock_post:
            response = client.generate(
                messages=[Message(role="user", content="Hello")],
                response_format=self.SampleModel,
            )

            payload = mock_post.call_args.kwargs["json"]
            assert "response_format" not in payload

            messages = payload["messages"]
            assert messages[0]["role"] == "system"
            assert "valid JSON only" in messages[0]["content"]

            assert response.parsed is not None
            assert response.parsed.field == "value"


class TestModelGardenClientAsync:
    """Tests for async API methods."""

    @pytest.fixture
    def client(self):
        """Create a Model Garden client instance."""
        client = ModelGardenClient(base_url="http://localhost:8000/v1")
        yield client
        client.close()

    @pytest.fixture
    def sample_api_response(self):
        """Sample API response for mocking."""
        return {
            "id": "chatcmpl-async",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Async response",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        }

    @pytest.mark.asyncio
    async def test_acall_api_handles_http_error(self, client):
        """Test that async API call handles HTTP errors correctly."""
        # Create a mock response that simulates an HTTP error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"detail": "Internal server error"}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=Mock(), response=mock_response
        )

        # Create async mock client where post returns a coroutine
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(client, "_get_async_client", return_value=mock_client):
            messages = [Message(role="user", content="Hello")]

            with pytest.raises(ProviderError) as exc_info:
                await client._acall_api(messages)

            assert "Model Garden API error" in str(exc_info.value)


class TestModelGardenClientCarbonStats:
    """Tests for carbon statistics methods."""

    @pytest.fixture
    def client(self):
        """Create a Model Garden client instance."""
        client = ModelGardenClient(base_url="http://localhost:8000/v1")
        yield client
        client.close()

    def test_get_carbon_stats_success(self, client):
        """Test getting carbon stats successfully."""
        stats_response = {
            "emissions_kg_co2": 0.001,
            "energy_kwh": 0.01,
            "request_count": 10,
            "total_tokens": 500,
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = stats_response

        with patch.object(client._http_client, "get", return_value=mock_response):
            stats = client.get_carbon_stats()

            assert stats is not None
            assert stats["emissions_kg_co2"] == 0.001
            assert stats["request_count"] == 10

    def test_get_carbon_stats_failure(self, client):
        """Test getting carbon stats when endpoint fails."""
        mock_response = Mock()
        mock_response.status_code = 404

        with patch.object(client._http_client, "get", return_value=mock_response):
            stats = client.get_carbon_stats()

            assert stats is None

    def test_get_carbon_stats_exception(self, client):
        """Test getting carbon stats when exception occurs."""
        with patch.object(client._http_client, "get", side_effect=Exception("Network error")):
            stats = client.get_carbon_stats()

            assert stats is None

    def test_get_emissions_summary_success(self, client):
        """Test getting emissions summary successfully."""
        summary_response = {
            "total_emissions_kg_co2": 1.5,
            "inference_emissions_kg_co2": 0.5,
            "training_emissions_kg_co2": 1.0,
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = summary_response

        with patch.object(client._http_client, "get", return_value=mock_response):
            summary = client.get_emissions_summary()

            assert summary is not None
            assert summary["total_emissions_kg_co2"] == 1.5


class TestModelGardenClientIntegration:
    """Integration-style tests for full generate flow."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a Model Garden client instance."""
        client = ModelGardenClient(
            base_url="http://localhost:8000/v1", cache_dir=str(tmp_path / "cache")
        )
        yield client
        client.close()

    def test_generate_full_flow(self, client):
        """Test full generate flow with mocked API."""
        api_response = {
            "id": "chatcmpl-full",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I'm an AI assistant.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 7,
                "total_tokens": 17,
            },
            "x_carbon_trace": {
                "emissions_g_co2": 0.002,
                "energy_consumed_wh": 0.05,
                "cpu_power_watts": 50.0,
                "gpu_power_watts": 200.0,
                "ram_power_watts": 15.0,
                "cpu_energy_wh": 0.01,
                "gpu_energy_wh": 0.035,
                "ram_energy_wh": 0.005,
                "duration_seconds": 2.0,
                "measured": True,
                "tracking_active": True,
            },
        }

        mock_response = Mock()
        mock_response.json.return_value = api_response
        mock_response.raise_for_status = Mock()

        with patch.object(client._http_client, "post", return_value=mock_response):
            response = client.generate(messages=[Message(role="user", content="Hello!")])

            # Verify response content
            assert response.content == "Hello! I'm an AI assistant."
            assert response.model == "Qwen/Qwen2.5-3B-Instruct"

            # Verify tracking data
            tracking = response.tracking
            assert tracking is not None
            assert tracking.tracking_method == "codecarbon"
            assert tracking.provider == "model_garden"

            # Verify energy and emissions
            assert tracking.energy_kwh == pytest.approx(0.00005, rel=1e-3)
            assert tracking.gwp_kgco2eq == pytest.approx(0.000002, rel=1e-3)


@pytest.mark.integration
class TestModelGardenClientLiveStructuredOutputs:
    """Live structured-output tests against a running Model Garden server."""

    class Person(BaseModel):
        name: str
        age: int

    @pytest.fixture
    def live_client(self) -> ModelGardenClient:
        base_url = os.getenv("MODEL_GARDEN_BASE_URL")
        model = os.getenv("MODEL_GARDEN_MODEL", "default")
        api_key = os.getenv("MODEL_GARDEN_API_KEY")
        rf_mode = os.getenv("MODEL_GARDEN_RESPONSE_FORMAT_MODE", "auto")

        if not base_url:
            pytest.skip("MODEL_GARDEN_BASE_URL required for live test")

        client = ModelGardenClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            supports_response_format=True,
            response_format_mode=rf_mode,
            temperature=0,
            max_tokens=256,
        )
        yield client
        client.close()

    def test_structured_output_live(self, live_client: ModelGardenClient):
        """Verify backend-enforced structured outputs end-to-end."""
        messages = [
            Message(
                role="user",
                content=(
                    "Return a JSON object with fields 'name' (string) and 'age' (integer). "
                    "Keep it short."
                ),
            )
        ]

        response = live_client.generate(messages=messages, response_format=self.Person)

        assert isinstance(response.parsed, self.Person)
        assert response.parsed.name
        assert response.parsed.age >= 0


class TestModelGardenCumulativeTracking:
    """Tests for cumulative tracking with ModelGardenClient (CodeCarbon)."""

    @pytest.fixture
    def client(self):
        """Create a Model Garden client instance."""
        client = ModelGardenClient(base_url="http://localhost:8000/v1")
        yield client
        client.close()

    @pytest.fixture
    def sample_response_with_codecarbon(self):
        """Create a sample API response with x_carbon_trace."""
        return {
            "id": "chatcmpl-123",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
            "x_carbon_trace": {
                "emissions_g_co2": 0.52,  # 0.00052 kg
                "energy_consumed_wh": 1.5,  # 0.0015 kWh
                "cpu_energy_wh": 0.5,
                "gpu_energy_wh": 0.8,
                "ram_energy_wh": 0.2,
                "cpu_power_watts": 85.0,
                "gpu_power_watts": 250.0,
                "ram_power_watts": 15.0,
                "duration_seconds": 0.5,
                "measured": True,
                "tracking_active": True,
            },
        }

    def test_client_has_cumulative_tracking(self, client):
        """Test that ModelGardenClient has cumulative tracking initialized."""
        from seeds_clients.core.types import CumulativeTracking

        assert hasattr(client, "cumulative_tracking")
        assert isinstance(client.cumulative_tracking, CumulativeTracking)

    def test_codecarbon_request_accumulates(self, client, sample_response_with_codecarbon):
        """Test that CodeCarbon responses accumulate correctly."""
        mock_response = Mock()
        mock_response.json.return_value = sample_response_with_codecarbon
        mock_response.raise_for_status = Mock()

        with patch.object(client._http_client, "post", return_value=mock_response):
            response = client.generate(
                messages=[Message(role="user", content="Hello!")],
                use_cache=False,
            )

            tracking = client.cumulative_tracking
            assert tracking.api_request_count == 1
            assert tracking.cached_request_count == 0
            assert tracking.api_gwp_kgco2eq == pytest.approx(0.00052)
            assert tracking.api_energy_kwh == pytest.approx(0.0015)

            # CodeCarbon tracks usage phase only
            assert tracking.api_gwp_usage_kgco2eq == pytest.approx(0.00052)
            assert tracking.api_energy_usage_kwh == pytest.approx(0.0015)

            # Embodied should be 0 (CodeCarbon doesn't track it)
            assert tracking.api_gwp_embodied_kgco2eq == 0.0

    def test_multiple_codecarbon_requests_accumulate(self, client, sample_response_with_codecarbon):
        """Test that multiple CodeCarbon requests accumulate correctly."""
        mock_response = Mock()
        mock_response.json.return_value = sample_response_with_codecarbon
        mock_response.raise_for_status = Mock()

        with patch.object(client._http_client, "post", return_value=mock_response):
            # Make 3 requests
            for i in range(3):
                client.generate(
                    messages=[Message(role="user", content=f"Hello {i}!")],
                    use_cache=False,
                )

            tracking = client.cumulative_tracking
            assert tracking.api_request_count == 3
            assert tracking.api_gwp_kgco2eq == pytest.approx(0.00156)  # 0.00052 * 3
            assert tracking.api_energy_kwh == pytest.approx(0.0045)  # 0.0015 * 3
            assert tracking.api_prompt_tokens == 30  # 10 * 3

    def test_reset_cumulative_tracking(self, client, sample_response_with_codecarbon):
        """Test resetting cumulative tracking for ModelGardenClient."""
        mock_response = Mock()
        mock_response.json.return_value = sample_response_with_codecarbon
        mock_response.raise_for_status = Mock()

        with patch.object(client._http_client, "post", return_value=mock_response):
            client.generate(
                messages=[Message(role="user", content="Hello!")],
                use_cache=False,
            )

            # Verify we have data
            assert client.cumulative_tracking.api_request_count == 1
            assert client.cumulative_tracking.api_gwp_kgco2eq > 0

            # Reset
            client.reset_cumulative_tracking()

            # Verify reset
            assert client.cumulative_tracking.api_request_count == 0
            assert client.cumulative_tracking.api_gwp_kgco2eq == 0.0

    @pytest.mark.asyncio
    async def test_async_codecarbon_request_accumulates(
        self, client, sample_response_with_codecarbon
    ):
        """Test that async CodeCarbon requests accumulate correctly."""
        mock_async_response = Mock()
        mock_async_response.json.return_value = sample_response_with_codecarbon
        mock_async_response.raise_for_status = Mock()

        mock_async_client = Mock()
        mock_async_client.post = AsyncMock(return_value=mock_async_response)

        with patch.object(client, "_get_async_client", return_value=mock_async_client):
            response = await client.agenerate(
                messages=[Message(role="user", content="Hello!")],
                use_cache=False,
            )

            tracking = client.cumulative_tracking
            assert tracking.api_request_count == 1
            assert tracking.api_gwp_kgco2eq == pytest.approx(0.00052)
            assert tracking.api_gwp_usage_kgco2eq == pytest.approx(0.00052)
            assert tracking.api_gwp_embodied_kgco2eq == 0.0
