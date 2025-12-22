"""Tests for Google Gemini client."""

import json
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from PIL import Image
from pydantic import BaseModel, Field

from seeds_clients import Message
from seeds_clients.core.exceptions import ConfigurationError, ProviderError, ValidationError


# We need to mock the google-genai imports since they may not be installed
@pytest.fixture
def mock_genai():
    """Mock the google-genai module."""
    mock_types = MagicMock()
    mock_genai_module = MagicMock()
    mock_genai_module.Client.return_value = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "google": MagicMock(),
            "google.genai": mock_genai_module,
            "google.genai.types": mock_types,
        },
    ):
        yield mock_genai_module, mock_types


class TestGoogleClientInit:
    """Test Google client initialization."""

    def test_init_with_api_key(self, mock_genai) -> None:
        """Test initialization with explicit API key."""
        mock_genai_module, mock_types = mock_genai

        # Import after mocking
        from seeds_clients.providers.google import GoogleClient

        client = GoogleClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.model == "gemini-2.5-flash"

    def test_init_from_gemini_api_key_env(
        self, mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test initialization from GEMINI_API_KEY environment variable."""
        mock_genai_module, mock_types = mock_genai
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        from seeds_clients.providers.google import GoogleClient

        client = GoogleClient()
        assert client.api_key == "env-key"

    def test_init_from_google_api_key_env(
        self, mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test initialization from GOOGLE_API_KEY environment variable."""
        mock_genai_module, mock_types = mock_genai
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-env-key")

        from seeds_clients.providers.google import GoogleClient

        client = GoogleClient()
        assert client.api_key == "google-env-key"

    def test_init_without_key_raises_error(
        self, mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test initialization without API key raises error."""
        mock_genai_module, mock_types = mock_genai
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        from seeds_clients.providers.google import GoogleClient

        with pytest.raises(ConfigurationError) as exc_info:
            GoogleClient()
        assert "API key required" in str(exc_info.value)

    def test_init_with_custom_model(self, mock_genai) -> None:
        """Test initialization with custom model."""
        mock_genai_module, mock_types = mock_genai

        from seeds_clients.providers.google import GoogleClient

        client = GoogleClient(api_key="test-key", model="gemini-2.5-pro")
        assert client.model == "gemini-2.5-pro"

    def test_init_with_custom_params(self, mock_genai) -> None:
        """Test initialization with custom parameters."""
        mock_genai_module, mock_types = mock_genai

        from seeds_clients.providers.google import GoogleClient

        client = GoogleClient(
            api_key="test-key",
            model="gemini-3-pro-preview",
            max_output_tokens=1000,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            system_instruction="You are a helpful assistant.",
            ttl_hours=48.0,
        )
        assert client.model == "gemini-3-pro-preview"
        assert client.max_output_tokens == 1000
        assert client.temperature == 0.5
        assert client.top_p == 0.9
        assert client.top_k == 40
        assert client.system_instruction == "You are a helpful assistant."

    @pytest.mark.skip(reason="Complex to test package import failure without affecting other tests")
    def test_init_without_genai_package_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization without google-genai package raises error."""
        # This test checks if ConfigurationError is raised when google-genai is not installed
        # Skipped because properly mocking the import failure requires complex setup
        # that can affect other tests
        pass


class TestGoogleClientGenerate:
    """Test Google client generate method."""

    @pytest.fixture
    def client(self, mock_genai) -> Generator:
        """Create test client."""
        mock_genai_module, mock_types = mock_genai

        from seeds_clients.providers.google import GoogleClient

        cache_dir = tempfile.mkdtemp(prefix="test_cache_")
        client = GoogleClient(api_key="test-key", cache_dir=cache_dir)

        yield client

        # Cleanup
        if client.cache:
            client.cache.close()
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.fixture
    def mock_api_response(self, mock_genai) -> MagicMock:
        """Create mock API response."""
        mock_genai_module, mock_types = mock_genai

        # Create mock response structure that mimics google-genai response
        mock_response = MagicMock()
        mock_response.text = "Hello! How can I help?"

        # Mock candidates
        mock_part = MagicMock()
        mock_part.text = "Hello! How can I help?"
        mock_part.inline_data = None

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"
        mock_candidate.safety_ratings = []

        mock_response.candidates = [mock_candidate]

        # Mock usage metadata
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 5
        mock_usage.total_token_count = 15
        mock_response.usage_metadata = mock_usage

        mock_response.model_version = "gemini-2.5-flash"

        return mock_response

    def test_generate_text_message(self, client, mock_api_response) -> None:
        """Test generating response from text message."""
        client._client.models.generate_content.return_value = mock_api_response

        messages = [Message(role="user", content="Hello")]
        response = client.generate(messages)

        assert response.content == "Hello! How can I help?"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15
        assert not response.cached

    def test_generate_with_system_message(self, client, mock_api_response) -> None:
        """Test generating with system message via system_instruction."""
        client._client.models.generate_content.return_value = mock_api_response

        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
        ]
        client.generate(messages, system_instruction="Custom instruction")

        # Check the API was called with system_instruction in config
        call_args = client._client.models.generate_content.call_args
        config = call_args.kwargs.get("config")
        # Note: system_instruction handling depends on _build_generation_config

    def test_generate_with_kwargs(self, client, mock_api_response) -> None:
        """Test generating with additional kwargs."""
        client._client.models.generate_content.return_value = mock_api_response

        messages = [Message(role="user", content="Hello")]
        client.generate(
            messages,
            temperature=0.7,
            max_output_tokens=500,
        )

        # Check parameters were passed to config
        call_args = client._client.models.generate_content.call_args
        assert call_args is not None

    def test_generate_caching(self, client, mock_api_response) -> None:
        """Test response caching."""
        client._client.models.generate_content.return_value = mock_api_response

        messages = [Message(role="user", content="Hello")]

        # First call - should hit API
        response1 = client.generate(messages, use_cache=True)
        assert client._client.models.generate_content.call_count == 1
        assert not response1.cached

        # Second call - should use cache
        response2 = client.generate(messages, use_cache=True)
        assert client._client.models.generate_content.call_count == 1  # No additional API call
        assert response2.cached
        assert response2.content == response1.content

        # Clear cache for cleanup
        if client.cache:
            client.cache.clear()

    def test_generate_multimodal_message(self, client, mock_api_response) -> None:
        """Test generating with multimodal (text + image) message."""
        client._client.models.generate_content.return_value = mock_api_response

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "source": "https://example.com/image.jpg"},
                ],
            )
        ]
        response = client.generate(messages)

        # Check multimodal content was formatted and sent
        assert client._client.models.generate_content.called


class TestGoogleClientErrors:
    """Test Google client error handling."""

    @pytest.fixture
    def client(self, mock_genai):
        """Create test client."""
        mock_genai_module, mock_types = mock_genai

        from seeds_clients.providers.google import GoogleClient

        return GoogleClient(api_key="test-key")

    def test_api_error(self, client) -> None:
        """Test handling API errors."""
        error = Exception("API Error: Invalid request")
        client._client.models.generate_content.side_effect = error

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError) as exc_info:
            client.generate(messages)

        err = exc_info.value
        assert err.provider == "google"
        assert "API Error" in str(err)

    def test_api_error_with_status_code(self, client) -> None:
        """Test handling API error with status code attribute."""
        error = Exception("Rate limit exceeded")
        error.status_code = 429  # type: ignore
        client._client.models.generate_content.side_effect = error

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError) as exc_info:
            client.generate(messages)

        err = exc_info.value
        assert err.provider == "google"
        assert err.status_code == 429


class TestGoogleClientImageFormatting:
    """Test image formatting for multimodal messages."""

    @pytest.fixture
    def client(self, mock_genai):
        """Create test client."""
        mock_genai_module, mock_types = mock_genai

        from seeds_clients.providers.google import GoogleClient

        return GoogleClient(api_key="test-key")

    def test_format_image_url(self, client) -> None:
        """Test formatting HTTP image URL."""
        url = "https://example.com/image.jpg"
        part = client._format_image_part(url)
        # Should return a Part from URI
        assert part is not None

    def test_format_image_gs_uri(self, client) -> None:
        """Test formatting Google Cloud Storage URI."""
        gs_uri = "gs://bucket/image.jpg"
        part = client._format_image_part(gs_uri)
        # Should return a Part from URI
        assert part is not None

    def test_format_pil_image(self, client) -> None:
        """Test formatting PIL Image."""
        img = Image.new("RGB", (10, 10), color="blue")
        part = client._format_image_part(img)
        # Should return a Part from bytes
        assert part is not None

    def test_format_image_bytes(self, client) -> None:
        """Test formatting image bytes."""
        import io

        img = Image.new("RGB", (10, 10), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        part = client._format_image_part(img_bytes)
        # Should return a Part from bytes
        assert part is not None

    def test_format_data_url(self, client) -> None:
        """Test formatting data URL."""
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"
        part = client._format_image_part(data_url)
        # Should extract base64 and create Part from bytes
        assert part is not None

    def test_format_image_file_path(self, client, tmp_path: Path) -> None:
        """Test formatting image from file path."""
        # Create a test image
        img = Image.new("RGB", (10, 10), color="red")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        part = client._format_image_part(str(img_path))
        # Should return a Part from bytes
        assert part is not None


class TestGoogleStructuredOutputs:
    """Test Google structured outputs with Pydantic models."""

    @pytest.fixture
    def client(self, mock_genai) -> Generator:
        """Create test client."""
        mock_genai_module, mock_types = mock_genai

        from seeds_clients.providers.google import GoogleClient

        cache_dir = tempfile.mkdtemp(prefix="test_cache_")
        client = GoogleClient(api_key="test-key", cache_dir=cache_dir)

        yield client

        if client.cache:
            client.cache.close()
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_structured_output_simple_model(self, client) -> None:
        """Test structured output with a simple Pydantic model."""

        class Person(BaseModel):
            name: str
            age: int

        # Create mock response with JSON content
        mock_response = MagicMock()
        mock_response.text = json.dumps({"name": "John Doe", "age": 30})

        mock_part = MagicMock()
        mock_part.text = json.dumps({"name": "John Doe", "age": 30})
        mock_part.inline_data = None

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"
        mock_candidate.safety_ratings = []

        mock_response.candidates = [mock_candidate]

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 5
        mock_usage.total_token_count = 15
        mock_response.usage_metadata = mock_usage

        # Important: set model_version to a string, not MagicMock
        mock_response.model_version = "gemini-2.5-flash"

        client._client.models.generate_content.return_value = mock_response

        messages = [Message(role="user", content="Extract person info")]
        response = client.generate(messages, response_format=Person)

        # Check parsed output
        assert response.parsed is not None
        assert isinstance(response.parsed, Person)
        assert response.parsed.name == "John Doe"
        assert response.parsed.age == 30

    def test_structured_output_nested_model(self, client) -> None:
        """Test structured output with nested models."""

        class Address(BaseModel):
            street: str
            city: str
            zip_code: str

        class PersonWithAddress(BaseModel):
            name: str
            age: int
            address: Address

        mock_data = {
            "name": "Jane Smith",
            "age": 25,
            "address": {
                "street": "123 Main St",
                "city": "Boston",
                "zip_code": "02101",
            },
        }

        mock_response = MagicMock()
        mock_response.text = json.dumps(mock_data)

        mock_part = MagicMock()
        mock_part.text = json.dumps(mock_data)
        mock_part.inline_data = None

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"
        mock_candidate.safety_ratings = []

        mock_response.candidates = [mock_candidate]

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 15
        mock_usage.candidates_token_count = 10
        mock_usage.total_token_count = 25
        mock_response.usage_metadata = mock_usage

        # Important: set model_version to a string
        mock_response.model_version = "gemini-2.5-flash"

        client._client.models.generate_content.return_value = mock_response

        messages = [Message(role="user", content="Extract person with address")]
        response = client.generate(messages, response_format=PersonWithAddress)

        assert response.parsed is not None
        assert isinstance(response.parsed, PersonWithAddress)
        assert response.parsed.name == "Jane Smith"
        assert response.parsed.age == 25
        assert isinstance(response.parsed.address, Address)
        assert response.parsed.address.city == "Boston"

    def test_structured_output_invalid_json(self, client) -> None:
        """Test error handling for invalid JSON in structured output."""

        class Person(BaseModel):
            name: str
            age: int

        mock_response = MagicMock()
        mock_response.text = "This is not JSON"

        mock_part = MagicMock()
        mock_part.text = "This is not JSON"
        mock_part.inline_data = None

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"
        mock_candidate.safety_ratings = []

        mock_response.candidates = [mock_candidate]

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 5
        mock_usage.total_token_count = 15
        mock_response.usage_metadata = mock_usage

        # Important: set model_version to a string
        mock_response.model_version = "gemini-2.5-flash"

        client._client.models.generate_content.return_value = mock_response

        messages = [Message(role="user", content="Extract person")]

        with pytest.raises(ValidationError) as exc_info:
            client.generate(messages, response_format=Person)

        assert "Failed to parse structured output" in str(exc_info.value)
        raw = exc_info.value.raw_response
        assert isinstance(raw, dict)
        assert raw.get("text") == "This is not JSON"
        assert raw.get("model_version") == "gemini-2.5-flash"
        assert raw.get("usage_metadata", {}).get("prompt_token_count") == 10

    def test_structured_output_caching(self, client) -> None:
        """Test that structured outputs work with caching."""

        class Person(BaseModel):
            name: str
            age: int

        mock_response = MagicMock()
        mock_response.text = json.dumps({"name": "Cached Person", "age": 40})

        mock_part = MagicMock()
        mock_part.text = json.dumps({"name": "Cached Person", "age": 40})
        mock_part.inline_data = None

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"
        mock_candidate.safety_ratings = []

        mock_response.candidates = [mock_candidate]

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 5
        mock_usage.total_token_count = 15
        mock_response.usage_metadata = mock_usage

        # Important: set model_version to a string
        mock_response.model_version = "gemini-2.5-flash"

        client._client.models.generate_content.return_value = mock_response

        messages = [Message(role="user", content="Extract cached person")]

        # First call
        response1 = client.generate(messages, response_format=Person, use_cache=True)
        assert client._client.models.generate_content.call_count == 1
        assert response1.parsed is not None
        assert response1.parsed.name == "Cached Person"
        assert not response1.cached

        # Second call - should use cache
        response2 = client.generate(messages, response_format=Person, use_cache=True)
        assert client._client.models.generate_content.call_count == 1  # No additional API call
        assert response2.parsed is not None
        assert response2.parsed.name == "Cached Person"
        assert response2.cached


class TestGoogleCostTracking:
    """Tests for cost tracking in Google client."""

    @pytest.fixture
    def client(self, mock_genai) -> Generator:
        """Create test client."""
        mock_genai_module, mock_types = mock_genai

        from seeds_clients.providers.google import GoogleClient

        cache_dir = tempfile.mkdtemp(prefix="test_cache_")
        client = GoogleClient(api_key="test-key", cache_dir=cache_dir, model="gemini-2.5-flash")

        yield client

        if client.cache:
            client.cache.close()
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_cost_tracking_enabled(self, client) -> None:
        """Test that cost tracking is automatically enabled."""
        mock_response = MagicMock()
        mock_response.text = "Test response"

        mock_part = MagicMock()
        mock_part.text = "Test response"
        mock_part.inline_data = None

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"
        mock_candidate.safety_ratings = []

        mock_response.candidates = [mock_candidate]

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 1000
        mock_usage.candidates_token_count = 500
        mock_usage.total_token_count = 1500
        mock_response.usage_metadata = mock_usage

        mock_response.model_version = "gemini-2.5-flash"

        client._client.models.generate_content.return_value = mock_response

        messages = [Message(role="user", content="Test")]
        response = client.generate(messages)

        # Check tracking data is present
        assert response.tracking is not None
        assert response.tracking.cost_usd > 0
        assert response.tracking.prompt_tokens == 1000
        assert response.tracking.completion_tokens == 500
        assert response.tracking.provider == "google"

        # Check cost calculation for gemini-2.5-flash
        # $0.30 per 1M input, $2.50 per 1M output
        # (1000/1M * 0.30) + (500/1M * 2.50) = 0.0003 + 0.00125 = 0.00155
        expected_cost = (1000 / 1_000_000 * 0.30) + (500 / 1_000_000 * 2.50)
        assert response.tracking.cost_usd == pytest.approx(expected_cost)

    def test_cost_tracking_unknown_model(self, client) -> None:
        """Test cost tracking with unknown model falls back to zero cost."""
        mock_response = MagicMock()
        mock_response.text = "Test"

        mock_part = MagicMock()
        mock_part.text = "Test"
        mock_part.inline_data = None

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"
        mock_candidate.safety_ratings = []

        mock_response.candidates = [mock_candidate]

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 1000
        mock_usage.candidates_token_count = 500
        mock_usage.total_token_count = 1500
        mock_response.usage_metadata = mock_usage

        mock_response.model_version = "unknown-model-xyz"

        client._client.models.generate_content.return_value = mock_response

        messages = [Message(role="user", content="Test")]
        response = client.generate(messages)

        # Should still have tracking data but with zero cost
        assert response.tracking is not None
        assert response.tracking.cost_usd == 0.0
        assert response.tracking.prompt_tokens == 1000
        assert response.tracking.completion_tokens == 500


class TestGoogleClientAsync:
    """Test Google client async methods."""

    @pytest.fixture
    def client(self, mock_genai) -> Generator:
        """Create test client."""
        mock_genai_module, mock_types = mock_genai

        from seeds_clients.providers.google import GoogleClient

        cache_dir = tempfile.mkdtemp(prefix="test_cache_")
        client = GoogleClient(api_key="test-key", cache_dir=cache_dir)

        yield client

        if client.cache:
            client.cache.close()
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_agenerate_text_message(self, client) -> None:
        """Test async generating response from text message."""
        # Create mock async response
        mock_response = MagicMock()
        mock_response.text = "Hello async!"

        mock_part = MagicMock()
        mock_part.text = "Hello async!"
        mock_part.inline_data = None

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"
        mock_candidate.safety_ratings = []

        mock_response.candidates = [mock_candidate]

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 5
        mock_usage.total_token_count = 15
        mock_response.usage_metadata = mock_usage

        # Important: set model_version to a string
        mock_response.model_version = "gemini-2.5-flash"

        # Make the async client return the mock
        client._async_client.models.generate_content = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hello")]
        response = await client.agenerate(messages)

        assert response.content == "Hello async!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert not response.cached

    @pytest.mark.asyncio
    async def test_agenerate_with_structured_output(self, client) -> None:
        """Test async structured output generation."""

        class Product(BaseModel):
            name: str
            price: float

        mock_data = {"name": "Widget", "price": 29.99}

        mock_response = MagicMock()
        mock_response.text = json.dumps(mock_data)

        mock_part = MagicMock()
        mock_part.text = json.dumps(mock_data)
        mock_part.inline_data = None

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"
        mock_candidate.safety_ratings = []

        mock_response.candidates = [mock_candidate]

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 5
        mock_usage.total_token_count = 15
        mock_response.usage_metadata = mock_usage

        # Important: set model_version to a string
        mock_response.model_version = "gemini-2.5-flash"

        client._async_client.models.generate_content = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Extract product")]
        response = await client.agenerate(messages, response_format=Product)

        assert response.parsed is not None
        assert isinstance(response.parsed, Product)
        assert response.parsed.name == "Widget"
        assert response.parsed.price == 29.99

    @pytest.mark.asyncio
    async def test_agenerate_caching(self, client) -> None:
        """Test async response caching."""
        mock_response = MagicMock()
        mock_response.text = "Cached response"

        mock_part = MagicMock()
        mock_part.text = "Cached response"
        mock_part.inline_data = None

        mock_content = MagicMock()
        mock_content.role = "model"
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"
        mock_candidate.safety_ratings = []

        mock_response.candidates = [mock_candidate]

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 5
        mock_usage.total_token_count = 15
        mock_response.usage_metadata = mock_usage

        # Important: set model_version to a string
        mock_response.model_version = "gemini-2.5-flash"

        client._async_client.models.generate_content = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Cache test")]

        # First call - should hit API
        response1 = await client.agenerate(messages, use_cache=True)
        assert client._async_client.models.generate_content.call_count == 1
        assert not response1.cached

        # Second call - should use cache
        response2 = await client.agenerate(messages, use_cache=True)
        assert (
            client._async_client.models.generate_content.call_count == 1
        )  # No additional API call
        assert response2.cached
        assert response2.content == response1.content


class TestGoogleProviderName:
    """Test Google client provider name methods."""

    @pytest.fixture
    def client(self, mock_genai):
        """Create test client."""
        mock_genai_module, mock_types = mock_genai

        from seeds_clients.providers.google import GoogleClient

        return GoogleClient(api_key="test-key")

    def test_get_provider_name(self, client) -> None:
        """Test _get_provider_name returns correct value."""
        assert client._get_provider_name() == "google"

    def test_get_ecologits_provider(self, client) -> None:
        """Test _get_ecologits_provider returns correct value."""
        # EcoLogits uses "google_genai" for Google Gemini models
        assert client._get_ecologits_provider() == "google_genai"


class TestGoogleModelNormalization:
    """Test model name normalization for pricing lookup."""

    @pytest.fixture
    def client(self, mock_genai):
        """Create test client."""
        mock_genai_module, mock_types = mock_genai

        from seeds_clients.providers.google import GoogleClient

        return GoogleClient(api_key="test-key")

    def test_normalize_versioned_model(self, client) -> None:
        """Test normalizing versioned model names."""
        # Model with version suffix
        assert client._normalize_model_for_pricing("gemini-2.5-flash-001") == "gemini-2.5-flash"
        assert client._normalize_model_for_pricing("gemini-2.5-pro-002") == "gemini-2.5-pro"

    def test_normalize_base_model(self, client) -> None:
        """Test normalizing base model names (no change needed)."""
        assert client._normalize_model_for_pricing("gemini-2.5-flash") == "gemini-2.5-flash"
        assert client._normalize_model_for_pricing("gemini-3-pro-preview") == "gemini-3-pro-preview"
