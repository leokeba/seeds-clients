"""Tests for Anthropic Claude client."""

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


# We need to mock the anthropic imports since they may not be installed
@pytest.fixture
def mock_anthropic():
    """Mock the anthropic module."""
    mock_anthropic_module = MagicMock()
    mock_anthropic_module.Anthropic.return_value = MagicMock()
    mock_anthropic_module.AsyncAnthropic.return_value = MagicMock()

    # Mock exception classes
    mock_anthropic_module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    mock_anthropic_module.RateLimitError = type("RateLimitError", (Exception,), {})
    mock_anthropic_module.APIStatusError = type(
        "APIStatusError",
        (Exception,),
        {
            "__init__": lambda self, msg, status_code=500: setattr(self, "status_code", status_code)
            or None
        },
    )

    with patch.dict(
        "sys.modules",
        {
            "anthropic": mock_anthropic_module,
        },
    ):
        yield mock_anthropic_module


class TestAnthropicClientInit:
    """Test Anthropic client initialization."""

    def test_init_with_api_key(self, mock_anthropic) -> None:
        """Test initialization with explicit API key."""
        from seeds_clients.providers.anthropic import AnthropicClient

        client = AnthropicClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.model == "claude-sonnet-4-20250514"

    def test_init_from_env(self, mock_anthropic, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization from ANTHROPIC_API_KEY environment variable."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")

        from seeds_clients.providers.anthropic import AnthropicClient

        client = AnthropicClient()
        assert client.api_key == "env-key"

    def test_init_without_key_raises_error(
        self, mock_anthropic, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test initialization without API key raises error."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from seeds_clients.providers.anthropic import AnthropicClient

        with pytest.raises(ConfigurationError) as exc_info:
            AnthropicClient()
        assert "API key required" in str(exc_info.value)

    def test_init_with_custom_model(self, mock_anthropic) -> None:
        """Test initialization with custom model."""
        from seeds_clients.providers.anthropic import AnthropicClient

        client = AnthropicClient(api_key="test-key", model="claude-3-opus-20240229")
        assert client.model == "claude-3-opus-20240229"

    def test_init_with_custom_params(self, mock_anthropic) -> None:
        """Test initialization with custom parameters."""
        from seeds_clients.providers.anthropic import AnthropicClient

        client = AnthropicClient(
            api_key="test-key",
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            system="You are a helpful assistant.",
            stop_sequences=["###"],
            ttl_hours=48.0,
        )
        assert client.model == "claude-3-5-sonnet-20241022"
        assert client.max_tokens == 2000
        assert client.temperature == 0.5
        assert client.top_p == 0.9
        assert client.top_k == 40
        assert client.system == "You are a helpful assistant."
        assert client.stop_sequences == ["###"]


class TestAnthropicClientGenerate:
    """Test Anthropic client generate method."""

    @pytest.fixture
    def client(self, mock_anthropic) -> Generator:
        """Create test client."""
        from seeds_clients.providers.anthropic import AnthropicClient

        cache_dir = tempfile.mkdtemp(prefix="test_cache_")
        client = AnthropicClient(api_key="test-key", cache_dir=cache_dir)

        yield client

        # Cleanup
        if client.cache:
            client.cache.close()
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.fixture
    def mock_api_response(self, mock_anthropic) -> MagicMock:
        """Create mock API response."""
        # Create mock response structure that mimics Anthropic response
        mock_response = MagicMock()
        mock_response.id = "msg_test123"
        mock_response.type = "message"
        mock_response.role = "assistant"
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"
        mock_response.stop_sequence = None

        # Mock content blocks
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello! How can I help?"
        mock_response.content = [mock_text_block]

        # Mock usage
        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5
        mock_response.usage = mock_usage

        return mock_response

    def test_generate_text_message(self, client, mock_api_response) -> None:
        """Test generating response from text message."""
        client._client.messages.create.return_value = mock_api_response

        messages = [Message(role="user", content="Hello")]
        response = client.generate(messages)

        assert response.content == "Hello! How can I help?"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15
        assert not response.cached

    def test_generate_with_system_prompt(self, client, mock_api_response) -> None:
        """Test generating with system prompt."""
        client._client.messages.create.return_value = mock_api_response

        messages = [Message(role="user", content="Hello")]
        client.generate(messages, system="You are a helpful assistant.")

        # Check the API was called with system parameter
        call_args = client._client.messages.create.call_args
        assert call_args.kwargs.get("system") == "You are a helpful assistant."

    def test_generate_with_kwargs(self, client, mock_api_response) -> None:
        """Test generating with additional kwargs."""
        client._client.messages.create.return_value = mock_api_response

        messages = [Message(role="user", content="Hello")]
        client.generate(
            messages,
            temperature=0.7,
            max_tokens=500,
        )

        # Check parameters were passed
        call_args = client._client.messages.create.call_args
        assert call_args.kwargs.get("temperature") == 0.7
        assert call_args.kwargs.get("max_tokens") == 500

    def test_generate_caching(self, client, mock_api_response) -> None:
        """Test response caching."""
        client._client.messages.create.return_value = mock_api_response

        messages = [Message(role="user", content="Hello")]

        # First call - should hit API
        response1 = client.generate(messages, use_cache=True)
        assert client._client.messages.create.call_count == 1
        assert not response1.cached

        # Second call - should use cache
        response2 = client.generate(messages, use_cache=True)
        assert client._client.messages.create.call_count == 1  # No additional API call
        assert response2.cached
        assert response2.content == response1.content

        # Clear cache for cleanup
        if client.cache:
            client.cache.clear()

    def test_generate_multimodal_message(self, client, mock_api_response) -> None:
        """Test generating with multimodal (text + image) message."""
        client._client.messages.create.return_value = mock_api_response

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
        assert client._client.messages.create.called

    def test_generate_filters_system_messages(self, client, mock_api_response) -> None:
        """Test that system messages are filtered from messages list."""
        client._client.messages.create.return_value = mock_api_response

        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
        ]
        client.generate(messages)

        # Check the messages passed to API don't include system
        call_args = client._client.messages.create.call_args
        api_messages = call_args.kwargs.get("messages", [])
        assert len(api_messages) == 1
        assert api_messages[0]["role"] == "user"


class TestAnthropicClientAsync:
    """Test Anthropic client async methods."""

    @pytest.fixture
    def client(self, mock_anthropic) -> Generator:
        """Create test client."""
        from seeds_clients.providers.anthropic import AnthropicClient

        cache_dir = tempfile.mkdtemp(prefix="test_cache_")
        client = AnthropicClient(api_key="test-key", cache_dir=cache_dir)

        yield client

        if client.cache:
            client.cache.close()
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.fixture
    def mock_async_response(self) -> MagicMock:
        """Create mock async API response."""
        mock_response = MagicMock()
        mock_response.id = "msg_async123"
        mock_response.type = "message"
        mock_response.role = "assistant"
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"
        mock_response.stop_sequence = None

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Async response"
        mock_response.content = [mock_text_block]

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5
        mock_response.usage = mock_usage

        return mock_response

    @pytest.mark.asyncio
    async def test_agenerate_text_message(self, client, mock_async_response) -> None:
        """Test async generation."""
        client._async_client.messages.create = AsyncMock(return_value=mock_async_response)

        messages = [Message(role="user", content="Hello")]
        response = await client.agenerate(messages)

        assert response.content == "Async response"
        assert not response.cached

    @pytest.mark.asyncio
    async def test_agenerate_caching(self, client, mock_async_response) -> None:
        """Test async response caching."""
        client._async_client.messages.create = AsyncMock(return_value=mock_async_response)

        messages = [Message(role="user", content="Hello async")]

        # First call - should hit API
        response1 = await client.agenerate(messages, use_cache=True)
        assert client._async_client.messages.create.call_count == 1
        assert not response1.cached

        # Second call - should use cache
        response2 = await client.agenerate(messages, use_cache=True)
        assert client._async_client.messages.create.call_count == 1
        assert response2.cached

        if client.cache:
            client.cache.clear()


class TestAnthropicClientErrors:
    """Test Anthropic client error handling."""

    @pytest.fixture
    def client(self, mock_anthropic):
        """Create test client."""
        from seeds_clients.providers.anthropic import AnthropicClient

        return AnthropicClient(api_key="test-key")

    def test_api_error(self, client, mock_anthropic) -> None:
        """Test handling API errors."""
        error = Exception("API Error: Invalid request")
        client._client.messages.create.side_effect = error

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError) as exc_info:
            client.generate(messages)

        err = exc_info.value
        assert err.provider == "anthropic"
        assert "API Error" in str(err)

    def test_connection_error(self, client, mock_anthropic) -> None:
        """Test handling connection errors."""
        error = client._anthropic.APIConnectionError("Connection failed")
        client._client.messages.create.side_effect = error

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError) as exc_info:
            client.generate(messages)

        err = exc_info.value
        assert err.provider == "anthropic"
        assert "connection" in str(err).lower()

    def test_rate_limit_error(self, client, mock_anthropic) -> None:
        """Test handling rate limit errors."""
        error = client._anthropic.RateLimitError("Rate limit exceeded")
        client._client.messages.create.side_effect = error

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ProviderError) as exc_info:
            client.generate(messages)

        err = exc_info.value
        assert err.provider == "anthropic"
        assert err.status_code == 429


class TestAnthropicClientImageFormatting:
    """Test image formatting for multimodal messages."""

    @pytest.fixture
    def client(self, mock_anthropic):
        """Create test client."""
        from seeds_clients.providers.anthropic import AnthropicClient

        return AnthropicClient(api_key="test-key")

    def test_format_image_url(self, client) -> None:
        """Test formatting HTTP image URL."""
        url = "https://example.com/image.jpg"
        block = client._format_image_part(url)
        assert block is not None
        assert block["type"] == "image"
        assert block["source"]["type"] == "url"
        assert block["source"]["url"] == url

    def test_format_pil_image(self, client) -> None:
        """Test formatting PIL Image."""
        img = Image.new("RGB", (10, 10), color="blue")
        block = client._format_image_part(img)
        assert block is not None
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"
        assert "data" in block["source"]

    def test_format_image_bytes(self, client) -> None:
        """Test formatting image bytes."""
        import io

        img = Image.new("RGB", (10, 10), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        block = client._format_image_part(img_bytes)
        assert block is not None
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"

    def test_format_data_url(self, client) -> None:
        """Test formatting data URL."""
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"
        block = client._format_image_part(data_url)
        assert block is not None
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"

    def test_format_image_file_path(self, client, tmp_path: Path) -> None:
        """Test formatting image from file path."""
        # Create a test image
        img = Image.new("RGB", (10, 10), color="red")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        block = client._format_image_part(str(img_path))
        assert block is not None
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"


class TestAnthropicStructuredOutputs:
    """Test Anthropic structured outputs with Pydantic models."""

    @pytest.fixture
    def client(self, mock_anthropic) -> Generator:
        """Create test client."""
        from seeds_clients.providers.anthropic import AnthropicClient

        cache_dir = tempfile.mkdtemp(prefix="test_cache_")
        client = AnthropicClient(api_key="test-key", cache_dir=cache_dir)

        yield client

        if client.cache:
            client.cache.close()
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_structured_output_simple_model(self, client) -> None:
        """Test structured output with a simple Pydantic model."""

        class Person(BaseModel):
            name: str
            age: int

        # Create mock response with tool_use block
        mock_response = MagicMock()
        mock_response.id = "msg_struct123"
        mock_response.type = "message"
        mock_response.role = "assistant"
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "tool_use"
        mock_response.stop_sequence = None

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_123"
        mock_tool_block.name = "Person"
        mock_tool_block.input = {"name": "John Doe", "age": 30}
        mock_response.content = [mock_tool_block]

        mock_usage = MagicMock()
        mock_usage.input_tokens = 20
        mock_usage.output_tokens = 15
        mock_response.usage = mock_usage

        client._client.messages.create.return_value = mock_response

        messages = [Message(role="user", content="Extract: John Doe is 30 years old")]
        response = client.generate(messages, response_format=Person)

        assert response.parsed is not None
        assert response.parsed.name == "John Doe"
        assert response.parsed.age == 30

    def test_structured_output_nested_model(self, client) -> None:
        """Test structured output with nested Pydantic model."""

        class Address(BaseModel):
            city: str
            country: str

        class Company(BaseModel):
            name: str
            employees: int
            address: Address

        # Create mock response with tool_use block
        mock_response = MagicMock()
        mock_response.id = "msg_nested123"
        mock_response.type = "message"
        mock_response.role = "assistant"
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "tool_use"
        mock_response.stop_sequence = None

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_456"
        mock_tool_block.name = "Company"
        mock_tool_block.input = {
            "name": "Acme Corp",
            "employees": 100,
            "address": {"city": "Paris", "country": "France"},
        }
        mock_response.content = [mock_tool_block]

        mock_usage = MagicMock()
        mock_usage.input_tokens = 30
        mock_usage.output_tokens = 25
        mock_response.usage = mock_usage

        client._client.messages.create.return_value = mock_response

        messages = [Message(role="user", content="Extract company info")]
        response = client.generate(messages, response_format=Company)

        assert response.parsed is not None
        assert response.parsed.name == "Acme Corp"
        assert response.parsed.employees == 100
        assert response.parsed.address.city == "Paris"

    def test_structured_output_tool_choice(self, client) -> None:
        """Test that tool_choice is set correctly for structured output."""

        class Item(BaseModel):
            name: str
            quantity: int

        mock_response = MagicMock()
        mock_response.id = "msg_tool123"
        mock_response.type = "message"
        mock_response.role = "assistant"
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "tool_use"
        mock_response.stop_sequence = None

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_789"
        mock_tool_block.name = "Item"
        mock_tool_block.input = {"name": "Widget", "quantity": 5}
        mock_response.content = [mock_tool_block]

        mock_usage = MagicMock()
        mock_usage.input_tokens = 15
        mock_usage.output_tokens = 10
        mock_response.usage = mock_usage

        client._client.messages.create.return_value = mock_response

        messages = [Message(role="user", content="Extract item")]
        client.generate(messages, response_format=Item)

        # Check tools and tool_choice were set
        call_args = client._client.messages.create.call_args
        tools = call_args.kwargs.get("tools")
        tool_choice = call_args.kwargs.get("tool_choice")

        assert tools is not None
        assert len(tools) == 1
        assert tools[0]["name"] == "Item"
        assert tool_choice == {"type": "tool", "name": "Item"}


class TestAnthropicMessageFormatting:
    """Test message formatting for Anthropic API."""

    @pytest.fixture
    def client(self, mock_anthropic):
        """Create test client."""
        from seeds_clients.providers.anthropic import AnthropicClient

        return AnthropicClient(api_key="test-key")

    def test_format_text_messages(self, client) -> None:
        """Test formatting text messages."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]
        formatted = client._format_messages(messages)

        assert len(formatted) == 3
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"] == "Hello"
        assert formatted[1]["role"] == "assistant"
        assert formatted[2]["role"] == "user"

    def test_format_filters_system_messages(self, client) -> None:
        """Test that system messages are filtered."""
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
        ]
        formatted = client._format_messages(messages)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"

    def test_format_multimodal_messages(self, client) -> None:
        """Test formatting multimodal messages."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What's this?"},
                    {"type": "image", "source": "https://example.com/img.jpg"},
                ],
            )
        ]
        formatted = client._format_messages(messages)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"
        assert isinstance(formatted[0]["content"], list)
        assert len(formatted[0]["content"]) == 2


class TestAnthropicModelNormalization:
    """Test model name normalization for pricing."""

    @pytest.fixture
    def client(self, mock_anthropic):
        """Create test client."""
        from seeds_clients.providers.anthropic import AnthropicClient

        return AnthropicClient(api_key="test-key")

    def test_normalize_claude_sonnet_4(self, client) -> None:
        """Test normalizing Claude Sonnet 4 model names."""
        assert client._normalize_model_for_pricing("claude-sonnet-4-20250514") == "claude-sonnet-4"

    def test_normalize_claude_3_5_sonnet(self, client) -> None:
        """Test normalizing Claude 3.5 Sonnet model names."""
        assert (
            client._normalize_model_for_pricing("claude-3-5-sonnet-20241022") == "claude-3-5-sonnet"
        )
        assert (
            client._normalize_model_for_pricing("claude-3-5-sonnet-20240620") == "claude-3-5-sonnet"
        )

    def test_normalize_claude_3_5_haiku(self, client) -> None:
        """Test normalizing Claude 3.5 Haiku model names."""
        assert (
            client._normalize_model_for_pricing("claude-3-5-haiku-20241022") == "claude-3-5-haiku"
        )

    def test_normalize_claude_3_opus(self, client) -> None:
        """Test normalizing Claude 3 Opus model names."""
        assert client._normalize_model_for_pricing("claude-3-opus-20240229") == "claude-3-opus"

    def test_normalize_claude_3_sonnet(self, client) -> None:
        """Test normalizing Claude 3 Sonnet model names."""
        assert client._normalize_model_for_pricing("claude-3-sonnet-20240229") == "claude-3-sonnet"

    def test_normalize_claude_3_haiku(self, client) -> None:
        """Test normalizing Claude 3 Haiku model names."""
        assert client._normalize_model_for_pricing("claude-3-haiku-20240307") == "claude-3-haiku"

    def test_normalize_unknown_model(self, client) -> None:
        """Test normalizing unknown model names."""
        assert client._normalize_model_for_pricing("claude-4-opus") == "claude-4-opus"


class TestAnthropicClientTracking:
    """Test tracking data in Anthropic client."""

    @pytest.fixture
    def client(self, mock_anthropic) -> Generator:
        """Create test client."""
        from seeds_clients.providers.anthropic import AnthropicClient

        cache_dir = tempfile.mkdtemp(prefix="test_cache_")
        client = AnthropicClient(api_key="test-key", cache_dir=cache_dir)

        yield client

        if client.cache:
            client.cache.close()
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.fixture
    def mock_response_with_usage(self, mock_anthropic) -> MagicMock:
        """Create mock response with usage data."""
        mock_response = MagicMock()
        mock_response.id = "msg_track123"
        mock_response.type = "message"
        mock_response.role = "assistant"
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "end_turn"
        mock_response.stop_sequence = None

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response with tracking"
        mock_response.content = [mock_text_block]

        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_response.usage = mock_usage

        return mock_response

    def test_tracking_data_populated(self, client, mock_response_with_usage) -> None:
        """Test that tracking data is populated."""
        client._client.messages.create.return_value = mock_response_with_usage

        messages = [Message(role="user", content="Hello")]
        response = client.generate(messages)

        assert response.tracking is not None
        assert response.tracking.provider == "anthropic"
        assert response.tracking.prompt_tokens == 100
        assert response.tracking.completion_tokens == 50

    def test_cost_calculation(self, client, mock_response_with_usage) -> None:
        """Test that cost is calculated."""
        client._client.messages.create.return_value = mock_response_with_usage

        messages = [Message(role="user", content="Hello")]
        response = client.generate(messages)

        # Cost should be calculated based on pricing data
        assert response.tracking is not None
        # Cost may be 0 if model not in pricing data, but should not raise
        assert response.tracking.cost_usd >= 0


class TestAnthropicClientCleanup:
    """Test client cleanup methods."""

    def test_close(self, mock_anthropic) -> None:
        """Test close method."""
        from seeds_clients.providers.anthropic import AnthropicClient

        client = AnthropicClient(api_key="test-key")
        client.close()

        # Verify close was called on the internal client
        client._client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_aclose(self, mock_anthropic) -> None:
        """Test async close method."""
        from seeds_clients.providers.anthropic import AnthropicClient

        client = AnthropicClient(api_key="test-key")
        client._async_client.close = AsyncMock()
        await client.aclose()

        # Verify close was called on the async client
        client._async_client.close.assert_called_once()


class TestAnthropicProviderNames:
    """Test provider name methods."""

    def test_provider_name(self, mock_anthropic) -> None:
        """Test _get_provider_name returns 'anthropic'."""
        from seeds_clients.providers.anthropic import AnthropicClient

        client = AnthropicClient(api_key="test-key")
        assert client._get_provider_name() == "anthropic"

    def test_ecologits_provider(self, mock_anthropic) -> None:
        """Test _get_ecologits_provider returns 'anthropic'."""
        from seeds_clients.providers.anthropic import AnthropicClient

        client = AnthropicClient(api_key="test-key")
        assert client._get_ecologits_provider() == "anthropic"


class TestAnthropicPydanticSchemaConversion:
    """Test Pydantic to JSON schema conversion."""

    @pytest.fixture
    def client(self, mock_anthropic):
        """Create test client."""
        from seeds_clients.providers.anthropic import AnthropicClient

        return AnthropicClient(api_key="test-key")

    def test_simple_schema_conversion(self, client) -> None:
        """Test converting simple Pydantic model to JSON schema."""

        class SimpleModel(BaseModel):
            name: str
            count: int

        schema = client._pydantic_to_json_schema(SimpleModel)

        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]

    def test_nested_schema_conversion(self, client) -> None:
        """Test converting nested Pydantic model to JSON schema."""

        class Inner(BaseModel):
            value: str

        class Outer(BaseModel):
            inner: Inner
            label: str

        schema = client._pydantic_to_json_schema(Outer)

        assert "properties" in schema
        assert "inner" in schema["properties"]
        assert "label" in schema["properties"]
        # $defs should be resolved
        assert "$defs" not in schema

    def test_schema_with_descriptions(self, client) -> None:
        """Test schema conversion preserves Field descriptions."""

        class DescribedModel(BaseModel):
            name: str = Field(..., description="The person's name")
            age: int = Field(..., description="The person's age in years")

        schema = client._pydantic_to_json_schema(DescribedModel)

        assert "properties" in schema
        name_prop = schema["properties"]["name"]
        assert name_prop.get("description") == "The person's name"
