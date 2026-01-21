"""Tests for OpenAI client."""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from pydantic import BaseModel, Field

from seeds_clients import Message, OpenAIClient
from seeds_clients.core.exceptions import ConfigurationError, ProviderError, ValidationError


class TestOpenAIClientInit:
    """Test OpenAI client initialization."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        client = OpenAIClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.model == "gpt-4.1"
        assert client.base_url == "https://api.openai.com/v1"

    def test_init_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization from environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        client = OpenAIClient()
        assert client.api_key == "env-key"

    def test_init_without_key_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization without API key raises error."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigurationError) as exc_info:
            OpenAIClient()
        assert "API key required" in str(exc_info.value)

    def test_init_with_custom_model(self) -> None:
        """Test initialization with custom model."""
        client = OpenAIClient(api_key="test-key", model="gpt-4.1-mini")
        assert client.model == "gpt-4.1-mini"

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        client = OpenAIClient(
            api_key="test-key",
            model="gpt-3.5-turbo",
            base_url="https://custom.openai.com/v1",
            max_tokens=1000,
            temperature=0.5,
            ttl_hours=48.0,
        )
        assert client.model == "gpt-3.5-turbo"
        assert client.base_url == "https://custom.openai.com/v1"
        assert client.max_tokens == 1000
        assert client.temperature == 0.5


class TestOpenAIClientGenerate:
    """Test OpenAI client generate method."""

    @pytest.fixture
    def client(self) -> Generator[OpenAIClient, None, None]:
        """Create test client."""
        import shutil
        import tempfile

        # Use a unique temp directory for each test
        cache_dir = tempfile.mkdtemp(prefix="test_cache_")
        client = OpenAIClient(api_key="test-key", cache_dir=cache_dir)

        yield client

        # Cleanup
        client.close()
        if client.cache:
            client.cache.close()
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.fixture
    def mock_response(self) -> dict:
        """Create mock API response."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello! How can I help?"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    def test_generate_text_message(self, client: OpenAIClient, mock_response: dict) -> None:
        """Test generating response from text message."""
        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=mock_response),
                raise_for_status=Mock(),
            )

            messages = [Message(role="user", content="Hello")]
            response = client.generate(messages)

            assert response.content == "Hello! How can I help?"
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 5
            assert response.usage.total_tokens == 15
            assert response.model == "gpt-4.1"
            assert response.finish_reason == "stop"
            assert response.response_id == "chatcmpl-123"
            assert not response.cached

    def test_generate_with_system_message(self, client: OpenAIClient, mock_response: dict) -> None:
        """Test generating with system message."""
        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=mock_response),
                raise_for_status=Mock(),
            )

            messages = [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello"),
            ]
            client.generate(messages)

            # Check the API was called with correct format
            call_args = mock_post.call_args
            payload = call_args.kwargs["json"]
            assert len(payload["messages"]) == 2
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][1]["role"] == "user"

    def test_generate_with_kwargs(self, client: OpenAIClient, mock_response: dict) -> None:
        """Test generating with additional kwargs."""
        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=mock_response),
                raise_for_status=Mock(),
            )

            messages = [Message(role="user", content="Hello")]
            client.generate(
                messages,
                temperature=0.7,
                max_tokens=500,
            )

            # Check parameters were passed to API
            call_args = mock_post.call_args
            payload = call_args.kwargs["json"]
            assert payload["temperature"] == 0.7
            assert payload["max_tokens"] == 500

    def test_generate_caching(self, client: OpenAIClient, mock_response: dict) -> None:
        """Test response caching."""
        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=mock_response),
                raise_for_status=Mock(),
            )

            messages = [Message(role="user", content="Hello")]

            # First call - should hit API
            response1 = client.generate(messages, use_cache=True)
            assert mock_post.call_count == 1
            assert not response1.cached

            # Second call - should use cache
            response2 = client.generate(messages, use_cache=True)
            assert mock_post.call_count == 1  # No additional API call
            assert response2.cached
            assert response2.content == response1.content

            # Clear cache for cleanup
        client.close()
        if client.cache:
                client.cache.clear()

    def test_generate_multimodal_message(self, client: OpenAIClient, mock_response: dict) -> None:
        """Test generating with multimodal (text + image) message."""
        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=mock_response),
                raise_for_status=Mock(),
            )

            messages = [
                Message(
                    role="user",
                    content=[
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image", "source": "https://example.com/image.jpg"},
                    ],
                )
            ]
            client.generate(messages)

            # Check multimodal content was formatted correctly
            call_args = mock_post.call_args
            payload = call_args.kwargs["json"]
            content = payload["messages"][0]["content"]
            assert isinstance(content, list)
            assert len(content) == 2
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image_url"


class TestOpenAIClientErrors:
    """Test OpenAI client error handling."""

    @pytest.fixture
    def client(self) -> Generator[OpenAIClient, None, None]:
        """Create test client."""
        client = OpenAIClient(api_key="test-key")
        yield client
        client.close()
    def test_api_error_with_json_response(self, client: OpenAIClient) -> None:
        """Test handling API error with JSON error message."""
        import httpx

        error_response = Mock(
            status_code=429,
            json=Mock(return_value={"error": {"message": "Rate limit exceeded"}}),
        )

        with patch.object(client._http_client, "post") as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "Error",
                request=Mock(),
                response=error_response,
            )

            messages = [Message(role="user", content="Hello")]

            with pytest.raises(ProviderError) as exc_info:
                client.generate(messages)

            assert exc_info.value.provider == "openai"
            assert exc_info.value.status_code == 429
            assert "Rate limit exceeded" in str(exc_info.value)

    def test_api_error_with_text_response(self, client: OpenAIClient) -> None:
        """Test handling API error with plain text response."""
        import httpx

        error_response = Mock(
            status_code=500,
            json=Mock(side_effect=Exception("Not JSON")),
            text="Internal server error",
        )

        with patch.object(client._http_client, "post") as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "Error",
                request=Mock(),
                response=error_response,
            )

            messages = [Message(role="user", content="Hello")]

            with pytest.raises(ProviderError) as exc_info:
                client.generate(messages)

            assert exc_info.value.provider == "openai"
            assert exc_info.value.status_code == 500
            assert "Internal server error" in str(exc_info.value)

    def test_request_error(self, client: OpenAIClient) -> None:
        """Test handling request errors (network, timeout, etc.)."""
        import httpx

        with patch.object(client._http_client, "post") as mock_post:
            mock_post.side_effect = httpx.RequestError("Connection failed")

            messages = [Message(role="user", content="Hello")]

            with pytest.raises(ProviderError) as exc_info:
                client.generate(messages)

            assert exc_info.value.provider == "openai"
            assert exc_info.value.status_code is None
            assert "Request failed" in str(exc_info.value)

    def test_invalid_response_format(self, client: OpenAIClient) -> None:
        """Test handling invalid response format."""
        invalid_response: dict[str, list] = {"choices": []}  # Missing required fields

        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=invalid_response),
                raise_for_status=Mock(),
            )

            messages = [Message(role="user", content="Hello")]

            with pytest.raises(ProviderError) as exc_info:
                client.generate(messages)

            assert exc_info.value.provider == "openai"
            assert "Invalid response format" in str(exc_info.value)


class TestOpenAIClientImageFormatting:
    """Test image formatting for multimodal messages."""

    @pytest.fixture
    def client(self) -> Generator[OpenAIClient, None, None]:
        """Create test client."""
        client = OpenAIClient(api_key="test-key")
        yield client
        client.close()
    def test_format_image_url(self, client: OpenAIClient) -> None:
        """Test formatting image URL."""
        url = "https://example.com/image.jpg"
        formatted = client._format_image(url)
        assert formatted == url

    def test_format_image_file_path(self, client: OpenAIClient, tmp_path: Path) -> None:
        """Test formatting image from file path."""
        # Create a test image
        img = Image.new("RGB", (10, 10), color="red")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        formatted = client._format_image(str(img_path))
        assert formatted.startswith("data:image/png;base64,")

    def test_format_pil_image(self, client: OpenAIClient) -> None:
        """Test formatting PIL Image."""
        img = Image.new("RGB", (10, 10), color="blue")
        formatted = client._format_image(img)
        assert formatted.startswith("data:image/png;base64,")

    def test_format_image_bytes(self, client: OpenAIClient) -> None:
        """Test formatting image bytes."""
        img = Image.new("RGB", (10, 10), color="green")
        import io

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        formatted = client._format_image(img_bytes)
        assert formatted.startswith("data:image/png;base64,")

    def test_format_data_url(self, client: OpenAIClient) -> None:
        """Test passing through existing data URL."""
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"
        formatted = client._format_image(data_url)
        assert formatted == data_url


class TestOpenAIStructuredOutputs:
    """Test OpenAI structured outputs with Pydantic models."""

    @pytest.fixture
    def client(self) -> Generator[OpenAIClient, None, None]:
        """Create test client."""
        import shutil
        import tempfile

        cache_dir = tempfile.mkdtemp(prefix="test_cache_")
        client = OpenAIClient(api_key="test-key", cache_dir=cache_dir)

        yield client

        client.close()
        if client.cache:
            client.cache.close()
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_structured_output_simple_model(self, client: OpenAIClient) -> None:
        """Test structured output with a simple Pydantic model."""

        # Define test model
        class Person(BaseModel):
            name: str
            age: int

        # Mock response with JSON content
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"name": "John Doe", "age": 30}),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=mock_response),
                raise_for_status=Mock(),
            )

            messages = [Message(role="user", content="Extract person info")]
            response = client.generate(messages, response_format=Person)

            # Check structured output was configured
            call_args = mock_post.call_args
            payload = call_args.kwargs["json"]
            assert "response_format" in payload
            assert payload["response_format"]["type"] == "json_schema"
            assert payload["response_format"]["json_schema"]["name"] == "Person"
            assert "schema" in payload["response_format"]["json_schema"]

            # Check parsed output
            assert response.parsed is not None
            assert isinstance(response.parsed, Person)
            assert response.parsed.name == "John Doe"
            assert response.parsed.age == 30
            assert response.content == json.dumps({"name": "John Doe", "age": 30})

    def test_structured_output_nested_model(self, client: OpenAIClient) -> None:
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

        mock_response = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(mock_data),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 10,
                "total_tokens": 25,
            },
        }

        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=mock_response),
                raise_for_status=Mock(),
            )

            messages = [Message(role="user", content="Extract person with address")]
            response = client.generate(messages, response_format=PersonWithAddress)

            assert response.parsed is not None
            assert isinstance(response.parsed, PersonWithAddress)
            assert response.parsed.name == "Jane Smith"
            assert response.parsed.age == 25
            assert isinstance(response.parsed.address, Address)
            assert response.parsed.address.city == "Boston"

    def test_structured_output_with_field_descriptions(self, client: OpenAIClient) -> None:
        """Test structured output with Field descriptions."""

        class Product(BaseModel):
            name: str = Field(description="Product name")
            price: float = Field(description="Price in USD", ge=0)
            in_stock: bool = Field(description="Whether product is in stock")

        mock_data = {"name": "Widget", "price": 29.99, "in_stock": True}

        mock_response = {
            "id": "chatcmpl-789",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(mock_data),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        }

        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=mock_response),
                raise_for_status=Mock(),
            )

            messages = [Message(role="user", content="Extract product info")]
            response = client.generate(messages, response_format=Product)

            # Check schema includes field descriptions
            call_args = mock_post.call_args
            schema = call_args.kwargs["json"]["response_format"]["json_schema"]["schema"]
            assert "properties" in schema

            # Verify parsed output
            assert response.parsed is not None
            assert response.parsed.name == "Widget"
            assert response.parsed.price == 29.99
            assert response.parsed.in_stock is True

    def test_structured_output_enforces_no_additional_properties(
        self, client: OpenAIClient
    ) -> None:
        """Ensure OpenAI strict schema sets additionalProperties=false everywhere."""

        class ToolCall(BaseModel):
            name: str
            args: dict[str, Any]

        class StepA(BaseModel):
            kind: str
            tool: ToolCall

        class StepB(BaseModel):
            kind: str
            value: str

        class Payload(BaseModel):
            step: StepA | StepB

        schema = client._pydantic_to_json_schema(Payload)

        def assert_no_additional_properties(node: Any) -> None:
            if isinstance(node, dict):
                is_object = node.get("type") == "object" or "properties" in node
                if is_object:
                    assert node.get("additionalProperties") is False
                for value in node.values():
                    assert_no_additional_properties(value)
            elif isinstance(node, list):
                for item in node:
                    assert_no_additional_properties(item)

        assert_no_additional_properties(schema)

    def test_structured_output_invalid_json(self, client: OpenAIClient) -> None:
        """Test error handling for invalid JSON in structured output."""

        class Person(BaseModel):
            name: str
            age: int

        # Mock response with invalid JSON
        mock_response = {
            "id": "chatcmpl-999",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is not JSON",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=mock_response),
                raise_for_status=Mock(),
            )

            messages = [Message(role="user", content="Extract person")]

            with pytest.raises(ValidationError) as exc_info:
                client.generate(messages, response_format=Person)

            assert "Failed to parse structured output" in str(exc_info.value)
            assert exc_info.value.raw_response == mock_response

    def test_structured_output_validation_error(self, client: OpenAIClient) -> None:
        """Test error handling for Pydantic validation errors."""

        class Person(BaseModel):
            name: str
            age: int  # Should be int but API returns string

        # Mock response with wrong type
        mock_response = {
            "id": "chatcmpl-888",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"name": "John", "age": "thirty"}),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=mock_response),
                raise_for_status=Mock(),
            )

            messages = [Message(role="user", content="Extract person")]

            with pytest.raises(ValidationError) as exc_info:
                client.generate(messages, response_format=Person)

            assert "Failed to parse structured output" in str(exc_info.value)
            assert exc_info.value.raw_response == mock_response

    def test_structured_output_caching(self, client: OpenAIClient) -> None:
        """Test that structured outputs work with caching."""

        class Person(BaseModel):
            name: str
            age: int

        mock_response = {
            "id": "chatcmpl-cache",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"name": "Cached Person", "age": 40}),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=mock_response),
                raise_for_status=Mock(),
            )

            messages = [Message(role="user", content="Extract cached person")]

            # First call
            response1 = client.generate(messages, response_format=Person, use_cache=True)
            assert mock_post.call_count == 1
            assert response1.parsed is not None
            assert response1.parsed.name == "Cached Person"
            assert not response1.cached

            # Second call - should use cache
            response2 = client.generate(messages, response_format=Person, use_cache=True)
            assert mock_post.call_count == 1  # No additional API call
            assert response2.parsed is not None
            assert response2.parsed.name == "Cached Person"
            assert response2.cached

    def test_json_schema_generation(self, client: OpenAIClient) -> None:
        """Test JSON schema generation from Pydantic models."""

        class TestModel(BaseModel):
            name: str
            count: int
            active: bool

        schema = client._pydantic_to_json_schema(TestModel)

        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]
        assert "active" in schema["properties"]
        assert "required" in schema
        assert set(schema["required"]) == {"name", "count", "active"}

    def test_json_schema_has_additional_properties_false(self, client: OpenAIClient) -> None:
        """Test that JSON schema includes additionalProperties: false for OpenAI compatibility."""

        class SimpleModel(BaseModel):
            name: str
            value: int

        schema = client._pydantic_to_json_schema(SimpleModel)

        # Top-level object should have additionalProperties: false
        assert schema.get("additionalProperties") is False

    def test_json_schema_has_all_properties_required(self, client: OpenAIClient) -> None:
        """Test that JSON schema includes ALL properties in required array."""

        class ModelWithOptional(BaseModel):
            name: str
            age: int
            nickname: str | None = None  # Optional field

        schema = client._pydantic_to_json_schema(ModelWithOptional)

        # OpenAI strict mode requires ALL properties in required, even optional ones
        assert "required" in schema
        assert set(schema["required"]) == {"name", "age", "nickname"}

    def test_json_schema_nested_objects_have_additional_properties_false(
        self, client: OpenAIClient
    ) -> None:
        """Test that nested objects also have additionalProperties: false."""

        class Address(BaseModel):
            street: str
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        schema = client._pydantic_to_json_schema(Person)

        # Top-level should have additionalProperties: false
        assert schema.get("additionalProperties") is False

        # Nested Address object should also have additionalProperties: false
        address_schema = schema["properties"]["address"]
        assert address_schema.get("additionalProperties") is False

    def test_json_schema_nested_objects_have_all_properties_required(
        self, client: OpenAIClient
    ) -> None:
        """Test that nested objects have ALL properties in required."""

        class LineItem(BaseModel):
            description: str
            quantity: int = 1  # Has default, but still should be required
            price: float

        class Invoice(BaseModel):
            items: list[LineItem]
            total: float

        schema = client._pydantic_to_json_schema(Invoice)

        # Check nested LineItem schema (inside array)
        items_schema = schema["properties"]["items"]
        line_item_schema = items_schema.get("items", {})

        # ALL properties must be in required, even those with defaults
        assert "required" in line_item_schema
        assert set(line_item_schema["required"]) == {"description", "quantity", "price"}

    def test_json_schema_deeply_nested_has_additional_properties_false(
        self, client: OpenAIClient
    ) -> None:
        """Test additionalProperties: false on deeply nested structures."""

        class Item(BaseModel):
            name: str
            price: float

        class Order(BaseModel):
            items: list[Item]
            total: float

        class Invoice(BaseModel):
            order: Order
            customer_name: str

        schema = client._pydantic_to_json_schema(Invoice)

        # Top-level
        assert schema.get("additionalProperties") is False

        # Order
        order_schema = schema["properties"]["order"]
        assert order_schema.get("additionalProperties") is False

        # Item (inside array)
        items_schema = order_schema["properties"]["items"]
        item_schema = items_schema.get("items", {})
        assert item_schema.get("additionalProperties") is False


class TestOpenAICostTracking:
    """Tests for cost tracking in OpenAI client."""

    @pytest.fixture
    def client(self) -> Generator[OpenAIClient, None, None]:
        """Create test client."""
        import shutil
        import tempfile

        cache_dir = tempfile.mkdtemp(prefix="test_cache_")
        client = OpenAIClient(api_key="test-key", cache_dir=cache_dir)

        yield client

        client.close()
        if client.cache:
            client.cache.close()
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_cost_tracking_enabled(self, client: OpenAIClient) -> None:
        """Test that cost tracking is automatically enabled."""
        mock_response = {
            "id": "chatcmpl-cost",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
            },
        }

        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=mock_response),
                raise_for_status=Mock(),
            )

            messages = [Message(role="user", content="Test")]
            response = client.generate(messages)

            # Check tracking data is present
            assert response.tracking is not None
            assert response.tracking.cost_usd > 0
            assert response.tracking.prompt_tokens == 1000
            assert response.tracking.completion_tokens == 500
            assert response.tracking.provider == "openai"
            assert response.tracking.model == "gpt-4.1"

            # Check cost calculation
            # GPT-4.1: $2.00 per 1M input, $8.00 per 1M output
            # (1000/1M * 2.00) + (500/1M * 8.00) = 0.002 + 0.004 = 0.006
            assert response.tracking.cost_usd == pytest.approx(0.006)

    def test_cost_tracking_different_models(self, client: OpenAIClient) -> None:
        """Test cost tracking with different models."""
        test_cases: list[dict[str, Any]] = [
            {
                "model": "gpt-4.1-mini",
                "prompt_tokens": 10000,
                "completion_tokens": 5000,
                # gpt-4.1-mini: $0.40 per 1M input, $1.60 per 1M output
                "expected_cost": 0.012,  # (10000/1M * 0.40) + (5000/1M * 1.60)
            },
            {
                "model": "gpt-3.5-turbo",
                "prompt_tokens": 100000,
                "completion_tokens": 50000,
                "expected_cost": 0.125,  # (100000/1M * 0.50) + (50000/1M * 1.50)
            },
        ]

        for test_case in test_cases:
            prompt_tokens = int(test_case["prompt_tokens"])
            completion_tokens = int(test_case["completion_tokens"])

            with patch.object(client._http_client, "post") as mock_post:
                mock_response = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": test_case["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Test"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }

                mock_post.return_value = Mock(
                    json=Mock(return_value=mock_response),
                    raise_for_status=Mock(),
                )

                messages = [Message(role="user", content="Test")]
                response = client.generate(messages, use_cache=False)

                assert response.tracking is not None
                assert response.tracking.cost_usd == pytest.approx(test_case["expected_cost"])

    def test_cost_tracking_unknown_model(self, client: OpenAIClient) -> None:
        """Test cost tracking with unknown model falls back to zero cost."""
        mock_response = {
            "id": "chatcmpl-unknown",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "unknown-model-xyz",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
            },
        }

        with patch.object(client._http_client, "post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value=mock_response),
                raise_for_status=Mock(),
            )

            messages = [Message(role="user", content="Test")]
            response = client.generate(messages)

            # Should still have tracking data but with zero cost
            assert response.tracking is not None
            assert response.tracking.cost_usd == 0.0
            assert response.tracking.prompt_tokens == 1000
            assert response.tracking.completion_tokens == 500
