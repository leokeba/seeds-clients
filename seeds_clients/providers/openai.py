"""OpenAI client implementation."""

import json
import os
from contextlib import suppress
from typing import Any

import httpx
from PIL import Image
from pydantic import BaseModel

from seeds_clients.core.base_client import BaseClient
from seeds_clients.core.exceptions import ConfigurationError, ProviderError, ValidationError
from seeds_clients.core.types import Message, Response, TrackingData, Usage
from seeds_clients.providers.pricing import calculate_cost


class OpenAIClient(BaseClient):
    """
    OpenAI client implementation.

    Supports:
    - Chat completions (text and multimodal)
    - Structured outputs with response_format
    - Image inputs (URLs, file paths, PIL Images, base64)
    - Cost tracking and carbon impact measurement

    Example:
        ```python
        from seeds_clients import OpenAIClient, Message

        client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            cache_dir="cache"
        )

        response = client.generate(
            messages=[
                Message(role="user", content="What is 2+2?")
            ]
        )
        print(response.content)
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        cache_dir: str = "cache",
        ttl_hours: float | None = 24.0,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo").
            base_url: API base URL.
            cache_dir: Directory for caching responses.
            ttl_hours: Cache TTL in hours. None for no expiration.
            max_tokens: Maximum completion tokens.
            temperature: Sampling temperature (0-2).
            **kwargs: Additional arguments passed to BaseClient.

        Raises:
            ConfigurationError: If API key is not provided or found in environment.
        """
        # Get API key from parameter or environment
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ConfigurationError(
                "OpenAI API key required. Provide via api_key parameter or "
                "OPENAI_API_KEY environment variable."
            )

        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialize base client (will store api_key)
        super().__init__(
            model=model,
            api_key=resolved_api_key,
            cache_dir=cache_dir,
            ttl_hours=ttl_hours,
            **kwargs
        )

        # HTTP client for API calls
        self._http_client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def _get_provider_name(self) -> str:
        """Return provider name for tracking."""
        return "openai"

    def _setup_tracking(self) -> None:
        """Setup tracking (placeholder for Phase 3)."""
        pass

    def generate(
        self,
        messages: list[Message],
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Response[Any]:
        """
        Generate a response from the LLM.

        Supports structured outputs via the response_format kwarg.

        Args:
            messages: List of messages in the conversation.
            use_cache: Whether to use cache for this request.
            **kwargs: Additional API parameters. Special kwargs:
                - response_format: Pydantic model class for structured output
                - temperature: Sampling temperature
                - max_tokens: Maximum completion tokens

        Returns:
            Response object with content, usage, and optionally parsed structured data.

        Example:
            ```python
            from pydantic import BaseModel

            class Person(BaseModel):
                name: str
                age: int

            response = client.generate(
                messages=[Message(role="user", content="Extract: John is 30")],
                response_format=Person
            )
            print(response.parsed.name)  # "John"
            print(response.parsed.age)   # 30
            ```
        """
        # Extract response_format if provided
        response_format = kwargs.pop("response_format", None)

        # Add structured output configuration if provided
        if response_format is not None and isinstance(response_format, type) and issubclass(response_format, BaseModel):
            # Convert Pydantic model to OpenAI's JSON schema format
            schema = self._pydantic_to_json_schema(response_format)
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": schema,
                    "strict": True,
                },
            }

        # Call parent generate method
        response = super().generate(messages, use_cache=use_cache, **kwargs)

        # Parse structured output if response_format was provided
        if response_format is not None and response.content:
            try:
                parsed_data = json.loads(response.content)
                parsed_model = response_format(**parsed_data)
                # Create new response with parsed data
                response = Response(
                    content=response.content,
                    usage=response.usage,
                    model=response.model,
                    raw=response.raw,
                    tracking=response.tracking,
                    cached=response.cached,
                    finish_reason=response.finish_reason,
                    response_id=response.response_id,
                    parsed=parsed_model,
                )
            except (json.JSONDecodeError, ValueError) as e:
                raise ValidationError(
                    f"Failed to parse structured output: {str(e)}"
                ) from e

        return response

    def _pydantic_to_json_schema(self, model: type[BaseModel]) -> dict[str, Any]:
        """
        Convert Pydantic model to JSON schema for OpenAI API.

        Args:
            model: Pydantic model class.

        Returns:
            JSON schema dict with resolved references.
        """
        # Get Pydantic v2 JSON schema
        schema = model.model_json_schema()

        # Resolve $ref references by inlining $defs
        if "$defs" in schema:
            defs = schema.pop("$defs")
            schema = self._resolve_refs(schema, defs)

        return schema

    def _resolve_refs(self, schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively resolve $ref references in a schema.

        Args:
            schema: Schema dict that may contain $ref references.
            defs: Definitions dict to resolve references from.

        Returns:
            Schema with all references resolved.
        """
        if isinstance(schema, dict):
            if "$ref" in schema:
                # Extract definition name from #/$defs/Name format
                ref_path = schema["$ref"].split("/")
                if len(ref_path) == 3 and ref_path[0] == "#" and ref_path[1] == "$defs":
                    def_name = ref_path[2]
                    if def_name in defs:
                        # Replace reference with actual definition
                        resolved = defs[def_name].copy()
                        # Recursively resolve any nested refs
                        return self._resolve_refs(resolved, defs)
                return schema
            else:
                # Recursively process all values
                return {k: self._resolve_refs(v, defs) for k, v in schema.items()}
        elif isinstance(schema, list):
            return [self._resolve_refs(item, defs) for item in schema]
        return schema

    def __del__(self) -> None:
        """Clean up HTTP client."""
        if hasattr(self, "_http_client"):
            self._http_client.close()

    def _call_api(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Call OpenAI API.

        Args:
            messages: List of messages.
            **kwargs: Additional API parameters (max_tokens, temperature, etc.).

        Returns:
            Raw API response as dict.

        Raises:
            ProviderError: If API call fails.
        """
        # Build request payload
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
        }

        # Add optional parameters
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        # Override with kwargs
        payload.update(kwargs)

        try:
            response = self._http_client.post(
                "/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", "")
            except Exception:
                error_detail = e.response.text

            raise ProviderError(
                f"OpenAI API error: {error_detail}",
                provider="openai",
                status_code=e.response.status_code,
            ) from e

        except httpx.RequestError as e:
            raise ProviderError(
                f"Request failed: {str(e)}",
                provider="openai",
            ) from e

    def _parse_response(self, raw: dict[str, Any]) -> Response:
        """
        Parse OpenAI API response into Response object.

        Args:
            raw: Raw API response dict.

        Returns:
            Parsed Response object with cost tracking.

        Raises:
            ProviderError: If response format is invalid.
        """
        try:
            # Extract content from first choice
            choice = raw["choices"][0]
            message = choice["message"]
            content = message.get("content", "")

            # Extract usage
            usage_data = raw.get("usage", {})
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

            # Calculate cost
            model_name = raw.get("model", self.model)
            cost_usd = 0.0
            with suppress(ValueError):
                # Model not found in pricing data - cost remains 0
                cost_usd = calculate_cost(
                    model=model_name,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )

            # Create tracking data with cost
            # Note: energy_kwh and gwp_kgco2eq are set to 0 until EcoLogits integration
            tracking = TrackingData(
                energy_kwh=0.0,
                gwp_kgco2eq=0.0,
                cost_usd=cost_usd,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                provider="openai",
                model=model_name,
                tracking_method="none",
                duration_seconds=0.0,  # Will be set by base client if timing is enabled
            )

            # Extract optional fields
            finish_reason = choice.get("finish_reason")
            response_id = raw.get("id")

            return Response(
                content=content,
                usage=usage,
                model=model_name,
                raw=raw,
                tracking=tracking,
                finish_reason=finish_reason,
                response_id=response_id,
            )

        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(
                f"Invalid response format: {str(e)}",
                provider="openai",
            ) from e

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Format messages for OpenAI API.

        Args:
            messages: List of Message objects.

        Returns:
            List of message dicts in OpenAI format.
        """
        formatted = []

        for msg in messages:
            message_dict: dict[str, Any] = {"role": msg.role}

            # Handle text content
            if isinstance(msg.content, str):
                message_dict["content"] = msg.content

            # Handle multimodal content
            elif isinstance(msg.content, list):
                content_parts = []
                for item in msg.content:
                    item_type = item.get("type")
                    if item_type == "text":
                        content_parts.append({
                            "type": "text",
                            "text": item.get("text", ""),
                        })
                    elif item_type == "image":
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": self._format_image(item.get("source", "")),
                            },
                        })
                message_dict["content"] = content_parts

            formatted.append(message_dict)

        return formatted

    def _format_image(self, image: str | Image.Image | bytes) -> str:
        """
        Format image for OpenAI API.

        Args:
            image: Image as URL, file path, PIL Image, or bytes.

        Returns:
            Image URL or data URL.
        """
        import base64
        import io
        from pathlib import Path

        # Already a URL
        if isinstance(image, str) and (
            image.startswith("http://") or image.startswith("https://")
        ):
            return image

        # File path
        if isinstance(image, str):
            image_path = Path(image)
            if image_path.exists():
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
            else:
                # Assume it's a data URL
                return image

        # PIL Image
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

        # Raw bytes
        else:
            image_bytes = image

        # Convert to base64 data URL
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64_data}"
