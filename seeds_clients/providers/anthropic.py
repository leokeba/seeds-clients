"""Anthropic Claude client implementation using the anthropic SDK."""

import base64
import io
import json
import mimetypes
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

from seeds_clients.core.base_client import BaseClient
from seeds_clients.core.exceptions import ConfigurationError, ProviderError
from seeds_clients.core.types import Message, Response, TrackingData, Usage
from seeds_clients.tracking.ecologits_tracker import EcoLogitsMixin
from seeds_clients.utils.pricing import calculate_cost


class AnthropicClient(EcoLogitsMixin, BaseClient):
    """
    Anthropic Claude client implementation with carbon tracking.

    Uses the official anthropic SDK for the Claude API.

    Supports:
    - Chat completions (text and multimodal)
    - Structured outputs with Pydantic models via tool_use
    - Image inputs (URLs, file paths, PIL Images, base64)
    - Cost tracking and carbon impact measurement via EcoLogits
    - System prompts
    - Configurable generation parameters (temperature, max_tokens, etc.)

    Example:
        ```python
        from seeds_clients import AnthropicClient, Message

        client = AnthropicClient(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-sonnet-4-20250514",
            cache_dir="cache"
        )

        response = client.generate(
            messages=[
                Message(role="user", content="What is 2+2?")
            ]
        )
        print(response.content)
        ```

    Notes:
        - Claude models require max_tokens to be explicitly set
        - System prompts are passed separately from messages
        - Structured outputs use tool_use with a schema-based approach
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        cache_dir: str = "cache",
        ttl_hours: float | None = 24.0,
        max_tokens: int = 4096,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        system: str | None = None,
        stop_sequences: list[str] | None = None,
        electricity_mix_zone: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Anthropic Claude client.

        Args:
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
            model: Model name (e.g., "claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022",
                   "claude-3-opus-20240229").
            cache_dir: Directory for caching responses.
            ttl_hours: Cache TTL in hours. None for no expiration.
            max_tokens: Maximum output tokens. Required by Anthropic API.
            temperature: Sampling temperature (0-1). None for model default.
            top_p: Nucleus sampling probability. None for model default.
            top_k: Top-k sampling. None for model default.
            system: System prompt for the model.
            stop_sequences: List of stop sequences.
            electricity_mix_zone: ISO 3166-1 alpha-3 code for electricity mix zone
                                 (e.g., "FRA", "USA", "WOR"). Default is "WOR" (World).
            **kwargs: Additional arguments passed to BaseClient.

        Raises:
            ConfigurationError: If API key is not provided or found in environment.
        """
        # Import anthropic here to avoid import errors if not installed
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError as e:
            raise ConfigurationError(
                "anthropic package is required for AnthropicClient. "
                "Install it with: pip install anthropic"
            ) from e

        # Get API key from parameter or environment
        resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_api_key:
            raise ConfigurationError(
                "Anthropic API key required. Provide via api_key parameter or "
                "ANTHROPIC_API_KEY environment variable."
            )

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.system = system
        self.stop_sequences = stop_sequences

        # Initialize base client (will store api_key)
        super().__init__(
            model=model,
            api_key=resolved_api_key,
            cache_dir=cache_dir,
            ttl_hours=ttl_hours,
            electricity_mix_zone=electricity_mix_zone,
            **kwargs
        )

        # Set electricity mix zone for EcoLogits mixin
        self._set_electricity_mix_zone(electricity_mix_zone)

        # Create sync client
        self._client = anthropic.Anthropic(api_key=resolved_api_key)

        # Create async client
        self._async_client = anthropic.AsyncAnthropic(api_key=resolved_api_key)

    def _get_provider_name(self) -> str:
        """Return provider name for tracking."""
        return "anthropic"

    def _get_ecologits_provider(self) -> str:
        """Return provider name for EcoLogits tracking."""
        return "anthropic"

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
                - max_tokens: Maximum output tokens
                - system: Override system prompt for this request

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
        # Call parent generate method (handles caching and structured output parsing)
        return super().generate(messages, use_cache=use_cache, **kwargs)

    async def agenerate(
        self,
        messages: list[Message],
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Response[Any]:
        """
        Asynchronously generate a response from the LLM.

        Supports structured outputs via the response_format kwarg.

        Args:
            messages: List of messages in the conversation.
            use_cache: Whether to use cache for this request.
            **kwargs: Additional API parameters. Special kwargs:
                - response_format: Pydantic model class for structured output
                - temperature: Sampling temperature
                - max_tokens: Maximum output tokens

        Returns:
            Response object with content, usage, and optionally parsed structured data.

        Example:
            ```python
            import asyncio

            async def main():
                response = await client.agenerate(
                    messages=[Message(role="user", content="Hello!")]
                )
                print(response.content)

            asyncio.run(main())
            ```
        """
        # Call parent agenerate method (handles caching and structured output parsing)
        return await super().agenerate(messages, use_cache=use_cache, **kwargs)

    def _pydantic_to_json_schema(self, model: type[BaseModel]) -> dict[str, Any]:
        """
        Convert a Pydantic model to JSON schema for Anthropic tool_use.

        Args:
            model: Pydantic model class.

        Returns:
            JSON schema dict compatible with Anthropic's tool input_schema.
        """
        schema = model.model_json_schema()

        # Remove $defs if present and inline definitions
        if "$defs" in schema:
            defs = schema.pop("$defs")
            schema = self._resolve_refs(schema, defs)

        return schema

    def _resolve_refs(self, schema: Any, defs: dict[str, Any]) -> Any:
        """
        Resolve $ref references in JSON schema.

        Args:
            schema: Schema with potential $ref references.
            defs: Definitions to resolve references from.

        Returns:
            Schema with resolved references.
        """
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref_path = schema["$ref"].split("/")[-1]
                if ref_path in defs:
                    return self._resolve_refs(defs[ref_path], defs)
                return schema

            return {k: self._resolve_refs(v, defs) for k, v in schema.items()}
        elif isinstance(schema, list):
            return [self._resolve_refs(item, defs) for item in schema]
        return schema

    def _build_request_params(
        self,
        messages: list[Message],
        response_format: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Build request parameters for the API call.

        Args:
            messages: List of messages.
            response_format: Pydantic model for structured output.
            **kwargs: Additional parameters.

        Returns:
            Dict of parameters for the API call.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.pop("max_tokens", None) or self.max_tokens,
        }

        # System prompt (from init or override)
        system = kwargs.pop("system", None) or self.system
        if system:
            params["system"] = system

        # Temperature
        temperature = kwargs.pop("temperature", None)
        if temperature is None and self.temperature is not None:
            temperature = self.temperature
        if temperature is not None:
            params["temperature"] = temperature

        # Top-p
        top_p = kwargs.pop("top_p", None) or self.top_p
        if top_p is not None:
            params["top_p"] = top_p

        # Top-k
        top_k = kwargs.pop("top_k", None) or self.top_k
        if top_k is not None:
            params["top_k"] = top_k

        # Stop sequences
        stop_sequences = kwargs.pop("stop_sequences", None) or self.stop_sequences
        if stop_sequences:
            params["stop_sequences"] = stop_sequences

        # Format messages for Anthropic API
        params["messages"] = self._format_messages(messages)

        # Handle structured output via tool_use
        if response_format is not None and isinstance(response_format, type) and issubclass(response_format, BaseModel):
            tool_name = response_format.__name__
            tool_schema = self._pydantic_to_json_schema(response_format)

            params["tools"] = [{
                "name": tool_name,
                "description": f"Extract structured data as {tool_name}",
                "input_schema": tool_schema,
            }]
            params["tool_choice"] = {"type": "tool", "name": tool_name}

        # Pass through any remaining kwargs
        params.update(kwargs)

        return params

    def _call_api(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Call Anthropic Claude API.

        Args:
            messages: List of messages.
            **kwargs: Additional API parameters.

        Returns:
            Raw API response as dict.

        Raises:
            ProviderError: If API call fails.
        """
        # Extract response_format for structured output
        response_format = kwargs.pop("response_format", None)

        # Build request parameters
        params = self._build_request_params(messages, response_format=response_format, **kwargs)

        # Track start time for duration
        start_time = time.time()

        try:
            # Make the API call
            response = self._client.messages.create(**params)

            # Calculate request duration
            duration_seconds = time.time() - start_time

            # Convert response to dict for caching
            raw_response = self._response_to_dict(response, response_format)
            raw_response["_duration_seconds"] = duration_seconds

            # Calculate EcoLogits carbon impacts
            output_tokens = response.usage.output_tokens if response.usage else 0

            impacts = self._calculate_ecologits_impacts(
                model_name=self.model,
                output_tokens=output_tokens,
                request_latency=duration_seconds,
                electricity_mix_zone=self.electricity_mix_zone,
            )
            if impacts:
                raw_response["_ecologits_impacts"] = impacts

            return raw_response

        except self._anthropic.APIConnectionError as e:
            raise ProviderError(
                f"Anthropic API connection error: {str(e)}",
                provider="anthropic",
            ) from e
        except self._anthropic.RateLimitError as e:
            raise ProviderError(
                f"Anthropic API rate limit exceeded: {str(e)}",
                provider="anthropic",
                status_code=429,
            ) from e
        except self._anthropic.APIStatusError as e:
            raise ProviderError(
                f"Anthropic API error: {str(e)}",
                provider="anthropic",
                status_code=e.status_code,
            ) from e
        except Exception as e:
            raise ProviderError(
                f"Anthropic API error: {str(e)}",
                provider="anthropic",
            ) from e

    async def _acall_api(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Asynchronously call Anthropic Claude API.

        Args:
            messages: List of messages.
            **kwargs: Additional API parameters.

        Returns:
            Raw API response as dict.

        Raises:
            ProviderError: If API call fails.
        """
        # Extract response_format for structured output
        response_format = kwargs.pop("response_format", None)

        # Build request parameters
        params = self._build_request_params(messages, response_format=response_format, **kwargs)

        # Track start time for duration
        start_time = time.time()

        try:
            # Make the async API call
            response = await self._async_client.messages.create(**params)

            # Calculate request duration
            duration_seconds = time.time() - start_time

            # Convert response to dict for caching
            raw_response = self._response_to_dict(response, response_format)
            raw_response["_duration_seconds"] = duration_seconds

            # Calculate EcoLogits carbon impacts
            output_tokens = response.usage.output_tokens if response.usage else 0

            impacts = self._calculate_ecologits_impacts(
                model_name=self.model,
                output_tokens=output_tokens,
                request_latency=duration_seconds,
                electricity_mix_zone=self.electricity_mix_zone,
            )
            if impacts:
                raw_response["_ecologits_impacts"] = impacts

            return raw_response

        except self._anthropic.APIConnectionError as e:
            raise ProviderError(
                f"Anthropic API connection error: {str(e)}",
                provider="anthropic",
            ) from e
        except self._anthropic.RateLimitError as e:
            raise ProviderError(
                f"Anthropic API rate limit exceeded: {str(e)}",
                provider="anthropic",
                status_code=429,
            ) from e
        except self._anthropic.APIStatusError as e:
            raise ProviderError(
                f"Anthropic API error: {str(e)}",
                provider="anthropic",
                status_code=e.status_code,
            ) from e
        except Exception as e:
            raise ProviderError(
                f"Anthropic API error: {str(e)}",
                provider="anthropic",
            ) from e

    def _response_to_dict(
        self,
        response: Any,
        response_format: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """
        Convert anthropic response object to a dictionary for caching.

        Args:
            response: Message response from the API.
            response_format: Optional Pydantic model for structured output extraction.

        Returns:
            Dictionary representation of the response.
        """
        result: dict[str, Any] = {
            "id": response.id,
            "type": response.type,
            "role": response.role,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "stop_sequence": response.stop_sequence,
        }

        # Extract content blocks
        content_blocks = []
        text_content = ""
        tool_use_content = None

        if response.content:
            for block in response.content:
                block_dict: dict[str, Any] = {"type": block.type}

                if block.type == "text":
                    block_dict["text"] = block.text
                    text_content = block.text
                elif block.type == "tool_use":
                    block_dict["id"] = block.id
                    block_dict["name"] = block.name
                    block_dict["input"] = block.input
                    # If we have a response_format and this is the matching tool,
                    # extract the structured output
                    if response_format and block.name == response_format.__name__:
                        tool_use_content = block.input

                content_blocks.append(block_dict)

        result["content"] = content_blocks

        # If we have structured output from tool_use, store it as text for parsing
        if tool_use_content:
            result["text"] = json.dumps(tool_use_content)
        else:
            result["text"] = text_content

        # Extract usage
        if response.usage:
            result["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        return result

    def _parse_response(self, raw: dict[str, Any]) -> Response:
        """
        Parse API response into Response object.

        Args:
            raw: Raw API response dict.

        Returns:
            Parsed Response object with cost and carbon tracking.

        Raises:
            ProviderError: If response format is invalid.
        """
        try:
            # Extract content - prefer text field which may contain tool_use output
            content = raw.get("text", "")

            # If no direct text, try to extract from content blocks
            if not content and "content" in raw:
                for block in raw["content"]:
                    if block.get("type") == "text":
                        content = block.get("text", "")
                        break

            # Extract usage
            usage_data = raw.get("usage", {})
            input_tokens = usage_data.get("input_tokens", 0)
            output_tokens = usage_data.get("output_tokens", 0)
            usage = Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )

            # Calculate cost
            model_name = raw.get("model", self.model)
            pricing_model = self._normalize_model_for_pricing(model_name)
            cost_usd = 0.0
            with suppress(ValueError):
                cost_usd = calculate_cost(
                    model=pricing_model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    provider="anthropic",
                )

            # Extract EcoLogits carbon impact data
            ecologits_impacts = raw.get("_ecologits_impacts")
            duration_seconds = raw.get("_duration_seconds", 0.0)

            # Extract full metrics from EcoLogits
            metrics = self._extract_full_ecologits_metrics(ecologits_impacts)

            # Create tracking data with cost and carbon metrics
            tracking = TrackingData(
                # Total metrics
                energy_kwh=metrics.energy_kwh,
                gwp_kgco2eq=metrics.gwp_kgco2eq,
                adpe_kgsbeq=metrics.adpe_kgsbeq,
                pe_mj=metrics.pe_mj,
                # Cost
                cost_usd=cost_usd,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                # Metadata
                provider="anthropic",
                model=model_name,
                tracking_method=metrics.tracking_method,
                electricity_mix_zone=self.electricity_mix_zone,
                duration_seconds=duration_seconds,
                # Usage phase breakdown
                energy_usage_kwh=metrics.energy_usage_kwh,
                gwp_usage_kgco2eq=metrics.gwp_usage_kgco2eq,
                adpe_usage_kgsbeq=metrics.adpe_usage_kgsbeq,
                pe_usage_mj=metrics.pe_usage_mj,
                # Embodied phase breakdown
                gwp_embodied_kgco2eq=metrics.gwp_embodied_kgco2eq,
                adpe_embodied_kgsbeq=metrics.adpe_embodied_kgsbeq,
                pe_embodied_mj=metrics.pe_embodied_mj,
                # Status messages
                ecologits_warnings=metrics.warnings,
                ecologits_errors=metrics.errors,
            )

            # Extract finish reason (stop_reason in Anthropic)
            finish_reason = raw.get("stop_reason")

            # Response ID
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
                provider="anthropic",
            ) from e

    def _normalize_model_for_pricing(self, model_name: str) -> str:
        """
        Normalize model name for pricing lookup.

        Anthropic model names include dates (e.g., "claude-3-5-sonnet-20241022")
        but pricing is typically keyed by base name.

        Args:
            model_name: Model name from API response.

        Returns:
            Normalized model name for pricing lookup.
        """
        import re

        # Map of known model patterns to pricing keys
        model_mappings = {
            r"claude-sonnet-4-\d+": "claude-sonnet-4",
            r"claude-3-5-sonnet-\d+": "claude-3-5-sonnet",
            r"claude-3-5-haiku-\d+": "claude-3-5-haiku",
            r"claude-3-opus-\d+": "claude-3-opus",
            r"claude-3-sonnet-\d+": "claude-3-sonnet",
            r"claude-3-haiku-\d+": "claude-3-haiku",
        }

        for pattern, normalized in model_mappings.items():
            if re.match(pattern, model_name):
                return normalized

        # Return as-is if no pattern matches
        return model_name

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Format messages for Anthropic Claude API.

        Note: System messages should be passed via the 'system' parameter,
        not in the messages list. This method will filter out system messages.

        Args:
            messages: List of Message objects.

        Returns:
            List of message dicts in Anthropic format.
        """
        formatted = []

        for msg in messages:
            # Skip system messages (they're handled separately via 'system' param)
            if msg.role == "system":
                continue

            message_dict: dict[str, Any] = {
                "role": msg.role,
            }

            # Handle text content
            if isinstance(msg.content, str):
                message_dict["content"] = msg.content

            # Handle multimodal content
            elif isinstance(msg.content, list):
                content_blocks = []
                for item in msg.content:
                    item_type = item.get("type")
                    if item_type == "text":
                        content_blocks.append({
                            "type": "text",
                            "text": item.get("text", ""),
                        })
                    elif item_type == "image":
                        image_block = self._format_image_part(item.get("source", ""))
                        if image_block:
                            content_blocks.append(image_block)

                message_dict["content"] = content_blocks

            formatted.append(message_dict)

        return formatted

    def _format_image_part(self, image: str | Image.Image | bytes) -> dict[str, Any] | None:
        """
        Format image for Anthropic Claude API.

        Args:
            image: Image as URL, file path, PIL Image, or bytes.

        Returns:
            Dict with image block for Anthropic API, or None if invalid.
        """
        # URL image
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                return {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image,
                    },
                }
            elif image.startswith("data:"):
                # Data URL - extract base64 data
                # Format: data:image/png;base64,<data>
                try:
                    header, data = image.split(",", 1)
                    media_type = header.split(";")[0].split(":")[1]
                    return {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    }
                except (ValueError, IndexError):
                    return None
            else:
                # Assume file path
                image_path = Path(image)
                if image_path.exists():
                    mime_type, _ = mimetypes.guess_type(str(image_path))
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    return {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type or "image/jpeg",
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        },
                    }
                return None

        # PIL Image
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                },
            }

        # Raw bytes
        elif isinstance(image, bytes):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(image).decode("utf-8"),
                },
            }

        return None

    def close(self) -> None:
        """Close the client and clean up resources."""
        if hasattr(self, "_client") and self._client:
            self._client.close()
        super().close()

    async def aclose(self) -> None:
        """Asynchronously close the client and clean up resources."""
        if hasattr(self, "_async_client") and self._async_client:
            await self._async_client.close()
        super().close()

    def __del__(self) -> None:
        """Clean up resources."""
        if hasattr(self, "_client") and self._client:
            try:
                self._client.close()
            except Exception:
                pass
