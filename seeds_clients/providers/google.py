"""Google Gemini client implementation using the google-genai SDK."""

import os
import time
from typing import Any

from PIL import Image
from pydantic import BaseModel

from seeds_clients.core.base_client import BaseClient
from seeds_clients.core.exceptions import ConfigurationError, ProviderError
from seeds_clients.core.types import Message, Response, Usage
from seeds_clients.tracking.ecologits_tracker import EcoLogitsMixin
from seeds_clients.utils.logging_utils import get_logger
from seeds_clients.utils.pricing import calculate_cost

logger = get_logger(__name__)


class GoogleClient(EcoLogitsMixin, BaseClient):
    """
    Google Gemini client implementation with carbon tracking.

    Uses the official google-genai SDK for the Gemini API.

    Supports:
    - Chat completions (text and multimodal)
    - Structured outputs with Pydantic models (response_schema)
    - Image inputs (URLs, file paths, PIL Images, base64)
    - Cost tracking and carbon impact measurement via EcoLogits
    - System instructions
    - Configurable generation parameters (temperature, max_output_tokens, etc.)

    Example:
        ```python
        from seeds_clients import GoogleClient, Message

        client = GoogleClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-2.5-flash",
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
        - Gemini 2.5+ models have "thinking" enabled by default
        - For Gemini 3 models, keep temperature at 1.0 for best results
        - Structured outputs require response_mime_type='application/json'
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        cache_dir: str | None = None,
        ttl_hours: float | None = 24.0,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        system_instruction: str | None = None,
        electricity_mix_zone: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Google Gemini client.

        Args:
            api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY
                    or GOOGLE_API_KEY env var.
            model: Model name (e.g., "gemini-2.5-flash", "gemini-2.5-pro",
                   "gemini-3-pro-preview").
            cache_dir: Directory for caching responses.
            ttl_hours: Cache TTL in hours. None for no expiration.
            max_output_tokens: Maximum output tokens. None for model default.
            temperature: Sampling temperature. None for model default.
                        Note: Keep at 1.0 for Gemini 3 models.
            top_p: Nucleus sampling probability. None for model default.
            top_k: Top-k sampling. None for model default.
            system_instruction: System instruction for the model.
            electricity_mix_zone: ISO 3166-1 alpha-3 code for electricity mix zone
                                 (e.g., "FRA", "USA", "WOR"). Default is "WOR" (World).
            **kwargs: Additional arguments passed to BaseClient.

        Raises:
            ConfigurationError: If API key is not provided or found in environment.
        """
        # Import google-genai here to avoid import errors if not installed
        try:
            from google import genai
            from google.genai import types

            self._genai = genai
            self._types = types
        except ImportError as e:
            raise ConfigurationError(
                "google-genai package is required for GoogleClient. "
                "Install it with: pip install google-genai"
            ) from e

        # Get API key from parameter or environment
        # google-genai accepts GEMINI_API_KEY or GOOGLE_API_KEY
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise ConfigurationError(
                "Google API key required. Provide via api_key parameter or "
                "GEMINI_API_KEY/GOOGLE_API_KEY environment variable."
            )

        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.system_instruction = system_instruction

        # Initialize base client (will store api_key)
        super().__init__(
            model=model,
            api_key=resolved_api_key,
            cache_dir=cache_dir,
            ttl_hours=ttl_hours,
            electricity_mix_zone=electricity_mix_zone,
            **kwargs,
        )

        # Set electricity mix zone for EcoLogits mixin
        self._set_electricity_mix_zone(electricity_mix_zone)

        # Create sync client
        self._client = genai.Client(api_key=resolved_api_key)

        # Async client accessor (available via .aio attribute)
        self._async_client = self._client.aio

    def _get_provider_name(self) -> str:
        """Return provider name for tracking."""
        return "google"

    def _get_ecologits_provider(self) -> str:
        """Return provider name for EcoLogits tracking."""
        # EcoLogits uses "google_genai" for Google Gemini models
        return "google_genai"

    def _setup_tracking(self) -> None:
        """Setup tracking (placeholder for Phase 3)."""
        pass

    def generate(
        self,
        messages: list[Message],
        use_cache: bool = False,
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
                - max_output_tokens: Maximum output tokens
                - system_instruction: Override system instruction for this request

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
        response_format = kwargs.get("response_format")
        if (
            response_format is not None
            and isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
        ):
            kwargs = kwargs.copy()
            kwargs["_original_response_format"] = response_format

        # Call parent generate method (handles caching and structured output parsing)
        return super().generate(messages, use_cache=use_cache, **kwargs)

    async def agenerate(
        self,
        messages: list[Message],
        use_cache: bool = False,
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
                - max_output_tokens: Maximum output tokens

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
        response_format = kwargs.get("response_format")
        if (
            response_format is not None
            and isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
        ):
            kwargs = kwargs.copy()
            kwargs["_original_response_format"] = response_format

        # Call parent agenerate method (handles caching and structured output parsing)
        return await super().agenerate(messages, use_cache=use_cache, **kwargs)

    def _build_generation_config(
        self,
        response_format: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Build GenerateContentConfig for the API request.

        Args:
            response_format: Pydantic model for structured output.
            **kwargs: Additional generation parameters.

        Returns:
            types.GenerateContentConfig object
        """
        types = self._types

        config_params: dict[str, Any] = {}

        # System instruction (from init or override)
        system_instruction = kwargs.pop("system_instruction", None) or self.system_instruction
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        # Max output tokens
        max_output_tokens = kwargs.pop("max_output_tokens", None) or self.max_output_tokens
        if max_output_tokens is not None:
            config_params["max_output_tokens"] = max_output_tokens

        # Temperature
        temperature = kwargs.pop("temperature", None)
        if temperature is None and self.temperature is not None:
            temperature = self.temperature
        if temperature is not None:
            config_params["temperature"] = temperature

        # Top-p
        top_p = kwargs.pop("top_p", None) or self.top_p
        if top_p is not None:
            config_params["top_p"] = top_p

        # Top-k
        top_k = kwargs.pop("top_k", None) or self.top_k
        if top_k is not None:
            config_params["top_k"] = top_k

        # Structured output configuration
        if (
            response_format is not None
            and isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
        ):
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = response_format

        # Pass through any remaining kwargs
        config_params.update(kwargs)

        return types.GenerateContentConfig(**config_params) if config_params else None

    def _call_api(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Call Google Gemini API.

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

        # Build generation config
        config = self._build_generation_config(response_format=response_format, **kwargs)

        # Format messages for the API
        contents = self._format_messages(messages)

        start_time = time.time()

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            raw_response = self._response_to_dict(response)
            raw_response = self._apply_response_tracking(
                raw_response,
                start_time,
                model_name=self.model,
                output_tokens=getattr(
                    getattr(response, "usage_metadata", None), "candidates_token_count", 0
                )
                or 0,
            )

            return raw_response

        except Exception as e:
            error_msg = str(e)

            # Try to extract more specific error info
            if hasattr(e, "message"):
                error_msg = getattr(e, "message", error_msg)
            elif hasattr(e, "status_code"):
                raise ProviderError(
                    f"Google Gemini API error: {error_msg}",
                    provider="google",
                    status_code=getattr(e, "status_code", None),
                ) from e

            raise ProviderError(
                f"Google Gemini API error: {error_msg}",
                provider="google",
            ) from e

    async def _acall_api(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Asynchronously call Google Gemini API.

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

        # Build generation config
        config = self._build_generation_config(response_format=response_format, **kwargs)

        # Format messages for the API
        contents = self._format_messages(messages)

        start_time = time.time()

        try:
            response = await self._async_client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            raw_response = self._response_to_dict(response)
            raw_response = self._apply_response_tracking(
                raw_response,
                start_time,
                model_name=self.model,
                output_tokens=getattr(
                    getattr(response, "usage_metadata", None), "candidates_token_count", 0
                )
                or 0,
            )

            return raw_response

        except Exception as e:
            error_msg = str(e)

            # Try to extract more specific error info
            if hasattr(e, "message"):
                error_msg = getattr(e, "message", error_msg)
            elif hasattr(e, "status_code"):
                raise ProviderError(
                    f"Google Gemini API error: {error_msg}",
                    provider="google",
                    status_code=getattr(e, "status_code", None),
                ) from e

            raise ProviderError(
                f"Google Gemini API error: {error_msg}",
                provider="google",
            ) from e

    def _response_to_dict(self, response: Any) -> dict[str, Any]:
        """
        Convert google-genai response object to a dictionary for caching.

        Args:
            response: GenerateContentResponse from the API.

        Returns:
            Dictionary representation of the response.
        """
        result: dict[str, Any] = {}

        # Extract text content
        if hasattr(response, "text"):
            result["text"] = response.text

        # Extract candidates
        if hasattr(response, "candidates") and response.candidates:
            candidates = []
            for candidate in response.candidates:
                cand_dict: dict[str, Any] = {}

                # Content
                if hasattr(candidate, "content") and candidate.content:
                    content = candidate.content
                    parts_list = []
                    if hasattr(content, "parts"):
                        for part in content.parts:
                            part_dict: dict[str, Any] = {}
                            if hasattr(part, "text"):
                                part_dict["text"] = part.text
                            if hasattr(part, "inline_data") and part.inline_data:
                                part_dict["inline_data"] = {
                                    "mime_type": getattr(part.inline_data, "mime_type", None),
                                    "data": getattr(part.inline_data, "data", None),
                                }
                            parts_list.append(part_dict)
                    cand_dict["content"] = {
                        "role": getattr(content, "role", "model"),
                        "parts": parts_list,
                    }

                # Finish reason
                if hasattr(candidate, "finish_reason"):
                    cand_dict["finish_reason"] = str(candidate.finish_reason)

                # Safety ratings
                if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                    ratings = []
                    for rating in candidate.safety_ratings:
                        ratings.append(
                            {
                                "category": str(getattr(rating, "category", "")),
                                "probability": str(getattr(rating, "probability", "")),
                            }
                        )
                    cand_dict["safety_ratings"] = ratings

                candidates.append(cand_dict)
            result["candidates"] = candidates

        # Extract usage metadata
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            result["usage_metadata"] = {
                "prompt_token_count": getattr(usage, "prompt_token_count", 0),
                "candidates_token_count": getattr(usage, "candidates_token_count", 0),
                "total_token_count": getattr(usage, "total_token_count", 0),
            }

        # Model version if available
        if hasattr(response, "model_version"):
            result["model_version"] = response.model_version

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
            # Extract content
            content = raw.get("text", "")

            # If no direct text, try to extract from candidates
            if not content and "candidates" in raw:
                candidates = raw["candidates"]
                if candidates:
                    first_candidate = candidates[0]
                    if "content" in first_candidate:
                        parts = first_candidate["content"].get("parts", [])
                        if parts:
                            content = parts[0].get("text", "")

            # Extract usage
            usage_data = raw.get("usage_metadata", {})
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_token_count", 0),
                completion_tokens=usage_data.get("candidates_token_count", 0),
                total_tokens=usage_data.get("total_token_count", 0),
            )

            # Calculate cost
            model_name = raw.get("model_version", self.model)
            pricing_model = self._normalize_model_for_pricing(model_name)
            cost_usd = 0.0
            try:
                cost_usd = calculate_cost(
                    model=pricing_model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    provider="google",
                )
            except ValueError as e:
                logger.warning(f"Failed to calculate cost: {e}")

            # Extract EcoLogits carbon impact data
            ecologits_impacts = raw.get("_ecologits_impacts")
            duration_seconds = raw.get("_duration_seconds", 0.0)

            # Extract full metrics from EcoLogits
            metrics = self._extract_full_ecologits_metrics(ecologits_impacts)

            # Create tracking data with cost and carbon metrics
            tracking = self._create_tracking_data(
                metrics=metrics,
                usage=usage,
                provider="google",
                model_name=model_name,
                cost_usd=cost_usd,
                duration_seconds=duration_seconds,
            )

            # Extract finish reason
            finish_reason = None
            if "candidates" in raw and raw["candidates"]:
                finish_reason = raw["candidates"][0].get("finish_reason")

            return Response(
                content=content,
                usage=usage,
                model=model_name,
                raw=raw,
                tracking=tracking,
                finish_reason=finish_reason,
            )

        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(
                f"Invalid response format: {str(e)}",
                provider="google",
            ) from e

    def _normalize_model_for_pricing(self, model_name: str) -> str:
        """
        Normalize model name for pricing lookup.

        The API may return a versioned model name (e.g., "gemini-2.5-flash-001")
        but pricing is typically keyed by base name (e.g., "gemini-2.5-flash").

        Args:
            model_name: Model name from API response.

        Returns:
            Normalized model name for pricing lookup.
        """
        # First check if the exact model exists
        # If not, try progressively shorter versions
        # Common patterns: gemini-2.5-flash-001, gemini-2.5-flash-preview-09-2025

        # Remove common suffixes like -001, -002, etc.
        import re

        normalized = re.sub(r"-\d{3}$", "", model_name)

        return normalized

    def _format_messages(self, messages: list[Message]) -> list[Any]:
        """
        Format messages for Google Gemini API.

        Args:
            messages: List of Message objects.

        Returns:
            List of Content objects in Gemini format.
        """
        types = self._types
        contents = []

        for msg in messages:
            # Map role to Gemini format
            # Note: Gemini uses "user" and "model" roles
            role = "user" if msg.role in ("user", "system") else "model"

            # Handle text content
            if isinstance(msg.content, str):
                content = types.Content(role=role, parts=[types.Part.from_text(text=msg.content)])
                contents.append(content)

            # Handle multimodal content
            elif isinstance(msg.content, list):
                parts = []
                for item in msg.content:
                    item_type = item.get("type")
                    if item_type == "text":
                        parts.append(types.Part.from_text(text=item.get("text", "")))
                    elif item_type == "image":
                        parts.append(self._format_image_part(item.get("source", "")))

                content = types.Content(role=role, parts=parts)
                contents.append(content)

        return contents

    def _format_image_part(self, image: str | Image.Image | bytes) -> Any:
        """
        Format image for Google Gemini API.

        Args:
            image: Image as URL, file path, PIL Image, or bytes.

        Returns:
            types.Part object for the image.
        """
        import base64
        import io
        import mimetypes
        from pathlib import Path

        types = self._types

        # Already a URL (Google Cloud Storage or HTTP)
        if isinstance(image, str):
            if image.startswith("gs://"):
                # Google Cloud Storage URI
                return types.Part.from_uri(file_uri=image, mime_type="image/jpeg")
            elif image.startswith("http://") or image.startswith("https://"):
                # HTTP URL - download and convert to bytes
                # For simplicity, treat as URI (Gemini can handle URLs in some cases)
                return types.Part.from_uri(file_uri=image, mime_type="image/jpeg")
            elif image.startswith("data:"):
                # Data URL - extract base64 data
                # Format: data:image/png;base64,<data>
                header, data = image.split(",", 1)
                mime_type = header.split(";")[0].split(":")[1]
                image_bytes = base64.b64decode(data)
                return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            else:
                # Assume file path
                image_path = Path(image)
                if image_path.exists():
                    guessed_type, _ = mimetypes.guess_type(str(image_path))
                    file_mime_type: str = guessed_type or "image/jpeg"
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    return types.Part.from_bytes(data=image_bytes, mime_type=file_mime_type)
                raise ValueError(f"Unsupported image format or invalid source: {str(image)[:100]}")

        # PIL Image
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            return types.Part.from_bytes(data=image_bytes, mime_type="image/png")

        # Raw bytes
        elif isinstance(image, bytes):
            return types.Part.from_bytes(data=image, mime_type="image/jpeg")

        raise ValueError(f"Unsupported image format or invalid source: {str(image)[:100]}")

    def close(self) -> None:
        """Close the client and clean up resources."""
        if hasattr(self, "_client") and self._client:
            self._client.close()
        super().close()

    async def aclose(self) -> None:
        """Asynchronously close the client and clean up resources."""
        # The async client is accessed via .aio attribute
        # Close the main client which handles both sync and async
        if hasattr(self, "_client") and self._client:
            self._client.close()
        await super().aclose()

    def __del__(self) -> None:
        """Clean up resources."""
        if hasattr(self, "_client") and self._client:
            try:
                self._client.close()
            except Exception:
                pass
