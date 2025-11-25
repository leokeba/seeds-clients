"""OpenRouter client implementation.

OpenRouter provides access to multiple AI models through a single API
that's compatible with OpenAI's format. This client allows you to use
various models from different providers through OpenRouter's unified interface.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from seeds_clients.core.exceptions import ConfigurationError, ProviderError
from seeds_clients.core.types import Message, Response, TrackingData, Usage
from seeds_clients.providers.openai import OpenAIClient
from seeds_clients.utils.pricing import calculate_cost


@dataclass
class OpenRouterCostData:
    """Detailed cost information from OpenRouter's generation API.

    OpenRouter provides detailed cost breakdown for each request, including
    native vs normalized token counts which is important for accurate
    carbon tracking with models that use different tokenizers.

    Attributes:
        generation_id: Unique identifier for the generation.
        total_cost: Total cost in USD.
        model: Model used for generation.
        prompt_tokens: Normalized prompt token count.
        completion_tokens: Normalized completion token count.
        native_prompt_tokens: Native prompt tokens (model's actual tokenizer).
        native_completion_tokens: Native completion tokens (model's actual tokenizer).
        provider_name: Upstream provider name (e.g., "openai", "anthropic").
        latency: Request latency in milliseconds.
        cached: Whether this was from cache.
        timestamp: Unix timestamp when recorded.
    """

    generation_id: str
    total_cost: float
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    native_prompt_tokens: int | None = None
    native_completion_tokens: int | None = None
    provider_name: str | None = None
    latency: float | None = None
    cached: bool = False
    timestamp: float = field(default_factory=time.time)


class OpenRouterClient(OpenAIClient):
    """
    OpenRouter client for accessing multiple AI models through a unified API.

    OpenRouter provides access to various AI models including:
    - OpenAI: openai/gpt-4.1, openai/gpt-4.1-mini
    - Anthropic: anthropic/claude-3-5-sonnet, anthropic/claude-3-opus
    - Meta: meta-llama/llama-3.1-405b-instruct
    - Google: google/gemini-pro-1.5
    - Mistral: mistralai/mistral-large
    - And many more...

    The client extends OpenAIClient and adds:
    - Automatic provider/model extraction for EcoLogits tracking
    - Optional real-time cost fetching from OpenRouter's API
    - Support for OpenRouter-specific headers (site URL, app name)

    Example:
        ```python
        from seeds_clients import OpenRouterClient, Message

        # Using environment variable (OPENROUTER_API_KEY)
        client = OpenRouterClient(
            model="anthropic/claude-3-5-sonnet",
            cache_dir="cache"
        )

        response = client.generate(
            messages=[Message(role="user", content="Hello!")]
        )

        print(response.content)
        print(f"Cost: ${response.tracking.cost_usd:.6f}")
        print(f"Carbon: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq")
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-4.1",
        cache_dir: str = "cache",
        ttl_hours: float | None = 24.0,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        electricity_mix_zone: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        fetch_cost_data: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var.
            model: Model name in format "provider/model" (e.g., "anthropic/claude-3-5-sonnet").
            cache_dir: Directory for caching responses.
            ttl_hours: Cache TTL in hours. None for no expiration.
            max_tokens: Maximum completion tokens.
            temperature: Sampling temperature (0-2).
            electricity_mix_zone: ISO 3166-1 alpha-3 code for electricity mix zone.
            site_url: Your site URL for OpenRouter rankings (optional).
            app_name: Your app name for OpenRouter rankings (optional).
            fetch_cost_data: Whether to fetch detailed cost data from OpenRouter API.
                            This adds a small latency but provides accurate cost tracking.
            **kwargs: Additional arguments passed to BaseClient.

        Raises:
            ConfigurationError: If API key is not provided or found in environment.
        """
        # Get API key from parameter or environment
        resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not resolved_api_key:
            raise ConfigurationError(
                "OpenRouter API key required. Provide via api_key parameter or "
                "OPENROUTER_API_KEY environment variable. "
                "Get your key from: https://openrouter.ai/keys"
            )

        self.site_url = site_url
        self.app_name = app_name
        self.fetch_cost_data = fetch_cost_data

        # Track detailed cost data if fetching is enabled
        self._cost_data_history: list[OpenRouterCostData] = []

        # Call parent with OpenRouter base URL
        # Note: We pass api_key directly to skip OpenAI's env var check
        super().__init__(
            api_key=resolved_api_key,
            model=model,
            base_url="https://openrouter.ai/api/v1",
            cache_dir=cache_dir,
            ttl_hours=ttl_hours,
            max_tokens=max_tokens,
            temperature=temperature,
            electricity_mix_zone=electricity_mix_zone,
            **kwargs,
        )

        # Update HTTP client headers with OpenRouter-specific headers
        self._update_http_headers()

    def _update_http_headers(self) -> None:
        """Update HTTP client headers with OpenRouter-specific headers."""
        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            extra_headers["X-Title"] = self.app_name

        if extra_headers:
            # Recreate HTTP client with additional headers
            self._http_client.close()
            self._http_client = httpx.Client(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    **extra_headers,
                },
                timeout=60.0,
            )

    def _get_provider_name(self) -> str:
        """Return provider name for tracking."""
        return "openrouter"

    def _get_ecologits_provider(self) -> str:
        """
        Extract provider name from OpenRouter model string for EcoLogits.

        OpenRouter models are formatted as "provider/model" (e.g., "openai/gpt-4.1").
        This extracts the provider part for EcoLogits compatibility.

        Returns:
            Provider name for EcoLogits (e.g., "openai", "anthropic").
        """
        if "/" in self.model:
            provider, _ = self.model.split("/", 1)
            return provider
        return "openrouter"

    def _get_ecologits_model(self) -> str:
        """
        Extract model name from OpenRouter model string for EcoLogits.

        Returns:
            Model name for EcoLogits (e.g., "gpt-4.1", "claude-3-5-sonnet").
        """
        if "/" in self.model:
            _, model_name = self.model.split("/", 1)
            return model_name
        return self.model

    def _parse_response(self, raw: dict[str, Any]) -> Response:
        """
        Parse OpenRouter API response into Response object.

        Extends OpenAI parsing to handle OpenRouter-specific fields and
        optionally fetch detailed cost data.

        Args:
            raw: Raw API response dict.

        Returns:
            Parsed Response object with cost and carbon tracking.
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

            # Get generation ID for cost fetching
            generation_id = raw.get("id")

            # Calculate cost - try fetching from OpenRouter API if enabled
            cost_usd = 0.0
            if self.fetch_cost_data and generation_id:
                cost_data = self._fetch_cost_data(generation_id)
                if cost_data:
                    cost_usd = cost_data.total_cost
                    self._cost_data_history.append(cost_data)
            else:
                # Fall back to local pricing calculation
                # Try to use the actual model name from response
                model_name = raw.get("model", self.model)
                # For EcoLogits, we need just the model part
                ecologits_model = self._get_ecologits_model()

                # Try provider-specific pricing first
                try:
                    cost_usd = calculate_cost(
                        model=ecologits_model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                    )
                except ValueError:
                    # Model not in pricing database, cost stays 0
                    pass

            # Extract EcoLogits carbon impact data
            ecologits_impacts = raw.get("_ecologits_impacts")
            duration_seconds = raw.get("_duration_seconds", 0.0)

            # Extract full metrics from EcoLogits
            metrics = self._extract_full_ecologits_metrics(ecologits_impacts)

            # Get the actual model returned by OpenRouter
            model_name = raw.get("model", self.model)

            # Create tracking data
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
                provider="openrouter",
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
                provider="openrouter",
            ) from e

    def _call_api(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Call OpenRouter API.

        Extends OpenAI API call to use the correct EcoLogits provider/model.

        Args:
            messages: List of messages.
            **kwargs: Additional API parameters.

        Returns:
            Raw API response as dict.
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

        # Track start time for duration
        start_time = time.time()

        try:
            response = self._http_client.post(
                "/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()

            # Calculate request duration
            duration_seconds = time.time() - start_time
            result["_duration_seconds"] = duration_seconds

            # Calculate EcoLogits carbon impacts using extracted provider/model
            usage = result.get("usage", {})
            output_tokens = usage.get("completion_tokens", 0)

            # Use extracted model name for EcoLogits
            # The provider is obtained via _get_ecologits_provider() which we override
            ecologits_model = self._get_ecologits_model()

            impacts = self._calculate_ecologits_impacts(
                model_name=ecologits_model,
                output_tokens=output_tokens,
                request_latency=duration_seconds,
                electricity_mix_zone=self.electricity_mix_zone,
            )
            if impacts:
                result["_ecologits_impacts"] = impacts

            return result

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", "")
            except Exception:
                error_detail = e.response.text

            raise ProviderError(
                f"OpenRouter API error: {error_detail}",
                provider="openrouter",
                status_code=e.response.status_code,
            ) from e

        except httpx.RequestError as e:
            raise ProviderError(
                f"Request failed: {str(e)}",
                provider="openrouter",
            ) from e

    async def _acall_api(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Asynchronously call OpenRouter API.

        Args:
            messages: List of messages.
            **kwargs: Additional API parameters.

        Returns:
            Raw API response as dict.
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

        # Track start time for duration
        start_time = time.time()

        try:
            client = self._get_async_client()
            response = await client.post(
                "/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()

            # Calculate request duration
            duration_seconds = time.time() - start_time
            result["_duration_seconds"] = duration_seconds

            # Calculate EcoLogits carbon impacts using extracted provider/model
            usage = result.get("usage", {})
            output_tokens = usage.get("completion_tokens", 0)

            # Use extracted model name for EcoLogits
            # The provider is obtained via _get_ecologits_provider() which we override
            ecologits_model = self._get_ecologits_model()

            impacts = self._calculate_ecologits_impacts(
                model_name=ecologits_model,
                output_tokens=output_tokens,
                request_latency=duration_seconds,
                electricity_mix_zone=self.electricity_mix_zone,
            )
            if impacts:
                result["_ecologits_impacts"] = impacts

            return result

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", "")
            except Exception:
                error_detail = e.response.text

            raise ProviderError(
                f"OpenRouter API error: {error_detail}",
                provider="openrouter",
                status_code=e.response.status_code,
            ) from e

        except httpx.RequestError as e:
            raise ProviderError(
                f"Request failed: {str(e)}",
                provider="openrouter",
            ) from e

    def _fetch_cost_data(
        self,
        generation_id: str,
        max_retries: int = 2,
    ) -> OpenRouterCostData | None:
        """
        Fetch detailed cost information from OpenRouter's generation API.

        OpenRouter provides a dedicated API endpoint to retrieve detailed
        cost information for each generation, including native token counts
        which are important for accurate carbon tracking.

        Args:
            generation_id: The generation ID to fetch cost data for.
            max_retries: Maximum number of retry attempts for 404 errors.

        Returns:
            OpenRouterCostData with detailed cost info, or None if failed.
        """
        url = "https://openrouter.ai/api/v1/generation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        params = {"id": generation_id}

        for attempt in range(max_retries + 1):
            try:
                # Add delay for retries - data might not be immediately available
                if attempt > 0:
                    time.sleep(1.5)

                response = httpx.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=10.0,
                )

                # If 404, data might not be ready yet - retry
                if response.status_code == 404 and attempt < max_retries:
                    continue

                response.raise_for_status()
                data = response.json()

                if "data" not in data:
                    return None

                gen_data = data["data"]

                return OpenRouterCostData(
                    generation_id=generation_id,
                    total_cost=gen_data.get("total_cost", 0.0),
                    model=gen_data.get("model", "unknown"),
                    prompt_tokens=gen_data.get("tokens_prompt", 0),
                    completion_tokens=gen_data.get("tokens_completion", 0),
                    native_prompt_tokens=gen_data.get("native_tokens_prompt"),
                    native_completion_tokens=gen_data.get("native_tokens_completion"),
                    provider_name=gen_data.get("provider_name"),
                    latency=gen_data.get("latency"),
                    cached=False,
                )

            except httpx.TimeoutException:
                if attempt == max_retries:
                    return None
            except httpx.HTTPStatusError:
                if attempt == max_retries:
                    return None
            except Exception:
                return None

        return None

    def get_cost_summary(self) -> dict[str, Any]:
        """
        Get a summary of all costs incurred if fetch_cost_data is enabled.

        Returns:
            Dictionary with cost summary statistics.
        """
        if not self._cost_data_history:
            return {
                "total_cost_usd": 0.0,
                "total_requests": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "cost_by_model": {},
                "cost_by_provider": {},
            }

        total_cost = sum(c.total_cost for c in self._cost_data_history)
        total_prompt_tokens = sum(c.prompt_tokens for c in self._cost_data_history)
        total_completion_tokens = sum(c.completion_tokens for c in self._cost_data_history)

        # Aggregate by model and provider
        cost_by_model: dict[str, dict[str, Any]] = {}
        cost_by_provider: dict[str, dict[str, Any]] = {}

        for cost_data in self._cost_data_history:
            # By model
            model = cost_data.model
            if model not in cost_by_model:
                cost_by_model[model] = {"cost": 0.0, "requests": 0}
            cost_by_model[model]["cost"] += cost_data.total_cost
            cost_by_model[model]["requests"] += 1

            # By provider
            provider = cost_data.provider_name or "unknown"
            if provider not in cost_by_provider:
                cost_by_provider[provider] = {"cost": 0.0, "requests": 0}
            cost_by_provider[provider]["cost"] += cost_data.total_cost
            cost_by_provider[provider]["requests"] += 1

        return {
            "total_cost_usd": total_cost,
            "total_requests": len(self._cost_data_history),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "cost_by_model": cost_by_model,
            "cost_by_provider": cost_by_provider,
        }

    def reset_cost_tracking(self) -> None:
        """Reset cost tracking data."""
        self._cost_data_history.clear()
