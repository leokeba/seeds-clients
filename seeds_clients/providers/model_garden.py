"""Model Garden client implementation.

Model Garden is a platform for fine-tuning and serving LLMs with integrated
CodeCarbon carbon tracking. It provides an OpenAI-compatible API with
hardware-measured carbon emissions data in responses.

This client extends OpenAIClient to work with Model Garden servers and
extracts CodeCarbon tracking data from the `x_carbon_trace` field in responses.

See: https://github.com/leokeba/model-garden
"""

import os
import time
from contextlib import suppress
from typing import Any

import httpx

from seeds_clients.core.exceptions import ConfigurationError, ProviderError
from seeds_clients.core.types import Message, Response, TrackingData, Usage
from seeds_clients.providers.openai import OpenAIClient
from seeds_clients.tracking.codecarbon_tracker import CodeCarbonMixin
from seeds_clients.utils.pricing import calculate_cost


class ModelGardenClient(CodeCarbonMixin, OpenAIClient):  # type: ignore[misc]
    """
    Client for Model Garden servers with CodeCarbon carbon tracking.
    
    Model Garden provides an OpenAI-compatible API with integrated carbon
    emissions tracking using CodeCarbon. This client extracts hardware-measured
    emissions data from server responses.
    
    Unlike EcoLogits (model-based estimates), CodeCarbon provides actual
    hardware measurements including GPU, CPU, and RAM power consumption.
    
    Features:
        - OpenAI-compatible API (chat completions, structured outputs)
        - Hardware-measured carbon emissions from CodeCarbon
        - Per-request emissions tracking via x_carbon_trace
        - Session-level aggregate statistics
        - Support for local and remote Model Garden deployments
    
    Example:
        ```python
        from seeds_clients import ModelGardenClient, Message
        
        # Connect to local Model Garden server
        client = ModelGardenClient(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen2.5-3B-Instruct",  # Model loaded in Model Garden
        )
        
        response = client.generate(
            messages=[Message(role="user", content="Hello!")]
        )
        
        print(response.content)
        
        # Access hardware-measured carbon data
        if response.tracking:
            print(f"Carbon: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq")
            print(f"Energy: {response.tracking.energy_kwh:.6f} kWh")
            print(f"Method: {response.tracking.tracking_method}")  # "codecarbon"
            
            # Hardware power measurements
            if response.tracking.gpu_power_watts:
                print(f"GPU Power: {response.tracking.gpu_power_watts:.1f} W")
        ```
    
    Note:
        Model Garden requires a model to be loaded before making requests.
        Use the Model Garden CLI or API to load a model first:
        
        ```bash
        model-garden serve --model Qwen/Qwen2.5-3B-Instruct
        ```
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        model: str = "default",
        cache_dir: str = "cache",
        ttl_hours: float | None = 24.0,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Model Garden client.
        
        Args:
            base_url: Model Garden API base URL (default: http://localhost:8000/v1).
            api_key: API key (optional, Model Garden doesn't require auth by default).
            model: Model name as loaded in Model Garden (e.g., "Qwen/Qwen2.5-3B-Instruct").
                   Use "default" to use whatever model is currently loaded.
            cache_dir: Directory for caching responses.
            ttl_hours: Cache TTL in hours. None for no expiration.
            max_tokens: Maximum completion tokens.
            temperature: Sampling temperature (0-2).
            **kwargs: Additional arguments passed to OpenAIClient.
        
        Note:
            Model Garden doesn't require API authentication by default.
            If you've configured authentication on your server, provide the api_key.
        """
        # Model Garden doesn't require API key by default
        # Use a placeholder if not provided to satisfy parent class
        resolved_api_key = api_key or os.getenv("MODEL_GARDEN_API_KEY") or "not-needed"
        
        # Initialize parent class (OpenAIClient)
        # We skip EcoLogits tracking since we use CodeCarbon from the server
        super().__init__(
            api_key=resolved_api_key,
            model=model,
            base_url=base_url,
            cache_dir=cache_dir,
            ttl_hours=ttl_hours,
            max_tokens=max_tokens,
            temperature=temperature,
            electricity_mix_zone=None,  # Server handles this
            **kwargs,
        )
        
        # Override the HTTP client headers (Model Garden may not need auth)
        self._http_client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {resolved_api_key}"} if api_key else {}),
            },
            timeout=120.0,  # Longer timeout for local inference
        )
    
    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client for Model Garden."""
        if self._async_http_client is None:
            api_key = self.api_key if self.api_key != "not-needed" else None
            self._async_http_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
                },
                timeout=120.0,
            )
        return self._async_http_client
    
    def _get_provider_name(self) -> str:
        """Return provider name for tracking."""
        return "model_garden"
    
    def _get_ecologits_provider(self) -> str:
        """Return provider name for EcoLogits tracking.
        
        Note: Model Garden uses CodeCarbon instead of EcoLogits,
        but we return a value for compatibility.
        """
        return "model_garden"
    
    def _parse_response(self, raw: dict[str, Any]) -> Response:
        """
        Parse Model Garden API response into Response object.
        
        Extracts CodeCarbon tracking data from x_carbon_trace field
        instead of using EcoLogits estimates.
        
        Args:
            raw: Raw API response dict.
            
        Returns:
            Parsed Response object with CodeCarbon tracking data.
            
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
            
            # Try to calculate cost (may not be in pricing DB for custom models)
            model_name = raw.get("model", self.model)
            cost_usd = 0.0
            with suppress(ValueError):
                cost_usd = calculate_cost(
                    model=model_name,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )
            
            # Extract CodeCarbon metrics from x_carbon_trace
            codecarbon_metrics = self._extract_codecarbon_metrics(raw)
            codecarbon_fields = self._codecarbon_to_tracking_fields(codecarbon_metrics)
            
            # Get duration from CodeCarbon or from raw response
            duration_seconds = codecarbon_fields.get(
                "duration_seconds", 
                raw.get("_duration_seconds", 0.0)
            )
            
            # Create tracking data with CodeCarbon metrics
            tracking = TrackingData(
                # Total metrics from CodeCarbon
                energy_kwh=codecarbon_fields.get("energy_kwh", 0.0),
                gwp_kgco2eq=codecarbon_fields.get("gwp_kgco2eq", 0.0),
                
                # Cost (local inference is typically free)
                cost_usd=cost_usd,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                
                # Metadata
                provider="model_garden",
                model=model_name,
                tracking_method=codecarbon_fields.get("tracking_method") or "none",
                electricity_mix_zone=None,  # Server-side
                duration_seconds=duration_seconds,
                
                # Usage phase (all hardware-measured)
                energy_usage_kwh=codecarbon_fields.get("energy_usage_kwh"),
                gwp_usage_kgco2eq=codecarbon_fields.get("gwp_usage_kgco2eq"),
                
                # Embodied not tracked by CodeCarbon
                gwp_embodied_kgco2eq=None,
                adpe_kgsbeq=None,
                pe_mj=None,
                
                # CodeCarbon hardware measurements
                cpu_energy_kwh=codecarbon_fields.get("cpu_energy_kwh"),
                gpu_energy_kwh=codecarbon_fields.get("gpu_energy_kwh"),
                ram_energy_kwh=codecarbon_fields.get("ram_energy_kwh"),
                cpu_power_watts=codecarbon_fields.get("cpu_power_watts"),
                gpu_power_watts=codecarbon_fields.get("gpu_power_watts"),
                ram_power_watts=codecarbon_fields.get("ram_power_watts"),
            )
            
            # Add CodeCarbon-specific fields to raw for access
            if codecarbon_metrics:
                raw["_codecarbon_metrics"] = {
                    "cpu_energy_kwh": codecarbon_fields.get("cpu_energy_kwh"),
                    "gpu_energy_kwh": codecarbon_fields.get("gpu_energy_kwh"),
                    "ram_energy_kwh": codecarbon_fields.get("ram_energy_kwh"),
                    "cpu_power_watts": codecarbon_fields.get("cpu_power_watts"),
                    "gpu_power_watts": codecarbon_fields.get("gpu_power_watts"),
                    "ram_power_watts": codecarbon_fields.get("ram_power_watts"),
                    "measured": codecarbon_metrics.measured,
                    "tracking_active": codecarbon_metrics.tracking_active,
                    "session_total_kg_co2": codecarbon_metrics.session_total_kg_co2,
                    "session_requests": codecarbon_metrics.session_requests,
                    "session_tokens": codecarbon_metrics.session_tokens,
                }
            
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
                provider="model_garden",
            ) from e
    
    def _call_api(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Call Model Garden API.
        
        Args:
            messages: List of messages.
            **kwargs: Additional API parameters.
            
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
        
        # Track start time
        start_time = time.time()
        
        try:
            response = self._http_client.post(
                "/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            
            # Add duration for tracking
            result["_duration_seconds"] = time.time() - start_time
            
            return result
            
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("detail", str(error_data))
            except Exception:
                error_detail = e.response.text
            
            raise ProviderError(
                f"Model Garden API error: {error_detail}",
                provider="model_garden",
                status_code=e.response.status_code,
            ) from e
            
        except httpx.RequestError as e:
            raise ProviderError(
                f"Request failed: {str(e)}. Is Model Garden running at {self.base_url}?",
                provider="model_garden",
            ) from e
    
    async def _acall_api(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Asynchronously call Model Garden API.
        
        Args:
            messages: List of messages.
            **kwargs: Additional API parameters.
            
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
        
        # Track start time
        start_time = time.time()
        
        try:
            client = self._get_async_client()
            response = await client.post(
                "/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            
            # Add duration for tracking
            result["_duration_seconds"] = time.time() - start_time
            
            return result
            
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("detail", str(error_data))
            except Exception:
                error_detail = e.response.text
            
            raise ProviderError(
                f"Model Garden API error: {error_detail}",
                provider="model_garden",
                status_code=e.response.status_code,
            ) from e
            
        except httpx.RequestError as e:
            raise ProviderError(
                f"Request failed: {str(e)}. Is Model Garden running at {self.base_url}?",
                provider="model_garden",
            ) from e
    
    def get_carbon_stats(self) -> dict[str, Any] | None:
        """
        Get current carbon tracking statistics from the server.
        
        Returns aggregate statistics for the current inference session
        including total emissions, requests, and tokens.
        
        Returns:
            Dictionary with carbon statistics, or None if unavailable.
            
        Example:
            ```python
            stats = client.get_carbon_stats()
            if stats:
                print(f"Total emissions: {stats['emissions_kg_co2']:.6f} kg CO2")
                print(f"Total requests: {stats['request_count']}")
            ```
        """
        try:
            response = self._http_client.get("/api/v1/carbon/inference/stats")
            if response.status_code == 200:
                result: dict[str, Any] = response.json()
                return result
            return None
        except Exception:
            return None
    
    async def aget_carbon_stats(self) -> dict[str, Any] | None:
        """
        Asynchronously get current carbon tracking statistics from the server.
        
        Returns:
            Dictionary with carbon statistics, or None if unavailable.
        """
        try:
            client = self._get_async_client()
            response = await client.get("/api/v1/carbon/inference/stats")
            if response.status_code == 200:
                result: dict[str, Any] = response.json()
                return result
            return None
        except Exception:
            return None
    
    def get_emissions_summary(self) -> dict[str, Any] | None:
        """
        Get aggregate emissions summary from the server.
        
        Returns total emissions across all jobs (training and inference).
        
        Returns:
            Dictionary with emissions summary, or None if unavailable.
        """
        try:
            response = self._http_client.get("/api/v1/carbon/summary")
            if response.status_code == 200:
                result: dict[str, Any] = response.json()
                return result
            return None
        except Exception:
            return None
