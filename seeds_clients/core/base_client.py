"""Abstract base client for all LLM providers."""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from seeds_clients.core.batch import BatchResult
from seeds_clients.core.cache import CacheManager
from seeds_clients.core.exceptions import ValidationError
from seeds_clients.core.types import CumulativeTracking, Message, Response, TrackingData
from seeds_clients.utils.logging_utils import get_logger

# Type variable for structured outputs
T = TypeVar("T", bound=BaseModel)

logger = get_logger(__name__)


class BaseClient(ABC):
    """Abstract base class for LLM clients.

    Provides unified interface with caching, tracking, and error handling.
    Subclasses implement provider-specific API calls and response parsing.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        cache_dir: Path | str | None = None,
        ttl_hours: float | None = 24,
        cache_ttl_hours: float | None = None,
        enable_tracking: bool = True,
        tracking_method: str = "ecologits",
        electricity_mix_zone: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize base client.

        Args:
            model: Model identifier (e.g., 'gpt-4.1', 'claude-3-5-sonnet')
            api_key: API key for the provider (can also come from env vars)
            cache_dir: Directory for cache storage (None = no caching)
            ttl_hours: Cache time-to-live in hours (None = no expiration)
            cache_ttl_hours: Deprecated alias for ttl_hours; preserved for
                             backward compatibility
            enable_tracking: Whether to enable carbon/cost tracking
            tracking_method: Tracking method ('ecologits', 'codecarbon', 'none')
            electricity_mix_zone: ISO 3166-1 alpha-3 code for electricity mix
                                 (e.g., 'FRA', 'USA', 'WOR'). Default is 'WOR' (World).
            **kwargs: Additional provider-specific arguments
        """
        self.model = model
        self.api_key = api_key
        self.enable_tracking = enable_tracking
        self.tracking_method = tracking_method
        self.electricity_mix_zone = electricity_mix_zone or "WOR"
        self.kwargs = kwargs

        effective_ttl_hours = ttl_hours if ttl_hours is not None else cache_ttl_hours

        # Initialize cache if directory provided
        self.cache: CacheManager | None = None
        if cache_dir:
            self.cache = CacheManager(cache_dir, ttl_hours=effective_ttl_hours)

        # Initialize cumulative tracking
        self._cumulative_tracking = CumulativeTracking()

        # Initialize tracking (will be set up by subclasses)
        self.tracker: Any = None
        if enable_tracking and tracking_method != "none":
            self._setup_tracking()

    @abstractmethod
    def _setup_tracking(self) -> None:
        """Set up tracking based on tracking_method.

        Should be implemented by subclasses or mixins.
        """
        pass

    @abstractmethod
    def _call_api(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call provider API and return raw response.

        Args:
            messages: List of conversation messages
            **kwargs: Additional API parameters

        Returns:
            Raw API response as dictionary

        Raises:
            ProviderError: If API call fails
        """
        pass

    @abstractmethod
    async def _acall_api(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call provider API asynchronously and return raw response.

        Args:
            messages: List of conversation messages
            **kwargs: Additional API parameters

        Returns:
            Raw API response as dictionary

        Raises:
            ProviderError: If API call fails
        """
        pass

    @abstractmethod
    def _parse_response(self, raw: dict[str, Any]) -> Response:
        """Parse raw API response into standardized Response.

        Args:
            raw: Raw API response dictionary

        Returns:
            Standardized Response object

        Raises:
            ValidationError: If response format is invalid
        """
        pass

    def generate(
        self,
        messages: list[Message],
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Response[Any]:
        """Generate a response from the LLM.

        Main entry point for text generation with automatic caching and tracking.
        Supports structured outputs via the response_format kwarg.

        Args:
            messages: List of conversation messages
            use_cache: Whether to use cache for this request
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
                Special kwargs:
                - response_format: Pydantic model class for structured output

        Returns:
            Response object with generated text and metadata.
            If response_format is provided, response.parsed contains the validated model.

        Raises:
            ProviderError: If API call fails
            ValidationError: If response is invalid or structured output parsing fails

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
        # Extract response_format if provided (keep copy for parsing later)
        # Check for _original_response_format first (set by providers that transform the format)
        response_format = kwargs.pop("_original_response_format", None) or kwargs.get(
            "response_format", None
        )
        # Ensure response_format is a Pydantic model class, not a transformed dict
        if response_format is not None and not (
            isinstance(response_format, type) and issubclass(response_format, BaseModel)
        ):
            response_format = None

        # Generate cache key
        cache_key = self._compute_cache_key(messages, kwargs)
        provider_name = self._get_provider_name()

        logger.debug(
            "generate called",
            extra={
                "provider": provider_name,
                "model": self.model,
                "use_cache": use_cache,
                "response_format": bool(response_format),
            },
        )

        # Try cache first
        if use_cache and self.cache:
            cached_raw = self.cache.get(cache_key)
            if cached_raw:
                logger.debug(
                    "cache hit",
                    extra={"provider": provider_name, "model": self.model, "cache_key": cache_key},
                )
                response = self._parse_response(cached_raw)
                response.cached = True
                # Parse structured output if response_format was provided
                if response_format is not None and response.content:
                    response = self._parse_structured_output(response, response_format)
                # Accumulate cached tracking data
                if response.tracking is not None:
                    self._cumulative_tracking.accumulate(response.tracking, cached=True)
                return response

        # Call API with timing
        logger.debug(
            "calling provider",
            extra={"provider": provider_name, "model": self.model, "use_cache": use_cache},
        )
        start_time = time.time()
        raw_response = self._call_api(messages, **kwargs)
        duration = time.time() - start_time
        logger.debug(
            "provider response",
            extra={
                "provider": provider_name,
                "model": self.model,
                "duration_seconds": round(duration, 3),
                "cached": False,
            },
        )

        # Parse response
        response = self._parse_response(raw_response)

        # Add tracking if enabled (only if not already set by provider)
        if response.tracking is None and self.enable_tracking and self.tracker:
            tracking_data = self._track_request(raw_response, duration)
            response.tracking = tracking_data

        response.cached = False

        # Cache the raw response
        if use_cache and self.cache:
            metadata = {
                "model": self.model,
                "provider": provider_name,
                "duration_seconds": duration,
            }
            self.cache.set(cache_key, raw_response, metadata)
            logger.debug(
                "cache store",
                extra={"provider": provider_name, "model": self.model, "cache_key": cache_key},
            )

        # Accumulate API tracking data
        if response.tracking is not None:
            self._cumulative_tracking.accumulate(response.tracking, cached=False)

        # Parse structured output if response_format was provided
        if response_format is not None and response.content:
            response = self._parse_structured_output(response, response_format)

        return response

    async def agenerate(
        self,
        messages: list[Message],
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Response[Any]:
        """Generate a response asynchronously.

        Async version of generate() for concurrent request handling.
        Supports structured outputs via the response_format kwarg.

        Args:
            messages: List of conversation messages
            use_cache: Whether to use cache for this request
            **kwargs: Additional generation parameters
                Special kwargs:
                - response_format: Pydantic model class for structured output

        Returns:
            Response object with generated text and metadata.
            If response_format is provided, response.parsed contains the validated model.

        Raises:
            ProviderError: If API call fails
            ValidationError: If response is invalid or structured output parsing fails

        Example:
            ```python
            from pydantic import BaseModel

            class Person(BaseModel):
                name: str
                age: int

            response = await client.agenerate(
                messages=[Message(role="user", content="Extract: John is 30")],
                response_format=Person
            )
            print(response.parsed.name)  # "John"
            ```
        """
        # Extract response_format if provided (keep copy for parsing later)
        # Check for _original_response_format first (set by providers that transform the format)
        response_format = kwargs.pop("_original_response_format", None) or kwargs.get(
            "response_format", None
        )
        # Ensure response_format is a Pydantic model class, not a transformed dict
        if response_format is not None and not (
            isinstance(response_format, type) and issubclass(response_format, BaseModel)
        ):
            response_format = None

        # Generate cache key
        cache_key = self._compute_cache_key(messages, kwargs)
        provider_name = self._get_provider_name()

        logger.debug(
            "agenerate called",
            extra={
                "provider": provider_name,
                "model": self.model,
                "use_cache": use_cache,
                "response_format": bool(response_format),
            },
        )

        # Try cache first
        if use_cache and self.cache:
            cached_raw = self.cache.get(cache_key)
            if cached_raw:
                logger.debug(
                    "cache hit",
                    extra={"provider": provider_name, "model": self.model, "cache_key": cache_key},
                )
                response = self._parse_response(cached_raw)
                response.cached = True
                # Parse structured output if response_format was provided
                if response_format is not None and response.content:
                    response = self._parse_structured_output(response, response_format)
                # Accumulate cached tracking data
                if response.tracking is not None:
                    self._cumulative_tracking.accumulate(response.tracking, cached=True)
                return response

        # Call API with timing
        logger.debug(
            "calling provider",
            extra={"provider": provider_name, "model": self.model, "use_cache": use_cache},
        )
        start_time = time.time()
        raw_response = await self._acall_api(messages, **kwargs)
        duration = time.time() - start_time
        logger.debug(
            "provider response",
            extra={
                "provider": provider_name,
                "model": self.model,
                "duration_seconds": round(duration, 3),
                "cached": False,
            },
        )

        # Parse response
        response = self._parse_response(raw_response)

        # Add tracking if enabled
        if response.tracking is None and self.enable_tracking and self.tracker:
            tracking_data = self._track_request(raw_response, duration)
            response.tracking = tracking_data

        response.cached = False

        # Cache the raw response
        if use_cache and self.cache:
            metadata = {
                "model": self.model,
                "provider": provider_name,
                "duration_seconds": duration,
            }
            self.cache.set(cache_key, raw_response, metadata)
            logger.debug(
                "cache store",
                extra={"provider": provider_name, "model": self.model, "cache_key": cache_key},
            )

        # Accumulate API tracking data
        if response.tracking is not None:
            self._cumulative_tracking.accumulate(response.tracking, cached=False)

        # Parse structured output if response_format was provided
        if response_format is not None and response.content:
            response = self._parse_structured_output(response, response_format)

        return response

    async def batch_generate(
        self,
        messages_list: list[list[Message]],
        max_concurrent: int = 5,
        use_cache: bool = True,
        on_progress: Callable[[int, int, Response | Exception], None] | None = None,
        **kwargs: Any,
    ) -> BatchResult:
        """Generate responses for multiple prompts concurrently.

        Processes a batch of requests with configurable concurrency limit.
        Aggregates metrics across all successful requests.

        Args:
            messages_list: List of message lists (one per request)
            max_concurrent: Maximum concurrent requests (default: 5)
            use_cache: Whether to use cache for requests
            on_progress: Optional callback(completed, total, result_or_error)
            **kwargs: Additional generation parameters for all requests

        Returns:
            BatchResult with responses and aggregated metrics

        Example:
            ```python
            prompts = [
                [Message(role="user", content="Explain Python")],
                [Message(role="user", content="Explain JavaScript")],
                [Message(role="user", content="Explain Rust")],
            ]

            result = await client.batch_generate(
                prompts,
                max_concurrent=3,
                on_progress=lambda done, total, r: print(f"{done}/{total}")
            )

            print(f"Success: {result.successful_count}/{result.total_count}")
            print(f"Total cost: ${result.total_cost_usd:.4f}")
            ```
        """
        batch_start = time.time()
        result = BatchResult()
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0
        provider_name = self._get_provider_name()

        logger.debug(
            "batch_generate start",
            extra={
                "provider": provider_name,
                "model": self.model,
                "requests": len(messages_list),
                "max_concurrent": max_concurrent,
                "use_cache": use_cache,
            },
        )

        async def process_one(
            index: int, messages: list[Message]
        ) -> tuple[int, Response | Exception]:
            nonlocal completed
            async with semaphore:
                try:
                    response = await self.agenerate(messages, use_cache=use_cache, **kwargs)
                    completed += 1
                    if on_progress:
                        on_progress(completed, len(messages_list), response)
                    return index, response
                except Exception as e:
                    completed += 1
                    if on_progress:
                        on_progress(completed, len(messages_list), e)
                    return index, e

        # Run all tasks concurrently
        tasks = [process_one(i, msgs) for i, msgs in enumerate(messages_list)]
        results = await asyncio.gather(*tasks)

        # Process results in order
        for index, res in sorted(results, key=lambda x: x[0]):
            if isinstance(res, Exception):
                result.errors.append((index, res))
            else:
                result.responses.append(res)
                # Aggregate metrics
                if res.usage:
                    result.total_prompt_tokens += res.usage.prompt_tokens or 0
                    result.total_completion_tokens += res.usage.completion_tokens or 0
                if res.tracking:
                    result.total_cost_usd += res.tracking.cost_usd or 0.0
                    result.total_energy_kwh += res.tracking.energy_kwh or 0.0
                    result.total_gwp_kgco2eq += res.tracking.gwp_kgco2eq or 0.0

        result.total_duration_seconds = time.time() - batch_start
        logger.debug(
            "batch_generate done",
            extra={
                "provider": provider_name,
                "model": self.model,
                "duration_seconds": round(result.total_duration_seconds, 3),
                "successes": len(result.responses),
                "errors": len(result.errors),
            },
        )
        return result

    async def batch_generate_iter(
        self,
        messages_list: list[list[Message]],
        max_concurrent: int = 5,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, Response | Exception]]:
        """Generate responses as an async iterator, yielding as completed.

        Unlike batch_generate, this yields results as they complete rather
        than waiting for all to finish. Useful for progress tracking or
        early processing.

        Args:
            messages_list: List of message lists (one per request)
            max_concurrent: Maximum concurrent requests (default: 5)
            use_cache: Whether to use cache for requests
            **kwargs: Additional generation parameters for all requests

        Yields:
            Tuples of (index, response_or_exception) as they complete

        Example:
            ```python
            async for idx, result in client.batch_generate_iter(prompts):
                if isinstance(result, Exception):
                    print(f"Request {idx} failed: {result}")
                else:
                    print(f"Request {idx}: {result.content[:50]}...")
            ```
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        queue: asyncio.Queue[tuple[int, Response | Exception] | None] = asyncio.Queue()

        async def process_one(index: int, messages: list[Message]) -> None:
            async with semaphore:
                try:
                    response = await self.agenerate(messages, use_cache=use_cache, **kwargs)
                    await queue.put((index, response))
                except Exception as e:
                    await queue.put((index, e))

        async def producer() -> None:
            tasks = [
                asyncio.create_task(process_one(i, msgs)) for i, msgs in enumerate(messages_list)
            ]
            await asyncio.gather(*tasks)
            await queue.put(None)  # Signal completion

        producer_task = asyncio.create_task(producer())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass

    # Parameters to exclude from cache key (don't affect output content)
    _CACHE_EXCLUDE_PARAMS = frozenset(
        {
            "stream",  # Delivery method, not content
            "stream_options",  # Streaming configuration
            "timeout",  # Network timeout
            "user",  # User identifier for logging
            "n",  # Number of completions (we cache single responses)
            "_original_response_format",  # Internal tracking parameter
        }
    )

    def _compute_cache_key(
        self,
        messages: list[Message],
        params: dict[str, Any],
    ) -> str:
        """Compute deterministic cache key from request parameters.

        Includes all parameters that affect output content, excluding only
        parameters related to delivery method or internal tracking.

        Args:
            messages: Conversation messages
            params: Generation parameters

        Returns:
            Cache key string
        """
        # Build cache data structure
        cache_data = {
            "provider": self._get_provider_name(),
            "model": self.model,
            "messages": [
                {
                    "role": m.role,
                    "content": self._hash_content(m.content),
                }
                for m in messages
            ],
            # Include all params except those that don't affect output
            "params": {
                k: self._normalize_param_for_cache(k, v)
                for k, v in params.items()
                if k not in self._CACHE_EXCLUDE_PARAMS
            },
        }

        return CacheManager.generate_key(cache_data)

    def _normalize_param_for_cache(self, key: str, value: Any) -> Any:
        """Normalize parameter values for deterministic cache key generation.

        Handles special cases like Pydantic models and complex objects.

        Args:
            key: Parameter name
            value: Parameter value

        Returns:
            JSON-serializable representation of the value
        """
        # Handle response_format which may be a Pydantic model class
        if key == "response_format" and isinstance(value, type) and issubclass(value, BaseModel):
            # Use the schema as the cache key component
            return {"type": "json_schema", "schema": value.model_json_schema()}

        # Handle tools/functions which may contain complex objects
        if key in ("tools", "functions") and isinstance(value, list):
            # Tools are already dicts, but ensure consistent serialization
            return value

        return value

    def _hash_content(self, content: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        """Hash content for cache key generation.

        For text content, returns as-is.
        For multimodal content with images, hashes the image data.

        Args:
            content: Message content

        Returns:
            Hashable representation of content
        """
        if isinstance(content, str):
            return content

        # Handle multimodal content
        hashed = []
        for part in content:
            if part.get("type") == "image":
                # Hash image data for cache key
                import hashlib
                from typing import TYPE_CHECKING

                if TYPE_CHECKING:
                    from PIL import Image as PILImage  # pragma: no cover

                source = part.get("source", "")
                image_bytes: bytes | None = None
                image_hash = ""

                # Normalize PIL images to deterministic raw bytes so the cache
                # key is stable across processes (avoids __repr__ memory addresses).
                try:
                    from PIL import Image as PILImage
                except Exception:  # pragma: no cover - fallback when Pillow missing
                    PILImage = None

                if isinstance(source, bytes):
                    image_bytes = source
                elif PILImage is not None and isinstance(source, PILImage.Image):
                    normalized = source.convert("RGBA")
                    hasher = hashlib.sha256()
                    hasher.update(normalized.mode.encode())
                    hasher.update(str(normalized.size).encode())
                    hasher.update(normalized.tobytes())
                    image_hash = hasher.hexdigest()[:16]
                elif isinstance(source, str):
                    image_hash = source[:100]
                else:
                    image_hash = repr(source)[:100]

                if image_bytes is not None:
                    image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
                hashed.append({"type": "image", "hash": image_hash})
            else:
                hashed.append(part)

        return hashed

    @abstractmethod
    def _get_provider_name(self) -> str:
        """Get the provider name for this client.

        Returns:
            Provider name (e.g., 'openai', 'anthropic')
        """
        pass

    def _parse_structured_output(
        self,
        response: Response[Any],
        response_format: type[T],
    ) -> Response[T]:
        """Parse response content as structured output.

        Takes a Response with JSON content and parses it into the provided
        Pydantic model, returning a new Response with the parsed field populated.

        Args:
            response: Response object with JSON content
            response_format: Pydantic model class to parse content into

        Returns:
            New Response object with parsed field populated

        Raises:
            ValidationError: If JSON parsing or Pydantic validation fails
        """
        try:
            parsed_data = json.loads(response.content)
            parsed_model = response_format(**parsed_data)
            # Create new response with parsed data
            return Response(
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
                f"Failed to parse structured output: {str(e)}",
                raw_response=response.raw,
            ) from e

    def _track_request(
        self,
        raw_response: dict[str, Any],
        duration: float,
    ) -> TrackingData | None:
        """Track carbon and cost for a request.

        Args:
            raw_response: Raw API response
            duration: Request duration in seconds

        Returns:
            TrackingData if tracking is enabled, None otherwise
        """
        # Will be implemented by tracking mixins
        return None

    def clear_cache(self) -> None:
        """Clear all cached responses."""
        if self.cache:
            self.cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if self.cache:
            return self.cache.stats()
        return {"enabled": False}

    @property
    def cumulative_tracking(self) -> CumulativeTracking:
        """Get cumulative tracking data for this client's lifecycle.

        Returns:
            CumulativeTracking object with aggregated emissions, costs, and tokens
            separated into API requests vs cached requests, and usage vs embodied phases.

        Example:
            ```python
            client = OpenAIClient(model="gpt-4.1", cache_dir="./cache")

            # Make some requests
            response1 = client.generate([Message(role="user", content="Hello")])
            response2 = client.generate([Message(role="user", content="Hello")])  # cached

            # Access cumulative data
            tracking = client.cumulative_tracking
            print(f"Total GWP: {tracking.total_gwp_kgco2eq} kgCO2eq")
            print(f"API GWP: {tracking.api_gwp_kgco2eq} kgCO2eq")
            print(f"Avoided emissions: {tracking.emissions_avoided_kgco2eq} kgCO2eq")
            print(f"Usage phase: {tracking.total_gwp_usage_kgco2eq} kgCO2eq")
            print(f"Embodied phase: {tracking.total_gwp_embodied_kgco2eq} kgCO2eq")
            ```
        """
        return self._cumulative_tracking

    def get_cumulative_tracking(self) -> CumulativeTracking:
        """Get cumulative tracking data for this client's lifecycle.

        This is an alias for the cumulative_tracking property.

        Returns:
            CumulativeTracking object with aggregated emissions, costs, and tokens.
        """
        return self._cumulative_tracking

    def reset_cumulative_tracking(self) -> None:
        """Reset cumulative tracking data to zero.

        Use this to start a new tracking period without creating a new client.

        Example:
            ```python
            client = OpenAIClient(model="gpt-4.1")

            # First batch of work
            response1 = client.generate([Message(role="user", content="Task 1")])
            print(f"Batch 1 emissions: {client.cumulative_tracking.total_gwp_kgco2eq}")

            # Reset and start fresh
            client.reset_cumulative_tracking()

            # Second batch of work
            response2 = client.generate([Message(role="user", content="Task 2")])
            print(f"Batch 2 emissions: {client.cumulative_tracking.total_gwp_kgco2eq}")
            ```
        """
        self._cumulative_tracking.reset()

    def export_boamps_report(
        self,
        output_path: str | Path,
        *,
        publisher_name: str | None = None,
        task_description: str | None = None,
        task_family: str = "textGeneration",
        include_summary: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Export a BoAmps-compliant energy consumption report.

        Generates a standardized JSON report following the BoAmps format
        for energy consumption of LLM inference tasks.

        See: https://github.com/Boavizta/BoAmps

        Args:
            output_path: Path where to save the JSON report.
            publisher_name: Name of the organization publishing the report.
            task_description: Free-form description of the task.
            task_family: Family of the task (textGeneration, imageClassification, etc.).
            include_summary: Whether to print a summary to console.
            **kwargs: Additional arguments passed to BoAmpsReporter.

        Returns:
            BoAmpsReport object containing all energy consumption data.

        Example:
            ```python
            client = OpenAIClient(model="gpt-4.1", cache_dir="./cache")

            # Make some requests
            for i in range(10):
                client.generate([Message(role="user", content=f"Question {i}")])

            # Export BoAmps report
            report = client.export_boamps_report(
                "energy_report.json",
                publisher_name="My Organization",
                task_description="LLM inference for question answering",
            )

            # Access report data
            print(f"Total energy: {report.measures[0].powerConsumption} kWh")
            ```
        """
        from seeds_clients.tracking.boamps_reporter import export_boamps_report

        return export_boamps_report(
            self,
            output_path,
            publisher_name=publisher_name,
            task_description=task_description,
            task_family=task_family,
            include_summary=include_summary,
            **kwargs,
        )

    def close(self) -> None:
        """Close client and clean up resources."""
        if self.cache:
            self.cache.close()

    async def aclose(self) -> None:
        """Close client asynchronously and clean up resources.

        Subclasses should override this to close async HTTP clients.
        """
        self.close()

    def __enter__(self) -> "BaseClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    async def __aenter__(self) -> "BaseClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.aclose()

    def __del__(self) -> None:
        """Ensure underlying resources are cleaned up on GC."""
        try:
            self.close()
        except Exception:
            # Avoid raising during interpreter shutdown
            pass
