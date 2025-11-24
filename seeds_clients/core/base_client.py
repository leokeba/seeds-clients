"""Abstract base client for all LLM providers."""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from seeds_clients.core.cache import CacheManager
from seeds_clients.core.types import Message, Response, TrackingData


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
        cache_ttl_hours: float | None = 24,
        enable_tracking: bool = True,
        tracking_method: str = "ecologits",
        **kwargs: Any,
    ) -> None:
        """Initialize base client.

        Args:
            model: Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet')
            api_key: API key for the provider (can also come from env vars)
            cache_dir: Directory for cache storage (None = no caching)
            cache_ttl_hours: Cache time-to-live in hours (None = no expiration)
            enable_tracking: Whether to enable carbon/cost tracking
            tracking_method: Tracking method ('ecologits', 'codecarbon', 'none')
            **kwargs: Additional provider-specific arguments
        """
        self.model = model
        self.api_key = api_key
        self.enable_tracking = enable_tracking
        self.tracking_method = tracking_method
        self.kwargs = kwargs

        # Initialize cache if directory provided
        self.cache: CacheManager | None = None
        if cache_dir:
            self.cache = CacheManager(cache_dir, ttl_hours=cache_ttl_hours)

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
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Response:
        """Generate a response from the LLM.

        Main entry point for text generation with automatic caching and tracking.

        Args:
            messages: List of conversation messages
            use_cache: Whether to use cache for this request
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            Response object with generated text and metadata

        Raises:
            ProviderError: If API call fails
            ValidationError: If response is invalid
        """
        # Generate cache key
        cache_key = self._compute_cache_key(messages, kwargs)

        # Try cache first
        if use_cache and self.cache:
            cached_raw = self.cache.get(cache_key)
            if cached_raw:
                response = self._parse_response(cached_raw)
                response.cached = True
                return response

        # Call API with timing
        start_time = time.time()
        raw_response = self._call_api(messages, **kwargs)
        duration = time.time() - start_time

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
                "provider": self._get_provider_name(),
                "duration_seconds": duration,
            }
            self.cache.set(cache_key, raw_response, metadata)

        return response

    def _compute_cache_key(
        self,
        messages: list[Message],
        params: dict[str, Any],
    ) -> str:
        """Compute deterministic cache key from request parameters.

        Args:
            messages: Conversation messages
            params: Generation parameters

        Returns:
            Cache key string
        """
        # Build cache data structure
        cache_data = {
            "model": self.model,
            "messages": [
                {
                    "role": m.role,
                    "content": self._hash_content(m.content),
                }
                for m in messages
            ],
            "params": {
                k: v
                for k, v in params.items()
                if k in ["temperature", "top_p", "max_tokens", "frequency_penalty"]
            },
        }

        return CacheManager.generate_key(cache_data)

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

                source = part.get("source", "")
                if isinstance(source, bytes):
                    image_hash = hashlib.sha256(source).hexdigest()[:16]
                else:
                    image_hash = str(source)[:100]  # Use URL/path prefix
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

    def close(self) -> None:
        """Close client and clean up resources."""
        if self.cache:
            self.cache.close()

    def __enter__(self) -> "BaseClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
