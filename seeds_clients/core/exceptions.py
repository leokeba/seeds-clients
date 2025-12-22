"""Custom exceptions for seeds-clients."""

from typing import Any


class SeedsClientError(Exception):
    """Base exception for all seeds-clients errors."""

    pass


class CacheError(SeedsClientError):
    """Exception raised for cache-related errors."""

    pass


class ProviderError(SeedsClientError):
    """Exception raised for provider API errors."""

    def __init__(self, message: str, provider: str, status_code: int | None = None) -> None:
        """Initialize provider error with context."""
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] {message}")


class ValidationError(SeedsClientError):
    """Exception raised for validation errors."""

    def __init__(self, message: str, raw_response: dict[str, Any] | None = None) -> None:
        """Initialize validation error with optional raw response context."""
        self.raw_response = raw_response
        super().__init__(message)


class TrackingError(SeedsClientError):
    """Exception raised for tracking-related errors."""

    pass


class ConfigurationError(SeedsClientError):
    """Exception raised for configuration errors."""

    pass
