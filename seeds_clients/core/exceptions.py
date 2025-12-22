"""Custom exceptions for seeds-clients."""

import json
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
    """Exception raised for validation errors with optional raw payload."""

    def __init__(self, message: str, raw_response: Any | None = None) -> None:
        self.raw_response = raw_response

        if raw_response is None:
            super().__init__(message)
            return

        preview = raw_response
        if isinstance(preview, (dict, list)):
            try:
                preview = json.dumps(preview, ensure_ascii=True)
            except Exception:
                preview = str(preview)
        if isinstance(preview, str) and len(preview) > 2000:
            preview = f"{preview[:2000]}...[truncated]"

        super().__init__(f"{message}\nRaw response: {preview}")


class TrackingError(SeedsClientError):
    """Exception raised for tracking-related errors."""

    pass


class ConfigurationError(SeedsClientError):
    """Exception raised for configuration errors."""

    pass
