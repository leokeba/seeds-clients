"""Tests for custom exceptions."""

import pytest

from seeds_clients.core.exceptions import (
    CacheError,
    ConfigurationError,
    ProviderError,
    SeedsClientError,
    TrackingError,
    ValidationError,
)


class TestExceptions:
    """Test custom exception hierarchy."""

    def test_base_exception(self) -> None:
        """Test base SeedsClientError."""
        error = SeedsClientError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_cache_error(self) -> None:
        """Test CacheError."""
        error = CacheError("Cache failed")
        assert isinstance(error, SeedsClientError)
        assert str(error) == "Cache failed"

    def test_provider_error(self) -> None:
        """Test ProviderError with context."""
        error = ProviderError("API call failed", provider="openai", status_code=429)
        assert isinstance(error, SeedsClientError)
        assert error.provider == "openai"
        assert error.status_code == 429
        assert "API call failed" in str(error)

    def test_provider_error_without_status(self) -> None:
        """Test ProviderError without status code."""
        error = ProviderError("Generic error", provider="test")
        assert error.provider == "test"
        assert error.status_code is None

    def test_validation_error(self) -> None:
        """Test ValidationError."""
        error = ValidationError("Invalid input")
        assert isinstance(error, SeedsClientError)
        assert str(error) == "Invalid input"

        # Raw response context is optional
        raw = {"foo": "bar"}
        error_with_raw = ValidationError("Invalid input", raw_response=raw)
        assert error_with_raw.raw_response == raw

    def test_tracking_error(self) -> None:
        """Test TrackingError."""
        error = TrackingError("Tracking failed")
        assert isinstance(error, SeedsClientError)
        assert str(error) == "Tracking failed"

    def test_configuration_error(self) -> None:
        """Test ConfigurationError."""
        error = ConfigurationError("Missing API key")
        assert isinstance(error, SeedsClientError)
        assert str(error) == "Missing API key"

    def test_exception_can_be_raised(self) -> None:
        """Test exceptions can be raised and caught."""
        with pytest.raises(ProviderError) as exc_info:
            raise ProviderError("Test", provider="test", status_code=500)

        assert exc_info.value.provider == "test"
        assert exc_info.value.status_code == 500

    def test_inheritance_chain(self) -> None:
        """Test exception inheritance."""
        # All custom exceptions should inherit from SeedsClientError
        exceptions = [
            CacheError("test"),
            ProviderError("test", provider="test"),
            ValidationError("test"),
            TrackingError("test"),
            ConfigurationError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, SeedsClientError)
            assert isinstance(exc, Exception)
