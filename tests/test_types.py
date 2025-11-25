"""Tests for core type definitions."""

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from seeds_clients.core.types import Message, Response, TrackingData, Usage


class TestMessage:
    """Tests for Message type."""

    def test_text_message(self) -> None:
        """Test creating a simple text message."""
        msg = Message(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"

    def test_multimodal_message(self) -> None:
        """Test creating a multimodal message."""
        content = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "source": "https://example.com/image.jpg"},
        ]
        msg = Message(role="user", content=content)
        assert msg.role == "user"
        assert len(msg.content) == 2

    def test_invalid_role(self) -> None:
        """Test that invalid roles are rejected."""
        with pytest.raises(PydanticValidationError):
            Message(role="invalid", content="test")  # type: ignore[arg-type]

    def test_immutable(self) -> None:
        """Test that Message is immutable."""
        msg = Message(role="user", content="test")
        with pytest.raises(PydanticValidationError):
            msg.role = "assistant"


class TestUsage:
    """Tests for Usage type."""

    def test_usage_with_total(self) -> None:
        """Test creating Usage with explicit total."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_usage_auto_total(self) -> None:
        """Test that total_tokens is calculated automatically."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        assert usage.total_tokens == 30

    def test_negative_tokens(self) -> None:
        """Test that negative token counts are rejected."""
        with pytest.raises(PydanticValidationError):
            Usage(prompt_tokens=-1, completion_tokens=20)


class TestTrackingData:
    """Tests for TrackingData type."""

    def test_minimal_tracking_data(self) -> None:
        """Test creating minimal tracking data."""
        tracking = TrackingData(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            cost_usd=0.002,
            prompt_tokens=10,
            completion_tokens=20,
            provider="openai",
            model="gpt-4.1",
            tracking_method="ecologits",
            duration_seconds=1.5,
        )
        assert tracking.energy_kwh == 0.001
        assert tracking.gwp_kgco2eq == 0.0005
        assert tracking.cost_usd == 0.002
        assert tracking.provider == "openai"
        assert tracking.model == "gpt-4.1"

    def test_tracking_with_optional_fields(self) -> None:
        """Test tracking data with optional fields."""
        tracking = TrackingData(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            adpe_kgsbeq=0.00001,
            pe_mj=0.05,
            cost_usd=0.002,
            prompt_tokens=10,
            completion_tokens=20,
            provider="openai",
            model="gpt-4.1",
            tracking_method="ecologits",
            electricity_mix_zone="FRA",
            duration_seconds=1.5,
        )
        assert tracking.adpe_kgsbeq == 0.00001
        assert tracking.pe_mj == 0.05
        assert tracking.electricity_mix_zone == "FRA"

    def test_tracking_timestamp(self) -> None:
        """Test that measured_at timestamp is set automatically."""
        tracking = TrackingData(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            cost_usd=0.002,
            prompt_tokens=10,
            completion_tokens=20,
            provider="openai",
            model="gpt-4.1",
            tracking_method="ecologits",
            duration_seconds=1.5,
        )
        assert isinstance(tracking.measured_at, datetime)


class TestResponse:
    """Tests for Response type."""

    def test_basic_response(self) -> None:
        """Test creating a basic response."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response: Response[Any] = Response(
            content="Hello!",
            usage=usage,
            model="gpt-4.1",
            raw={"choices": [{"message": {"content": "Hello!"}}]},
        )
        assert response.content == "Hello!"
        assert response.usage.total_tokens == 30
        assert response.model == "gpt-4.1"
        assert not response.cached

    def test_response_with_tracking(self) -> None:
        """Test response with tracking data."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        tracking = TrackingData(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            cost_usd=0.002,
            prompt_tokens=10,
            completion_tokens=20,
            provider="openai",
            model="gpt-4.1",
            tracking_method="ecologits",
            duration_seconds=1.5,
        )
        response: Response[Any] = Response(
            content="Hello!",
            usage=usage,
            model="gpt-4.1",
            raw={"choices": [{"message": {"content": "Hello!"}}]},
            tracking=tracking,
        )
        assert response.tracking is not None
        assert response.tracking.cost_usd == 0.002

    def test_cached_response(self) -> None:
        """Test response marked as cached."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        response: Response[Any] = Response(
            content="Hello!",
            usage=usage,
            model="gpt-4.1",
            raw={"choices": [{"message": {"content": "Hello!"}}]},
            cached=True,
        )
        assert response.cached
