"""Tests for core type definitions."""

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from seeds_clients.core.types import (
    CumulativeTracking,
    Message,
    Response,
    TrackingData,
    Usage,
)


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


class TestCumulativeTracking:
    """Tests for CumulativeTracking type."""

    def test_default_values(self) -> None:
        """Test default values are all zero."""
        tracking = CumulativeTracking()
        
        # Request counts
        assert tracking.api_request_count == 0
        assert tracking.cached_request_count == 0
        
        # API metrics
        assert tracking.api_energy_kwh == 0.0
        assert tracking.api_gwp_kgco2eq == 0.0
        assert tracking.api_cost_usd == 0.0
        assert tracking.api_prompt_tokens == 0
        assert tracking.api_completion_tokens == 0
        
        # Cached metrics
        assert tracking.cached_energy_kwh == 0.0
        assert tracking.cached_gwp_kgco2eq == 0.0
        assert tracking.cached_cost_usd == 0.0

    def test_total_properties(self) -> None:
        """Test computed total properties."""
        tracking = CumulativeTracking(
            api_request_count=3,
            cached_request_count=2,
            api_energy_kwh=0.003,
            cached_energy_kwh=0.002,
            api_gwp_kgco2eq=0.0015,
            cached_gwp_kgco2eq=0.001,
            api_cost_usd=0.06,
            cached_cost_usd=0.04,
            api_prompt_tokens=300,
            cached_prompt_tokens=200,
            api_completion_tokens=150,
            cached_completion_tokens=100,
        )
        
        assert tracking.total_request_count == 5
        assert tracking.total_energy_kwh == pytest.approx(0.005)
        assert tracking.total_gwp_kgco2eq == pytest.approx(0.0025)
        assert tracking.total_cost_usd == pytest.approx(0.10)
        assert tracking.total_prompt_tokens == 500
        assert tracking.total_completion_tokens == 250

    def test_cache_hit_rate(self) -> None:
        """Test cache hit rate calculation."""
        # No requests
        tracking = CumulativeTracking()
        assert tracking.cache_hit_rate == 0.0
        
        # Some cached
        tracking = CumulativeTracking(
            api_request_count=3,
            cached_request_count=2,
        )
        assert tracking.cache_hit_rate == pytest.approx(0.4)
        
        # All cached
        tracking = CumulativeTracking(
            api_request_count=0,
            cached_request_count=5,
        )
        assert tracking.cache_hit_rate == 1.0

    def test_emissions_avoided(self) -> None:
        """Test emissions avoided property."""
        tracking = CumulativeTracking(
            cached_gwp_kgco2eq=0.005,
        )
        assert tracking.emissions_avoided_kgco2eq == 0.005

    def test_usage_embodied_breakdown(self) -> None:
        """Test usage vs embodied phase breakdown."""
        tracking = CumulativeTracking(
            api_gwp_usage_kgco2eq=0.0008,
            api_gwp_embodied_kgco2eq=0.0002,
            cached_gwp_usage_kgco2eq=0.0004,
            cached_gwp_embodied_kgco2eq=0.0001,
        )
        
        assert tracking.total_gwp_usage_kgco2eq == pytest.approx(0.0012)
        assert tracking.total_gwp_embodied_kgco2eq == pytest.approx(0.0003)

    def test_accumulate_api_request(self) -> None:
        """Test accumulating an API request."""
        tracking = CumulativeTracking()
        
        request_tracking = TrackingData(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            cost_usd=0.02,
            prompt_tokens=100,
            completion_tokens=50,
            provider="openai",
            model="gpt-4.1",
            tracking_method="ecologits",
            duration_seconds=1.5,
            gwp_usage_kgco2eq=0.0003,
            gwp_embodied_kgco2eq=0.0002,
            energy_usage_kwh=0.0008,
        )
        
        tracking.accumulate(request_tracking, cached=False)
        
        assert tracking.api_request_count == 1
        assert tracking.api_energy_kwh == 0.001
        assert tracking.api_gwp_kgco2eq == 0.0005
        assert tracking.api_cost_usd == 0.02
        assert tracking.api_prompt_tokens == 100
        assert tracking.api_completion_tokens == 50
        assert tracking.api_gwp_usage_kgco2eq == 0.0003
        assert tracking.api_gwp_embodied_kgco2eq == 0.0002
        assert tracking.api_energy_usage_kwh == 0.0008
        
        # Cached should be unchanged
        assert tracking.cached_request_count == 0
        assert tracking.cached_gwp_kgco2eq == 0.0

    def test_accumulate_cached_request(self) -> None:
        """Test accumulating a cached request."""
        tracking = CumulativeTracking()
        
        request_tracking = TrackingData(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            cost_usd=0.02,
            prompt_tokens=100,
            completion_tokens=50,
            provider="openai",
            model="gpt-4.1",
            tracking_method="ecologits",
            duration_seconds=1.5,
            gwp_usage_kgco2eq=0.0003,
            gwp_embodied_kgco2eq=0.0002,
            energy_usage_kwh=0.0008,
        )
        
        tracking.accumulate(request_tracking, cached=True)
        
        assert tracking.cached_request_count == 1
        assert tracking.cached_energy_kwh == 0.001
        assert tracking.cached_gwp_kgco2eq == 0.0005
        assert tracking.cached_cost_usd == 0.02
        assert tracking.cached_prompt_tokens == 100
        assert tracking.cached_completion_tokens == 50
        assert tracking.cached_gwp_usage_kgco2eq == 0.0003
        assert tracking.cached_gwp_embodied_kgco2eq == 0.0002
        assert tracking.cached_energy_usage_kwh == 0.0008
        
        # API should be unchanged
        assert tracking.api_request_count == 0
        assert tracking.api_gwp_kgco2eq == 0.0

    def test_accumulate_multiple_requests(self) -> None:
        """Test accumulating multiple requests."""
        tracking = CumulativeTracking()
        
        # Create tracking data for requests
        def make_tracking(energy: float, gwp: float, cost: float) -> TrackingData:
            return TrackingData(
                energy_kwh=energy,
                gwp_kgco2eq=gwp,
                cost_usd=cost,
                prompt_tokens=100,
                completion_tokens=50,
                provider="openai",
                model="gpt-4.1",
                tracking_method="ecologits",
                duration_seconds=1.0,
            )
        
        # Accumulate API requests
        tracking.accumulate(make_tracking(0.001, 0.0005, 0.02), cached=False)
        tracking.accumulate(make_tracking(0.002, 0.001, 0.03), cached=False)
        
        # Accumulate cached request
        tracking.accumulate(make_tracking(0.001, 0.0005, 0.02), cached=True)
        
        assert tracking.api_request_count == 2
        assert tracking.cached_request_count == 1
        assert tracking.total_request_count == 3
        
        assert tracking.api_energy_kwh == pytest.approx(0.003)
        assert tracking.api_gwp_kgco2eq == pytest.approx(0.0015)
        assert tracking.api_cost_usd == pytest.approx(0.05)
        
        assert tracking.cached_energy_kwh == pytest.approx(0.001)
        assert tracking.cached_gwp_kgco2eq == pytest.approx(0.0005)
        
        assert tracking.total_gwp_kgco2eq == pytest.approx(0.002)

    def test_accumulate_with_none_optional_fields(self) -> None:
        """Test accumulating when optional fields are None."""
        tracking = CumulativeTracking()
        
        # Tracking without optional breakdown fields
        request_tracking = TrackingData(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            cost_usd=0.02,
            prompt_tokens=100,
            completion_tokens=50,
            provider="openai",
            model="gpt-4.1",
            tracking_method="ecologits",
            duration_seconds=1.5,
            # gwp_usage_kgco2eq=None (default)
            # gwp_embodied_kgco2eq=None (default)
        )
        
        tracking.accumulate(request_tracking, cached=False)
        
        # Should not crash and should leave breakdown at 0
        assert tracking.api_gwp_usage_kgco2eq == 0.0
        assert tracking.api_gwp_embodied_kgco2eq == 0.0

    def test_reset(self) -> None:
        """Test resetting cumulative tracking."""
        tracking = CumulativeTracking(
            api_request_count=5,
            cached_request_count=3,
            api_energy_kwh=0.005,
            api_gwp_kgco2eq=0.0025,
            api_cost_usd=0.10,
            cached_energy_kwh=0.003,
            cached_gwp_kgco2eq=0.0015,
        )
        
        tracking.reset()
        
        assert tracking.api_request_count == 0
        assert tracking.cached_request_count == 0
        assert tracking.api_energy_kwh == 0.0
        assert tracking.api_gwp_kgco2eq == 0.0
        assert tracking.api_cost_usd == 0.0
        assert tracking.cached_energy_kwh == 0.0
        assert tracking.cached_gwp_kgco2eq == 0.0

    def test_repr(self) -> None:
        """Test string representation."""
        tracking = CumulativeTracking(
            api_request_count=3,
            cached_request_count=2,
            api_gwp_kgco2eq=0.0015,
            cached_gwp_kgco2eq=0.001,
            api_gwp_usage_kgco2eq=0.001,
            api_gwp_embodied_kgco2eq=0.0005,
            cached_gwp_usage_kgco2eq=0.0006,
            cached_gwp_embodied_kgco2eq=0.0004,
            api_cost_usd=0.05,
            cached_cost_usd=0.03,
        )
        
        repr_str = repr(tracking)
        assert "requests=5" in repr_str
        assert "api=3" in repr_str
        assert "cached=2" in repr_str
        assert "gwp=" in repr_str
        assert "usage=" in repr_str
        assert "embodied=" in repr_str
        assert "cost=$" in repr_str
