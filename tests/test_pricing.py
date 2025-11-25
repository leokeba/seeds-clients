"""Tests for pricing module."""

import warnings
from datetime import datetime

import pytest

from seeds_clients.utils.pricing import (
    OPENAI_PRICING,
    calculate_cost,
    check_pricing_update_needed,
    get_model_pricing,
    get_pricing_metadata,
    reload_pricing_data,
    warn_if_update_needed,
)


class TestCalculateCost:
    """Tests for calculate_cost function."""

    def test_gpt4o_cost(self) -> None:
        """Test cost calculation for GPT-4.1."""
        cost = calculate_cost(
            model="gpt-4.1",
            prompt_tokens=1_000,
            completion_tokens=500,
        )
        # gpt-4.1: $2.00 per 1M input, $8.00 per 1M output
        # (1000 / 1M * $2.00) + (500 / 1M * $8.00)
        # = $0.002 + $0.004 = $0.006
        assert cost == pytest.approx(0.006)

    def test_gpt4o_mini_cost(self) -> None:
        """Test cost calculation for GPT-4.1 mini."""
        cost = calculate_cost(
            model="gpt-4.1-mini",
            prompt_tokens=10_000,
            completion_tokens=5_000,
        )
        # gpt-4.1-mini: $0.40 per 1M input, $1.60 per 1M output
        # (10000 / 1M * $0.40) + (5000 / 1M * $1.60)
        # = $0.004 + $0.008 = $0.012
        assert cost == pytest.approx(0.012)

    def test_gpt35_turbo_cost(self) -> None:
        """Test cost calculation for GPT-3.5 Turbo."""
        cost = calculate_cost(
            model="gpt-3.5-turbo",
            prompt_tokens=100_000,
            completion_tokens=50_000,
        )
        # (100000 / 1M * $0.50) + (50000 / 1M * $1.50)
        # = $0.05 + $0.075 = $0.125
        assert cost == pytest.approx(0.125)

    def test_large_token_count(self) -> None:
        """Test cost calculation with large token counts."""
        cost = calculate_cost(
            model="gpt-4.1",
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
        )
        # gpt-4.1: $2.00 per 1M input, $8.00 per 1M output
        # (1M / 1M * $2.00) + (500K / 1M * $8.00)
        # = $2.00 + $4.00 = $6.00
        assert cost == pytest.approx(6.00)

    def test_zero_tokens(self) -> None:
        """Test cost calculation with zero tokens."""
        cost = calculate_cost(
            model="gpt-4.1",
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert cost == 0.0

    def test_model_not_found(self) -> None:
        """Test error when model not in pricing data."""
        with pytest.raises(ValueError, match="Model 'invalid-model' not found"):
            calculate_cost(
                model="invalid-model",
                prompt_tokens=1000,
                completion_tokens=500,
            )

    def test_custom_pricing(self) -> None:
        """Test cost calculation with custom pricing."""
        custom_pricing = {
            "custom-model": {
                "input": 1.0,
                "output": 2.0,
            }
        }
        cost = calculate_cost(
            model="custom-model",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            pricing=custom_pricing,
        )
        # (1M / 1M * $1.00) + (1M / 1M * $2.00) = $3.00
        assert cost == pytest.approx(3.0)


class TestGetModelPricing:
    """Tests for get_model_pricing function."""

    def test_get_gpt4o_pricing(self) -> None:
        """Test getting pricing for GPT-4.1."""
        pricing = get_model_pricing("gpt-4.1")
        assert pricing == {"input": 2.00, "output": 8.00}

    def test_get_gpt4o_mini_pricing(self) -> None:
        """Test getting pricing for GPT-4.1 mini."""
        pricing = get_model_pricing("gpt-4.1-mini")
        assert pricing == {"input": 0.40, "output": 1.60}

    def test_get_gpt35_turbo_pricing(self) -> None:
        """Test getting pricing for GPT-3.5 Turbo."""
        pricing = get_model_pricing("gpt-3.5-turbo")
        assert pricing == {"input": 0.50, "output": 1.50}

    def test_model_not_found(self) -> None:
        """Test error when model not in pricing data."""
        with pytest.raises(ValueError, match="Model 'invalid-model' not found"):
            get_model_pricing("invalid-model")

    def test_custom_pricing(self) -> None:
        """Test getting pricing from custom dict."""
        custom_pricing = {
            "custom-model": {
                "input": 1.0,
                "output": 2.0,
            }
        }
        pricing = get_model_pricing("custom-model", pricing=custom_pricing)
        assert pricing == {"input": 1.0, "output": 2.0}


class TestOpenAIPricing:
    """Tests for OPENAI_PRICING data structure."""

    def test_all_models_have_input_output(self) -> None:
        """Test that all models have both input and output pricing."""
        for model, pricing in OPENAI_PRICING.items():
            assert "input" in pricing, f"Model {model} missing 'input' pricing"
            assert "output" in pricing, f"Model {model} missing 'output' pricing"
            assert isinstance(pricing["input"], (int, float))
            assert isinstance(pricing["output"], (int, float))
            assert pricing["input"] >= 0
            assert pricing["output"] >= 0

    def test_common_models_present(self) -> None:
        """Test that common models are present in pricing data."""
        common_models = [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ]
        for model in common_models:
            assert model in OPENAI_PRICING, f"Common model {model} not in pricing data"


class TestPricingDataManagement:
    """Tests for pricing data loading and reloading."""

    def test_reload_pricing_data(self) -> None:
        """Test that pricing data can be reloaded."""
        # Get initial pricing
        initial_pricing = get_model_pricing("gpt-4.1")

        # Reload pricing data
        reload_pricing_data()

        # Should still have the same pricing
        reloaded_pricing = get_model_pricing("gpt-4.1")
        assert reloaded_pricing == initial_pricing

    def test_pricing_file_structure(self) -> None:
        """Test that pricing file has correct structure."""
        # Should have openai provider
        from seeds_clients.utils.pricing import _PRICING_DATA

        assert "openai" in _PRICING_DATA
        assert isinstance(_PRICING_DATA["openai"], dict)

        # Each model should have input and output pricing
        for model, pricing in _PRICING_DATA["openai"].items():
            assert "input" in pricing, f"Model {model} missing 'input' pricing"
            assert "output" in pricing, f"Model {model} missing 'output' pricing"
            assert isinstance(pricing["input"], (int, float))
            assert isinstance(pricing["output"], (int, float))


class TestPricingMetadata:
    """Tests for pricing metadata functionality."""

    def test_get_pricing_metadata(self) -> None:
        """Test that metadata can be retrieved."""
        metadata = get_pricing_metadata()
        assert isinstance(metadata, dict)
        assert "last_updated" in metadata
        assert "update_frequency_days" in metadata
        assert "source" in metadata

    def test_metadata_has_valid_date(self) -> None:
        """Test that last_updated is a valid date."""
        metadata = get_pricing_metadata()
        last_updated_str = metadata["last_updated"]

        # Should be parseable as a date
        try:
            datetime.strptime(last_updated_str, "%Y-%m-%d")
        except ValueError:
            pytest.fail(f"Invalid date format: {last_updated_str}")

    def test_check_pricing_update_needed_current(self) -> None:
        """Test update check when pricing is current."""
        metadata = get_pricing_metadata()
        last_updated = datetime.strptime(metadata["last_updated"], "%Y-%m-%d")
        days_since_update = (datetime.now() - last_updated).days

        # Only test if pricing is actually current
        if days_since_update < metadata["update_frequency_days"]:
            needs_update, message = check_pricing_update_needed()
            assert not needs_update
            assert "current" in message.lower()

    def test_check_pricing_update_needed_structure(self) -> None:
        """Test that update check returns correct structure."""
        needs_update, message = check_pricing_update_needed()
        assert isinstance(needs_update, bool)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_warn_if_update_needed_warning(self) -> None:
        """Test that warning is issued if update is needed."""
        # Get current metadata
        metadata = get_pricing_metadata()
        last_updated = datetime.strptime(metadata["last_updated"], "%Y-%m-%d")
        days_since_update = (datetime.now() - last_updated).days

        # Only test warning if pricing is actually outdated
        if days_since_update >= metadata["update_frequency_days"]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                warn_if_update_needed()
                assert len(w) >= 1
                assert issubclass(w[0].category, UserWarning)
                assert "days old" in str(w[0].message)
