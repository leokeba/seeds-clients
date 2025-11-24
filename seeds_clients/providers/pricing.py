"""Pricing information for LLM providers."""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Load pricing data from JSON file
_PRICING_FILE = Path(__file__).parent / "pricing_data.json"


def _load_pricing_data() -> dict[str, Any]:
    """Load pricing data from JSON file.

    Returns:
        Dictionary with metadata and provider -> model -> pricing structure
    """
    with open(_PRICING_FILE) as f:
        data: dict[str, Any] = json.load(f)
        return data


def _extract_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata from pricing data."""
    metadata: dict[str, Any] = data.get("_metadata", {})
    return metadata


def _extract_providers(data: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    """Extract provider pricing data (excluding metadata)."""
    return {k: v for k, v in data.items() if k != "_metadata"}


# Cache the pricing data
_PRICING_DATA_RAW = _load_pricing_data()
_METADATA = _extract_metadata(_PRICING_DATA_RAW)
_PRICING_DATA = _extract_providers(_PRICING_DATA_RAW)

# OpenAI pricing per 1M tokens
OPENAI_PRICING: dict[str, dict[str, float]] = _PRICING_DATA["openai"]


def check_pricing_update_needed() -> tuple[bool, str]:
    """Check if pricing data needs to be updated based on last_updated date.

    Returns:
        tuple: (needs_update: bool, message: str)
    """
    if not _METADATA:
        return False, "No metadata found in pricing data"

    last_updated_str = _METADATA.get("last_updated")
    if not last_updated_str:
        return False, "No last_updated date found in metadata"

    update_frequency_days = _METADATA.get("update_frequency_days", 30)

    try:
        last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d")
        days_since_update = (datetime.now() - last_updated).days

        if days_since_update >= update_frequency_days:
            return True, (
                f"Pricing data is {days_since_update} days old "
                f"(last updated: {last_updated_str}). "
                f"Consider updating from {_METADATA.get('source', 'OpenAI pricing page')}."
            )
        return False, f"Pricing data is current (last updated: {last_updated_str})"
    except ValueError as e:
        logger.warning(f"Invalid date format in metadata: {e}")
        return False, f"Invalid date format: {last_updated_str}"


def warn_if_update_needed() -> None:
    """Issue a warning if pricing data needs to be updated."""
    needs_update, message = check_pricing_update_needed()
    if needs_update:
        warnings.warn(message, UserWarning, stacklevel=2)


def get_pricing_metadata() -> dict[str, Any]:
    """Get pricing metadata including last update date and source.

    Returns:
        Dictionary with metadata fields: last_updated, update_frequency_days, source
    """
    return _METADATA.copy()


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: dict[str, dict[str, float]] | None = None,
    provider: str = "openai",
) -> float:
    """Calculate cost for a given model and token usage.

    Args:
        model: Model identifier
        prompt_tokens: Number of input/prompt tokens
        completion_tokens: Number of output/completion tokens
        pricing: Optional custom pricing dict (defaults to provider's pricing)
        provider: Provider name (default: "openai")

    Returns:
        Total cost in USD

    Raises:
        ValueError: If model not found in pricing data
    """
    if pricing is None:
        if provider not in _PRICING_DATA:
            raise ValueError(
                f"Provider '{provider}' not found in pricing data. "
                f"Available providers: {', '.join(sorted(_PRICING_DATA.keys()))}"
            )
        pricing = _PRICING_DATA[provider]

    if model not in pricing:
        raise ValueError(
            f"Model '{model}' not found in pricing data. "
            f"Available models: {', '.join(sorted(pricing.keys()))}"
        )

    model_pricing = pricing[model]
    input_cost = (prompt_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * model_pricing["output"]

    return input_cost + output_cost


def get_model_pricing(
    model: str,
    pricing: dict[str, dict[str, float]] | None = None,
    provider: str = "openai",
) -> dict[str, Any]:
    """Get pricing information for a specific model.

    Args:
        model: Model identifier
        pricing: Optional custom pricing dict (defaults to provider's pricing)
        provider: Provider name (default: "openai")

    Returns:
        Dictionary with input and output pricing per 1M tokens

    Raises:
        ValueError: If model not found in pricing data
    """
    if pricing is None:
        if provider not in _PRICING_DATA:
            raise ValueError(
                f"Provider '{provider}' not found in pricing data. "
                f"Available providers: {', '.join(sorted(_PRICING_DATA.keys()))}"
            )
        pricing = _PRICING_DATA[provider]

    if model not in pricing:
        raise ValueError(
            f"Model '{model}' not found in pricing data. "
            f"Available models: {', '.join(sorted(pricing.keys()))}"
        )

    return pricing[model]


def reload_pricing_data() -> None:
    """Reload pricing data from JSON file.

    Useful if pricing file is updated at runtime.
    """
    global _PRICING_DATA_RAW, _METADATA, _PRICING_DATA, OPENAI_PRICING
    _PRICING_DATA_RAW = _load_pricing_data()
    _METADATA = _extract_metadata(_PRICING_DATA_RAW)
    _PRICING_DATA = _extract_providers(_PRICING_DATA_RAW)
    OPENAI_PRICING = _PRICING_DATA["openai"]
    logger.info("Pricing data reloaded from file")


# Check for updates when module is imported
warn_if_update_needed()
