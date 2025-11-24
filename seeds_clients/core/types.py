"""Core type definitions for seeds-clients."""

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

# Type variable for structured outputs
T = TypeVar("T", bound=BaseModel)


class Message(BaseModel):
    """A message in a conversation.

    Supports both text-only and multimodal (text + images) content.
    """

    role: Literal["system", "user", "assistant"] = Field(
        description="The role of the message sender"
    )
    content: str | list[dict[str, Any]] = Field(
        description="Message content - either plain text or list of content parts for multimodal"
    )

    model_config = {"frozen": True}


class Usage(BaseModel):
    """Token usage information for a request."""

    prompt_tokens: int = Field(ge=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(ge=0, description="Number of tokens in the completion")
    total_tokens: int = Field(ge=0, description="Total number of tokens used")

    def __init__(self, **data: Any) -> None:
        """Initialize with automatic total calculation if not provided."""
        if "total_tokens" not in data:
            data["total_tokens"] = data.get("prompt_tokens", 0) + data.get(
                "completion_tokens", 0
            )
        super().__init__(**data)


class TrackingData(BaseModel):
    """Comprehensive tracking data for a request.

    Includes both carbon/energy metrics and cost metrics.
    """

    # Carbon metrics
    energy_kwh: float = Field(ge=0, description="Energy consumption in kilowatt-hours")
    gwp_kgco2eq: float = Field(ge=0, description="Global Warming Potential in kg CO2 equivalent")
    adpe_kgsbeq: float | None = Field(
        default=None,
        ge=0,
        description="Abiotic Depletion Potential (elements) in kg Sb equivalent",
    )
    pe_mj: float | None = Field(default=None, ge=0, description="Primary Energy in megajoules")

    # Cost metrics
    cost_usd: float = Field(ge=0, description="Cost in US dollars")
    prompt_tokens: int = Field(ge=0, description="Number of prompt tokens")
    completion_tokens: int = Field(ge=0, description="Number of completion tokens")

    # Metadata
    provider: str = Field(description="LLM provider name (e.g., 'openai', 'anthropic')")
    model: str = Field(description="Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet')")
    tracking_method: Literal["ecologits", "codecarbon", "none"] = Field(
        description="Method used for carbon tracking"
    )
    electricity_mix_zone: str | None = Field(
        default=None, description="Electricity mix zone code (e.g., 'FRA', 'USA')"
    )

    # Timestamps
    measured_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp when measurement was taken"
    )
    duration_seconds: float = Field(ge=0, description="Request duration in seconds")

    # Additional energy breakdown (optional)
    energy_usage_kwh: float | None = Field(
        default=None, ge=0, description="Energy from usage phase"
    )
    gwp_usage_kgco2eq: float | None = Field(
        default=None, ge=0, description="GWP from usage phase"
    )
    gwp_embodied_kgco2eq: float | None = Field(
        default=None, ge=0, description="GWP from embodied emissions"
    )


class Response(BaseModel, Generic[T]):
    """Standardized response from any LLM provider.

    Generic type parameter T is used for structured outputs.
    When response_format is provided, parsed contains the validated model.
    """

    content: str = Field(description="Generated text content")
    usage: Usage = Field(description="Token usage information")
    model: str = Field(description="Model that generated the response")
    raw: dict[str, Any] = Field(description="Original raw API response for debugging/caching")
    tracking: TrackingData | None = Field(
        default=None, description="Tracking data if enabled"
    )
    cached: bool = Field(default=False, description="Whether response was served from cache")

    # Structured output support
    parsed: T | None = Field(
        default=None, description="Parsed structured output (when response_format is used)"
    )

    # Provider-specific fields (optional)
    finish_reason: str | None = Field(
        default=None, description="Reason the generation stopped"
    )
    response_id: str | None = Field(
        default=None, description="Unique response identifier from provider"
    )


class ImageContent(BaseModel):
    """Image content for multimodal messages."""

    type: Literal["image"] = "image"
    source: str | bytes = Field(
        description="Image source - URL, base64 string, or raw bytes"
    )
    detail: Literal["auto", "low", "high"] = Field(
        default="auto", description="Level of detail for image processing"
    )


class TextContent(BaseModel):
    """Text content for multimodal messages."""

    type: Literal["text"] = "text"
    text: str = Field(description="Text content")
