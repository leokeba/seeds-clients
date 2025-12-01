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

    Includes both carbon/energy metrics and cost metrics, with breakdown
    into usage phase (electricity during inference) and embodied phase
    (manufacturing, resource extraction, transportation).

    Environmental Impact Metrics:
    - energy_kwh: Total energy consumption
    - gwp_kgco2eq: Global Warming Potential (carbon footprint)
    - adpe_kgsbeq: Abiotic Depletion Potential (resource depletion)
    - pe_mj: Primary Energy consumption

    Cost Metrics:
    - cost_usd: Total cost in US dollars
    - prompt_tokens/completion_tokens: Token counts

    See https://ecologits.ai/latest/tutorial/impacts/ for details on metrics.
    """

    # Carbon metrics - totals
    energy_kwh: float = Field(ge=0, description="Total energy consumption in kilowatt-hours")
    gwp_kgco2eq: float = Field(
        ge=0, description="Total Global Warming Potential in kg CO2 equivalent"
    )
    adpe_kgsbeq: float | None = Field(
        default=None,
        ge=0,
        description="Abiotic Depletion Potential (elements) in kg Sb equivalent",
    )
    pe_mj: float | None = Field(
        default=None, ge=0, description="Primary Energy consumption in megajoules"
    )

    # Cost metrics
    cost_usd: float = Field(ge=0, description="Cost in US dollars")
    prompt_tokens: int = Field(ge=0, description="Number of prompt tokens")
    completion_tokens: int = Field(ge=0, description="Number of completion tokens")

    # Metadata
    provider: str = Field(description="LLM provider name (e.g., 'openai', 'anthropic')")
    model: str = Field(description="Model identifier (e.g., 'gpt-4.1', 'claude-3-5-sonnet')")
    tracking_method: Literal["ecologits", "codecarbon", "codecarbon_estimated", "none"] | None = Field(
        default=None, description="Method used for carbon tracking"
    )
    electricity_mix_zone: str | None = Field(
        default=None,
        description="ISO 3166-1 alpha-3 code for electricity mix zone (e.g., 'FRA', 'USA', 'WOR')",
    )

    # Timestamps
    measured_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp when measurement was taken"
    )
    duration_seconds: float = Field(ge=0, description="Request duration in seconds")

    # Usage phase breakdown (electricity consumption during inference)
    energy_usage_kwh: float | None = Field(
        default=None, ge=0, description="Energy from usage phase in kWh"
    )
    gwp_usage_kgco2eq: float | None = Field(
        default=None, ge=0, description="GWP from usage phase in kgCO2eq"
    )
    adpe_usage_kgsbeq: float | None = Field(
        default=None, ge=0, description="ADPe from usage phase in kgSbeq"
    )
    pe_usage_mj: float | None = Field(
        default=None, ge=0, description="PE from usage phase in MJ"
    )

    # Embodied phase breakdown (manufacturing, resource extraction, transport)
    gwp_embodied_kgco2eq: float | None = Field(
        default=None, ge=0, description="GWP from embodied phase in kgCO2eq"
    )
    adpe_embodied_kgsbeq: float | None = Field(
        default=None, ge=0, description="ADPe from embodied phase in kgSbeq"
    )
    pe_embodied_mj: float | None = Field(
        default=None, ge=0, description="PE from embodied phase in MJ"
    )

    # EcoLogits status messages
    ecologits_warnings: list[str] | None = Field(
        default=None, description="Warning messages from EcoLogits calculation"
    )
    ecologits_errors: list[str] | None = Field(
        default=None, description="Error messages from EcoLogits calculation"
    )

    # CodeCarbon hardware-measured fields (when tracking_method is codecarbon)
    cpu_energy_kwh: float | None = Field(
        default=None, ge=0, description="CPU energy consumption in kWh (CodeCarbon)"
    )
    gpu_energy_kwh: float | None = Field(
        default=None, ge=0, description="GPU energy consumption in kWh (CodeCarbon)"
    )
    ram_energy_kwh: float | None = Field(
        default=None, ge=0, description="RAM energy consumption in kWh (CodeCarbon)"
    )
    cpu_power_watts: float | None = Field(
        default=None, ge=0, description="CPU power usage in watts (CodeCarbon)"
    )
    gpu_power_watts: float | None = Field(
        default=None, ge=0, description="GPU power usage in watts (CodeCarbon)"
    )
    ram_power_watts: float | None = Field(
        default=None, ge=0, description="RAM power usage in watts (CodeCarbon)"
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


class CumulativeTracking(BaseModel):
    """Cumulative tracking data across a client's lifecycle.

    Tracks total emissions, costs, and token usage separately for:
    - API requests (non-cached): Actual requests that incurred real emissions
    - Cached requests: Requests served from cache (avoided emissions)

    Also provides breakdown between usage and embodied emissions.

    Usage Phase: Emissions from electricity consumption during inference
    Embodied Phase: Emissions from manufacturing, resource extraction, transport

    Example:
        ```python
        client = OpenAIClient(model="gpt-4.1", cache_dir="./cache")

        # Make some requests
        response1 = client.generate([Message(role="user", content="Hello")])
        response2 = client.generate([Message(role="user", content="Hello")])  # cached

        # Get cumulative tracking
        tracking = client.get_cumulative_tracking()
        print(f"Total emissions: {tracking.total_gwp_kgco2eq} kgCO2eq")
        print(f"API emissions: {tracking.api_gwp_kgco2eq} kgCO2eq")
        print(f"Avoided emissions (cached): {tracking.cached_gwp_kgco2eq} kgCO2eq")
        print(f"Usage phase: {tracking.total_gwp_usage_kgco2eq} kgCO2eq")
        print(f"Embodied phase: {tracking.total_gwp_embodied_kgco2eq} kgCO2eq")
        ```
    """

    # Request counts
    api_request_count: int = Field(
        default=0, ge=0, description="Number of actual API requests made"
    )
    cached_request_count: int = Field(
        default=0, ge=0, description="Number of requests served from cache"
    )

    # API requests (non-cached) - Total metrics
    api_energy_kwh: float = Field(
        default=0.0, ge=0, description="Total energy from API requests in kWh"
    )
    api_gwp_kgco2eq: float = Field(
        default=0.0, ge=0, description="Total GWP from API requests in kgCO2eq"
    )
    api_cost_usd: float = Field(
        default=0.0, ge=0, description="Total cost from API requests in USD"
    )
    api_prompt_tokens: int = Field(
        default=0, ge=0, description="Total prompt tokens from API requests"
    )
    api_completion_tokens: int = Field(
        default=0, ge=0, description="Total completion tokens from API requests"
    )

    # API requests - Usage phase breakdown
    api_gwp_usage_kgco2eq: float = Field(
        default=0.0, ge=0, description="GWP from API requests usage phase in kgCO2eq"
    )
    api_energy_usage_kwh: float = Field(
        default=0.0, ge=0, description="Energy from API requests usage phase in kWh"
    )

    # API requests - Embodied phase breakdown
    api_gwp_embodied_kgco2eq: float = Field(
        default=0.0, ge=0, description="GWP from API requests embodied phase in kgCO2eq"
    )

    # Cached requests - Total metrics (would have been incurred if not cached)
    cached_energy_kwh: float = Field(
        default=0.0, ge=0, description="Estimated energy if cached requests were API calls in kWh"
    )
    cached_gwp_kgco2eq: float = Field(
        default=0.0, ge=0, description="Estimated GWP if cached requests were API calls in kgCO2eq"
    )
    cached_cost_usd: float = Field(
        default=0.0, ge=0, description="Estimated cost if cached requests were API calls in USD"
    )
    cached_prompt_tokens: int = Field(
        default=0, ge=0, description="Total prompt tokens from cached requests"
    )
    cached_completion_tokens: int = Field(
        default=0, ge=0, description="Total completion tokens from cached requests"
    )

    # Cached requests - Usage phase breakdown
    cached_gwp_usage_kgco2eq: float = Field(
        default=0.0, ge=0, description="Estimated GWP usage phase for cached requests in kgCO2eq"
    )
    cached_energy_usage_kwh: float = Field(
        default=0.0, ge=0, description="Estimated energy usage phase for cached requests in kWh"
    )

    # Cached requests - Embodied phase breakdown
    cached_gwp_embodied_kgco2eq: float = Field(
        default=0.0, ge=0, description="Estimated GWP embodied phase for cached requests in kgCO2eq"
    )

    @property
    def total_request_count(self) -> int:
        """Total number of requests (API + cached)."""
        return self.api_request_count + self.cached_request_count

    @property
    def total_energy_kwh(self) -> float:
        """Total energy across all requests (API + cached estimates)."""
        return self.api_energy_kwh + self.cached_energy_kwh

    @property
    def total_gwp_kgco2eq(self) -> float:
        """Total GWP across all requests (API + cached estimates)."""
        return self.api_gwp_kgco2eq + self.cached_gwp_kgco2eq

    @property
    def total_cost_usd(self) -> float:
        """Total cost across all requests (API + cached estimates)."""
        return self.api_cost_usd + self.cached_cost_usd

    @property
    def total_prompt_tokens(self) -> int:
        """Total prompt tokens across all requests."""
        return self.api_prompt_tokens + self.cached_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        """Total completion tokens across all requests."""
        return self.api_completion_tokens + self.cached_completion_tokens

    @property
    def total_gwp_usage_kgco2eq(self) -> float:
        """Total GWP from usage phase across all requests."""
        return self.api_gwp_usage_kgco2eq + self.cached_gwp_usage_kgco2eq

    @property
    def total_gwp_embodied_kgco2eq(self) -> float:
        """Total GWP from embodied phase across all requests."""
        return self.api_gwp_embodied_kgco2eq + self.cached_gwp_embodied_kgco2eq

    @property
    def total_energy_usage_kwh(self) -> float:
        """Total energy from usage phase across all requests."""
        return self.api_energy_usage_kwh + self.cached_energy_usage_kwh

    @property
    def cache_hit_rate(self) -> float:
        """Proportion of requests served from cache (0.0 to 1.0)."""
        if self.total_request_count == 0:
            return 0.0
        return self.cached_request_count / self.total_request_count

    @property
    def emissions_avoided_kgco2eq(self) -> float:
        """GWP emissions avoided by using cache (same as cached_gwp_kgco2eq).

        This represents the emissions that would have been incurred if
        all cached requests had been actual API calls.
        """
        return self.cached_gwp_kgco2eq

    def accumulate(self, tracking: "TrackingData", cached: bool = False) -> None:
        """Accumulate tracking data from a response.

        Args:
            tracking: TrackingData from a response
            cached: Whether this was a cached response
        """
        if cached:
            self.cached_request_count += 1
            self.cached_energy_kwh += tracking.energy_kwh
            self.cached_gwp_kgco2eq += tracking.gwp_kgco2eq
            self.cached_cost_usd += tracking.cost_usd
            self.cached_prompt_tokens += tracking.prompt_tokens
            self.cached_completion_tokens += tracking.completion_tokens
            # Usage phase
            if tracking.gwp_usage_kgco2eq is not None:
                self.cached_gwp_usage_kgco2eq += tracking.gwp_usage_kgco2eq
            if tracking.energy_usage_kwh is not None:
                self.cached_energy_usage_kwh += tracking.energy_usage_kwh
            # Embodied phase
            if tracking.gwp_embodied_kgco2eq is not None:
                self.cached_gwp_embodied_kgco2eq += tracking.gwp_embodied_kgco2eq
        else:
            self.api_request_count += 1
            self.api_energy_kwh += tracking.energy_kwh
            self.api_gwp_kgco2eq += tracking.gwp_kgco2eq
            self.api_cost_usd += tracking.cost_usd
            self.api_prompt_tokens += tracking.prompt_tokens
            self.api_completion_tokens += tracking.completion_tokens
            # Usage phase
            if tracking.gwp_usage_kgco2eq is not None:
                self.api_gwp_usage_kgco2eq += tracking.gwp_usage_kgco2eq
            if tracking.energy_usage_kwh is not None:
                self.api_energy_usage_kwh += tracking.energy_usage_kwh
            # Embodied phase
            if tracking.gwp_embodied_kgco2eq is not None:
                self.api_gwp_embodied_kgco2eq += tracking.gwp_embodied_kgco2eq

    def reset(self) -> None:
        """Reset all cumulative tracking to zero."""
        self.api_request_count = 0
        self.cached_request_count = 0
        self.api_energy_kwh = 0.0
        self.api_gwp_kgco2eq = 0.0
        self.api_cost_usd = 0.0
        self.api_prompt_tokens = 0
        self.api_completion_tokens = 0
        self.api_gwp_usage_kgco2eq = 0.0
        self.api_energy_usage_kwh = 0.0
        self.api_gwp_embodied_kgco2eq = 0.0
        self.cached_energy_kwh = 0.0
        self.cached_gwp_kgco2eq = 0.0
        self.cached_cost_usd = 0.0
        self.cached_prompt_tokens = 0
        self.cached_completion_tokens = 0
        self.cached_gwp_usage_kgco2eq = 0.0
        self.cached_energy_usage_kwh = 0.0
        self.cached_gwp_embodied_kgco2eq = 0.0

    def __repr__(self) -> str:
        return (
            f"CumulativeTracking("
            f"requests={self.total_request_count} (api={self.api_request_count}, cached={self.cached_request_count}), "
            f"gwp={self.total_gwp_kgco2eq:.6f}kgCO2eq (usage={self.total_gwp_usage_kgco2eq:.6f}, embodied={self.total_gwp_embodied_kgco2eq:.6f}), "
            f"cost=${self.total_cost_usd:.4f})"
        )
