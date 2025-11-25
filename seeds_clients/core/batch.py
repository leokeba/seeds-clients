"""Batch processing types and utilities."""

from dataclasses import dataclass, field

from seeds_clients.core.types import Response


@dataclass
class BatchResult:
    """Result of a batch generation operation.

    Contains individual responses and aggregated metrics across all requests.

    Attributes:
        responses: Individual responses for each successful request.
        errors: List of (index, exception) tuples for failed requests.
        total_prompt_tokens: Aggregated prompt tokens across all successful requests.
        total_completion_tokens: Aggregated completion tokens.
        total_cost_usd: Aggregated cost in USD.
        total_energy_kwh: Aggregated energy consumption in kWh.
        total_gwp_kgco2eq: Aggregated carbon emissions in kgCO2eq.
        total_duration_seconds: Wall clock time for the entire batch operation.

    Example:
        ```python
        result = await client.batch_generate(messages_list)

        print(f"Successful: {result.successful_count}/{result.total_count}")
        print(f"Total cost: ${result.total_cost_usd:.4f}")
        print(f"Total carbon: {result.total_gwp_kgco2eq:.6f} kgCO2eq")

        for response in result.responses:
            print(response.content)

        for idx, error in result.errors:
            print(f"Request {idx} failed: {error}")
        ```
    """

    responses: list[Response] = field(default_factory=list)
    """Individual responses for each successful request in the batch."""

    errors: list[tuple[int, Exception]] = field(default_factory=list)
    """List of (index, exception) tuples for failed requests."""

    # Aggregated metrics
    total_prompt_tokens: int = 0
    """Total prompt tokens across all successful requests."""

    total_completion_tokens: int = 0
    """Total completion tokens across all successful requests."""

    total_cost_usd: float = 0.0
    """Total cost in USD across all successful requests."""

    total_energy_kwh: float = 0.0
    """Total energy consumption in kWh across all successful requests."""

    total_gwp_kgco2eq: float = 0.0
    """Total carbon emissions in kgCO2eq across all successful requests."""

    total_duration_seconds: float = 0.0
    """Wall clock time for the entire batch operation."""

    @property
    def successful_count(self) -> int:
        """Number of successful requests."""
        return len(self.responses)

    @property
    def failed_count(self) -> int:
        """Number of failed requests."""
        return len(self.errors)

    @property
    def total_count(self) -> int:
        """Total number of requests (successful + failed)."""
        return self.successful_count + self.failed_count

    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + completion) across all successful requests."""
        return self.total_prompt_tokens + self.total_completion_tokens

    def __repr__(self) -> str:
        return (
            f"BatchResult(successful={self.successful_count}, "
            f"failed={self.failed_count}, "
            f"cost=${self.total_cost_usd:.4f}, "
            f"carbon={self.total_gwp_kgco2eq:.6f}kgCO2eq)"
        )
