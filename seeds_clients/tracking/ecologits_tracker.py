"""EcoLogits carbon tracking mixin for LLM clients.

This module provides comprehensive carbon impact tracking using the EcoLogits library.
It calculates energy consumption, greenhouse gas emissions, and other environmental
impact metrics based on model parameters, token counts, and request latency.

EcoLogits metrics include:
- Energy consumption (kWh)
- Global Warming Potential (kgCO2eq)
- Abiotic Depletion Potential for Elements (kgSbeq)
- Primary Energy (MJ)

These are broken down into usage phase (electricity consumption during inference)
and embodied phase (resource extraction, manufacturing, transportation).
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

from ecologits.tracers.utils import llm_impacts

from seeds_clients.core.types import TrackingData, Usage
from seeds_clients.tracking.base import CarbonMetrics, TrackingMethod


@dataclass
class EcoLogitsMetrics(CarbonMetrics):
    """Container for all EcoLogits impact metrics.

    Extends CarbonMetrics with EcoLogits-specific environmental impact metrics
    including Abiotic Depletion Potential (ADPe) and Primary Energy (PE), as well
    as detailed phase breakdowns.

    Attributes:
        energy_kwh: Total energy consumption in kilowatt-hours.
        gwp_kgco2eq: Total Global Warming Potential in kg CO2 equivalent.
        tracking_method: The method used for tracking ("ecologits" or "none").
        duration_seconds: Request duration in seconds (inherited from CarbonMetrics).

        # EcoLogits-specific environmental metrics
        adpe_kgsbeq: Abiotic Depletion Potential (elements) in kg Sb equivalent.
        pe_mj: Primary Energy consumption in megajoules.

        # Usage phase breakdown (electricity consumption during inference)
        energy_usage_kwh: Energy from usage phase only.
        gwp_usage_kgco2eq: GWP from usage phase only.
        adpe_usage_kgsbeq: ADPe from usage phase only.
        pe_usage_mj: PE from usage phase only.

        # Embodied phase breakdown (manufacturing, resource extraction)
        gwp_embodied_kgco2eq: GWP from embodied phase (manufacturing, etc.).
        adpe_embodied_kgsbeq: ADPe from embodied phase.
        pe_embodied_mj: PE from embodied phase.

        # Status messages
        warnings: List of warning messages from EcoLogits.
        errors: List of error messages from EcoLogits.
    """

    # EcoLogits-specific environmental metrics
    adpe_kgsbeq: float | None = None
    pe_mj: float | None = None

    # Extended usage phase breakdown (ADPe and PE)
    adpe_usage_kgsbeq: float | None = None
    pe_usage_mj: float | None = None

    # Embodied phase breakdown (ADPe and PE)
    adpe_embodied_kgsbeq: float | None = None
    pe_embodied_mj: float | None = None

    # Status messages from EcoLogits
    warnings: list[str] | None = None
    errors: list[str] | None = None

    def to_tracking_fields(self) -> dict[str, Any]:
        """Convert EcoLogits metrics to a dictionary for TrackingData.

        Returns fields that can be used to populate TrackingData, including
        all EcoLogits-specific environmental impact metrics.

        Returns:
            Dictionary with fields for TrackingData.
        """
        # Get base fields
        fields = super().to_tracking_fields()

        # Add EcoLogits-specific fields
        fields.update(
            {
                "adpe_kgsbeq": self.adpe_kgsbeq,
                "pe_mj": self.pe_mj,
                "adpe_usage_kgsbeq": self.adpe_usage_kgsbeq,
                "pe_usage_mj": self.pe_usage_mj,
                "adpe_embodied_kgsbeq": self.adpe_embodied_kgsbeq,
                "pe_embodied_mj": self.pe_embodied_mj,
                "ecologits_warnings": self.warnings,
                "ecologits_errors": self.errors,
            }
        )

        return fields


class EcoLogitsMixin:
    """
    Mixin providing EcoLogits carbon impact tracking for LLM clients.

    This mixin adds comprehensive carbon footprint tracking to any LLM client by
    calculating energy consumption and greenhouse gas emissions based on model
    parameters, token counts, and request latency using the EcoLogits library.

    Features:
    - Automatic carbon impact calculation per request
    - Energy consumption tracking (kWh)
    - GHG emissions tracking (kgCO2eq)
    - Abiotic Depletion Potential tracking (kgSbeq)
    - Primary Energy tracking (MJ)
    - Usage vs Embodied phase breakdown
    - Configurable electricity mix zone
    - Handles unknown models gracefully
    - Minimal performance overhead

    Usage:
        ```python
        class MyLLMClient(EcoLogitsMixin, BaseClient):
            def _get_ecologits_provider(self) -> str:
                return "openai"  # or "anthropic", "google_genai", etc.
        ```

    Supported electricity mix zones:
        - "WOR": World average (default)
        - "FRA": France
        - "USA": United States
        - "DEU": Germany
        - And many more ISO 3166-1 alpha-3 country codes

    See https://ecologits.ai/latest/tutorial/impacts/#electricity-mix for details.
    """

    # Default electricity mix zone (world average)
    _electricity_mix_zone: str = "WOR"

    def _get_ecologits_provider(self) -> str:
        """
        Get the provider name for EcoLogits tracking.

        Must be implemented by subclasses to return the provider identifier
        used by EcoLogits (e.g., "openai", "anthropic", "google_genai").

        Returns:
            str: Provider name for EcoLogits

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement _get_ecologits_provider()")

    def _get_electricity_mix_zone(self) -> str | None:
        """
        Get the electricity mix zone for carbon calculations.

        Returns:
            str | None: ISO 3166-1 alpha-3 code for the electricity mix zone,
                       or None to use default (WOR - World).
        """
        return getattr(self, "_electricity_mix_zone", None)

    def _set_electricity_mix_zone(self, zone: str | None) -> None:
        """
        Set the electricity mix zone for carbon calculations.

        Args:
            zone: ISO 3166-1 alpha-3 code (e.g., "FRA", "USA", "DEU")
                 or None to use default (WOR - World).
        """
        self._electricity_mix_zone = zone or "WOR"

    def _calculate_ecologits_impacts(
        self,
        model_name: str,
        output_tokens: int,
        request_latency: float,
        electricity_mix_zone: str | None = None,
    ) -> Any:
        """
        Calculate carbon impacts using EcoLogits.

        Args:
            model_name: Name of the LLM model (e.g., "gpt-4.1", "claude-3-5-sonnet")
            output_tokens: Number of completion/output tokens
            request_latency: Request duration in seconds
            electricity_mix_zone: ISO 3166-1 alpha-3 code for electricity mix.
                                 If None, uses the instance default or "WOR".

        Returns:
            EcoLogits ImpactsOutput object, or None if calculation fails.
            The ImpactsOutput contains energy, gwp, adpe, pe metrics with
            usage and embodied phase breakdowns.
        """
        if output_tokens <= 0:
            return None

        # Use provided zone, instance default, or global default
        zone = electricity_mix_zone or self._get_electricity_mix_zone()

        try:
            provider = self._get_ecologits_provider()
            impacts = llm_impacts(
                provider=provider,
                model_name=model_name,
                output_token_count=output_tokens,
                request_latency=request_latency,
                electricity_mix_zone=zone,
            )
            return impacts
        except Exception as e:
            # Don't fail the request if carbon tracking fails
            warnings.warn(f"EcoLogits carbon tracking failed: {e}", stacklevel=2)
            return None

    def _extract_impact_value(self, impact_obj: Any) -> float:
        """
        Extract numerical value from an EcoLogits impact object.

        Handles both direct float values and RangeValue objects.
        For RangeValue, uses the mean of min and max values.

        Args:
            impact_obj: EcoLogits impact object (Energy, GWP, ADPe, PE)
                       or None

        Returns:
            float: The impact value, or 0.0 if None or invalid

        Example:
            >>> mixin._extract_impact_value(impacts.gwp)
            0.00034  # kgCO2eq
        """
        if impact_obj is None:
            return 0.0

        value: Any = getattr(impact_obj, "value", 0.0)

        # Handle RangeValue objects (use mean)
        if hasattr(value, "mean"):
            return float(value.mean)
        elif hasattr(value, "min") and hasattr(value, "max"):
            return (float(value.min) + float(value.max)) / 2
        else:
            return float(value) if value is not None else 0.0

    def _extract_impact_value_optional(self, impact_obj: Any) -> float | None:
        """
        Extract numerical value from an EcoLogits impact object.

        Same as _extract_impact_value but returns None instead of 0.0
        when the impact object is None.

        Args:
            impact_obj: EcoLogits impact object (Energy, GWP, ADPe, PE)
                       or None

        Returns:
            float | None: The impact value, or None if not available
        """
        if impact_obj is None:
            return None
        return self._extract_impact_value(impact_obj)

    def _extract_ecologits_metrics(
        self, impacts: Any
    ) -> tuple[float, float, Literal["ecologits", "codecarbon", "none"]]:
        """
        Extract basic energy and GHG emissions from EcoLogits impacts.

        This is a simplified method for backwards compatibility.
        Use _extract_full_ecologits_metrics for complete data.

        Args:
            impacts: EcoLogits ImpactsOutput object

        Returns:
            Tuple of (energy_kwh, gwp_kgco2eq, tracking_method)
        """
        if not impacts:
            return 0.0, 0.0, "none"

        energy_kwh = self._extract_impact_value(impacts.energy)
        gwp_kgco2eq = self._extract_impact_value(impacts.gwp)

        return energy_kwh, gwp_kgco2eq, "ecologits"

    def _extract_full_ecologits_metrics(self, impacts: Any) -> EcoLogitsMetrics:
        """
        Extract all available metrics from EcoLogits impacts.

        This method extracts comprehensive environmental impact data including:
        - Total energy, GWP, ADPe, and PE
        - Usage phase breakdown (from electricity consumption)
        - Embodied phase breakdown (from manufacturing, etc.)
        - Any warnings or errors from the calculation

        Args:
            impacts: EcoLogits ImpactsOutput object

        Returns:
            EcoLogitsMetrics dataclass with all available metrics

        Example:
            ```python
            impacts = self._calculate_ecologits_impacts(
                model_name="gpt-4.1",
                output_tokens=100,
                request_latency=1.5,
            )
            metrics = self._extract_full_ecologits_metrics(impacts)
            print(f"Energy: {metrics.energy_kwh} kWh")
            print(f"GWP: {metrics.gwp_kgco2eq} kgCO2eq")
            print(f"  - Usage: {metrics.gwp_usage_kgco2eq} kgCO2eq")
            print(f"  - Embodied: {metrics.gwp_embodied_kgco2eq} kgCO2eq")
            ```
        """
        if not impacts:
            return EcoLogitsMetrics(tracking_method="none")

        # Extract total values
        energy_kwh = self._extract_impact_value(impacts.energy)
        gwp_kgco2eq = self._extract_impact_value(impacts.gwp)
        adpe_kgsbeq = self._extract_impact_value_optional(impacts.adpe)
        pe_mj = self._extract_impact_value_optional(impacts.pe)

        # Extract usage phase values
        usage = getattr(impacts, "usage", None)
        energy_usage_kwh = None
        gwp_usage_kgco2eq = None
        adpe_usage_kgsbeq = None
        pe_usage_mj = None

        if usage:
            energy_usage_kwh = self._extract_impact_value_optional(getattr(usage, "energy", None))
            gwp_usage_kgco2eq = self._extract_impact_value_optional(getattr(usage, "gwp", None))
            adpe_usage_kgsbeq = self._extract_impact_value_optional(getattr(usage, "adpe", None))
            pe_usage_mj = self._extract_impact_value_optional(getattr(usage, "pe", None))

        # Extract embodied phase values
        embodied = getattr(impacts, "embodied", None)
        gwp_embodied_kgco2eq = None
        adpe_embodied_kgsbeq = None
        pe_embodied_mj = None

        if embodied:
            gwp_embodied_kgco2eq = self._extract_impact_value_optional(
                getattr(embodied, "gwp", None)
            )
            adpe_embodied_kgsbeq = self._extract_impact_value_optional(
                getattr(embodied, "adpe", None)
            )
            pe_embodied_mj = self._extract_impact_value_optional(getattr(embodied, "pe", None))

        # Extract warnings and errors
        warning_list = None
        error_list = None

        if hasattr(impacts, "warnings") and impacts.warnings:
            warning_list = [str(w) for w in impacts.warnings]

        if hasattr(impacts, "errors") and impacts.errors:
            error_list = [str(e) for e in impacts.errors]

        return EcoLogitsMetrics(
            energy_kwh=energy_kwh,
            gwp_kgco2eq=gwp_kgco2eq,
            adpe_kgsbeq=adpe_kgsbeq,
            pe_mj=pe_mj,
            energy_usage_kwh=energy_usage_kwh,
            gwp_usage_kgco2eq=gwp_usage_kgco2eq,
            adpe_usage_kgsbeq=adpe_usage_kgsbeq,
            pe_usage_mj=pe_usage_mj,
            gwp_embodied_kgco2eq=gwp_embodied_kgco2eq,
            adpe_embodied_kgsbeq=adpe_embodied_kgsbeq,
            pe_embodied_mj=pe_embodied_mj,
            tracking_method="ecologits",
            warnings=warning_list,
            errors=error_list,
        )

    def extract_carbon_metrics(
        self,
        raw_response: dict[str, Any],
        **kwargs: Any,
    ) -> EcoLogitsMetrics | None:
        """Extract carbon metrics using EcoLogits calculations.

        This method implements the CarbonTrackingMixin protocol, providing
        a unified interface for carbon tracking.

        For EcoLogits, the metrics are calculated (not extracted from response),
        so this method requires additional kwargs:

        Args:
            raw_response: Raw API response dict. Used to extract model name and
                         output token count if not provided in kwargs.
            **kwargs: Required parameters for EcoLogits calculation:
                - model_name: Model name (falls back to raw_response["model"])
                - output_tokens: Output token count (falls back to raw_response["usage"]["completion_tokens"])
                - request_latency: Request duration in seconds (falls back to raw_response["_duration_seconds"])
                - electricity_mix_zone: Optional electricity mix zone code

        Returns:
            EcoLogitsMetrics with calculated environmental impacts, or None
            if calculation fails or output_tokens is 0.

        Example:
            ```python
            metrics = mixin.extract_carbon_metrics(
                raw_response,
                model_name="gpt-4.1",
                output_tokens=100,
                request_latency=1.5,
            )
            if metrics:
                print(f"Energy: {metrics.energy_kwh} kWh")
                print(f"Carbon: {metrics.gwp_kgco2eq} kgCO2eq")
            ```
        """
        # Extract parameters from kwargs or raw_response
        model_name = kwargs.get("model_name") or raw_response.get("model", "")

        usage = raw_response.get("usage", {})
        output_tokens = kwargs.get("output_tokens") or usage.get("completion_tokens", 0)

        request_latency = kwargs.get("request_latency") or raw_response.get(
            "_duration_seconds", 0.0
        )

        electricity_mix_zone = kwargs.get("electricity_mix_zone")

        # Calculate impacts using EcoLogits
        impacts = self._calculate_ecologits_impacts(
            model_name=model_name,
            output_tokens=output_tokens,
            request_latency=request_latency,
            electricity_mix_zone=electricity_mix_zone,
        )

        if not impacts:
            return None

        # Extract full metrics
        metrics = self._extract_full_ecologits_metrics(impacts)

        # Add duration to metrics
        metrics.duration_seconds = request_latency

        return metrics

    def _create_tracking_data(
        self,
        metrics: EcoLogitsMetrics,
        usage: Usage,
        provider: str,
        model_name: str,
        cost_usd: float,
        duration_seconds: float,
    ) -> TrackingData:
        """Build TrackingData from metrics and usage."""
        return TrackingData(
            # Total metrics
            energy_kwh=metrics.energy_kwh,
            gwp_kgco2eq=metrics.gwp_kgco2eq,
            adpe_kgsbeq=metrics.adpe_kgsbeq,
            pe_mj=metrics.pe_mj,
            # Cost & tokens
            cost_usd=cost_usd,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            # Metadata
            provider=provider,
            model=model_name,
            tracking_method=metrics.tracking_method,
            electricity_mix_zone=self._get_electricity_mix_zone(),
            duration_seconds=duration_seconds,
            # Usage phase
            energy_usage_kwh=metrics.energy_usage_kwh,
            gwp_usage_kgco2eq=metrics.gwp_usage_kgco2eq,
            adpe_usage_kgsbeq=metrics.adpe_usage_kgsbeq,
            pe_usage_mj=metrics.pe_usage_mj,
            # Embodied phase
            gwp_embodied_kgco2eq=metrics.gwp_embodied_kgco2eq,
            adpe_embodied_kgsbeq=metrics.adpe_embodied_kgsbeq,
            pe_embodied_mj=metrics.pe_embodied_mj,
            # Status
            ecologits_warnings=metrics.warnings,
            ecologits_errors=metrics.errors,
        )

    def _apply_response_tracking(
        self,
        result: dict[str, Any],
        start_time: float,
        model_name: str,
        output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Inject duration and EcoLogits impacts into raw response."""
        duration_seconds = time.time() - start_time
        result["_duration_seconds"] = duration_seconds

        # Determine tokens if not provided
        if output_tokens is None:
            usage = result.get("usage", {})
            output_tokens = usage.get("completion_tokens", 0)

        impacts = self._calculate_ecologits_impacts(
            model_name=model_name,
            output_tokens=output_tokens or 0,
            request_latency=duration_seconds,
            electricity_mix_zone=self._get_electricity_mix_zone(),
        )
        if impacts:
            result["_ecologits_impacts"] = impacts

        return result
