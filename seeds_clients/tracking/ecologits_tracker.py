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

import warnings
from dataclasses import dataclass
from typing import Any, Literal

from ecologits.tracers.utils import llm_impacts


@dataclass
class EcoLogitsMetrics:
    """Container for all EcoLogits impact metrics.

    Attributes:
        energy_kwh: Total energy consumption in kilowatt-hours.
        gwp_kgco2eq: Total Global Warming Potential in kg CO2 equivalent.
        adpe_kgsbeq: Abiotic Depletion Potential (elements) in kg Sb equivalent.
        pe_mj: Primary Energy consumption in megajoules.
        energy_usage_kwh: Energy from usage phase only.
        gwp_usage_kgco2eq: GWP from usage phase only.
        gwp_embodied_kgco2eq: GWP from embodied phase (manufacturing, etc.).
        adpe_usage_kgsbeq: ADPe from usage phase only.
        adpe_embodied_kgsbeq: ADPe from embodied phase.
        pe_usage_mj: PE from usage phase only.
        pe_embodied_mj: PE from embodied phase.
        tracking_method: The method used for tracking ("ecologits" or "none").
        warnings: List of warning messages from EcoLogits.
        errors: List of error messages from EcoLogits.
    """

    energy_kwh: float = 0.0
    gwp_kgco2eq: float = 0.0
    adpe_kgsbeq: float | None = None
    pe_mj: float | None = None

    # Usage phase breakdown
    energy_usage_kwh: float | None = None
    gwp_usage_kgco2eq: float | None = None
    adpe_usage_kgsbeq: float | None = None
    pe_usage_mj: float | None = None

    # Embodied phase breakdown
    gwp_embodied_kgco2eq: float | None = None
    adpe_embodied_kgsbeq: float | None = None
    pe_embodied_mj: float | None = None

    tracking_method: Literal["ecologits", "codecarbon", "none"] = "none"
    warnings: list[str] | None = None
    errors: list[str] | None = None


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
        raise NotImplementedError(
            "Subclasses must implement _get_ecologits_provider()"
        )

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
            model_name: Name of the LLM model (e.g., "gpt-4o", "claude-3-5-sonnet")
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

        value = getattr(impact_obj, "value", 0.0)

        # Handle RangeValue objects (use mean)
        if hasattr(value, "mean"):
            return float(value.mean)
        elif hasattr(value, "min") and hasattr(value, "max"):
            min_val = value.min
            max_val = value.max
            return float((min_val + max_val) / 2)
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
                model_name="gpt-4o",
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
            energy_usage_kwh = self._extract_impact_value_optional(
                getattr(usage, "energy", None)
            )
            gwp_usage_kgco2eq = self._extract_impact_value_optional(
                getattr(usage, "gwp", None)
            )
            adpe_usage_kgsbeq = self._extract_impact_value_optional(
                getattr(usage, "adpe", None)
            )
            pe_usage_mj = self._extract_impact_value_optional(
                getattr(usage, "pe", None)
            )

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
            pe_embodied_mj = self._extract_impact_value_optional(
                getattr(embodied, "pe", None)
            )

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
