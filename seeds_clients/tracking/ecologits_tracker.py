"""EcoLogits carbon tracking mixin for LLM clients."""

import warnings
from typing import Any, Literal

from ecologits.tracers.utils import llm_impacts


class EcoLogitsMixin:
    """
    Mixin providing EcoLogits carbon impact tracking for LLM clients.

    This mixin adds carbon footprint tracking to any LLM client by calculating
    energy consumption and greenhouse gas emissions based on model parameters,
    token counts, and request latency using the EcoLogits library.

    Features:
    - Automatic carbon impact calculation per request
    - Energy consumption tracking (kWh)
    - GHG emissions tracking (kgCO2eq)
    - Handles unknown models gracefully
    - Minimal performance overhead

    Usage:
        class MyLLMClient(EcoLogitsMixin, BaseClient):
            def _get_ecologits_provider(self) -> str:
                return "openai"  # or "anthropic", "google", etc.
    """

    def _get_ecologits_provider(self) -> str:
        """
        Get the provider name for EcoLogits tracking.

        Must be implemented by subclasses to return the provider identifier
        used by EcoLogits (e.g., "openai", "anthropic", "google_genai").

        Returns:
            str: Provider name for EcoLogits
        """
        raise NotImplementedError(
            "Subclasses must implement _get_ecologits_provider()"
        )

    def _calculate_ecologits_impacts(
        self,
        model_name: str,
        output_tokens: int,
        request_latency: float,
    ) -> Any:
        """
        Calculate carbon impacts using EcoLogits.

        Args:
            model_name: Name of the LLM model
            output_tokens: Number of completion/output tokens
            request_latency: Request duration in seconds

        Returns:
            EcoLogits ImpactsOutput object, or None if calculation fails
        """
        if output_tokens <= 0:
            return None

        try:
            provider = self._get_ecologits_provider()
            impacts = llm_impacts(
                provider=provider,
                model_name=model_name,
                output_token_count=output_tokens,
                request_latency=request_latency,
            )
            return impacts
        except Exception as e:
            # Don't fail the request if carbon tracking fails
            warnings.warn(f"EcoLogits carbon tracking failed: {e}", stacklevel=2)
            return None

    def _extract_impact_value(self, impact_obj: Any) -> float:
        """
        Extract numerical value from an EcoLogits impact object.

        Handles both direct float values and RangeValue objects (uses mean).

        Args:
            impact_obj: EcoLogits impact object (Energy, GWP, ADPe, PE)

        Returns:
            float: The impact value, or 0.0 if None
        """
        if impact_obj is None:
            return 0.0

        value = getattr(impact_obj, 'value', 0.0)

        # Handle RangeValue objects (use mean)
        if hasattr(value, 'mean'):
            return float(value.mean)
        elif hasattr(value, 'min') and hasattr(value, 'max'):
            min_val = value.min
            max_val = value.max
            return float((min_val + max_val) / 2)
        else:
            return float(value) if value is not None else 0.0

    def _extract_ecologits_metrics(
        self, impacts: Any
    ) -> tuple[float, float, Literal["ecologits", "codecarbon", "none"]]:
        """
        Extract energy and GHG emissions from EcoLogits impacts.

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
