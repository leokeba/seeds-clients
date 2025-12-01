"""Tracking module for carbon and cost monitoring.

This module provides mixins and utilities for tracking the environmental impact
and cost of LLM requests.

Base Classes:
    CarbonMetrics: Base dataclass with common carbon tracking fields (energy, GWP).
    CarbonTrackingMixin: Protocol defining the unified interface for tracking mixins.
    TrackingMethod: Type alias for tracking method literals.

EcoLogits (Model-Based Estimates):
    EcoLogitsMixin: Mixin providing EcoLogits carbon impact tracking.
    EcoLogitsMetrics: Dataclass containing all EcoLogits impact metrics.

CodeCarbon (Hardware-Measured):
    CodeCarbonMixin: Mixin for extracting CodeCarbon tracking data from server responses.
    CodeCarbonMetrics: Dataclass containing CodeCarbon hardware-measured metrics.

Utility Functions:
    wh_to_kwh, kwh_to_wh: Convert between watt-hours and kilowatt-hours.
    g_to_kg, kg_to_g: Convert between grams and kilograms.

Example:
    ```python
    from seeds_clients.tracking import (
        CarbonMetrics,
        EcoLogitsMixin, 
        EcoLogitsMetrics,
        CodeCarbonMixin,
        CodeCarbonMetrics,
    )

    class MyClient(EcoLogitsMixin, BaseClient):
        def _get_ecologits_provider(self) -> str:
            return "openai"
        
        def _parse_response(self, raw: dict) -> Response:
            # Use the unified interface
            metrics = self.extract_carbon_metrics(
                raw,
                model_name=self.model,
                output_tokens=100,
                request_latency=1.5,
            )
            if metrics:
                # Convert to TrackingData fields
                fields = metrics.to_tracking_fields()
    ```
"""

from seeds_clients.tracking.base import (
    CarbonMetrics,
    CarbonTrackingMixin,
    TrackingMethod,
    g_to_kg,
    kg_to_g,
    kwh_to_wh,
    wh_to_kwh,
)
from seeds_clients.tracking.boamps_reporter import (
    BoAmpsReport,
    BoAmpsReporter,
    export_boamps_report,
)
from seeds_clients.tracking.codecarbon_tracker import CodeCarbonMetrics, CodeCarbonMixin
from seeds_clients.tracking.ecologits_tracker import EcoLogitsMetrics, EcoLogitsMixin

__all__ = [
    # Base classes and protocols
    "CarbonMetrics",
    "CarbonTrackingMixin",
    "TrackingMethod",
    # EcoLogits
    "EcoLogitsMixin",
    "EcoLogitsMetrics",
    # CodeCarbon
    "CodeCarbonMixin",
    "CodeCarbonMetrics",
    # BoAmps Reporter
    "BoAmpsReport",
    "BoAmpsReporter",
    "export_boamps_report",
    # Utility functions
    "wh_to_kwh",
    "kwh_to_wh",
    "g_to_kg",
    "kg_to_g",
]
