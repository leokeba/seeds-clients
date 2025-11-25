"""Tracking module for carbon and cost monitoring.

This module provides mixins and utilities for tracking the environmental impact
and cost of LLM requests.

Classes:
    EcoLogitsMixin: Mixin providing EcoLogits carbon impact tracking (model estimates).
    EcoLogitsMetrics: Dataclass containing all EcoLogits impact metrics.
    CodeCarbonMixin: Mixin for extracting CodeCarbon tracking data from server responses.
    CodeCarbonMetrics: Dataclass containing CodeCarbon hardware-measured metrics.

Example:
    ```python
    from seeds_clients.tracking import EcoLogitsMixin, EcoLogitsMetrics

    class MyClient(EcoLogitsMixin, BaseClient):
        def _get_ecologits_provider(self) -> str:
            return "openai"
    ```
"""

from seeds_clients.tracking.codecarbon_tracker import CodeCarbonMetrics, CodeCarbonMixin
from seeds_clients.tracking.ecologits_tracker import EcoLogitsMetrics, EcoLogitsMixin

__all__ = [
    "EcoLogitsMixin",
    "EcoLogitsMetrics",
    "CodeCarbonMixin",
    "CodeCarbonMetrics",
]
