"""Tracking module for carbon and cost monitoring.

This module provides mixins and utilities for tracking the environmental impact
and cost of LLM requests.

Classes:
    EcoLogitsMixin: Mixin providing EcoLogits carbon impact tracking.
    EcoLogitsMetrics: Dataclass containing all EcoLogits impact metrics.

Example:
    ```python
    from seeds_clients.tracking import EcoLogitsMixin, EcoLogitsMetrics

    class MyClient(EcoLogitsMixin, BaseClient):
        def _get_ecologits_provider(self) -> str:
            return "openai"
    ```
"""

from seeds_clients.tracking.ecologits_tracker import EcoLogitsMetrics, EcoLogitsMixin

__all__ = ["EcoLogitsMixin", "EcoLogitsMetrics"]
