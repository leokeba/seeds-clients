"""Base classes and protocols for carbon tracking.

This module provides unified interfaces and base classes for carbon tracking,
allowing EcoLogits (model-based estimates) and CodeCarbon (hardware-measured)
tracking to be used interchangeably where their capabilities overlap.

The unified structure:
- CarbonMetrics: Base dataclass with common fields (energy, GWP, tracking method)
- CarbonTrackingMixin: Protocol defining the interface for tracking mixins
- Helper functions for unit conversions
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


# Type alias for tracking methods
TrackingMethod = Literal["ecologits", "codecarbon", "codecarbon_estimated", "none"]


@dataclass
class CarbonMetrics:
    """Base class for carbon/energy tracking metrics.
    
    This provides the common fields that both EcoLogits and CodeCarbon tracking
    produce, allowing them to be used interchangeably for basic carbon accounting.
    
    All energy values are in kWh (kilowatt-hours).
    All GWP (Global Warming Potential) values are in kgCO2eq (kilograms CO2 equivalent).
    
    Attributes:
        energy_kwh: Total energy consumption in kilowatt-hours.
        gwp_kgco2eq: Total Global Warming Potential (carbon footprint) in kg CO2 equivalent.
        tracking_method: Method used for tracking ("ecologits", "codecarbon", "codecarbon_estimated", "none").
        duration_seconds: Request/operation duration in seconds.
        
        # Usage phase (energy consumption during inference)
        energy_usage_kwh: Energy from usage phase in kWh (if available).
        gwp_usage_kgco2eq: GWP from usage phase in kgCO2eq (if available).
        
        # Embodied phase (manufacturing, resource extraction - typically only EcoLogits)
        gwp_embodied_kgco2eq: GWP from embodied phase in kgCO2eq (if available).
    """
    
    # Core metrics (always present)
    energy_kwh: float = 0.0
    gwp_kgco2eq: float = 0.0
    tracking_method: TrackingMethod = "none"
    duration_seconds: float = 0.0
    
    # Usage phase breakdown (electricity consumption during inference)
    energy_usage_kwh: float | None = None
    gwp_usage_kgco2eq: float | None = None
    
    # Embodied phase breakdown (manufacturing, etc.)
    gwp_embodied_kgco2eq: float | None = None
    
    def to_tracking_fields(self) -> dict[str, Any]:
        """Convert metrics to a dictionary suitable for TrackingData.
        
        Returns fields that can be unpacked into TrackingData initialization.
        Subclasses should override this to add their specific fields.
        
        Returns:
            Dictionary with fields for TrackingData.
        """
        return {
            "energy_kwh": self.energy_kwh,
            "gwp_kgco2eq": self.gwp_kgco2eq,
            "tracking_method": self.tracking_method,
            "duration_seconds": self.duration_seconds,
            "energy_usage_kwh": self.energy_usage_kwh,
            "gwp_usage_kgco2eq": self.gwp_usage_kgco2eq,
            "gwp_embodied_kgco2eq": self.gwp_embodied_kgco2eq,
        }


@runtime_checkable
class CarbonTrackingMixin(Protocol):
    """Protocol defining the interface for carbon tracking mixins.
    
    Both EcoLogitsMixin and CodeCarbonMixin implement this protocol,
    allowing them to be used interchangeably where their common features
    are needed.
    
    Example:
        ```python
        def get_carbon_data(tracker: CarbonTrackingMixin, raw_response: dict) -> CarbonMetrics:
            return tracker.extract_carbon_metrics(raw_response)
        ```
    """
    
    @abstractmethod
    def extract_carbon_metrics(
        self,
        raw_response: dict[str, Any],
        **kwargs: Any,
    ) -> CarbonMetrics | None:
        """Extract carbon metrics from a response or calculation inputs.
        
        This is the unified interface for getting carbon metrics from either
        EcoLogits calculations or CodeCarbon server responses.
        
        Args:
            raw_response: Raw API response dict (for CodeCarbon) or dict with
                         model info and token counts (for EcoLogits).
            **kwargs: Additional arguments specific to the tracking method.
                     For EcoLogits: model_name, output_tokens, request_latency, electricity_mix_zone
                     For CodeCarbon: no additional args needed (data is in response)
        
        Returns:
            CarbonMetrics instance with extracted data, or None if tracking data
            is not available.
        """
        ...


# Unit conversion helpers

def wh_to_kwh(wh: float) -> float:
    """Convert watt-hours to kilowatt-hours.
    
    Args:
        wh: Energy in watt-hours.
        
    Returns:
        Energy in kilowatt-hours.
    """
    return wh / 1000.0


def kwh_to_wh(kwh: float) -> float:
    """Convert kilowatt-hours to watt-hours.
    
    Args:
        kwh: Energy in kilowatt-hours.
        
    Returns:
        Energy in watt-hours.
    """
    return kwh * 1000.0


def g_to_kg(g: float) -> float:
    """Convert grams to kilograms.
    
    Args:
        g: Mass in grams.
        
    Returns:
        Mass in kilograms.
    """
    return g / 1000.0


def kg_to_g(kg: float) -> float:
    """Convert kilograms to grams.
    
    Args:
        kg: Mass in kilograms.
        
    Returns:
        Mass in grams.
    """
    return kg * 1000.0
