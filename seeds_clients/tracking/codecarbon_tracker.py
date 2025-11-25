"""CodeCarbon tracker mixin for hardware-measured carbon emissions.

This module provides a mixin class for extracting carbon emissions data from
server responses that include CodeCarbon tracking. This is designed to work
with servers like Model Garden that return `x_carbon_trace` data in their
OpenAI-compatible API responses.

CodeCarbon provides hardware-measured emissions (actual GPU/CPU/RAM power)
as opposed to EcoLogits which provides model-based estimates.
"""

from dataclasses import dataclass
from typing import Any

from seeds_clients.tracking.base import CarbonMetrics, TrackingMethod, wh_to_kwh, g_to_kg


@dataclass
class CodeCarbonMetrics(CarbonMetrics):
    """Metrics extracted from CodeCarbon server response.
    
    Extends CarbonMetrics with hardware-measured values from the server running
    the model, providing actual power consumption data rather than estimates.
    
    All inherited energy values are in kWh (kilowatt-hours).
    All inherited GWP values are in kgCO2eq (kilograms CO2 equivalent).
    
    Attributes:
        # Inherited from CarbonMetrics:
        energy_kwh: Total energy consumption in kilowatt-hours.
        gwp_kgco2eq: Total Global Warming Potential in kg CO2 equivalent.
        tracking_method: "codecarbon" or "codecarbon_estimated".
        duration_seconds: Request duration in seconds.
        energy_usage_kwh: Energy from usage phase (same as total for CodeCarbon).
        gwp_usage_kgco2eq: GWP from usage phase (same as total for CodeCarbon).
        
        # Hardware component breakdown (CodeCarbon-specific):
        cpu_energy_kwh: CPU energy consumed in kilowatt-hours.
        gpu_energy_kwh: GPU energy consumed in kilowatt-hours.
        ram_energy_kwh: RAM energy consumed in kilowatt-hours.
        
        # Power measurements (CodeCarbon-specific):
        cpu_power_watts: CPU power usage in watts.
        gpu_power_watts: GPU power usage in watts.
        ram_power_watts: RAM power usage in watts.
        
        # Token and session tracking:
        completion_tokens: Number of completion tokens (if tracked).
        
        # Tracking status:
        measured: Whether this is measured (True) or estimated (False).
        tracking_active: Whether tracking was active on the server.
        
        # Session aggregates (if available):
        session_total_kg_co2: Total session emissions in kg CO2.
        session_requests: Total requests in session.
        session_tokens: Total tokens generated in session.
    """
    
    # Hardware component breakdown
    cpu_energy_kwh: float = 0.0
    gpu_energy_kwh: float = 0.0
    ram_energy_kwh: float = 0.0
    
    # Power measurements
    cpu_power_watts: float = 0.0
    gpu_power_watts: float = 0.0
    ram_power_watts: float = 0.0
    
    # Token tracking
    completion_tokens: int = 0
    
    # Tracking status
    measured: bool = False
    tracking_active: bool = False
    
    # Session aggregates (if available)
    session_total_kg_co2: float | None = None
    session_requests: int | None = None
    session_tokens: int | None = None
    
    def to_tracking_fields(self) -> dict[str, Any]:
        """Convert CodeCarbon metrics to a dictionary for TrackingData.
        
        Returns fields that can be used to populate TrackingData, including
        all CodeCarbon-specific hardware measurements.
        
        Returns:
            Dictionary with fields for TrackingData.
        """
        # Get base fields
        fields = super().to_tracking_fields()
        
        # Add CodeCarbon-specific fields
        fields.update({
            "cpu_energy_kwh": self.cpu_energy_kwh,
            "gpu_energy_kwh": self.gpu_energy_kwh,
            "ram_energy_kwh": self.ram_energy_kwh,
            "cpu_power_watts": self.cpu_power_watts,
            "gpu_power_watts": self.gpu_power_watts,
            "ram_power_watts": self.ram_power_watts,
            # CodeCarbon doesn't provide these environmental metrics
            "adpe_kgsbeq": None,
            "pe_mj": None,
        })
        
        return fields


class CodeCarbonMixin:
    """Mixin for extracting CodeCarbon metrics from server responses.
    
    This mixin extracts `x_carbon_trace` data from API responses, which
    contains hardware-measured carbon emissions data from servers running
    CodeCarbon tracking (like Model Garden).
    
    Usage:
        class MyClient(CodeCarbonMixin, BaseClient):
            def _parse_response(self, raw: dict) -> Response:
                # Extract CodeCarbon metrics using the unified interface
                metrics = self.extract_carbon_metrics(raw)
                if metrics:
                    fields = metrics.to_tracking_fields()
                    # Use fields in TrackingData...
    
    The mixin expects responses to include:
    ```json
    {
        "x_carbon_trace": {
            "emissions_g_co2": 0.00052,
            "energy_consumed_wh": 0.0015,
            "cpu_energy_wh": 0.0005,
            "gpu_energy_wh": 0.0008,
            "ram_energy_wh": 0.0002,
            "duration_seconds": 0.5,
            "cpu_power_watts": 85.0,
            "gpu_power_watts": 250.0,
            "ram_power_watts": 15.0,
            "completion_tokens": 100,
            "measured": true,
            "tracking_active": true
        }
    }
    ```
    """
    
    def extract_carbon_metrics(
        self,
        raw_response: dict[str, Any],
        **kwargs: Any,
    ) -> CodeCarbonMetrics | None:
        """Extract CodeCarbon metrics from API response.
        
        This method implements the CarbonTrackingMixin protocol, providing
        a unified interface for carbon tracking.
        
        For CodeCarbon, metrics are extracted directly from the response's
        `x_carbon_trace` field (no additional kwargs needed).
        
        Args:
            raw_response: Raw API response dict that may contain x_carbon_trace.
            **kwargs: Unused for CodeCarbon (data is in response).
            
        Returns:
            CodeCarbonMetrics if x_carbon_trace is present, None otherwise.
            All values are converted to standard units (kWh, kgCO2eq).
        
        Example:
            ```python
            metrics = mixin.extract_carbon_metrics(raw_response)
            if metrics:
                print(f"Energy: {metrics.energy_kwh} kWh")
                print(f"Carbon: {metrics.gwp_kgco2eq} kgCO2eq")
                print(f"GPU Power: {metrics.gpu_power_watts} W")
            ```
        """
        carbon_trace = raw_response.get("x_carbon_trace")
        
        if not carbon_trace:
            return None
        
        if not isinstance(carbon_trace, dict):
            return None
        
        # Extract raw values
        emissions_g = float(carbon_trace.get("emissions_g_co2", 0.0))
        energy_wh = float(carbon_trace.get("energy_consumed_wh", 0.0))
        cpu_energy_wh = float(carbon_trace.get("cpu_energy_wh", 0.0))
        gpu_energy_wh = float(carbon_trace.get("gpu_energy_wh", 0.0))
        ram_energy_wh = float(carbon_trace.get("ram_energy_wh", 0.0))
        duration = float(carbon_trace.get("duration_seconds", 0.0))
        measured = bool(carbon_trace.get("measured", False))
        
        # Convert to standard units (kWh, kg)
        energy_kwh = wh_to_kwh(energy_wh)
        gwp_kgco2eq = g_to_kg(emissions_g)
        cpu_kwh = wh_to_kwh(cpu_energy_wh)
        gpu_kwh = wh_to_kwh(gpu_energy_wh)
        ram_kwh = wh_to_kwh(ram_energy_wh)
        
        # Determine tracking method
        tracking_method: TrackingMethod = "codecarbon" if measured else "codecarbon_estimated"
        
        return CodeCarbonMetrics(
            # Base CarbonMetrics fields (in standard units)
            energy_kwh=energy_kwh,
            gwp_kgco2eq=gwp_kgco2eq,
            tracking_method=tracking_method,
            duration_seconds=duration,
            energy_usage_kwh=energy_kwh,  # For CodeCarbon, usage = total
            gwp_usage_kgco2eq=gwp_kgco2eq,  # For CodeCarbon, usage = total
            gwp_embodied_kgco2eq=None,  # CodeCarbon doesn't track embodied
            
            # Hardware component breakdown (in kWh)
            cpu_energy_kwh=cpu_kwh,
            gpu_energy_kwh=gpu_kwh,
            ram_energy_kwh=ram_kwh,
            
            # Power measurements (in watts)
            cpu_power_watts=float(carbon_trace.get("cpu_power_watts", 0.0)),
            gpu_power_watts=float(carbon_trace.get("gpu_power_watts", 0.0)),
            ram_power_watts=float(carbon_trace.get("ram_power_watts", 0.0)),
            
            # Token tracking
            completion_tokens=int(carbon_trace.get("completion_tokens", 0)),
            
            # Tracking status
            measured=measured,
            tracking_active=bool(carbon_trace.get("tracking_active", False)),
            
            # Session aggregates
            session_total_kg_co2=carbon_trace.get("session_total_kg_co2"),
            session_requests=carbon_trace.get("session_requests"),
            session_tokens=carbon_trace.get("session_tokens"),
        )
    
    def _extract_codecarbon_metrics(
        self, 
        raw_response: dict[str, Any]
    ) -> CodeCarbonMetrics | None:
        """Extract CodeCarbon metrics from API response.
        
        .. deprecated:: 
            Use `extract_carbon_metrics()` instead for the unified interface.
        
        Args:
            raw_response: Raw API response dict that may contain x_carbon_trace.
            
        Returns:
            CodeCarbonMetrics if x_carbon_trace is present, None otherwise.
        """
        return self.extract_carbon_metrics(raw_response)
    
    def _codecarbon_to_tracking_fields(
        self,
        metrics: CodeCarbonMetrics | None,
    ) -> dict[str, Any]:
        """Convert CodeCarbon metrics to TrackingData fields.
        
        .. deprecated:: 
            Use `metrics.to_tracking_fields()` instead.
        
        This converts hardware-measured values to the standard TrackingData
        format used by seeds-clients.
        
        Args:
            metrics: CodeCarbon metrics or None.
            
        Returns:
            Dictionary of fields to include in TrackingData.
        """
        if metrics is None:
            return {
                "tracking_method": None,
            }
        
        return metrics.to_tracking_fields()
