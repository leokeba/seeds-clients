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


@dataclass
class CodeCarbonMetrics:
    """Metrics extracted from CodeCarbon server response.
    
    These are hardware-measured values from the server running the model,
    providing actual power consumption data rather than estimates.
    
    Attributes:
        emissions_g_co2: Carbon emissions in grams CO2 for this request.
        energy_consumed_wh: Total energy consumed in watt-hours.
        cpu_energy_wh: CPU energy consumed in watt-hours.
        gpu_energy_wh: GPU energy consumed in watt-hours.
        ram_energy_wh: RAM energy consumed in watt-hours.
        duration_seconds: Request duration in seconds.
        cpu_power_watts: CPU power usage in watts.
        gpu_power_watts: GPU power usage in watts.
        ram_power_watts: RAM power usage in watts.
        completion_tokens: Number of completion tokens (if tracked).
        measured: Whether this is measured (True) or estimated (False).
        tracking_active: Whether tracking was active on the server.
        session_total_kg_co2: Total session emissions in kg CO2.
        session_requests: Total requests in session.
        session_tokens: Total tokens generated in session.
    """
    
    # Per-request metrics
    emissions_g_co2: float = 0.0
    energy_consumed_wh: float = 0.0
    cpu_energy_wh: float = 0.0
    gpu_energy_wh: float = 0.0
    ram_energy_wh: float = 0.0
    duration_seconds: float = 0.0
    
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


class CodeCarbonMixin:
    """Mixin for extracting CodeCarbon metrics from server responses.
    
    This mixin extracts `x_carbon_trace` data from API responses, which
    contains hardware-measured carbon emissions data from servers running
    CodeCarbon tracking (like Model Garden).
    
    Usage:
        class MyClient(CodeCarbonMixin, BaseClient):
            def _parse_response(self, raw: dict) -> Response:
                # Extract CodeCarbon metrics
                carbon_metrics = self._extract_codecarbon_metrics(raw)
                # Use metrics in tracking data...
    
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
    
    def _extract_codecarbon_metrics(
        self, 
        raw_response: dict[str, Any]
    ) -> CodeCarbonMetrics | None:
        """Extract CodeCarbon metrics from API response.
        
        Args:
            raw_response: Raw API response dict that may contain x_carbon_trace.
            
        Returns:
            CodeCarbonMetrics if x_carbon_trace is present, None otherwise.
        """
        carbon_trace = raw_response.get("x_carbon_trace")
        
        if not carbon_trace:
            return None
        
        if not isinstance(carbon_trace, dict):
            return None
        
        return CodeCarbonMetrics(
            # Per-request metrics
            emissions_g_co2=float(carbon_trace.get("emissions_g_co2", 0.0)),
            energy_consumed_wh=float(carbon_trace.get("energy_consumed_wh", 0.0)),
            cpu_energy_wh=float(carbon_trace.get("cpu_energy_wh", 0.0)),
            gpu_energy_wh=float(carbon_trace.get("gpu_energy_wh", 0.0)),
            ram_energy_wh=float(carbon_trace.get("ram_energy_wh", 0.0)),
            duration_seconds=float(carbon_trace.get("duration_seconds", 0.0)),
            
            # Power measurements
            cpu_power_watts=float(carbon_trace.get("cpu_power_watts", 0.0)),
            gpu_power_watts=float(carbon_trace.get("gpu_power_watts", 0.0)),
            ram_power_watts=float(carbon_trace.get("ram_power_watts", 0.0)),
            
            # Token tracking
            completion_tokens=int(carbon_trace.get("completion_tokens", 0)),
            
            # Tracking status
            measured=bool(carbon_trace.get("measured", False)),
            tracking_active=bool(carbon_trace.get("tracking_active", False)),
            
            # Session aggregates
            session_total_kg_co2=carbon_trace.get("session_total_kg_co2"),
            session_requests=carbon_trace.get("session_requests"),
            session_tokens=carbon_trace.get("session_tokens"),
        )
    
    def _codecarbon_to_tracking_fields(
        self,
        metrics: CodeCarbonMetrics | None,
    ) -> dict[str, Any]:
        """Convert CodeCarbon metrics to TrackingData fields.
        
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
        
        # Convert watt-hours to kilowatt-hours for consistency
        energy_kwh = metrics.energy_consumed_wh / 1000.0 if metrics.energy_consumed_wh else 0.0
        cpu_energy_kwh = metrics.cpu_energy_wh / 1000.0 if metrics.cpu_energy_wh else 0.0
        gpu_energy_kwh = metrics.gpu_energy_wh / 1000.0 if metrics.gpu_energy_wh else 0.0
        ram_energy_kwh = metrics.ram_energy_wh / 1000.0 if metrics.ram_energy_wh else 0.0
        
        # Convert grams to kilograms for GWP
        gwp_kgco2eq = metrics.emissions_g_co2 / 1000.0 if metrics.emissions_g_co2 else 0.0
        
        return {
            # Total metrics (hardware-measured)
            "energy_kwh": energy_kwh,
            "gwp_kgco2eq": gwp_kgco2eq,
            
            # Usage phase breakdown (all from hardware measurement)
            "energy_usage_kwh": energy_kwh,
            "gwp_usage_kgco2eq": gwp_kgco2eq,
            
            # Component breakdown
            "cpu_energy_kwh": cpu_energy_kwh,
            "gpu_energy_kwh": gpu_energy_kwh,
            "ram_energy_kwh": ram_energy_kwh,
            
            # Power measurements
            "cpu_power_watts": metrics.cpu_power_watts,
            "gpu_power_watts": metrics.gpu_power_watts,
            "ram_power_watts": metrics.ram_power_watts,
            
            # Tracking metadata
            "tracking_method": "codecarbon" if metrics.measured else "codecarbon_estimated",
            "duration_seconds": metrics.duration_seconds,
            
            # CodeCarbon doesn't provide embodied emissions
            "gwp_embodied_kgco2eq": None,
            "adpe_kgsbeq": None,
            "pe_mj": None,
        }
