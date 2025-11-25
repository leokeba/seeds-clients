"""Tests for CodeCarbon tracker mixin."""

import pytest

from seeds_clients.tracking.codecarbon_tracker import (
    CodeCarbonMetrics,
    CodeCarbonMixin,
)
from seeds_clients.tracking.base import wh_to_kwh, g_to_kg


class TestCodeCarbonMetrics:
    """Tests for CodeCarbonMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default values for CodeCarbonMetrics."""
        metrics = CodeCarbonMetrics()
        
        # Base CarbonMetrics fields
        assert metrics.energy_kwh == 0.0
        assert metrics.gwp_kgco2eq == 0.0
        assert metrics.tracking_method == "none"
        assert metrics.duration_seconds == 0.0
        
        # Hardware breakdown
        assert metrics.cpu_energy_kwh == 0.0
        assert metrics.gpu_energy_kwh == 0.0
        assert metrics.ram_energy_kwh == 0.0
        
        # Power measurements
        assert metrics.cpu_power_watts == 0.0
        assert metrics.gpu_power_watts == 0.0
        assert metrics.ram_power_watts == 0.0
        
        # Status
        assert metrics.completion_tokens == 0
        assert metrics.measured is False
        assert metrics.tracking_active is False
        
        # Session aggregates
        assert metrics.session_total_kg_co2 is None
        assert metrics.session_requests is None
        assert metrics.session_tokens is None

    def test_custom_values(self) -> None:
        """Test custom values for CodeCarbonMetrics."""
        metrics = CodeCarbonMetrics(
            # Base fields (in standard units: kWh and kg)
            energy_kwh=0.0015,  # 1.5 Wh in kWh
            gwp_kgco2eq=0.00052,  # 0.52g in kg
            tracking_method="codecarbon",
            duration_seconds=0.5,
            
            # Hardware breakdown (in kWh)
            cpu_energy_kwh=0.0005,
            gpu_energy_kwh=0.0008,
            ram_energy_kwh=0.0002,
            
            # Power measurements (in watts)
            cpu_power_watts=85.0,
            gpu_power_watts=250.0,
            ram_power_watts=15.0,
            
            # Status
            completion_tokens=100,
            measured=True,
            tracking_active=True,
            
            # Session aggregates
            session_total_kg_co2=0.0012,
            session_requests=50,
            session_tokens=5000,
        )
        
        assert metrics.energy_kwh == 0.0015
        assert metrics.gwp_kgco2eq == 0.00052
        assert metrics.gpu_power_watts == 250.0
        assert metrics.measured is True
        assert metrics.tracking_active is True
        assert metrics.session_total_kg_co2 == 0.0012
    
    def test_to_tracking_fields(self) -> None:
        """Test to_tracking_fields method."""
        metrics = CodeCarbonMetrics(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            tracking_method="codecarbon",
            duration_seconds=1.5,
            cpu_energy_kwh=0.0003,
            gpu_energy_kwh=0.0005,
            ram_energy_kwh=0.0002,
            cpu_power_watts=85.0,
            gpu_power_watts=250.0,
            ram_power_watts=15.0,
        )
        
        fields = metrics.to_tracking_fields()
        
        assert fields["energy_kwh"] == 0.001
        assert fields["gwp_kgco2eq"] == 0.0005
        assert fields["tracking_method"] == "codecarbon"
        assert fields["duration_seconds"] == 1.5
        assert fields["cpu_energy_kwh"] == 0.0003
        assert fields["gpu_power_watts"] == 250.0
        # CodeCarbon doesn't track these
        assert fields["adpe_kgsbeq"] is None
        assert fields["pe_mj"] is None


class TestCodeCarbonMixin:
    """Tests for CodeCarbonMixin class."""

    @pytest.fixture
    def mixin(self) -> CodeCarbonMixin:
        """Create a CodeCarbonMixin instance for testing."""
        return CodeCarbonMixin()

    def test_extract_codecarbon_metrics_none_response(
        self, mixin: CodeCarbonMixin
    ) -> None:
        """Test extracting metrics when no x_carbon_trace is present."""
        raw_response = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        
        metrics = mixin._extract_codecarbon_metrics(raw_response)
        assert metrics is None

    def test_extract_codecarbon_metrics_invalid_type(
        self, mixin: CodeCarbonMixin
    ) -> None:
        """Test extracting metrics when x_carbon_trace is not a dict."""
        raw_response = {
            "id": "chatcmpl-123",
            "x_carbon_trace": "invalid",
        }
        
        metrics = mixin._extract_codecarbon_metrics(raw_response)
        assert metrics is None

    def test_extract_codecarbon_metrics_valid(
        self, mixin: CodeCarbonMixin
    ) -> None:
        """Test extracting metrics from valid x_carbon_trace.
        
        Note: The mixin automatically converts units:
        - Wh -> kWh (divide by 1000)
        - g CO2 -> kg CO2 (divide by 1000)
        """
        raw_response = {
            "id": "chatcmpl-123",
            "x_carbon_trace": {
                "emissions_g_co2": 520.0,  # 520g -> 0.52 kg
                "energy_consumed_wh": 1500.0,  # 1500 Wh -> 1.5 kWh
                "cpu_energy_wh": 500.0,  # 500 Wh -> 0.5 kWh
                "gpu_energy_wh": 800.0,  # 800 Wh -> 0.8 kWh
                "ram_energy_wh": 200.0,  # 200 Wh -> 0.2 kWh
                "duration_seconds": 0.5,
                "cpu_power_watts": 85.0,
                "gpu_power_watts": 250.0,
                "ram_power_watts": 15.0,
                "completion_tokens": 100,
                "measured": True,
                "tracking_active": True,
            },
        }
        
        metrics = mixin._extract_codecarbon_metrics(raw_response)
        
        assert metrics is not None
        # Check converted values (now in kWh and kg)
        assert metrics.gwp_kgco2eq == pytest.approx(0.52)
        assert metrics.energy_kwh == pytest.approx(1.5)
        assert metrics.cpu_energy_kwh == pytest.approx(0.5)
        assert metrics.gpu_energy_kwh == pytest.approx(0.8)
        assert metrics.ram_energy_kwh == pytest.approx(0.2)
        assert metrics.duration_seconds == 0.5
        assert metrics.cpu_power_watts == 85.0
        assert metrics.gpu_power_watts == 250.0
        assert metrics.ram_power_watts == 15.0
        assert metrics.completion_tokens == 100
        assert metrics.measured is True
        assert metrics.tracking_active is True
        assert metrics.tracking_method == "codecarbon"

    def test_extract_codecarbon_metrics_partial(
        self, mixin: CodeCarbonMixin
    ) -> None:
        """Test extracting metrics when some fields are missing."""
        raw_response = {
            "x_carbon_trace": {
                "emissions_g_co2": 300.0,  # 300g -> 0.3 kg
                "measured": True,
            },
        }
        
        metrics = mixin._extract_codecarbon_metrics(raw_response)
        
        assert metrics is not None
        assert metrics.gwp_kgco2eq == pytest.approx(0.3)
        assert metrics.energy_kwh == 0.0  # Default
        assert metrics.measured is True
        assert metrics.tracking_method == "codecarbon"

    def test_extract_codecarbon_metrics_with_session_data(
        self, mixin: CodeCarbonMixin
    ) -> None:
        """Test extracting metrics with session aggregate data."""
        raw_response = {
            "x_carbon_trace": {
                "emissions_g_co2": 520.0,
                "measured": True,
                "tracking_active": True,
                "session_total_kg_co2": 0.0012,
                "session_requests": 50,
                "session_tokens": 5000,
            },
        }
        
        metrics = mixin._extract_codecarbon_metrics(raw_response)
        
        assert metrics is not None
        assert metrics.session_total_kg_co2 == 0.0012
        assert metrics.session_requests == 50
        assert metrics.session_tokens == 5000

    def test_codecarbon_to_tracking_fields_none(
        self, mixin: CodeCarbonMixin
    ) -> None:
        """Test converting None metrics to tracking fields."""
        fields = mixin._codecarbon_to_tracking_fields(None)
        
        assert fields["tracking_method"] is None

    def test_codecarbon_to_tracking_fields_measured(
        self, mixin: CodeCarbonMixin
    ) -> None:
        """Test converting measured metrics to tracking fields.
        
        Note: With the new unified structure, metrics are already in
        standard units (kWh, kg), so no conversion happens in to_tracking_fields().
        """
        metrics = CodeCarbonMetrics(
            # Already in standard units (converted during extraction)
            energy_kwh=1.5,
            gwp_kgco2eq=0.52,
            tracking_method="codecarbon",
            duration_seconds=0.5,
            # Usage phase (same as total for CodeCarbon)
            energy_usage_kwh=1.5,
            gwp_usage_kgco2eq=0.52,
            # Hardware breakdown
            cpu_energy_kwh=0.5,
            gpu_energy_kwh=0.8,
            ram_energy_kwh=0.2,
            cpu_power_watts=85.0,
            gpu_power_watts=250.0,
            ram_power_watts=15.0,
            measured=True,
        )
        
        fields = mixin._codecarbon_to_tracking_fields(metrics)
        
        assert fields["energy_kwh"] == pytest.approx(1.5)
        assert fields["gwp_kgco2eq"] == pytest.approx(0.52)
        assert fields["cpu_energy_kwh"] == pytest.approx(0.5)
        assert fields["gpu_energy_kwh"] == pytest.approx(0.8)
        assert fields["ram_energy_kwh"] == pytest.approx(0.2)
        assert fields["cpu_power_watts"] == 85.0
        assert fields["gpu_power_watts"] == 250.0
        assert fields["ram_power_watts"] == 15.0
        assert fields["tracking_method"] == "codecarbon"
        assert fields["duration_seconds"] == 0.5
        
        # Usage phase should match totals for CodeCarbon
        assert fields["energy_usage_kwh"] == pytest.approx(1.5)
        assert fields["gwp_usage_kgco2eq"] == pytest.approx(0.52)
        
        # Embodied not tracked by CodeCarbon
        assert fields["gwp_embodied_kgco2eq"] is None
        assert fields["adpe_kgsbeq"] is None
        assert fields["pe_mj"] is None

    def test_codecarbon_to_tracking_fields_estimated(
        self, mixin: CodeCarbonMixin
    ) -> None:
        """Test converting estimated (not measured) metrics to tracking fields."""
        metrics = CodeCarbonMetrics(
            gwp_kgco2eq=0.0003,  # Already in kg
            tracking_method="codecarbon_estimated",
            measured=False,
        )
        
        fields = mixin._codecarbon_to_tracking_fields(metrics)
        
        assert fields["tracking_method"] == "codecarbon_estimated"

    def test_codecarbon_to_tracking_fields_zero_values(
        self, mixin: CodeCarbonMixin
    ) -> None:
        """Test converting zero-value metrics."""
        metrics = CodeCarbonMetrics(
            energy_kwh=0.0,
            gwp_kgco2eq=0.0,
            tracking_method="codecarbon",
            measured=True,
        )
        
        fields = mixin._codecarbon_to_tracking_fields(metrics)
        
        assert fields["energy_kwh"] == 0.0
        assert fields["gwp_kgco2eq"] == 0.0
        assert fields["tracking_method"] == "codecarbon"

    def test_extract_carbon_metrics_unified_interface(
        self, mixin: CodeCarbonMixin
    ) -> None:
        """Test the unified extract_carbon_metrics interface."""
        raw_response = {
            "x_carbon_trace": {
                "emissions_g_co2": 100.0,  # 100g -> 0.1 kg
                "energy_consumed_wh": 500.0,  # 500 Wh -> 0.5 kWh
                "measured": True,
            },
        }
        
        # Use the unified interface
        metrics = mixin.extract_carbon_metrics(raw_response)
        
        assert metrics is not None
        assert metrics.energy_kwh == pytest.approx(0.5)
        assert metrics.gwp_kgco2eq == pytest.approx(0.1)
        assert metrics.tracking_method == "codecarbon"
        
        # Test to_tracking_fields
        fields = metrics.to_tracking_fields()
        assert fields["energy_kwh"] == pytest.approx(0.5)
        assert fields["gwp_kgco2eq"] == pytest.approx(0.1)
