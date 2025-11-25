"""Tests for CodeCarbon tracker mixin."""

import pytest

from seeds_clients.tracking.codecarbon_tracker import (
    CodeCarbonMetrics,
    CodeCarbonMixin,
)


class TestCodeCarbonMetrics:
    """Tests for CodeCarbonMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default values for CodeCarbonMetrics."""
        metrics = CodeCarbonMetrics()
        
        assert metrics.emissions_g_co2 == 0.0
        assert metrics.energy_consumed_wh == 0.0
        assert metrics.cpu_energy_wh == 0.0
        assert metrics.gpu_energy_wh == 0.0
        assert metrics.ram_energy_wh == 0.0
        assert metrics.duration_seconds == 0.0
        assert metrics.cpu_power_watts == 0.0
        assert metrics.gpu_power_watts == 0.0
        assert metrics.ram_power_watts == 0.0
        assert metrics.completion_tokens == 0
        assert metrics.measured is False
        assert metrics.tracking_active is False
        assert metrics.session_total_kg_co2 is None
        assert metrics.session_requests is None
        assert metrics.session_tokens is None

    def test_custom_values(self) -> None:
        """Test custom values for CodeCarbonMetrics."""
        metrics = CodeCarbonMetrics(
            emissions_g_co2=0.52,
            energy_consumed_wh=1.5,
            cpu_energy_wh=0.5,
            gpu_energy_wh=0.8,
            ram_energy_wh=0.2,
            duration_seconds=0.5,
            cpu_power_watts=85.0,
            gpu_power_watts=250.0,
            ram_power_watts=15.0,
            completion_tokens=100,
            measured=True,
            tracking_active=True,
            session_total_kg_co2=0.0012,
            session_requests=50,
            session_tokens=5000,
        )
        
        assert metrics.emissions_g_co2 == 0.52
        assert metrics.energy_consumed_wh == 1.5
        assert metrics.gpu_power_watts == 250.0
        assert metrics.measured is True
        assert metrics.tracking_active is True
        assert metrics.session_total_kg_co2 == 0.0012


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
        """Test extracting metrics from valid x_carbon_trace."""
        raw_response = {
            "id": "chatcmpl-123",
            "x_carbon_trace": {
                "emissions_g_co2": 0.52,
                "energy_consumed_wh": 1.5,
                "cpu_energy_wh": 0.5,
                "gpu_energy_wh": 0.8,
                "ram_energy_wh": 0.2,
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
        assert metrics.emissions_g_co2 == 0.52
        assert metrics.energy_consumed_wh == 1.5
        assert metrics.cpu_energy_wh == 0.5
        assert metrics.gpu_energy_wh == 0.8
        assert metrics.ram_energy_wh == 0.2
        assert metrics.duration_seconds == 0.5
        assert metrics.cpu_power_watts == 85.0
        assert metrics.gpu_power_watts == 250.0
        assert metrics.ram_power_watts == 15.0
        assert metrics.completion_tokens == 100
        assert metrics.measured is True
        assert metrics.tracking_active is True

    def test_extract_codecarbon_metrics_partial(
        self, mixin: CodeCarbonMixin
    ) -> None:
        """Test extracting metrics when some fields are missing."""
        raw_response = {
            "x_carbon_trace": {
                "emissions_g_co2": 0.3,
                "measured": True,
            },
        }
        
        metrics = mixin._extract_codecarbon_metrics(raw_response)
        
        assert metrics is not None
        assert metrics.emissions_g_co2 == 0.3
        assert metrics.energy_consumed_wh == 0.0  # Default
        assert metrics.measured is True

    def test_extract_codecarbon_metrics_with_session_data(
        self, mixin: CodeCarbonMixin
    ) -> None:
        """Test extracting metrics with session aggregate data."""
        raw_response = {
            "x_carbon_trace": {
                "emissions_g_co2": 0.52,
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
        """Test converting measured metrics to tracking fields."""
        metrics = CodeCarbonMetrics(
            emissions_g_co2=520.0,  # 520 mg = 0.52g = 0.00052 kg
            energy_consumed_wh=1500.0,  # 1.5 kWh
            cpu_energy_wh=500.0,
            gpu_energy_wh=800.0,
            ram_energy_wh=200.0,
            cpu_power_watts=85.0,
            gpu_power_watts=250.0,
            ram_power_watts=15.0,
            duration_seconds=0.5,
            measured=True,
        )
        
        fields = mixin._codecarbon_to_tracking_fields(metrics)
        
        # Check unit conversions (Wh to kWh, g to kg)
        assert fields["energy_kwh"] == pytest.approx(1.5)  # 1500 Wh -> 1.5 kWh
        assert fields["gwp_kgco2eq"] == pytest.approx(0.52)  # 520g -> 0.52 kg
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
            emissions_g_co2=0.3,
            measured=False,  # Not hardware-measured
        )
        
        fields = mixin._codecarbon_to_tracking_fields(metrics)
        
        assert fields["tracking_method"] == "codecarbon_estimated"

    def test_codecarbon_to_tracking_fields_zero_values(
        self, mixin: CodeCarbonMixin
    ) -> None:
        """Test converting zero-value metrics."""
        metrics = CodeCarbonMetrics(
            emissions_g_co2=0.0,
            energy_consumed_wh=0.0,
            measured=True,
        )
        
        fields = mixin._codecarbon_to_tracking_fields(metrics)
        
        assert fields["energy_kwh"] == 0.0
        assert fields["gwp_kgco2eq"] == 0.0
        assert fields["tracking_method"] == "codecarbon"
