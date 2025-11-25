"""Tests for the base tracking module."""

import pytest

from seeds_clients.tracking.base import (
    CarbonMetrics,
    CarbonTrackingMixin,
    TrackingMethod,
    g_to_kg,
    kg_to_g,
    kwh_to_wh,
    wh_to_kwh,
)
from seeds_clients.tracking import (
    EcoLogitsMetrics,
    CodeCarbonMetrics,
    EcoLogitsMixin,
    CodeCarbonMixin,
)


class TestCarbonMetrics:
    """Tests for the CarbonMetrics base class."""

    def test_default_values(self) -> None:
        """Test default values for CarbonMetrics."""
        metrics = CarbonMetrics()
        
        assert metrics.energy_kwh == 0.0
        assert metrics.gwp_kgco2eq == 0.0
        assert metrics.tracking_method == "none"
        assert metrics.duration_seconds == 0.0
        assert metrics.energy_usage_kwh is None
        assert metrics.gwp_usage_kgco2eq is None
        assert metrics.gwp_embodied_kgco2eq is None

    def test_custom_values(self) -> None:
        """Test custom values for CarbonMetrics."""
        metrics = CarbonMetrics(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            tracking_method="ecologits",
            duration_seconds=1.5,
            energy_usage_kwh=0.0008,
            gwp_usage_kgco2eq=0.0003,
            gwp_embodied_kgco2eq=0.0002,
        )
        
        assert metrics.energy_kwh == 0.001
        assert metrics.gwp_kgco2eq == 0.0005
        assert metrics.tracking_method == "ecologits"
        assert metrics.duration_seconds == 1.5
        assert metrics.energy_usage_kwh == 0.0008
        assert metrics.gwp_usage_kgco2eq == 0.0003
        assert metrics.gwp_embodied_kgco2eq == 0.0002

    def test_to_tracking_fields(self) -> None:
        """Test to_tracking_fields method."""
        metrics = CarbonMetrics(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            tracking_method="ecologits",
            duration_seconds=1.5,
            energy_usage_kwh=0.0008,
            gwp_usage_kgco2eq=0.0003,
            gwp_embodied_kgco2eq=0.0002,
        )
        
        fields = metrics.to_tracking_fields()
        
        assert fields["energy_kwh"] == 0.001
        assert fields["gwp_kgco2eq"] == 0.0005
        assert fields["tracking_method"] == "ecologits"
        assert fields["duration_seconds"] == 1.5
        assert fields["energy_usage_kwh"] == 0.0008
        assert fields["gwp_usage_kgco2eq"] == 0.0003
        assert fields["gwp_embodied_kgco2eq"] == 0.0002


class TestInheritance:
    """Test that EcoLogitsMetrics and CodeCarbonMetrics inherit from CarbonMetrics."""

    def test_ecologits_is_carbon_metrics(self) -> None:
        """Test that EcoLogitsMetrics is a subclass of CarbonMetrics."""
        assert issubclass(EcoLogitsMetrics, CarbonMetrics)
        
        metrics = EcoLogitsMetrics(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            tracking_method="ecologits",
        )
        assert isinstance(metrics, CarbonMetrics)

    def test_codecarbon_is_carbon_metrics(self) -> None:
        """Test that CodeCarbonMetrics is a subclass of CarbonMetrics."""
        assert issubclass(CodeCarbonMetrics, CarbonMetrics)
        
        metrics = CodeCarbonMetrics(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            tracking_method="codecarbon",
        )
        assert isinstance(metrics, CarbonMetrics)

    def test_common_fields_accessible(self) -> None:
        """Test that common fields are accessible on both types."""
        eco_metrics = EcoLogitsMetrics(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            tracking_method="ecologits",
            duration_seconds=1.0,
        )
        
        cc_metrics = CodeCarbonMetrics(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            tracking_method="codecarbon",
            duration_seconds=1.0,
        )
        
        # Both should have the same base fields
        for metrics in [eco_metrics, cc_metrics]:
            assert hasattr(metrics, "energy_kwh")
            assert hasattr(metrics, "gwp_kgco2eq")
            assert hasattr(metrics, "tracking_method")
            assert hasattr(metrics, "duration_seconds")
            assert hasattr(metrics, "to_tracking_fields")


class TestUnitConversions:
    """Tests for unit conversion functions."""

    def test_wh_to_kwh(self) -> None:
        """Test watt-hours to kilowatt-hours conversion."""
        assert wh_to_kwh(1000.0) == 1.0
        assert wh_to_kwh(500.0) == 0.5
        assert wh_to_kwh(0.0) == 0.0
        assert wh_to_kwh(1.5) == 0.0015

    def test_kwh_to_wh(self) -> None:
        """Test kilowatt-hours to watt-hours conversion."""
        assert kwh_to_wh(1.0) == 1000.0
        assert kwh_to_wh(0.5) == 500.0
        assert kwh_to_wh(0.0) == 0.0
        assert kwh_to_wh(0.0015) == 1.5

    def test_g_to_kg(self) -> None:
        """Test grams to kilograms conversion."""
        assert g_to_kg(1000.0) == 1.0
        assert g_to_kg(500.0) == 0.5
        assert g_to_kg(0.0) == 0.0
        assert g_to_kg(0.52) == pytest.approx(0.00052)

    def test_kg_to_g(self) -> None:
        """Test kilograms to grams conversion."""
        assert kg_to_g(1.0) == 1000.0
        assert kg_to_g(0.5) == 500.0
        assert kg_to_g(0.0) == 0.0
        assert kg_to_g(0.00052) == pytest.approx(0.52)

    def test_round_trip_energy(self) -> None:
        """Test round-trip conversion for energy."""
        original = 1500.0  # Wh
        assert kwh_to_wh(wh_to_kwh(original)) == pytest.approx(original)

    def test_round_trip_mass(self) -> None:
        """Test round-trip conversion for mass."""
        original = 520.0  # g
        assert kg_to_g(g_to_kg(original)) == pytest.approx(original)


class TestTrackingMethodType:
    """Tests for the TrackingMethod type alias."""

    def test_valid_tracking_methods(self) -> None:
        """Test valid tracking method values."""
        valid_methods: list[TrackingMethod] = [
            "ecologits",
            "codecarbon", 
            "codecarbon_estimated",
            "none",
        ]
        
        for method in valid_methods:
            metrics = CarbonMetrics(tracking_method=method)
            assert metrics.tracking_method == method


class TestPolymorphicUsage:
    """Test that metrics can be used interchangeably via the base class."""

    def process_metrics(self, metrics: CarbonMetrics) -> dict:
        """Example function that accepts any CarbonMetrics."""
        return {
            "energy": metrics.energy_kwh,
            "carbon": metrics.gwp_kgco2eq,
            "method": metrics.tracking_method,
        }

    def test_process_ecologits_metrics(self) -> None:
        """Test processing EcoLogitsMetrics through common interface."""
        metrics = EcoLogitsMetrics(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            tracking_method="ecologits",
            adpe_kgsbeq=1e-9,  # EcoLogits-specific
        )
        
        result = self.process_metrics(metrics)
        
        assert result["energy"] == 0.001
        assert result["carbon"] == 0.0005
        assert result["method"] == "ecologits"

    def test_process_codecarbon_metrics(self) -> None:
        """Test processing CodeCarbonMetrics through common interface."""
        metrics = CodeCarbonMetrics(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            tracking_method="codecarbon",
            gpu_power_watts=250.0,  # CodeCarbon-specific
        )
        
        result = self.process_metrics(metrics)
        
        assert result["energy"] == 0.001
        assert result["carbon"] == 0.0005
        assert result["method"] == "codecarbon"

    def test_to_tracking_fields_polymorphism(self) -> None:
        """Test that to_tracking_fields works polymorphically."""
        eco = EcoLogitsMetrics(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            tracking_method="ecologits",
        )
        
        cc = CodeCarbonMetrics(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            tracking_method="codecarbon",
        )
        
        # Both should return dicts with common keys
        eco_fields = eco.to_tracking_fields()
        cc_fields = cc.to_tracking_fields()
        
        common_keys = ["energy_kwh", "gwp_kgco2eq", "tracking_method", "duration_seconds"]
        for key in common_keys:
            assert key in eco_fields
            assert key in cc_fields
