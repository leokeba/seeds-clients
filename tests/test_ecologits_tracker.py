"""Tests for EcoLogits carbon tracking."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from seeds_clients.tracking.ecologits_tracker import EcoLogitsMetrics, EcoLogitsMixin


class MockRangeValue:
    """Mock for EcoLogits RangeValue objects."""

    def __init__(self, min_val: float, max_val: float):
        self.min = min_val
        self.max = max_val
        self.mean = (min_val + max_val) / 2


@dataclass
class MockImpact:
    """Mock for EcoLogits impact objects (Energy, GWP, ADPe, PE)."""

    value: Any  # Can be float, RangeValue, or None


@dataclass
class MockUsage:
    """Mock for EcoLogits Usage phase."""

    energy: MockImpact | None = None
    gwp: MockImpact | None = None
    adpe: MockImpact | None = None
    pe: MockImpact | None = None


@dataclass
class MockEmbodied:
    """Mock for EcoLogits Embodied phase."""

    gwp: MockImpact | None = None
    adpe: MockImpact | None = None
    pe: MockImpact | None = None


@dataclass
class MockImpactsOutput:
    """Mock for EcoLogits ImpactsOutput."""

    energy: MockImpact | None = None
    gwp: MockImpact | None = None
    adpe: MockImpact | None = None
    pe: MockImpact | None = None
    usage: MockUsage | None = None
    embodied: MockEmbodied | None = None
    warnings: list[Any] | None = None
    errors: list[Any] | None = None


class ConcreteEcoLogitsMixin(EcoLogitsMixin):
    """Concrete implementation of EcoLogitsMixin for testing."""

    def __init__(self, provider: str = "openai"):
        self._provider = provider
        self._electricity_mix_zone = "WOR"

    def _get_ecologits_provider(self) -> str:
        return self._provider


class TestEcoLogitsMixinBasic:
    """Test basic EcoLogitsMixin functionality."""

    def test_get_ecologits_provider_not_implemented(self) -> None:
        """Test that abstract method raises NotImplementedError."""
        mixin = EcoLogitsMixin()
        with pytest.raises(NotImplementedError):
            mixin._get_ecologits_provider()

    def test_get_electricity_mix_zone_default(self) -> None:
        """Test default electricity mix zone."""
        mixin = ConcreteEcoLogitsMixin()
        assert mixin._get_electricity_mix_zone() == "WOR"

    def test_set_electricity_mix_zone(self) -> None:
        """Test setting electricity mix zone."""
        mixin = ConcreteEcoLogitsMixin()
        mixin._set_electricity_mix_zone("FRA")
        assert mixin._get_electricity_mix_zone() == "FRA"

    def test_set_electricity_mix_zone_none_resets_to_default(self) -> None:
        """Test that setting None resets to WOR."""
        mixin = ConcreteEcoLogitsMixin()
        mixin._set_electricity_mix_zone("FRA")
        mixin._set_electricity_mix_zone(None)
        assert mixin._get_electricity_mix_zone() == "WOR"


class TestExtractImpactValue:
    """Test _extract_impact_value method."""

    @pytest.fixture
    def mixin(self) -> ConcreteEcoLogitsMixin:
        return ConcreteEcoLogitsMixin()

    def test_extract_none_returns_zero(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test that None returns 0.0."""
        assert mixin._extract_impact_value(None) == 0.0

    def test_extract_direct_float_value(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test extraction of direct float value."""
        impact = MockImpact(value=0.00034)
        assert mixin._extract_impact_value(impact) == 0.00034

    def test_extract_range_value_with_mean(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test extraction of RangeValue using mean."""
        range_val = MockRangeValue(min_val=0.1, max_val=0.3)
        impact = MockImpact(value=range_val)
        assert mixin._extract_impact_value(impact) == pytest.approx(0.2)

    def test_extract_range_value_without_mean(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test extraction of RangeValue calculating mean from min/max."""

        class RangeNoMean:
            def __init__(self):
                self.min = 0.1
                self.max = 0.5

        impact = MockImpact(value=RangeNoMean())
        assert mixin._extract_impact_value(impact) == pytest.approx(0.3)

    def test_extract_none_value_in_impact(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test impact with None value returns 0."""
        impact = MockImpact(value=None)  # type: ignore
        assert mixin._extract_impact_value(impact) == 0.0


class TestExtractImpactValueOptional:
    """Test _extract_impact_value_optional method."""

    @pytest.fixture
    def mixin(self) -> ConcreteEcoLogitsMixin:
        return ConcreteEcoLogitsMixin()

    def test_extract_none_returns_none(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test that None returns None (not 0.0)."""
        assert mixin._extract_impact_value_optional(None) is None

    def test_extract_value_returns_float(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test that valid impact returns float."""
        impact = MockImpact(value=0.5)
        assert mixin._extract_impact_value_optional(impact) == 0.5


class TestExtractEcologitsMetrics:
    """Test _extract_ecologits_metrics method (simplified)."""

    @pytest.fixture
    def mixin(self) -> ConcreteEcoLogitsMixin:
        return ConcreteEcoLogitsMixin()

    def test_extract_none_impacts(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test extraction with None impacts."""
        energy, gwp, method = mixin._extract_ecologits_metrics(None)
        assert energy == 0.0
        assert gwp == 0.0
        assert method == "none"

    def test_extract_valid_impacts(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test extraction with valid impacts."""
        impacts = MockImpactsOutput(
            energy=MockImpact(value=0.001),
            gwp=MockImpact(value=0.0005),
        )
        energy, gwp, method = mixin._extract_ecologits_metrics(impacts)
        assert energy == 0.001
        assert gwp == 0.0005
        assert method == "ecologits"


class TestExtractFullEcologitsMetrics:
    """Test _extract_full_ecologits_metrics method."""

    @pytest.fixture
    def mixin(self) -> ConcreteEcoLogitsMixin:
        return ConcreteEcoLogitsMixin()

    def test_extract_none_impacts(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test extraction with None impacts."""
        metrics = mixin._extract_full_ecologits_metrics(None)
        assert isinstance(metrics, EcoLogitsMetrics)
        assert metrics.energy_kwh == 0.0
        assert metrics.gwp_kgco2eq == 0.0
        assert metrics.tracking_method == "none"

    def test_extract_total_values(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test extraction of total values."""
        impacts = MockImpactsOutput(
            energy=MockImpact(value=0.001),
            gwp=MockImpact(value=0.0005),
            adpe=MockImpact(value=1.5e-9),
            pe=MockImpact(value=0.02),
        )
        metrics = mixin._extract_full_ecologits_metrics(impacts)

        assert metrics.energy_kwh == 0.001
        assert metrics.gwp_kgco2eq == 0.0005
        assert metrics.adpe_kgsbeq == 1.5e-9
        assert metrics.pe_mj == 0.02
        assert metrics.tracking_method == "ecologits"

    def test_extract_usage_phase(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test extraction of usage phase values."""
        usage = MockUsage(
            energy=MockImpact(value=0.0008),
            gwp=MockImpact(value=0.0003),
            adpe=MockImpact(value=1e-9),
            pe=MockImpact(value=0.015),
        )
        impacts = MockImpactsOutput(
            energy=MockImpact(value=0.001),
            gwp=MockImpact(value=0.0005),
            usage=usage,
        )
        metrics = mixin._extract_full_ecologits_metrics(impacts)

        assert metrics.energy_usage_kwh == 0.0008
        assert metrics.gwp_usage_kgco2eq == 0.0003
        assert metrics.adpe_usage_kgsbeq == 1e-9
        assert metrics.pe_usage_mj == 0.015

    def test_extract_embodied_phase(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test extraction of embodied phase values."""
        embodied = MockEmbodied(
            gwp=MockImpact(value=0.0002),
            adpe=MockImpact(value=5e-10),
            pe=MockImpact(value=0.005),
        )
        impacts = MockImpactsOutput(
            energy=MockImpact(value=0.001),
            gwp=MockImpact(value=0.0005),
            embodied=embodied,
        )
        metrics = mixin._extract_full_ecologits_metrics(impacts)

        assert metrics.gwp_embodied_kgco2eq == 0.0002
        assert metrics.adpe_embodied_kgsbeq == 5e-10
        assert metrics.pe_embodied_mj == 0.005

    def test_extract_warnings_and_errors(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test extraction of warnings and errors."""

        class MockWarning:
            def __str__(self) -> str:
                return "Model not found, using default"

        class MockError:
            def __str__(self) -> str:
                return "Critical error occurred"

        impacts = MockImpactsOutput(
            energy=MockImpact(value=0.001),
            gwp=MockImpact(value=0.0005),
            warnings=[MockWarning()],
            errors=[MockError()],
        )
        metrics = mixin._extract_full_ecologits_metrics(impacts)

        assert metrics.warnings == ["Model not found, using default"]
        assert metrics.errors == ["Critical error occurred"]

    def test_extract_range_values(self, mixin: ConcreteEcoLogitsMixin) -> None:
        """Test extraction with RangeValue objects."""
        impacts = MockImpactsOutput(
            energy=MockImpact(value=MockRangeValue(0.0008, 0.0012)),
            gwp=MockImpact(value=MockRangeValue(0.0003, 0.0007)),
        )
        metrics = mixin._extract_full_ecologits_metrics(impacts)

        assert metrics.energy_kwh == pytest.approx(0.001)
        assert metrics.gwp_kgco2eq == pytest.approx(0.0005)


class TestCalculateEcologitsImpacts:
    """Test _calculate_ecologits_impacts method."""

    @pytest.fixture
    def mixin(self) -> ConcreteEcoLogitsMixin:
        return ConcreteEcoLogitsMixin()

    def test_zero_output_tokens_returns_none(
        self, mixin: ConcreteEcoLogitsMixin
    ) -> None:
        """Test that zero output tokens returns None."""
        result = mixin._calculate_ecologits_impacts(
            model_name="gpt-4o",
            output_tokens=0,
            request_latency=1.0,
        )
        assert result is None

    def test_negative_output_tokens_returns_none(
        self, mixin: ConcreteEcoLogitsMixin
    ) -> None:
        """Test that negative output tokens returns None."""
        result = mixin._calculate_ecologits_impacts(
            model_name="gpt-4o",
            output_tokens=-10,
            request_latency=1.0,
        )
        assert result is None

    @patch("seeds_clients.tracking.ecologits_tracker.llm_impacts")
    def test_successful_impact_calculation(
        self, mock_llm_impacts: MagicMock, mixin: ConcreteEcoLogitsMixin
    ) -> None:
        """Test successful impact calculation."""
        mock_impacts = MockImpactsOutput(
            energy=MockImpact(value=0.001),
            gwp=MockImpact(value=0.0005),
        )
        mock_llm_impacts.return_value = mock_impacts

        result = mixin._calculate_ecologits_impacts(
            model_name="gpt-4o",
            output_tokens=100,
            request_latency=1.5,
        )

        assert result == mock_impacts
        mock_llm_impacts.assert_called_once_with(
            provider="openai",
            model_name="gpt-4o",
            output_token_count=100,
            request_latency=1.5,
            electricity_mix_zone="WOR",
        )

    @patch("seeds_clients.tracking.ecologits_tracker.llm_impacts")
    def test_custom_electricity_mix_zone(
        self, mock_llm_impacts: MagicMock, mixin: ConcreteEcoLogitsMixin
    ) -> None:
        """Test impact calculation with custom electricity mix zone."""
        mock_impacts = MockImpactsOutput()
        mock_llm_impacts.return_value = mock_impacts

        mixin._calculate_ecologits_impacts(
            model_name="gpt-4o",
            output_tokens=100,
            request_latency=1.5,
            electricity_mix_zone="FRA",
        )

        mock_llm_impacts.assert_called_once_with(
            provider="openai",
            model_name="gpt-4o",
            output_token_count=100,
            request_latency=1.5,
            electricity_mix_zone="FRA",
        )

    @patch("seeds_clients.tracking.ecologits_tracker.llm_impacts")
    def test_uses_instance_electricity_mix_zone(
        self, mock_llm_impacts: MagicMock, mixin: ConcreteEcoLogitsMixin
    ) -> None:
        """Test that instance electricity mix zone is used when not provided."""
        mock_impacts = MockImpactsOutput()
        mock_llm_impacts.return_value = mock_impacts

        mixin._set_electricity_mix_zone("DEU")
        mixin._calculate_ecologits_impacts(
            model_name="gpt-4o",
            output_tokens=100,
            request_latency=1.5,
        )

        mock_llm_impacts.assert_called_once_with(
            provider="openai",
            model_name="gpt-4o",
            output_token_count=100,
            request_latency=1.5,
            electricity_mix_zone="DEU",
        )

    @patch("seeds_clients.tracking.ecologits_tracker.llm_impacts")
    def test_exception_handling(
        self, mock_llm_impacts: MagicMock, mixin: ConcreteEcoLogitsMixin
    ) -> None:
        """Test that exceptions are caught and return None."""
        mock_llm_impacts.side_effect = Exception("Model not found in database")

        with pytest.warns(UserWarning, match="EcoLogits carbon tracking failed"):
            result = mixin._calculate_ecologits_impacts(
                model_name="unknown-model",
                output_tokens=100,
                request_latency=1.5,
            )

        assert result is None

    @patch("seeds_clients.tracking.ecologits_tracker.llm_impacts")
    def test_different_providers(
        self, mock_llm_impacts: MagicMock
    ) -> None:
        """Test impact calculation with different providers."""
        mock_impacts = MockImpactsOutput()
        mock_llm_impacts.return_value = mock_impacts

        providers = [
            ("openai", "gpt-4o"),
            ("anthropic", "claude-3-5-sonnet"),
            ("google_genai", "gemini-pro"),
            ("mistralai", "mistral-large"),
        ]

        for provider, model in providers:
            mixin = ConcreteEcoLogitsMixin(provider=provider)
            mixin._calculate_ecologits_impacts(
                model_name=model,
                output_tokens=50,
                request_latency=1.0,
            )

            mock_llm_impacts.assert_called_with(
                provider=provider,
                model_name=model,
                output_token_count=50,
                request_latency=1.0,
                electricity_mix_zone="WOR",
            )


class TestEcoLogitsMetricsDataclass:
    """Test EcoLogitsMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        metrics = EcoLogitsMetrics()

        assert metrics.energy_kwh == 0.0
        assert metrics.gwp_kgco2eq == 0.0
        assert metrics.adpe_kgsbeq is None
        assert metrics.pe_mj is None
        assert metrics.energy_usage_kwh is None
        assert metrics.gwp_usage_kgco2eq is None
        assert metrics.adpe_usage_kgsbeq is None
        assert metrics.pe_usage_mj is None
        assert metrics.gwp_embodied_kgco2eq is None
        assert metrics.adpe_embodied_kgsbeq is None
        assert metrics.pe_embodied_mj is None
        assert metrics.tracking_method == "none"
        assert metrics.warnings is None
        assert metrics.errors is None

    def test_custom_values(self) -> None:
        """Test setting custom values."""
        metrics = EcoLogitsMetrics(
            energy_kwh=0.001,
            gwp_kgco2eq=0.0005,
            adpe_kgsbeq=1.5e-9,
            pe_mj=0.02,
            energy_usage_kwh=0.0008,
            gwp_usage_kgco2eq=0.0003,
            gwp_embodied_kgco2eq=0.0002,
            tracking_method="ecologits",
            warnings=["Test warning"],
            errors=["Test error"],
        )

        assert metrics.energy_kwh == 0.001
        assert metrics.gwp_kgco2eq == 0.0005
        assert metrics.adpe_kgsbeq == 1.5e-9
        assert metrics.pe_mj == 0.02
        assert metrics.energy_usage_kwh == 0.0008
        assert metrics.gwp_usage_kgco2eq == 0.0003
        assert metrics.gwp_embodied_kgco2eq == 0.0002
        assert metrics.tracking_method == "ecologits"
        assert metrics.warnings == ["Test warning"]
        assert metrics.errors == ["Test error"]


class TestEcoLogitsIntegration:
    """Integration tests for EcoLogits tracking with OpenAI client."""

    @pytest.fixture
    def mock_impacts(self) -> MockImpactsOutput:
        """Create comprehensive mock impacts."""
        return MockImpactsOutput(
            energy=MockImpact(value=0.001),
            gwp=MockImpact(value=0.0005),
            adpe=MockImpact(value=1.5e-9),
            pe=MockImpact(value=0.02),
            usage=MockUsage(
                energy=MockImpact(value=0.0008),
                gwp=MockImpact(value=0.0003),
                adpe=MockImpact(value=1e-9),
                pe=MockImpact(value=0.015),
            ),
            embodied=MockEmbodied(
                gwp=MockImpact(value=0.0002),
                adpe=MockImpact(value=5e-10),
                pe=MockImpact(value=0.005),
            ),
        )

    @patch("seeds_clients.tracking.ecologits_tracker.llm_impacts")
    def test_full_metrics_extraction(
        self, mock_llm_impacts: MagicMock, mock_impacts: MockImpactsOutput
    ) -> None:
        """Test full metrics extraction workflow."""
        mock_llm_impacts.return_value = mock_impacts
        mixin = ConcreteEcoLogitsMixin()

        # Calculate impacts
        impacts = mixin._calculate_ecologits_impacts(
            model_name="gpt-4o",
            output_tokens=100,
            request_latency=1.5,
            electricity_mix_zone="FRA",
        )

        # Extract full metrics
        metrics = mixin._extract_full_ecologits_metrics(impacts)

        # Verify all metrics
        assert metrics.energy_kwh == 0.001
        assert metrics.gwp_kgco2eq == 0.0005
        assert metrics.adpe_kgsbeq == 1.5e-9
        assert metrics.pe_mj == 0.02

        # Usage phase
        assert metrics.energy_usage_kwh == 0.0008
        assert metrics.gwp_usage_kgco2eq == 0.0003
        assert metrics.adpe_usage_kgsbeq == 1e-9
        assert metrics.pe_usage_mj == 0.015

        # Embodied phase
        assert metrics.gwp_embodied_kgco2eq == 0.0002
        assert metrics.adpe_embodied_kgsbeq == 5e-10
        assert metrics.pe_embodied_mj == 0.005

        assert metrics.tracking_method == "ecologits"
