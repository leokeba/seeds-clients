"""Tests for BoAmps reporter."""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from seeds_clients.core.types import CumulativeTracking
from seeds_clients.tracking.boamps_reporter import (
    Algorithm,
    BoAmpsReport,
    BoAmpsReporter,
    Dataset,
    Environment,
    Header,
    HardwareComponent,
    Infrastructure,
    Measure,
    Publisher,
    SoftwareInfo,
    SystemInfo,
    Task,
    export_boamps_report,
)


class TestBoAmpsModels:
    """Tests for BoAmps Pydantic models."""

    def test_publisher_defaults(self) -> None:
        """Test Publisher model defaults."""
        publisher = Publisher()
        assert publisher.confidentialityLevel == "public"
        assert publisher.name is None

    def test_header_defaults(self) -> None:
        """Test Header model defaults."""
        header = Header()
        assert header.licensing == "CC-BY-4.0"
        assert header.formatVersion == "1.0.0"
        assert header.reportStatus == "final"
        assert header.reportId is not None
        assert header.reportDatetime is not None

    def test_algorithm_defaults(self) -> None:
        """Test Algorithm model defaults."""
        algo = Algorithm()
        assert algo.algorithmType == "llm"
        assert algo.framework == "seeds-clients"

    def test_algorithm_with_model(self) -> None:
        """Test Algorithm with foundation model."""
        algo = Algorithm(
            foundationModelName="gpt-4.1",
            parametersNumber=1760.0,
        )
        assert algo.foundationModelName == "gpt-4.1"
        assert algo.parametersNumber == 1760.0

    def test_dataset_defaults(self) -> None:
        """Test Dataset model defaults."""
        dataset = Dataset()
        assert dataset.dataType == "text"

    def test_dataset_with_sizes(self) -> None:
        """Test Dataset with input/output sizes."""
        dataset = Dataset(
            inputSize=1000,
            outputSize=500,
            datasetName="test_dataset",
        )
        assert dataset.inputSize == 1000
        assert dataset.outputSize == 500

    def test_task_defaults(self) -> None:
        """Test Task model defaults."""
        task = Task()
        assert task.taskStage == "inference"
        assert task.taskFamily == "textGeneration"
        assert task.nbRequest == 0

    def test_measure_required_fields(self) -> None:
        """Test Measure with required fields."""
        measure = Measure(
            measurementMethod="ecologits",
            powerConsumption=0.001,
        )
        assert measure.measurementMethod == "ecologits"
        assert measure.powerConsumption == 0.001

    def test_measure_with_calibration(self) -> None:
        """Test Measure with calibration data."""
        measure = Measure(
            measurementMethod="codecarbon",
            powerConsumption=0.002,
            powerCalibrationMeasurement=0.0001,
            durationCalibrationMeasurement=60.0,
        )
        assert measure.powerCalibrationMeasurement == 0.0001
        assert measure.durationCalibrationMeasurement == 60.0

    def test_hardware_component(self) -> None:
        """Test HardwareComponent model."""
        component = HardwareComponent(
            componentType="gpu",
            componentName="NVIDIA A100",
            manufacturer="NVIDIA",
            tdp=400.0,
        )
        assert component.componentType == "gpu"
        assert component.componentName == "NVIDIA A100"

    def test_infrastructure_defaults(self) -> None:
        """Test Infrastructure model defaults."""
        infra = Infrastructure()
        assert infra.infraType == "publicCloud"

    def test_environment_defaults(self) -> None:
        """Test Environment model defaults."""
        env = Environment()
        assert env.country == "WOR"

    def test_system_info(self) -> None:
        """Test SystemInfo model."""
        system = SystemInfo(os="Linux", distributionVersion="5.4.0")
        assert system.os == "Linux"

    def test_software_info(self) -> None:
        """Test SoftwareInfo model."""
        software = SoftwareInfo(language="python", version="3.10.0")
        assert software.language == "python"


class TestBoAmpsReport:
    """Tests for BoAmpsReport model."""

    def test_report_defaults(self) -> None:
        """Test report with default values."""
        report = BoAmpsReport()
        assert report.header is not None
        assert report.task is not None
        assert report.measures == []
        assert report.infrastructure is not None
        assert report.quality == "medium"

    def test_report_to_json(self) -> None:
        """Test converting report to JSON."""
        report = BoAmpsReport(
            measures=[
                Measure(
                    measurementMethod="ecologits",
                    powerConsumption=0.001,
                )
            ]
        )
        json_str = report.to_json()
        data = json.loads(json_str)

        assert "header" in data
        assert "task" in data
        assert "measures" in data
        assert len(data["measures"]) == 1
        assert data["measures"][0]["powerConsumption"] == 0.001

    def test_report_save(self) -> None:
        """Test saving report to file."""
        report = BoAmpsReport(
            measures=[
                Measure(
                    measurementMethod="ecologits",
                    powerConsumption=0.002,
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            report.save(filepath)

            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)
            assert data["measures"][0]["powerConsumption"] == 0.002

    def test_report_excludes_none_values(self) -> None:
        """Test that None values are excluded from JSON."""
        report = BoAmpsReport()
        json_str = report.to_json()
        data = json.loads(json_str)

        # Publisher is None by default in header
        assert "publisher" not in data["header"]


class TestBoAmpsReporter:
    """Tests for BoAmpsReporter class."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock client with cumulative tracking."""
        client = MagicMock()
        client.model = "gpt-4.1"
        client.tracking_method = "ecologits"
        client.electricity_mix_zone = "FRA"
        client._get_provider_name.return_value = "openai"

        # Setup cumulative tracking
        tracking = CumulativeTracking(
            api_request_count=10,
            cached_request_count=5,
            api_energy_kwh=0.005,
            api_gwp_kgco2eq=0.0025,
            api_cost_usd=0.50,
            api_prompt_tokens=1000,
            api_completion_tokens=500,
        )
        client.cumulative_tracking = tracking
        client._cumulative_tracking = tracking

        return client

    def test_reporter_initialization(self, mock_client: MagicMock) -> None:
        """Test reporter initialization."""
        reporter = BoAmpsReporter(
            client=mock_client,
            publisher_name="Test Org",
            task_description="Test task",
        )
        assert reporter.publisher_name == "Test Org"
        assert reporter.task_description == "Test task"

    def test_generate_report(self, mock_client: MagicMock) -> None:
        """Test generating a complete report."""
        reporter = BoAmpsReporter(
            client=mock_client,
            publisher_name="Test Organization",
            task_description="LLM inference testing",
        )

        report = reporter.generate_report()

        # Check header
        assert report.header.publisher is not None
        assert report.header.publisher.name == "Test Organization"

        # Check task
        assert report.task.taskStage == "inference"
        assert report.task.nbRequest == 15  # 10 API + 5 cached
        assert len(report.task.algorithms) == 1
        assert report.task.algorithms[0].foundationModelName == "gpt-4.1"
        assert len(report.task.dataset) == 1
        assert report.task.dataset[0].inputSize == 1000 + 0  # Only API tokens tracked
        assert report.task.taskDescription == "LLM inference testing"

        # Check measures
        assert len(report.measures) == 1
        assert report.measures[0].measurementMethod == "ecologits"
        assert report.measures[0].powerConsumption == pytest.approx(0.005)

        # Check infrastructure
        assert report.infrastructure.infraType == "publicCloud"
        assert report.infrastructure.cloudService == "openai"

        # Check environment
        assert report.environment.country == "FRA"

        # Check system info
        assert report.system is not None
        assert report.system.os is not None

        # Check software info
        assert report.software is not None
        assert report.software.language == "python"

    def test_generate_report_without_publisher(self, mock_client: MagicMock) -> None:
        """Test generating report without publisher info."""
        reporter = BoAmpsReporter(client=mock_client)
        report = reporter.generate_report()

        assert report.header.publisher is None

    def test_generate_report_without_system_info(self, mock_client: MagicMock) -> None:
        """Test generating report without system info."""
        reporter = BoAmpsReporter(
            client=mock_client,
            include_system_info=False,
        )
        report = reporter.generate_report()

        assert report.system is None
        assert report.software is None

    def test_generate_report_with_calibration(self, mock_client: MagicMock) -> None:
        """Test generating report with calibration data."""
        reporter = BoAmpsReporter(
            client=mock_client,
            calibration_energy_kwh=0.0001,
            calibration_duration_seconds=120.0,
        )
        report = reporter.generate_report()

        assert report.measures[0].powerCalibrationMeasurement == 0.0001
        assert report.measures[0].durationCalibrationMeasurement == 120.0

    def test_export_report(self, mock_client: MagicMock) -> None:
        """Test exporting report to file."""
        reporter = BoAmpsReporter(client=mock_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            report = reporter.export(filepath, include_summary=False)

            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)
            assert data["measures"][0]["powerConsumption"] == pytest.approx(0.005)

    def test_export_report_with_summary(
        self, mock_client: MagicMock, capsys: Any
    ) -> None:
        """Test exporting report with summary output."""
        reporter = BoAmpsReporter(client=mock_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            reporter.export(filepath, include_summary=True)

            captured = capsys.readouterr()
            assert "BoAmps Report saved to:" in captured.out
            assert "Total requests: 15" in captured.out
            assert "API requests: 10" in captured.out

    def test_client_without_tracking_raises(self) -> None:
        """Test that client without tracking raises error."""
        client = MagicMock(spec=[])  # No tracking attributes
        reporter = BoAmpsReporter(client=client)

        with pytest.raises(ValueError, match="does not have cumulative tracking"):
            reporter.generate_report()


class TestExportBoampsReportFunction:
    """Tests for the convenience export function."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock client."""
        client = MagicMock()
        client.model = "claude-3-5-sonnet"
        client.tracking_method = "ecologits"
        client.electricity_mix_zone = "WOR"
        client._get_provider_name.return_value = "anthropic"

        tracking = CumulativeTracking(
            api_request_count=5,
            api_energy_kwh=0.003,
            api_gwp_kgco2eq=0.0015,
            api_cost_usd=0.25,
            api_prompt_tokens=500,
            api_completion_tokens=250,
        )
        client.cumulative_tracking = tracking

        return client

    def test_export_function(self, mock_client: MagicMock) -> None:
        """Test the export convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            report = export_boamps_report(
                mock_client,
                filepath,
                publisher_name="Test",
                include_summary=False,
            )

            assert filepath.exists()
            assert report.task.algorithms[0].foundationModelName == "claude-3-5-sonnet"


class TestModelParameterEstimation:
    """Tests for model parameter estimation."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock client."""
        client = MagicMock()
        client.tracking_method = "ecologits"
        client.electricity_mix_zone = "WOR"
        client._get_provider_name.return_value = "openai"
        client.cumulative_tracking = CumulativeTracking()
        return client

    @pytest.mark.parametrize(
        "model,expected_params",
        [
            ("gpt-4.1", 1760),
            ("gpt-4", 1760),
            ("gpt-3.5-turbo", 175),
            ("claude-3-opus", 175),
            ("claude-3-5-sonnet", 70),
            ("claude-3-haiku", 20),
            ("llama-3-70b", 70),
            ("llama-3-8b", 8),
            ("unknown-model", None),
        ],
    )
    def test_model_parameters(
        self,
        mock_client: MagicMock,
        model: str,
        expected_params: float | None,
    ) -> None:
        """Test parameter estimation for various models."""
        mock_client.model = model
        reporter = BoAmpsReporter(client=mock_client)
        report = reporter.generate_report()

        assert report.task.algorithms[0].parametersNumber == expected_params


class TestCloudProviderDetection:
    """Tests for cloud provider detection."""

    @pytest.fixture
    def base_client(self) -> MagicMock:
        """Create base mock client."""
        client = MagicMock()
        client.model = "test-model"
        client.tracking_method = "ecologits"
        client.electricity_mix_zone = "WOR"
        client.cumulative_tracking = CumulativeTracking()
        return client

    @pytest.mark.parametrize(
        "provider,expected_service",
        [
            ("openai", "openai"),
            ("anthropic", "anthropic"),
            ("google", "google"),
            ("openrouter", "openrouter"),
            ("unknown", None),
        ],
    )
    def test_cloud_provider_detection(
        self,
        base_client: MagicMock,
        provider: str,
        expected_service: str | None,
    ) -> None:
        """Test cloud service detection from provider."""
        base_client._get_provider_name.return_value = provider
        reporter = BoAmpsReporter(client=base_client)
        report = reporter.generate_report()

        assert report.infrastructure.cloudService == expected_service


class TestCodeCarbonTracking:
    """Tests for CodeCarbon tracking method."""

    @pytest.fixture
    def codecarbon_client(self) -> MagicMock:
        """Create mock client with CodeCarbon tracking."""
        client = MagicMock()
        client.model = "llama-3-70b"
        client.tracking_method = "codecarbon"
        client.electricity_mix_zone = "USA"
        client._get_provider_name.return_value = "model_garden"
        client.cumulative_tracking = CumulativeTracking(
            api_request_count=100,
            api_energy_kwh=0.05,
            api_gwp_kgco2eq=0.025,
        )
        return client

    def test_codecarbon_tracking_mode(self, codecarbon_client: MagicMock) -> None:
        """Test that CodeCarbon sets tracking modes."""
        reporter = BoAmpsReporter(client=codecarbon_client)
        report = reporter.generate_report()

        assert report.measures[0].measurementMethod == "codecarbon"
        assert report.measures[0].cpuTrackingMode == "measured"
        assert report.measures[0].gpuTrackingMode == "measured"
