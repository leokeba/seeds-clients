"""BoAmps reporter for standardized energy consumption reports.

This module provides functionality to generate BoAmps-compliant JSON reports
for energy consumption of LLM inference tasks. The BoAmps format is a standardized
way of reporting energy consumption of AI/ML tasks.

See: https://github.com/Boavizta/BoAmps

The report includes:
- Header: Report metadata and publisher information
- Task: Algorithm and dataset information
- Measures: Energy consumption measurements
- Infrastructure: Hardware components information
- Environment: Location and power source information
- System: Operating system information
- Software: Programming language information
"""

import json
import os
import platform
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from seeds_clients.core.types import CumulativeTracking


# ============================================================================
# BoAmps Schema Models
# ============================================================================


class Publisher(BaseModel):
    """Publisher information for the report."""

    name: str | None = Field(default=None, description="Name of the organization")
    division: str | None = Field(
        default=None, description="Name of the publishing department"
    )
    projectName: str | None = Field(
        default=None, description="Name of the project"
    )
    confidentialityLevel: Literal["public", "internal", "confidential", "secret"] = Field(
        default="public", description="Confidentiality level of the report"
    )


class Header(BaseModel):
    """Header section of the BoAmps report."""

    licensing: str = Field(
        default="CC-BY-4.0",
        description="Type of licensing applicable for sharing the report",
    )
    formatVersion: str = Field(
        default="1.0.0",
        description="Version of the BoAmps specification",
    )
    formatVersionSpecificationUri: str = Field(
        default="https://github.com/Boavizta/BoAmps",
        description="URI of the BoAmps specification",
    )
    reportId: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier of this report",
    )
    reportDatetime: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="Publishing date of this report",
    )
    reportStatus: Literal["draft", "final", "corrective", "other"] = Field(
        default="final", description="Status of this report"
    )
    publisher: Publisher | None = Field(default=None, description="Publisher details")


class Algorithm(BaseModel):
    """Algorithm information for the BoAmps report."""

    algorithmType: str = Field(
        default="llm",
        description="Type of algorithm (e.g., llm, embeddings, nlp)",
    )
    algorithmName: str | None = Field(
        default=None,
        description="Name of the algorithm",
    )
    foundationModelName: str | None = Field(
        default=None,
        description="Name of the foundation model (e.g., gpt-4.1, claude-3-5-sonnet)",
    )
    foundationModelUri: str | None = Field(
        default=None,
        description="URI of the foundation model",
    )
    parametersNumber: float | None = Field(
        default=None,
        description="Number of billions of parameters",
    )
    framework: str = Field(
        default="seeds-clients",
        description="Software framework used",
    )
    frameworkVersion: str | None = Field(
        default=None,
        description="Version of the framework",
    )
    quantization: str | None = Field(
        default=None,
        description="Type of quantization used (fp32, fp16, int8, etc.)",
    )


class Dataset(BaseModel):
    """Dataset information for the BoAmps report."""

    dataType: str = Field(
        default="text",
        description="Type of data (text, image, audio, video, tabular, etc.)",
    )
    dataFormat: str | None = Field(
        default=None,
        description="Format of the data (json, csv, etc.)",
    )
    inputSize: int | None = Field(
        default=None,
        description="Size of input data (tokens, pixels, samples, etc.)",
    )
    outputSize: int | None = Field(
        default=None,
        description="Size of output data",
    )
    datasetName: str | None = Field(
        default=None,
        description="Name of the dataset if applicable",
    )


class Task(BaseModel):
    """Task section of the BoAmps report."""

    taskStage: str = Field(
        default="inference",
        description="Stage of the task (inference, training, etc.)",
    )
    taskFamily: str = Field(
        default="textGeneration",
        description="Family of the task (textGeneration, imageClassification, etc.)",
    )
    nbRequest: int = Field(
        default=0,
        description="Number of inference requests",
    )
    algorithms: list[Algorithm] = Field(
        default_factory=list,
        description="List of algorithms used",
    )
    dataset: list[Dataset] = Field(
        default_factory=list,
        description="List of datasets used",
    )
    measuredAccuracy: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Measured accuracy of the model (0-1)",
    )
    taskDescription: str | None = Field(
        default=None,
        description="Free-form description of the task",
    )


class Measure(BaseModel):
    """Measure section of the BoAmps report."""

    measurementMethod: str = Field(
        description="Method used for measurement (ecologits, codecarbon, etc.)",
    )
    version: str | None = Field(
        default=None,
        description="Version of the measurement tool",
    )
    cpuTrackingMode: str | None = Field(
        default=None,
        description="CPU tracking mode (constant, rapl, etc.)",
    )
    gpuTrackingMode: str | None = Field(
        default=None,
        description="GPU tracking mode (constant, nvml, etc.)",
    )
    averageUtilizationCpu: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Average CPU utilization (0-1)",
    )
    averageUtilizationGpu: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Average GPU utilization (0-1)",
    )
    powerCalibrationMeasurement: float | None = Field(
        default=None,
        description="Power consumed during calibration (kWh)",
    )
    durationCalibrationMeasurement: float | None = Field(
        default=None,
        description="Duration of calibration (seconds)",
    )
    powerConsumption: float = Field(
        ge=0,
        description="Power consumption of the task (kWh)",
    )
    measurementDuration: float | None = Field(
        default=None,
        description="Duration of the measurement (seconds)",
    )
    measurementDateTime: str | None = Field(
        default=None,
        description="Date when measurement began (YYYY-MM-DD HH:MM:SS)",
    )


class HardwareComponent(BaseModel):
    """Hardware component information for the BoAmps report."""

    componentType: Literal["cpu", "gpu", "ram", "storage", "other"] = Field(
        description="Type of hardware component",
    )
    componentName: str | None = Field(
        default=None,
        description="Name/model of the component",
    )
    manufacturer: str | None = Field(
        default=None,
        description="Manufacturer of the component",
    )
    tdp: float | None = Field(
        default=None,
        description="Thermal Design Power in watts",
    )
    nbCores: int | None = Field(
        default=None,
        description="Number of cores (for CPU/GPU)",
    )
    memorySize: float | None = Field(
        default=None,
        description="Memory size in GB",
    )


class Infrastructure(BaseModel):
    """Infrastructure section of the BoAmps report."""

    infraType: Literal["publicCloud", "privateCloud", "onPremise", "other"] = Field(
        default="publicCloud",
        description="Type of infrastructure",
    )
    cloudProvider: str | None = Field(
        default=None,
        description="Cloud provider name (aws, azure, google, etc.)",
    )
    cloudInstance: str | None = Field(
        default=None,
        description="Cloud instance type",
    )
    cloudService: str | None = Field(
        default=None,
        description="Cloud AI service name (e.g., OpenAI API)",
    )
    components: list[HardwareComponent] = Field(
        default_factory=list,
        description="List of hardware components",
    )


class Environment(BaseModel):
    """Environment section of the BoAmps report."""

    country: str = Field(
        default="WOR",
        description="Country code (ISO 3166-1 alpha-3)",
    )
    latitude: float | None = Field(default=None, description="Latitude")
    longitude: float | None = Field(default=None, description="Longitude")
    location: str | None = Field(
        default=None,
        description="More precise location (city, region, datacenter)",
    )
    powerSupplierType: Literal["public", "private", "internal", "other"] | None = Field(
        default=None,
        description="Type of power supplier",
    )
    powerSource: Literal[
        "solar", "wind", "nuclear", "hydroelectric", "gas", "coal", "other"
    ] | None = Field(default=None, description="Primary power source")
    powerSourceCarbonIntensity: float | None = Field(
        default=None,
        description="Carbon intensity of electricity (gCO2eq/kWh)",
    )


class SystemInfo(BaseModel):
    """System information section of the BoAmps report."""

    os: str = Field(description="Operating system name")
    distribution: str | None = Field(
        default=None,
        description="Distribution of the OS",
    )
    distributionVersion: str | None = Field(
        default=None,
        description="Version of the distribution",
    )


class SoftwareInfo(BaseModel):
    """Software/programming language section of the BoAmps report."""

    language: str = Field(
        default="python",
        description="Programming language used",
    )
    version: str | None = Field(
        default=None,
        description="Version of the programming language",
    )


class BoAmpsReport(BaseModel):
    """Complete BoAmps energy consumption report."""

    header: Header = Field(default_factory=Header)
    task: Task = Field(default_factory=Task)
    measures: list[Measure] = Field(default_factory=list)
    infrastructure: Infrastructure = Field(default_factory=Infrastructure)
    environment: Environment = Field(default_factory=Environment)
    system: SystemInfo | None = Field(default=None)
    software: SoftwareInfo | None = Field(default=None)
    quality: Literal["high", "medium", "low"] | None = Field(
        default="medium",
        description="Quality of the information provided",
    )

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return self.model_dump_json(indent=indent, exclude_none=True)

    def save(self, path: str | Path) -> None:
        """Save report to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())


# ============================================================================
# BoAmps Reporter
# ============================================================================


def _get_cloud_provider_from_model(model: str, provider: str) -> str | None:
    """Infer cloud provider/service from model and provider name."""
    provider_lower = provider.lower()
    if provider_lower in ("openai", "openai_client"):
        return "openai"
    elif provider_lower in ("anthropic", "anthropic_client"):
        return "anthropic"
    elif provider_lower in ("google", "google_client"):
        return "google"
    elif provider_lower in ("openrouter",):
        return "openrouter"
    return None


def _get_model_parameters(model: str) -> float | None:
    """Estimate parameter count from model name."""
    model_lower = model.lower()
    
    # GPT models
    if "gpt-4" in model_lower or "gpt4" in model_lower:
        return 1760  # Estimated
    if "gpt-3.5" in model_lower:
        return 175
    
    # Claude models
    if "opus" in model_lower:
        return 175  # Estimated
    if "sonnet" in model_lower:
        return 70  # Estimated
    if "haiku" in model_lower:
        return 20  # Estimated
    
    # Gemini models
    if "gemini-2" in model_lower or "gemini-1.5" in model_lower:
        return 200  # Estimated
    
    # Llama models
    if "70b" in model_lower:
        return 70
    if "8b" in model_lower:
        return 8
    if "3b" in model_lower:
        return 3
    
    return None


class BoAmpsReporter:
    """Reporter for generating BoAmps-compliant energy consumption reports.

    This class generates standardized JSON reports following the BoAmps format
    for energy consumption of LLM inference tasks.

    Example:
        ```python
        from seeds_clients import OpenAIClient, Message
        from seeds_clients.tracking.boamps_reporter import BoAmpsReporter

        client = OpenAIClient(model="gpt-4.1", cache_dir="./cache")

        # Make some requests
        for i in range(10):
            client.generate([Message(role="user", content=f"Question {i}")])

        # Generate BoAmps report
        reporter = BoAmpsReporter(
            client=client,
            publisher_name="My Organization",
            task_description="LLM inference for question answering",
        )

        report = reporter.generate_report()
        report.save("energy_report.json")
        ```
    """

    infrastructure_type: Literal["publicCloud", "privateCloud", "onPremise", "other"]
    quality: Literal["high", "medium", "low"]

    def __init__(
        self,
        client: Any,
        publisher_name: str | None = None,
        publisher_division: str | None = None,
        project_name: str | None = None,
        task_description: str | None = None,
        task_family: str = "textGeneration",
        data_type: str = "text",
        infrastructure_type: Literal[
            "publicCloud", "privateCloud", "onPremise", "other"
        ] = "publicCloud",
        quality: Literal["high", "medium", "low"] = "medium",
        include_system_info: bool = True,
        calibration_energy_kwh: float | None = None,
        calibration_duration_seconds: float | None = None,
    ) -> None:
        """Initialize the BoAmps reporter.

        Args:
            client: The LLM client with cumulative tracking data.
            publisher_name: Name of the organization publishing the report.
            publisher_division: Division/department within the organization.
            project_name: Name of the project.
            task_description: Description of the task being measured.
            task_family: Family of the task (textGeneration, imageClassification, etc.).
            data_type: Type of data (text, image, audio, etc.).
            infrastructure_type: Type of infrastructure used.
            quality: Quality level of the information provided.
            include_system_info: Whether to include system/software info.
            calibration_energy_kwh: Energy consumed during calibration (if any).
            calibration_duration_seconds: Duration of calibration (if any).
        """
        self.client = client
        self.publisher_name = publisher_name
        self.publisher_division = publisher_division
        self.project_name = project_name
        self.task_description = task_description
        self.task_family = task_family
        self.data_type = data_type
        self.infrastructure_type = infrastructure_type
        self.quality = quality
        self.include_system_info = include_system_info
        self.calibration_energy_kwh = calibration_energy_kwh
        self.calibration_duration_seconds = calibration_duration_seconds

    def _get_tracking(self) -> CumulativeTracking:
        """Get cumulative tracking from client."""
        if hasattr(self.client, "cumulative_tracking"):
            return self.client.cumulative_tracking
        elif hasattr(self.client, "_cumulative_tracking"):
            return self.client._cumulative_tracking
        else:
            raise ValueError("Client does not have cumulative tracking data")

    def _get_model(self) -> str:
        """Get model name from client."""
        return getattr(self.client, "model", "unknown")

    def _get_provider(self) -> str:
        """Get provider name from client."""
        if hasattr(self.client, "_get_provider_name"):
            return self.client._get_provider_name()
        return getattr(self.client, "provider", "unknown")

    def _get_tracking_method(self) -> str:
        """Get tracking method from client."""
        return getattr(self.client, "tracking_method", "ecologits")

    def _get_electricity_mix_zone(self) -> str:
        """Get electricity mix zone from client."""
        return getattr(self.client, "electricity_mix_zone", "WOR")

    def _build_header(self) -> Header:
        """Build the header section."""
        publisher = None
        if self.publisher_name or self.publisher_division or self.project_name:
            publisher = Publisher(
                name=self.publisher_name,
                division=self.publisher_division,
                projectName=self.project_name,
            )

        return Header(publisher=publisher)

    def _build_task(self, tracking: CumulativeTracking) -> Task:
        """Build the task section."""
        model = self._get_model()
        provider = self._get_provider()

        algorithm = Algorithm(
            algorithmType="llm",
            foundationModelName=model,
            parametersNumber=_get_model_parameters(model),
        )

        # Build dataset info from tracking
        dataset = Dataset(
            dataType=self.data_type,
            inputSize=tracking.total_prompt_tokens,
            outputSize=tracking.total_completion_tokens,
        )

        return Task(
            taskStage="inference",
            taskFamily=self.task_family,
            nbRequest=tracking.total_request_count,
            algorithms=[algorithm],
            dataset=[dataset],
            taskDescription=self.task_description,
        )

    def _build_measures(self, tracking: CumulativeTracking) -> list[Measure]:
        """Build the measures section."""
        tracking_method = self._get_tracking_method()

        measure = Measure(
            measurementMethod=tracking_method,
            powerConsumption=tracking.api_energy_kwh,
            measurementDateTime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Add calibration if provided
        if self.calibration_energy_kwh is not None:
            measure.powerCalibrationMeasurement = self.calibration_energy_kwh
        if self.calibration_duration_seconds is not None:
            measure.durationCalibrationMeasurement = self.calibration_duration_seconds

        # Add tracking mode info for CodeCarbon
        if tracking_method == "codecarbon":
            measure.cpuTrackingMode = "measured"
            measure.gpuTrackingMode = "measured"

        return [measure]

    def _build_infrastructure(self) -> Infrastructure:
        """Build the infrastructure section."""
        model = self._get_model()
        provider = self._get_provider()

        cloud_service = _get_cloud_provider_from_model(model, provider)

        # For API-based LLMs, we don't know the exact hardware
        # We indicate this is a cloud service
        components = [
            HardwareComponent(
                componentType="gpu",
                componentName="Unknown (Cloud API)",
            )
        ]

        return Infrastructure(
            infraType=self.infrastructure_type,
            cloudService=cloud_service,
            components=components,
        )

    def _build_environment(self) -> Environment:
        """Build the environment section."""
        electricity_zone = self._get_electricity_mix_zone()

        return Environment(
            country=electricity_zone,
        )

    def _build_system_info(self) -> SystemInfo | None:
        """Build the system section."""
        if not self.include_system_info:
            return None

        system = platform.system()
        release = platform.release()

        return SystemInfo(
            os=system,
            distributionVersion=release,
        )

    def _build_software_info(self) -> SoftwareInfo | None:
        """Build the software section."""
        if not self.include_system_info:
            return None

        import sys

        return SoftwareInfo(
            language="python",
            version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )

    def generate_report(self) -> BoAmpsReport:
        """Generate a complete BoAmps report from the client's cumulative tracking.

        Returns:
            BoAmpsReport object containing all energy consumption data.

        Raises:
            ValueError: If client does not have cumulative tracking data.
        """
        tracking = self._get_tracking()

        return BoAmpsReport(
            header=self._build_header(),
            task=self._build_task(tracking),
            measures=self._build_measures(tracking),
            infrastructure=self._build_infrastructure(),
            environment=self._build_environment(),
            system=self._build_system_info(),
            software=self._build_software_info(),
            quality=self.quality,
        )

    def export(
        self,
        output_path: str | Path,
        include_summary: bool = False,
    ) -> BoAmpsReport:
        """Generate and save a BoAmps report.

        Args:
            output_path: Path where to save the JSON report.
            include_summary: Whether to print a summary to console.

        Returns:
            The generated BoAmpsReport object.
        """
        report = self.generate_report()
        report.save(output_path)

        if include_summary:
            tracking = self._get_tracking()
            print(f"BoAmps Report saved to: {output_path}")
            print(f"  Total requests: {tracking.total_request_count}")
            print(f"  API requests: {tracking.api_request_count}")
            print(f"  Cached requests: {tracking.cached_request_count}")
            print(f"  Energy (API): {tracking.api_energy_kwh:.6f} kWh")
            print(f"  GWP (API): {tracking.api_gwp_kgco2eq:.6f} kgCO2eq")
            print(f"  Cost (API): ${tracking.api_cost_usd:.4f}")

        return report


def export_boamps_report(
    client: Any,
    output_path: str | Path,
    *,
    publisher_name: str | None = None,
    task_description: str | None = None,
    task_family: str = "textGeneration",
    include_summary: bool = True,
    **kwargs: Any,
) -> BoAmpsReport:
    """Convenience function to generate and export a BoAmps report.

    Args:
        client: The LLM client with cumulative tracking data.
        output_path: Path where to save the JSON report.
        publisher_name: Name of the organization.
        task_description: Description of the task.
        task_family: Family of the task.
        include_summary: Whether to print a summary.
        **kwargs: Additional arguments passed to BoAmpsReporter.

    Returns:
        The generated BoAmpsReport object.

    Example:
        ```python
        from seeds_clients import OpenAIClient, Message
        from seeds_clients.tracking.boamps_reporter import export_boamps_report

        client = OpenAIClient(model="gpt-4.1")

        # Make some requests
        client.generate([Message(role="user", content="Hello!")])

        # Export report
        report = export_boamps_report(
            client,
            "energy_report.json",
            publisher_name="My Org",
            task_description="LLM inference testing",
        )
        ```
    """
    reporter = BoAmpsReporter(
        client=client,
        publisher_name=publisher_name,
        task_description=task_description,
        task_family=task_family,
        **kwargs,
    )
    return reporter.export(output_path, include_summary=include_summary)
