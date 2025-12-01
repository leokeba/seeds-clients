# BoAmps Reporter

Generate standardized energy consumption reports following the BoAmps format.

::: seeds_clients.tracking.boamps_reporter.BoAmpsReporter
    options:
      show_root_heading: true
      members:
        - __init__
        - generate_report
        - export

::: seeds_clients.tracking.boamps_reporter.BoAmpsReport
    options:
      show_root_heading: true
      members:
        - to_json
        - save

## Overview

[BoAmps](https://github.com/Boavizta/BoAmps) is a standardized format for reporting energy consumption of AI/ML tasks, developed by [Boavizta](https://boavizta.org/).

Reports include:

- **Header**: Metadata, licensing, publisher info
- **Task**: Algorithm, dataset, request counts
- **Measures**: Energy consumption, duration
- **Infrastructure**: Cloud provider, hardware
- **Environment**: Location, power source
- **System/Software**: OS and language info

## Quick Export

```python
from seeds_clients import OpenAIClient, Message

client = OpenAIClient(model="gpt-4.1", cache_dir="./cache")

# Make some requests
for i in range(10):
    client.generate([Message(role="user", content=f"Question {i}")])

# Export report
report = client.export_boamps_report(
    output_path="energy_report.json",
    publisher_name="My Organization",
    task_description="LLM inference for Q&A",
)
```

## Detailed Configuration

```python
from seeds_clients.tracking import BoAmpsReporter

reporter = BoAmpsReporter(
    client=client,
    
    # Publisher information
    publisher_name="Research Lab",
    publisher_division="AI Team",
    project_name="Sustainability Study",
    
    # Task description
    task_description="Text generation benchmark",
    task_family="textGeneration",
    data_type="text",
    
    # Infrastructure
    infrastructure_type="publicCloud",  # publicCloud, privateCloud, onPremise, other
    
    # Quality indicator
    quality="high",  # high, medium, low
    
    # System info
    include_system_info=True,
    
    # Calibration (optional)
    calibration_energy_kwh=0.0001,
    calibration_duration_seconds=60.0,
)

report = reporter.generate_report()
report.save("detailed_report.json")
```

## Report Structure

```json
{
  "header": {
    "licensing": "CC-BY-4.0",
    "formatVersion": "1.0.0",
    "reportId": "uuid-...",
    "reportDatetime": "2024-12-01 10:00:00",
    "reportStatus": "final",
    "publisher": {
      "name": "My Organization",
      "projectName": "AI Study"
    }
  },
  "task": {
    "taskStage": "inference",
    "taskFamily": "textGeneration",
    "nbRequest": 10,
    "algorithms": [{
      "algorithmType": "llm",
      "foundationModelName": "gpt-4.1",
      "framework": "seeds-clients"
    }],
    "dataset": [{
      "dataType": "text",
      "inputSize": 1000,
      "outputSize": 5000
    }]
  },
  "measures": [{
    "measurementMethod": "ecologits",
    "powerConsumption": 0.000123,
    "measurementDateTime": "2024-12-01 10:00:00"
  }],
  "infrastructure": {
    "infraType": "publicCloud",
    "cloudService": "openai"
  },
  "environment": {
    "country": "WOR"
  },
  "quality": "high"
}
```

## Accessing Report Data

```python
report = client.export_boamps_report("report.json")

# Header
print(f"Report ID: {report.header.reportId}")
print(f"Date: {report.header.reportDatetime}")

# Task
print(f"Requests: {report.task.nbRequest}")
print(f"Model: {report.task.algorithms[0].foundationModelName}")

# Measures
print(f"Energy: {report.measures[0].powerConsumption} kWh")

# Infrastructure
print(f"Provider: {report.infrastructure.cloudService}")
```

## Export Function

Convenience function for quick exports:

```python
from seeds_clients.tracking import export_boamps_report

report = export_boamps_report(
    client=client,
    output_path="report.json",
    publisher_name="My Org",
    task_description="LLM inference",
    include_summary=True,  # Print summary to console
)
```

## Resources

- [BoAmps Specification](https://github.com/Boavizta/BoAmps)
- [Boavizta](https://boavizta.org/)
