# EcoLogits Tracking

EcoLogits provides model-based carbon emission estimates for LLM API calls.

::: seeds_clients.tracking.ecologits_tracker.EcoLogitsMixin
    options:
      show_root_heading: true
      members:
        - extract_carbon_metrics

::: seeds_clients.tracking.ecologits_tracker.EcoLogitsMetrics
    options:
      show_root_heading: true

## Overview

EcoLogits estimates carbon emissions based on:

- Model architecture and size
- Token counts (input/output)
- Electricity mix of the deployment region
- Request latency

## How It Works

1. **Model Matching**: EcoLogits maps model names to known architectures
2. **Energy Estimation**: Computes energy based on model size and tokens
3. **Carbon Calculation**: Applies regional electricity mix carbon intensity
4. **Phase Breakdown**: Separates usage phase vs. embodied emissions

## Configuration

```python
from seeds_clients import OpenAIClient

client = OpenAIClient(
    model="gpt-4.1",
    enable_tracking=True,          # Enable EcoLogits (default)
    electricity_mix_zone="WOR",    # World average
)
```

### Electricity Mix Zones

| Zone | Description | Carbon Intensity |
|------|-------------|------------------|
| `WOR` | World average | ~500 gCO2/kWh |
| `USA` | United States | ~400 gCO2/kWh |
| `FRA` | France | ~50 gCO2/kWh (nuclear) |
| `DEU` | Germany | ~400 gCO2/kWh |

## Accessing Metrics

### Per-Request

```python
response = client.generate([Message(role="user", content="Hello")])

tracking = response.tracking

# Total metrics
print(f"Energy: {tracking.energy_kwh:.6f} kWh")
print(f"Carbon: {tracking.gwp_kgco2eq:.6f} kgCO2eq")

# Usage phase (electricity consumption)
print(f"Usage energy: {tracking.energy_usage_kwh:.6f} kWh")
print(f"Usage carbon: {tracking.gwp_usage_kgco2eq:.6f} kgCO2eq")

# Embodied phase (manufacturing, infrastructure)
print(f"Embodied carbon: {tracking.gwp_embodied_kgco2eq:.6f} kgCO2eq")

# Additional environmental metrics
print(f"Primary energy: {tracking.pe_mj} MJ")
print(f"Abiotic depletion: {tracking.adpe_kgsbeq} kgSbeq")
```

### Cumulative

```python
tracking = client.cumulative_tracking

# Separate API vs cached emissions
print(f"API emissions: {tracking.api_gwp_kgco2eq:.6f} kgCO2eq")
print(f"Emissions avoided (cached): {tracking.emissions_avoided_kgco2eq:.6f} kgCO2eq")
```

## Warnings and Errors

EcoLogits may return warnings for unsupported models:

```python
if response.tracking.ecologits_warnings:
    for warning in response.tracking.ecologits_warnings:
        print(f"Warning: {warning}")

if response.tracking.ecologits_errors:
    for error in response.tracking.ecologits_errors:
        print(f"Error: {error}")
```

## Supported Providers

EcoLogits supports these providers:

- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude 3, Claude 3.5)
- Google (Gemini)
- Mistral
- Cohere
- And more...

## Disabling Tracking

```python
client = OpenAIClient(
    model="gpt-4.1",
    enable_tracking=False,  # Disable EcoLogits
)

response = client.generate(messages)
# response.tracking will have None for energy/carbon values
```

## Resources

- [EcoLogits Documentation](https://ecologits.ai/)
- [EcoLogits GitHub](https://github.com/genai-impact/ecologits)
