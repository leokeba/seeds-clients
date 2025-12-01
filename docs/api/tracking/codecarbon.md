# CodeCarbon Tracking

CodeCarbon provides hardware-measured carbon emissions for self-hosted models.

::: seeds_clients.tracking.codecarbon_tracker.CodeCarbonMixin
    options:
      show_root_heading: true
      members:
        - extract_carbon_metrics

::: seeds_clients.tracking.codecarbon_tracker.CodeCarbonMetrics
    options:
      show_root_heading: true

## Overview

CodeCarbon measures actual power consumption from:

- CPU usage and power
- GPU usage and power
- RAM usage and power

This provides more accurate measurements than model-based estimates, but requires server instrumentation.

## How It Works

1. **Server Instrumentation**: The model server runs CodeCarbon tracking
2. **Response Metadata**: Carbon data is included in API responses as `x_carbon_trace`
3. **Client Extraction**: seeds-clients extracts and normalizes the data

## Response Format

CodeCarbon data is extracted from the `x_carbon_trace` field:

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
    "measured": true,
    "tracking_active": true
  }
}
```

## Accessing Metrics

```python
response = client.generate([Message(role="user", content="Hello")])

tracking = response.tracking

# Check if CodeCarbon data is available
if tracking.tracking_method == "codecarbon":
    # Total metrics
    print(f"Energy: {tracking.energy_kwh:.6f} kWh")
    print(f"Carbon: {tracking.gwp_kgco2eq:.6f} kgCO2eq")
    
    # Hardware breakdown
    print(f"CPU energy: {tracking.cpu_energy_kwh:.6f} kWh")
    print(f"GPU energy: {tracking.gpu_energy_kwh:.6f} kWh")
    print(f"RAM energy: {tracking.ram_energy_kwh:.6f} kWh")
    
    # Power measurements
    print(f"CPU power: {tracking.cpu_power_watts:.1f} W")
    print(f"GPU power: {tracking.gpu_power_watts:.1f} W")
```

## Use Cases

CodeCarbon is ideal for:

- **Self-hosted models**: Running Llama, Mistral, etc. on your own hardware
- **Model Garden**: Services that provide CodeCarbon instrumentation
- **Research**: When accurate measurements are required
- **Optimization**: Identifying power-hungry operations

## Comparison with EcoLogits

| Feature | EcoLogits | CodeCarbon |
|---------|-----------|------------|
| Measurement | Estimated | Measured |
| Server setup | None | Required |
| Provider support | API providers | Self-hosted |
| Hardware breakdown | No | Yes |
| Accuracy | Model-based | Hardware-based |

## Resources

- [CodeCarbon Documentation](https://codecarbon.io/)
- [CodeCarbon GitHub](https://github.com/mlco2/codecarbon)
