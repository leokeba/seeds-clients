# Carbon Tracking Guide

Understanding and using carbon tracking in seeds-clients.

## Overview

seeds-clients provides comprehensive carbon tracking through:

- **EcoLogits**: Model-based emission estimates (default)
- **CodeCarbon**: Hardware-measured emissions (for self-hosted)
- **Cumulative Tracking**: Aggregated metrics across requests
- **BoAmps Reports**: Standardized energy consumption reports

## Why Track Carbon?

AI models consume significant energy. Understanding the environmental impact helps:

- Make informed model selection decisions
- Optimize prompts and reduce unnecessary calls
- Report sustainability metrics
- Reduce emissions through caching

## EcoLogits Tracking

EcoLogits is enabled by default and estimates emissions based on:

- Model architecture and size
- Token counts (input/output)
- Regional electricity carbon intensity
- Request latency

### Configuration

```python
from seeds_clients import OpenAIClient

client = OpenAIClient(
    model="gpt-4.1",
    enable_tracking=True,           # Default: True
    electricity_mix_zone="WOR",     # Default: World average
)
```

### Electricity Mix Zones

The zone affects carbon calculations:

| Zone | Description | ~Carbon Intensity |
|------|-------------|-------------------|
| `WOR` | World average | 500 gCO2/kWh |
| `USA` | United States | 400 gCO2/kWh |
| `FRA` | France (nuclear) | 50 gCO2/kWh |
| `DEU` | Germany | 400 gCO2/kWh |
| `CHN` | China | 600 gCO2/kWh |

### Accessing Metrics

```python
response = client.generate([Message(role="user", content="Hello")])

# Basic metrics
print(f"Energy: {response.tracking.energy_kwh:.6f} kWh")
print(f"Carbon: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq")

# Phase breakdown
print(f"Usage phase: {response.tracking.gwp_usage_kgco2eq:.6f} kgCO2eq")
print(f"Embodied: {response.tracking.gwp_embodied_kgco2eq:.6f} kgCO2eq")

# Additional environmental metrics
print(f"Primary energy: {response.tracking.pe_mj} MJ")
print(f"Abiotic depletion: {response.tracking.adpe_kgsbeq} kgSbeq")
```

## Cumulative Tracking

Track totals across multiple requests:

```python
# Make requests
for question in ["What is AI?", "What is ML?", "What is AI?"]:
    client.generate([Message(role="user", content=question)])

# Access cumulative data
tracking = client.cumulative_tracking

# Request counts
print(f"Total requests: {tracking.total_request_count}")
print(f"API requests: {tracking.api_request_count}")
print(f"Cached requests: {tracking.cached_request_count}")
print(f"Cache hit rate: {tracking.cache_hit_rate:.1%}")

# Emissions
print(f"Total emissions: {tracking.total_gwp_kgco2eq:.6f} kgCO2eq")
print(f"API emissions: {tracking.api_gwp_kgco2eq:.6f} kgCO2eq")
print(f"Avoided (cached): {tracking.emissions_avoided_kgco2eq:.6f} kgCO2eq")

# Costs
print(f"Total cost: ${tracking.total_cost_usd:.4f}")
```

### Reset Tracking

```python
# Start fresh tracking
client.reset_cumulative_tracking()
```

## Caching Reduces Emissions

Caching avoids duplicate API calls and their emissions:

```python
# First request: hits API
response1 = client.generate([Message(role="user", content="Hello")])
print(f"Emissions: {response1.tracking.gwp_kgco2eq:.6f} kgCO2eq")

# Second identical request: from cache (no new emissions)
response2 = client.generate([Message(role="user", content="Hello")])
print(f"Cached: {response2.cached}")  # True

# Check emissions avoided
print(f"Avoided: {client.cumulative_tracking.emissions_avoided_kgco2eq:.6f} kgCO2eq")
```

## Model Comparison

Compare emissions across models:

```python
from seeds_clients import OpenAIClient, Message

models = ["gpt-4.1", "gpt-4.1-mini", "gpt-3.5-turbo"]
prompt = [Message(role="user", content="Explain photosynthesis")]

for model in models:
    client = OpenAIClient(model=model, cache_dir=f"./cache_{model}")
    response = client.generate(prompt)
    
    print(f"\n{model}:")
    print(f"  Energy: {response.tracking.energy_kwh:.6f} kWh")
    print(f"  Carbon: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq")
    print(f"  Cost: ${response.tracking.cost_usd:.4f}")
```

## BoAmps Reports

Export standardized reports:

```python
# After making requests
report = client.export_boamps_report(
    output_path="energy_report.json",
    publisher_name="My Organization",
    task_description="LLM inference benchmark",
)

print(f"Report saved with {report.task.nbRequest} requests")
print(f"Total energy: {report.measures[0].powerConsumption} kWh")
```

## Best Practices

### 1. Choose Efficient Models

Smaller models use less energy:

```python
# For simple tasks, use smaller models
client = OpenAIClient(model="gpt-4.1-mini")  # vs gpt-4.1
```

### 2. Enable Caching

```python
client = OpenAIClient(
    model="gpt-4.1",
    cache_dir="./cache",  # Enable caching
)
```

### 3. Optimize Prompts

Shorter prompts = fewer tokens = less energy:

```python
# Less efficient
messages = [Message(role="user", content="Please provide a detailed explanation of...")]

# More efficient
messages = [Message(role="user", content="Explain briefly:")]
```

### 4. Batch Similar Requests

Process related requests together:

```python
result = await client.batch_generate(prompts, max_concurrent=5)
print(f"Total carbon: {result.total_gwp_kgco2eq:.6f} kgCO2eq")
```

### 5. Choose Low-Carbon Regions

If you control deployment:

```python
client = OpenAIClient(
    model="gpt-4.1",
    electricity_mix_zone="FRA",  # France has low-carbon grid
)
```

### 6. Monitor and Report

Track emissions over time:

```python
# Daily reporting
tracking = client.cumulative_tracking
daily_report = {
    "date": "2024-12-01",
    "requests": tracking.total_request_count,
    "api_calls": tracking.api_request_count,
    "cache_hits": tracking.cached_request_count,
    "emissions_kg": tracking.total_gwp_kgco2eq,
    "avoided_kg": tracking.emissions_avoided_kgco2eq,
    "cost_usd": tracking.total_cost_usd,
}
```

## Resources

- [EcoLogits](https://ecologits.ai/) - Model-based tracking
- [CodeCarbon](https://codecarbon.io/) - Hardware measurement
- [BoAmps](https://github.com/Boavizta/BoAmps) - Reporting standard
- [Boavizta](https://boavizta.org/) - Sustainability organization
