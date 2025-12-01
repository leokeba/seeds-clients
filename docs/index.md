# 🌱 seeds-clients

A modern Python library providing unified LLM clients with integrated carbon impact tracking, cost monitoring, and intelligent local caching.

## Overview

`seeds-clients` provides a unified interface to multiple LLM providers (OpenAI, Anthropic, Google, etc.) with built-in environmental impact tracking and cost monitoring. The library is designed for researchers and developers who want to understand and optimize the carbon footprint of their AI applications.

## Key Features

- 🔌 **Unified API** across multiple LLM providers
- 🌍 **Carbon Tracking** with EcoLogits and CodeCarbon integration
- 💰 **Cost Monitoring** with detailed usage analytics
- 💾 **Smart Caching** using raw API responses for maximum flexibility
- 📊 **BoAmps Reporting** standardized energy consumption reports
- 🖼️ **Multimodal Support** for text and image inputs
- 🔧 **Structured Outputs** with Pydantic model validation
- 📈 **Batch Processing** with parallel execution and aggregated tracking

## Quick Example

```python
from seeds_clients import OpenAIClient, Message

# Initialize client with caching and tracking
client = OpenAIClient(
    model="gpt-4.1",
    cache_dir="./cache",
    enable_tracking=True,
)

# Generate text
messages = [Message(role="user", content="Explain quantum computing")]
response = client.generate(messages)

print(response.content)
print(f"Cost: ${response.tracking.cost_usd:.4f}")
print(f"Carbon: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq")
print(f"Energy: {response.tracking.energy_kwh:.6f} kWh")
```

## Installation

```bash
pip install seeds-clients
```

See the [Installation Guide](getting-started/installation.md) for more options.

## Supported Providers

| Provider | Status | Structured Outputs | Multimodal |
|----------|--------|-------------------|------------|
| OpenAI | ✅ Stable | ✅ | ✅ |
| Anthropic | ✅ Stable | ✅ | ✅ |
| Google | ✅ Stable | ✅ | ✅ |
| OpenRouter | ✅ Stable | ⚠️ Model-dependent | ⚠️ Model-dependent |
| Mistral | 🚧 Planned | - | - |

## Why seeds-clients?

### Environmental Impact Awareness

AI models consume significant energy. seeds-clients helps you:

- Track carbon emissions per request
- Compare environmental impact across models
- Generate standardized BoAmps reports
- Reduce emissions through intelligent caching

### Cost Optimization

- Automatic cost tracking per request
- Cumulative cost monitoring
- Cache hit rates to identify savings
- Per-provider cost comparison

### Developer Experience

- Consistent API across all providers
- Type-safe with Pydantic models
- Async support with batch processing
- Comprehensive error handling

## License

MIT License - see [LICENSE](https://github.com/leokeba/seeds-clients/blob/main/LICENSE) for details.
