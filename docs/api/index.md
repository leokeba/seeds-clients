# API Reference

This section contains the complete API documentation for seeds-clients.

## Core Types

The fundamental types used throughout the library:

- [**Message**](types.md#message) - Represents a conversation message
- [**Response**](types.md#response) - API response with content and tracking
- [**TrackingData**](types.md#trackingdata) - Per-request tracking metrics
- [**CumulativeTracking**](types.md#cumulativetracking) - Aggregated tracking across requests
- [**Usage**](types.md#usage) - Token usage information
- [**BatchResult**](types.md#batchresult) - Results from batch processing

## Providers

Client implementations for each LLM provider:

- [**OpenAIClient**](providers/openai.md) - OpenAI GPT models
- [**AnthropicClient**](providers/anthropic.md) - Anthropic Claude models
- [**GoogleClient**](providers/google.md) - Google Gemini models
- [**OpenRouterClient**](providers/openrouter.md) - Multi-provider routing

## Tracking

Carbon and cost tracking utilities:

- [**EcoLogitsMixin**](tracking/ecologits.md) - Model-based carbon estimates
- [**CodeCarbonMixin**](tracking/codecarbon.md) - Hardware-measured emissions
- [**BoAmpsReporter**](tracking/boamps.md) - Standardized energy reports

## Exceptions

Exception hierarchy for error handling:

```python
from seeds_clients import (
    SeedsClientError,    # Base exception
    CacheError,          # Cache-related errors
    ProviderError,       # Provider API errors
    ValidationError,     # Input validation errors
)
```

## Quick Links

### Common Operations

```python
from seeds_clients import OpenAIClient, Message

# Initialize
client = OpenAIClient(model="gpt-4.1", cache_dir="./cache")

# Generate
response = client.generate([Message(role="user", content="Hello")])

# Async generate
response = await client.agenerate([Message(role="user", content="Hello")])

# Batch generate
result = await client.batch_generate(messages_list, max_concurrent=5)

# Structured output
data = client.generate_structured(messages, response_model=MyModel)

# Export BoAmps report
report = client.export_boamps_report("report.json")
```

### Accessing Tracking

```python
# Per-request
response.tracking.cost_usd
response.tracking.energy_kwh
response.tracking.gwp_kgco2eq

# Cumulative
client.cumulative_tracking.total_cost_usd
client.cumulative_tracking.cache_hit_rate
client.cumulative_tracking.emissions_avoided_kgco2eq
```
