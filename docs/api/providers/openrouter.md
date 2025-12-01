# OpenRouterClient

Client for OpenRouter, providing access to multiple LLM providers through a unified API.

::: seeds_clients.providers.openrouter.OpenRouterClient
    options:
      show_root_heading: true
      members:
        - __init__
        - generate
        - agenerate
        - batch_generate
        - batch_generate_iter
        - cumulative_tracking

## Initialization

```python
from seeds_clients import OpenRouterClient

client = OpenRouterClient(
    model="openai/gpt-4.1",      # Required: provider/model format
    api_key="sk-or-...",         # Optional: uses OPENROUTER_API_KEY env var
    cache_dir="./cache",         # Optional: enables caching
    ttl_hours=24.0,              # Optional: cache TTL
    enable_tracking=True,        # Optional: enables carbon tracking
)
```

## Model Format

OpenRouter uses `provider/model` format:

```python
# OpenAI models
client = OpenRouterClient(model="openai/gpt-4.1")
client = OpenRouterClient(model="openai/gpt-4o")

# Anthropic models
client = OpenRouterClient(model="anthropic/claude-3.5-sonnet")
client = OpenRouterClient(model="anthropic/claude-3-opus")

# Google models
client = OpenRouterClient(model="google/gemini-pro-1.5")

# Meta models
client = OpenRouterClient(model="meta-llama/llama-3.1-70b-instruct")
```

## Basic Generation

```python
from seeds_clients import Message

response = client.generate([
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Hello!"),
])

print(response.content)
print(f"Cost: ${response.tracking.cost_usd:.4f}")
```

## Carbon Tracking

OpenRouter tracking uses the underlying model's provider for EcoLogits calculations:

```python
client = OpenRouterClient(model="openai/gpt-4.1")

response = client.generate([Message(role="user", content="Hello")])

# EcoLogits uses "openai" provider and "gpt-4.1" model
print(f"Energy: {response.tracking.energy_kwh:.6f} kWh")
print(f"Carbon: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq")
```

## Cost Tracking

OpenRouter provides cost data in the response:

```python
response = client.generate([Message(role="user", content="Hello")])

# Cost from OpenRouter's usage data
print(f"Cost: ${response.tracking.cost_usd:.4f}")
```

## Async Usage

```python
import asyncio

async def main():
    async with OpenRouterClient(model="openai/gpt-4.1-mini") as client:
        response = await client.agenerate([
            Message(role="user", content="Hello!")
        ])
        print(response.content)

asyncio.run(main())
```

## Provider Comparison

Use OpenRouter to compare models across providers:

```python
models = [
    "openai/gpt-4.1-mini",
    "anthropic/claude-3.5-haiku",
    "google/gemini-flash-1.5",
]

for model in models:
    client = OpenRouterClient(model=model, cache_dir=f"./cache/{model}")
    response = client.generate([Message(role="user", content="Hello")])
    print(f"{model}: ${response.tracking.cost_usd:.6f}")
```

## Generation Parameters

```python
response = client.generate(
    messages,
    temperature=0.7,      # Creativity
    max_tokens=1000,      # Maximum output length
    top_p=0.95,           # Nucleus sampling
    use_cache=True,       # Use cache (default: True)
)
```

## Limitations

- Structured outputs depend on the underlying model's support
- Multimodal support depends on the underlying model
- Some advanced features may not be available for all models
