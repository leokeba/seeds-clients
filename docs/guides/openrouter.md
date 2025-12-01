# OpenRouter Guide

Complete guide for using seeds-clients with OpenRouter's multi-provider API.

## Setup

### API Key

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### Initialize Client

```python
from seeds_clients import OpenRouterClient, Message

client = OpenRouterClient(
    model="openai/gpt-4.1",
    cache_dir="./cache",
)
```

## Model Format

OpenRouter uses `provider/model` format:

```python
# OpenAI models
client = OpenRouterClient(model="openai/gpt-4.1")
client = OpenRouterClient(model="openai/gpt-4o")
client = OpenRouterClient(model="openai/gpt-4.1-mini")

# Anthropic models
client = OpenRouterClient(model="anthropic/claude-3.5-sonnet")
client = OpenRouterClient(model="anthropic/claude-3-opus")

# Google models
client = OpenRouterClient(model="google/gemini-pro-1.5")

# Meta models
client = OpenRouterClient(model="meta-llama/llama-3.1-70b-instruct")
client = OpenRouterClient(model="meta-llama/llama-3.1-8b-instruct")

# Mistral models
client = OpenRouterClient(model="mistralai/mistral-large")
```

## Basic Usage

```python
response = client.generate([
    Message(role="user", content="Hello!")
])

print(response.content)
print(f"Cost: ${response.tracking.cost_usd:.4f}")
```

## Provider Comparison

Use OpenRouter to compare models:

```python
models = [
    "openai/gpt-4.1-mini",
    "anthropic/claude-3.5-haiku",
    "google/gemini-flash-1.5",
]

for model in models:
    client = OpenRouterClient(
        model=model,
        cache_dir=f"./cache/{model.replace('/', '_')}"
    )
    
    response = client.generate([
        Message(role="user", content="Write a haiku about AI")
    ])
    
    print(f"\n{model}:")
    print(response.content)
    print(f"Cost: ${response.tracking.cost_usd:.6f}")
```

## Carbon Tracking

OpenRouter carbon tracking uses the underlying model's provider:

```python
client = OpenRouterClient(model="openai/gpt-4.1")

response = client.generate([Message(role="user", content="Hello")])

# EcoLogits uses "openai" as provider
print(f"Energy: {response.tracking.energy_kwh:.6f} kWh")
print(f"Carbon: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq")
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

## Benefits of OpenRouter

1. **Single API key** for multiple providers
2. **Unified billing** across providers
3. **Model fallbacks** and routing
4. **Cost comparison** between providers
5. **Access to open models** (Llama, Mistral)
