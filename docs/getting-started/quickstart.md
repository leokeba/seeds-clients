# Quick Start

This guide will get you up and running with seeds-clients in just a few minutes.

## Basic Usage

### 1. Initialize a Client

```python
from seeds_clients import OpenAIClient, Message

client = OpenAIClient(
    model="gpt-4.1",           # Model to use
    cache_dir="./cache",       # Enable caching
    enable_tracking=True,      # Enable carbon/cost tracking
)
```

### 2. Generate Text

```python
messages = [
    Message(role="user", content="What is machine learning?")
]

response = client.generate(messages)
print(response.content)
```

### 3. Access Tracking Data

```python
# Cost and usage
print(f"Cost: ${response.tracking.cost_usd:.4f}")
print(f"Tokens: {response.tracking.prompt_tokens} + {response.tracking.completion_tokens}")

# Carbon impact
print(f"Energy: {response.tracking.energy_kwh:.6f} kWh")
print(f"Carbon: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq")
```

## Using Different Providers

### Anthropic (Claude)

```python
from seeds_clients import AnthropicClient, Message

client = AnthropicClient(
    model="claude-sonnet-4-20250514",
    cache_dir="./cache",
)

response = client.generate([
    Message(role="user", content="Explain neural networks")
])
```

### Google (Gemini)

```python
from seeds_clients import GoogleClient, Message

client = GoogleClient(
    model="gemini-2.5-flash",
    cache_dir="./cache",
)

response = client.generate([
    Message(role="user", content="What is deep learning?")
])
```

### OpenRouter (Multiple Providers)

```python
from seeds_clients import OpenRouterClient, Message

client = OpenRouterClient(
    model="anthropic/claude-3.5-sonnet",  # provider/model format
    cache_dir="./cache",
)

response = client.generate([
    Message(role="user", content="Hello!")
])
```

## Structured Outputs

Extract structured data using Pydantic models:

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

person = client.generate_structured(
    messages=[Message(role="user", content="Extract: Alice is a 28-year-old engineer")],
    response_model=Person,
)

print(f"Name: {person.name}")
print(f"Age: {person.age}")
print(f"Occupation: {person.occupation}")
```

## Multimodal (Images)

Send images along with text:

```python
from PIL import Image

messages = [
    Message(
        role="user",
        content=[
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "source": Image.open("photo.jpg")},
        ]
    )
]

response = client.generate(messages)
print(response.content)
```

## Async and Batch Processing

For processing multiple requests efficiently:

```python
import asyncio

async def main():
    async with OpenAIClient(model="gpt-4.1-mini") as client:
        # Single async request
        response = await client.agenerate(
            messages=[Message(role="user", content="Hello!")]
        )
        
        # Batch processing
        prompts = [
            [Message(role="user", content="What is 1+1?")],
            [Message(role="user", content="What is 2+2?")],
            [Message(role="user", content="What is 3+3?")],
        ]
        
        result = await client.batch_generate(prompts, max_concurrent=3)
        
        print(f"Total cost: ${result.total_cost_usd:.4f}")
        print(f"Total carbon: {result.total_gwp_kgco2eq:.6f} kgCO2eq")

asyncio.run(main())
```

## Caching

Responses are automatically cached when you provide a `cache_dir`:

```python
# First request - hits the API
response1 = client.generate(messages)
print(f"Cached: {response1.cached}")  # False

# Second identical request - from cache
response2 = client.generate(messages)
print(f"Cached: {response2.cached}")  # True
```

Skip cache for a specific request:

```python
response = client.generate(messages, use_cache=False)
```

## Cumulative Tracking

Track totals across multiple requests:

```python
# Make several requests
for question in ["What is AI?", "What is ML?", "What is AI?"]:
    client.generate([Message(role="user", content=question)])

# Get cumulative stats
tracking = client.cumulative_tracking

print(f"Total requests: {tracking.total_request_count}")
print(f"Cache hit rate: {tracking.cache_hit_rate:.1%}")
print(f"Total cost: ${tracking.total_cost_usd:.4f}")
print(f"Emissions avoided: {tracking.emissions_avoided_kgco2eq:.6f} kgCO2eq")
```

## Next Steps

- Learn about [Configuration](configuration.md) options
- Explore provider-specific guides:
  - [OpenAI Guide](../guides/openai.md)
  - [Anthropic Guide](../guides/anthropic.md)
  - [Google Guide](../guides/google.md)
- Understand [Carbon Tracking](../guides/carbon-tracking.md)
- Set up [Batch Processing](../guides/batch-processing.md)
