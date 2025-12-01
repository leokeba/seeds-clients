# AnthropicClient

Client for Anthropic's Claude models.

::: seeds_clients.providers.anthropic.AnthropicClient
    options:
      show_root_heading: true
      members:
        - __init__
        - generate
        - agenerate
        - generate_structured
        - batch_generate
        - batch_generate_iter
        - cumulative_tracking

## Initialization

```python
from seeds_clients import AnthropicClient

client = AnthropicClient(
    model="claude-sonnet-4-20250514",  # Required: model name
    api_key="sk-ant-...",              # Optional: uses ANTHROPIC_API_KEY env var
    cache_dir="./cache",               # Optional: enables caching
    ttl_hours=24.0,                    # Optional: cache TTL
    enable_tracking=True,              # Optional: enables carbon tracking
)
```

## Supported Models

| Model | Description |
|-------|-------------|
| `claude-sonnet-4-20250514` | Claude Sonnet 4 |
| `claude-3-5-sonnet-20241022` | Claude 3.5 Sonnet |
| `claude-3-5-haiku-20241022` | Claude 3.5 Haiku |
| `claude-3-opus-20240229` | Claude 3 Opus |

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

## Structured Outputs

Anthropic uses tool_use for structured outputs:

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

person = client.generate_structured(
    messages=[Message(role="user", content="Alice is 30 years old")],
    response_model=Person,
)

print(f"Name: {person.name}, Age: {person.age}")
```

## Multimodal (Vision)

```python
from PIL import Image

response = client.generate([
    Message(
        role="user",
        content=[
            {"type": "text", "text": "Describe this image"},
            {"type": "image", "source": Image.open("photo.jpg")},
        ]
    )
])
```

## Async Usage

```python
import asyncio

async def main():
    async with AnthropicClient(model="claude-sonnet-4-20250514") as client:
        response = await client.agenerate([
            Message(role="user", content="Hello!")
        ])
        print(response.content)

asyncio.run(main())
```

## System Messages

Anthropic handles system messages separately:

```python
response = client.generate([
    Message(role="system", content="You are a pirate."),
    Message(role="user", content="Hello!"),
])
# System message is extracted and passed to the API correctly
```

## Generation Parameters

```python
response = client.generate(
    messages,
    temperature=0.7,      # Creativity (0.0-1.0)
    max_tokens=1000,      # Maximum output length
    top_p=0.95,           # Nucleus sampling
    use_cache=True,       # Use cache (default: True)
)
```
