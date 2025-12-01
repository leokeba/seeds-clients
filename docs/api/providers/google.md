# GoogleClient

Client for Google's Gemini models.

::: seeds_clients.providers.google.GoogleClient
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
from seeds_clients import GoogleClient

client = GoogleClient(
    model="gemini-2.5-flash",    # Required: model name
    api_key="...",               # Optional: uses GEMINI_API_KEY env var
    cache_dir="./cache",         # Optional: enables caching
    ttl_hours=24.0,              # Optional: cache TTL
    enable_tracking=True,        # Optional: enables carbon tracking
)
```

## Supported Models

| Model | Description |
|-------|-------------|
| `gemini-2.5-flash` | Gemini 2.5 Flash |
| `gemini-2.5-pro` | Gemini 2.5 Pro |
| `gemini-1.5-flash` | Gemini 1.5 Flash |
| `gemini-1.5-pro` | Gemini 1.5 Pro |

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
    async with GoogleClient(model="gemini-2.5-flash") as client:
        response = await client.agenerate([
            Message(role="user", content="Hello!")
        ])
        print(response.content)

asyncio.run(main())
```

## Generation Parameters

```python
response = client.generate(
    messages,
    temperature=0.7,      # Creativity (0.0-2.0)
    max_tokens=1000,      # Maximum output length
    top_p=0.95,           # Nucleus sampling
    use_cache=True,       # Use cache (default: True)
)
```
