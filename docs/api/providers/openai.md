# OpenAIClient

Client for OpenAI's GPT models.

::: seeds_clients.providers.openai.OpenAIClient
    options:
      show_root_heading: true
      members:
        - __init__
        - generate
        - agenerate
        - generate_structured
        - batch_generate
        - batch_generate_iter
        - export_boamps_report
        - cumulative_tracking
        - reset_cumulative_tracking

## Initialization

```python
from seeds_clients import OpenAIClient

client = OpenAIClient(
    model="gpt-4.1",              # Required: model name
    api_key="sk-...",             # Optional: uses OPENAI_API_KEY env var
    cache_dir="./cache",          # Optional: enables caching
    ttl_hours=24.0,               # Optional: cache TTL
    enable_tracking=True,         # Optional: enables carbon tracking
    electricity_mix_zone="WOR",   # Optional: for carbon calculations
)
```

## Supported Models

| Model | Description |
|-------|-------------|
| `gpt-4.1` | Latest GPT-4.1 |
| `gpt-4.1-mini` | Smaller, faster GPT-4.1 |
| `gpt-4o` | GPT-4o multimodal |
| `gpt-4o-mini` | Smaller GPT-4o |
| `gpt-4-turbo` | GPT-4 Turbo |
| `gpt-3.5-turbo` | GPT-3.5 Turbo |

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
    async with OpenAIClient(model="gpt-4.1-mini") as client:
        response = await client.agenerate([
            Message(role="user", content="Hello!")
        ])
        print(response.content)

asyncio.run(main())
```

## Batch Processing

```python
async def process_batch():
    async with OpenAIClient(model="gpt-4.1-mini") as client:
        prompts = [
            [Message(role="user", content=f"Question {i}")]
            for i in range(10)
        ]
        
        result = await client.batch_generate(
            prompts,
            max_concurrent=5,
            on_progress=lambda done, total, r: print(f"{done}/{total}"),
        )
        
        print(f"Success: {result.successful_count}")
        print(f"Total cost: ${result.total_cost_usd:.4f}")
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
