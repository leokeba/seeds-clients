# Google Guide

Complete guide for using seeds-clients with Google's Gemini models.

## Setup

### API Key

Set your Google API key:

```bash
export GEMINI_API_KEY="..."
```

### Initialize Client

```python
from seeds_clients import GoogleClient, Message

client = GoogleClient(
    model="gemini-2.5-flash",
    cache_dir="./cache",
)
```

## Basic Generation

```python
response = client.generate([
    Message(role="user", content="What is the capital of France?")
])

print(response.content)
```

## Structured Outputs

```python
from pydantic import BaseModel

class Country(BaseModel):
    name: str
    capital: str
    population: int
    continent: str

country = client.generate_structured(
    messages=[Message(
        role="user",
        content="Tell me about Japan"
    )],
    response_model=Country,
)

print(f"Country: {country.name}")
print(f"Capital: {country.capital}")
```

## Multimodal

```python
from PIL import Image

response = client.generate([
    Message(
        role="user",
        content=[
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "source": Image.open("photo.jpg")},
        ]
    )
])
```

## Model Selection

| Model | Best For | Speed |
|-------|----------|-------|
| `gemini-2.5-flash` | Fast, efficient | Very Fast |
| `gemini-2.5-pro` | Complex tasks | Medium |
| `gemini-1.5-flash` | Legacy, fast | Fast |
| `gemini-1.5-pro` | Legacy, powerful | Medium |

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
