# Anthropic Guide

Complete guide for using seeds-clients with Anthropic's Claude models.

## Setup

### API Key

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Initialize Client

```python
from seeds_clients import AnthropicClient, Message

client = AnthropicClient(
    model="claude-sonnet-4-20250514",
    cache_dir="./cache",
)
```

## Basic Generation

```python
response = client.generate([
    Message(role="user", content="Explain quantum computing")
])

print(response.content)
```

### With System Message

```python
response = client.generate([
    Message(role="system", content="You are a helpful science tutor."),
    Message(role="user", content="What is photosynthesis?"),
])
```

## Structured Outputs

Anthropic uses tool_use for structured outputs:

```python
from pydantic import BaseModel
from typing import List

class Recipe(BaseModel):
    name: str
    ingredients: List[str]
    prep_time_minutes: int
    difficulty: str

recipe = client.generate_structured(
    messages=[Message(
        role="user",
        content="Give me a simple pasta recipe"
    )],
    response_model=Recipe,
)

print(f"Recipe: {recipe.name}")
print(f"Ingredients: {', '.join(recipe.ingredients)}")
print(f"Prep time: {recipe.prep_time_minutes} minutes")
```

## Multimodal (Vision)

```python
from PIL import Image

response = client.generate([
    Message(
        role="user",
        content=[
            {"type": "text", "text": "Describe this image in detail"},
            {"type": "image", "source": Image.open("photo.jpg")},
        ]
    )
])
```

## Model Selection

| Model | Best For | Speed |
|-------|----------|-------|
| `claude-sonnet-4-20250514` | Balanced performance | Fast |
| `claude-3-5-sonnet-20241022` | High quality, vision | Medium |
| `claude-3-5-haiku-20241022` | Fast, efficient | Very Fast |
| `claude-3-opus-20240229` | Complex reasoning | Slow |

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

## Best Practices

1. **System messages** are handled correctly and extracted
2. **Claude excels** at nuanced, thoughtful responses
3. **Use structured outputs** for reliable data extraction
4. **Vision capabilities** are excellent for image analysis
