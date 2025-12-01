# OpenAI Guide

Complete guide for using seeds-clients with OpenAI's GPT models.

## Setup

### API Key

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Or in a `.env` file:

```env
OPENAI_API_KEY=sk-...
```

### Initialize Client

```python
from seeds_clients import OpenAIClient, Message

client = OpenAIClient(
    model="gpt-4.1",
    cache_dir="./cache",
)
```

## Basic Generation

```python
response = client.generate([
    Message(role="user", content="Explain machine learning in simple terms")
])

print(response.content)
```

### With System Message

```python
response = client.generate([
    Message(role="system", content="You are a helpful coding assistant."),
    Message(role="user", content="How do I read a file in Python?"),
])
```

### Multi-turn Conversation

```python
messages = [
    Message(role="user", content="What is Python?"),
]

response1 = client.generate(messages)
print(response1.content)

# Continue the conversation
messages.append(Message(role="assistant", content=response1.content))
messages.append(Message(role="user", content="What are its main uses?"))

response2 = client.generate(messages)
print(response2.content)
```

## Structured Outputs

Extract structured data using Pydantic models:

```python
from pydantic import BaseModel, Field
from typing import List

class Movie(BaseModel):
    title: str
    year: int
    director: str
    genres: List[str]

movie = client.generate_structured(
    messages=[Message(
        role="user",
        content="The movie Inception was directed by Christopher Nolan in 2010. It's a sci-fi thriller."
    )],
    response_model=Movie,
)

print(f"Title: {movie.title}")
print(f"Year: {movie.year}")
print(f"Director: {movie.director}")
print(f"Genres: {', '.join(movie.genres)}")
```

### Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    address: Address

person = client.generate_structured(
    messages=[Message(
        role="user",
        content="John Doe is 35 years old and lives at 123 Main St, New York, USA"
    )],
    response_model=Person,
)
```

## Multimodal (Vision)

### From PIL Image

```python
from PIL import Image

image = Image.open("photo.jpg")

response = client.generate([
    Message(
        role="user",
        content=[
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "source": image},
        ]
    )
])
```

### From URL

```python
response = client.generate([
    Message(
        role="user",
        content=[
            {"type": "text", "text": "Describe this image"},
            {"type": "image", "source": "https://example.com/image.jpg"},
        ]
    )
])
```

### Multiple Images

```python
response = client.generate([
    Message(
        role="user",
        content=[
            {"type": "text", "text": "Compare these two images"},
            {"type": "image", "source": Image.open("image1.jpg")},
            {"type": "image", "source": Image.open("image2.jpg")},
        ]
    )
])
```

## Async Usage

### Single Request

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

### Batch Processing

```python
async def process_questions(questions: list[str]):
    async with OpenAIClient(model="gpt-4.1-mini") as client:
        prompts = [
            [Message(role="user", content=q)]
            for q in questions
        ]
        
        result = await client.batch_generate(
            prompts,
            max_concurrent=5,
            on_progress=lambda done, total, r: print(f"Progress: {done}/{total}"),
        )
        
        return result

questions = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?",
]

result = asyncio.run(process_questions(questions))

for i, response in enumerate(result.responses):
    print(f"Q: {questions[i]}")
    print(f"A: {response.content[:100]}...")
    print()
```

### Streaming Results

```python
async def stream_results():
    async with OpenAIClient(model="gpt-4.1-mini") as client:
        prompts = [
            [Message(role="user", content=f"Count to {i}")]
            for i in range(1, 6)
        ]
        
        async for idx, result in client.batch_generate_iter(prompts):
            print(f"Result {idx}: {result.content}")

asyncio.run(stream_results())
```

## Generation Parameters

```python
response = client.generate(
    messages,
    temperature=0.7,      # Creativity: 0.0 = deterministic, 2.0 = very creative
    max_tokens=1000,      # Maximum output tokens
    top_p=0.95,           # Nucleus sampling threshold
    use_cache=True,       # Use cached response if available
)
```

## Tracking

### Per-Request Tracking

```python
response = client.generate([Message(role="user", content="Hello")])

# Cost
print(f"Cost: ${response.tracking.cost_usd:.4f}")
print(f"Tokens: {response.tracking.prompt_tokens} + {response.tracking.completion_tokens}")

# Carbon
print(f"Energy: {response.tracking.energy_kwh:.6f} kWh")
print(f"Carbon: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq")
```

### Cumulative Tracking

```python
# After multiple requests
tracking = client.cumulative_tracking

print(f"Total requests: {tracking.total_request_count}")
print(f"Cache hit rate: {tracking.cache_hit_rate:.1%}")
print(f"Total cost: ${tracking.total_cost_usd:.4f}")
print(f"Emissions avoided: {tracking.emissions_avoided_kgco2eq:.6f} kgCO2eq")
```

## Model Selection

| Model | Best For | Cost |
|-------|----------|------|
| `gpt-4.1` | Complex reasoning, high quality | $$$ |
| `gpt-4.1-mini` | Balanced quality/cost | $$ |
| `gpt-4o` | Multimodal, vision | $$$ |
| `gpt-4o-mini` | Fast multimodal | $$ |
| `gpt-3.5-turbo` | Simple tasks, low cost | $ |

## Best Practices

1. **Use caching** for repeated or similar queries
2. **Choose the right model** for your task complexity
3. **Set appropriate max_tokens** to avoid unnecessary costs
4. **Use structured outputs** for data extraction
5. **Batch process** when handling multiple requests
6. **Monitor cumulative tracking** for cost/carbon awareness
