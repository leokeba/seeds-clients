# Batch Processing Guide

Efficiently process multiple requests with parallel execution and aggregated tracking.

## Overview

seeds-clients provides three methods for batch processing:

- `agenerate()` - Single async request
- `batch_generate()` - Parallel processing with aggregated results
- `batch_generate_iter()` - Streaming results as they complete

## Basic Async Usage

```python
import asyncio
from seeds_clients import OpenAIClient, Message

async def main():
    async with OpenAIClient(model="gpt-4.1-mini") as client:
        response = await client.agenerate([
            Message(role="user", content="Hello!")
        ])
        print(response.content)

asyncio.run(main())
```

## Batch Processing

Process multiple requests in parallel:

```python
async def process_batch():
    async with OpenAIClient(model="gpt-4.1-mini", cache_dir="./cache") as client:
        # Prepare prompts
        prompts = [
            [Message(role="user", content="What is Python?")],
            [Message(role="user", content="What is JavaScript?")],
            [Message(role="user", content="What is Rust?")],
            [Message(role="user", content="What is Go?")],
        ]
        
        # Process in parallel
        result = await client.batch_generate(
            prompts,
            max_concurrent=3,  # Max 3 requests at a time
        )
        
        return result

result = asyncio.run(process_batch())

# Check results
print(f"Successful: {result.successful_count}")
print(f"Failed: {result.failed_count}")
```

## Progress Tracking

Monitor progress during batch processing:

```python
def on_progress(completed: int, total: int, response):
    percentage = (completed / total) * 100
    print(f"Progress: {completed}/{total} ({percentage:.0f}%)")
    if response:
        print(f"  Latest: {response.content[:50]}...")

result = await client.batch_generate(
    prompts,
    max_concurrent=5,
    on_progress=on_progress,
)
```

## Streaming Results

Process results as they complete:

```python
async def stream_results():
    async with OpenAIClient(model="gpt-4.1-mini") as client:
        prompts = [
            [Message(role="user", content=f"Count to {i}")]
            for i in range(1, 11)
        ]
        
        async for idx, result in client.batch_generate_iter(prompts, max_concurrent=3):
            if isinstance(result, Exception):
                print(f"Request {idx} failed: {result}")
            else:
                print(f"Request {idx}: {result.content}")

asyncio.run(stream_results())
```

## Handling Errors

```python
result = await client.batch_generate(prompts, max_concurrent=5)

# Access successful responses
for i, response in enumerate(result.responses):
    if response is not None:
        print(f"Request {i}: {response.content}")

# Access errors
for idx, error in result.errors.items():
    print(f"Request {idx} failed: {error}")
```

## Aggregated Metrics

BatchResult includes aggregated tracking:

```python
result = await client.batch_generate(prompts, max_concurrent=5)

# Token usage
print(f"Total tokens: {result.total_tokens}")
print(f"Prompt tokens: {result.total_prompt_tokens}")
print(f"Completion tokens: {result.total_completion_tokens}")

# Cost
print(f"Total cost: ${result.total_cost_usd:.4f}")

# Carbon impact
print(f"Total energy: {result.total_energy_kwh:.6f} kWh")
print(f"Total carbon: {result.total_gwp_kgco2eq:.6f} kgCO2eq")
```

## Best Practices

### 1. Choose Appropriate Concurrency

```python
# Low concurrency for rate-limited APIs
result = await client.batch_generate(prompts, max_concurrent=2)

# Higher for APIs with generous limits
result = await client.batch_generate(prompts, max_concurrent=10)
```

### 2. Use Caching

```python
client = OpenAIClient(model="gpt-4.1-mini", cache_dir="./cache")

# First run: hits API
result1 = await client.batch_generate(prompts, max_concurrent=5)

# Second run: uses cache (instant)
result2 = await client.batch_generate(prompts, max_concurrent=5)
```

### 3. Handle Large Batches

```python
async def process_large_batch(prompts, batch_size=100):
    all_responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        result = await client.batch_generate(batch, max_concurrent=10)
        all_responses.extend(result.responses)
        print(f"Processed {i + len(batch)}/{len(prompts)}")
    
    return all_responses
```

### 4. Use Context Manager

Always use async context manager to ensure proper cleanup:

```python
async with OpenAIClient(model="gpt-4.1-mini") as client:
    result = await client.batch_generate(prompts)
    # Client is properly closed when exiting
```

## Example: Data Processing Pipeline

```python
import asyncio
from seeds_clients import OpenAIClient, Message
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    word_count: int

async def summarize_documents(documents: list[str]) -> list[Summary]:
    async with OpenAIClient(
        model="gpt-4.1-mini",
        cache_dir="./cache"
    ) as client:
        prompts = [
            [Message(
                role="user",
                content=f"Summarize this document:\n\n{doc}"
            )]
            for doc in documents
        ]
        
        # Process with structured output isn't directly supported in batch
        # So we process normally and parse
        result = await client.batch_generate(
            prompts,
            max_concurrent=5,
            on_progress=lambda d, t, r: print(f"Summarizing: {d}/{t}"),
        )
        
        print(f"\nTotal cost: ${result.total_cost_usd:.4f}")
        print(f"Total carbon: {result.total_gwp_kgco2eq:.6f} kgCO2eq")
        
        return [r.content for r in result.responses if r]

# Usage
documents = ["Doc 1 content...", "Doc 2 content...", "Doc 3 content..."]
summaries = asyncio.run(summarize_documents(documents))
```
