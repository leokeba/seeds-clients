# Caching Guide

Understanding and using the caching system in seeds-clients.

## Overview

seeds-clients caches raw API responses to:

- Avoid duplicate API calls
- Reduce costs
- Lower carbon emissions
- Speed up development and testing

## How Caching Works

1. **Cache Key Generation**: Based on model, messages, and parameters
2. **Raw Response Storage**: Complete API responses are stored
3. **TTL Expiration**: Cached entries expire after a configurable time
4. **Disk Persistence**: Cache survives restarts (using diskcache)

## Enabling Caching

```python
from seeds_clients import OpenAIClient

client = OpenAIClient(
    model="gpt-4.1",
    cache_dir="./cache",    # Enable caching
    ttl_hours=24.0,         # Cache TTL (default: 24 hours)
)
```

## Cache Behavior

### Identical Requests

```python
messages = [Message(role="user", content="Hello")]

# First request: hits API
response1 = client.generate(messages)
print(f"Cached: {response1.cached}")  # False

# Second request: from cache
response2 = client.generate(messages)
print(f"Cached: {response2.cached}")  # True

# Content is identical
assert response1.content == response2.content
```

### Different Parameters = Different Cache Keys

```python
messages = [Message(role="user", content="Hello")]

# Different temperatures = different cache entries
response1 = client.generate(messages, temperature=0.5)
response2 = client.generate(messages, temperature=0.9)

# Both are new API calls (different cache keys)
```

## Skipping Cache

```python
# Force API call even if cached
response = client.generate(messages, use_cache=False)
```

## Cache Key Components

Cache keys are based on:

1. **Model name**
2. **Messages** (role and content)
3. **Generation parameters** (temperature, max_tokens, top_p)

### Images in Cache Keys

Images are hashed for cache key generation:

```python
# Same image = same cache key
response1 = client.generate([Message(
    role="user",
    content=[
        {"type": "text", "text": "Describe this"},
        {"type": "image", "source": Image.open("photo.jpg")},
    ]
)])

# Cached if same image
response2 = client.generate([Message(
    role="user",
    content=[
        {"type": "text", "text": "Describe this"},
        {"type": "image", "source": Image.open("photo.jpg")},
    ]
)])
```

## Benefits

### 1. Cost Reduction

```python
# 100 identical requests
for _ in range(100):
    client.generate([Message(role="user", content="Hello")])

# Only 1 API call, 99 cache hits
tracking = client.cumulative_tracking
print(f"API calls: {tracking.api_request_count}")  # 1
print(f"Cached: {tracking.cached_request_count}")  # 99
print(f"Cost: ${tracking.total_cost_usd:.4f}")     # Cost of 1 call
```

### 2. Reduced Emissions

```python
print(f"Emissions: {tracking.api_gwp_kgco2eq:.6f} kgCO2eq")
print(f"Avoided: {tracking.emissions_avoided_kgco2eq:.6f} kgCO2eq")
```

### 3. Faster Development

Cached responses are instant, making iteration faster.

## Cache Statistics

```python
tracking = client.cumulative_tracking

print(f"Total requests: {tracking.total_request_count}")
print(f"Cache hit rate: {tracking.cache_hit_rate:.1%}")
```

## TTL Configuration

```python
# Short TTL for development
dev_client = OpenAIClient(
    model="gpt-4.1",
    cache_dir="./cache",
    ttl_hours=1.0,  # 1 hour
)

# Long TTL for production
prod_client = OpenAIClient(
    model="gpt-4.1",
    cache_dir="./cache",
    ttl_hours=168.0,  # 1 week
)
```

## Cache Directory Structure

```
cache/
├── cache.db           # SQLite database (diskcache)
├── cache.db-shm
├── cache.db-wal
└── ...
```

## Best Practices

### 1. Use Per-Model Cache Directories

```python
client1 = OpenAIClient(model="gpt-4.1", cache_dir="./cache/gpt4")
client2 = OpenAIClient(model="gpt-4.1-mini", cache_dir="./cache/gpt4mini")
```

### 2. Clear Cache for New Deployments

```python
import shutil

# Clear cache directory
shutil.rmtree("./cache", ignore_errors=True)
```

### 3. Separate Dev and Prod Caches

```python
import os

cache_dir = "./cache/dev" if os.getenv("ENV") == "dev" else "./cache/prod"
client = OpenAIClient(model="gpt-4.1", cache_dir=cache_dir)
```

### 4. Use Consistent Parameters

Parameters affect cache keys. For consistent caching:

```python
# Define defaults once
DEFAULTS = {"temperature": 0.7, "max_tokens": 1000}

response = client.generate(messages, **DEFAULTS)
```

## When Not to Cache

Skip caching when you need:

- Fresh responses every time
- Random/creative outputs
- Time-sensitive information

```python
# Skip cache for creative writing
response = client.generate(
    [Message(role="user", content="Write a random poem")],
    use_cache=False,
    temperature=1.0,
)
```

## Raw Response Storage

seeds-clients stores complete raw API responses, which means:

- **Future-proof**: Library updates don't invalidate cache
- **Flexible**: Re-process cached data with new logic
- **Debuggable**: Access original response for troubleshooting

```python
# Raw response is preserved
response = client.generate(messages)

# Full data available even after library updates
print(response.raw_response)  # Original API response
```
