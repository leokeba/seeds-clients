# Core Types

## Message

Represents a message in a conversation.

::: seeds_clients.core.types.Message
    options:
      show_root_heading: true
      members:
        - role
        - content

### Usage Examples

```python
from seeds_clients import Message

# Simple text message
message = Message(role="user", content="Hello!")

# System message
system = Message(role="system", content="You are a helpful assistant.")

# Multimodal message (text + image)
from PIL import Image

multimodal = Message(
    role="user",
    content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "source": Image.open("photo.jpg")},
    ]
)
```

---

## Response

The response from an LLM API call.

::: seeds_clients.core.types.Response
    options:
      show_root_heading: true

### Usage Examples

```python
response = client.generate([Message(role="user", content="Hello")])

# Access content
print(response.content)

# Check if cached
if response.cached:
    print("This was a cached response")

# Access usage
print(f"Tokens: {response.usage.total_tokens}")

# Access tracking
print(f"Cost: ${response.tracking.cost_usd}")
```

---

## TrackingData

Per-request tracking metrics including cost, energy, and carbon impact.

::: seeds_clients.core.types.TrackingData
    options:
      show_root_heading: true

### Key Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `cost_usd` | Request cost | USD |
| `energy_kwh` | Total energy consumed | kWh |
| `gwp_kgco2eq` | Total carbon emissions | kgCO2eq |
| `gwp_usage_kgco2eq` | Usage phase emissions | kgCO2eq |
| `gwp_embodied_kgco2eq` | Embodied emissions | kgCO2eq |
| `prompt_tokens` | Input tokens | count |
| `completion_tokens` | Output tokens | count |

### Usage Examples

```python
tracking = response.tracking

# Cost and tokens
print(f"Cost: ${tracking.cost_usd:.4f}")
print(f"Tokens: {tracking.prompt_tokens} + {tracking.completion_tokens}")

# Carbon metrics
print(f"Energy: {tracking.energy_kwh:.6f} kWh")
print(f"Carbon: {tracking.gwp_kgco2eq:.6f} kgCO2eq")

# Detailed breakdown
print(f"Usage phase: {tracking.gwp_usage_kgco2eq:.6f} kgCO2eq")
print(f"Embodied: {tracking.gwp_embodied_kgco2eq:.6f} kgCO2eq")
```

---

## CumulativeTracking

Aggregated tracking across the client's lifetime.

::: seeds_clients.core.types.CumulativeTracking
    options:
      show_root_heading: true

### Key Properties

| Property | Description |
|----------|-------------|
| `total_request_count` | Total requests made |
| `api_request_count` | Requests that hit the API |
| `cached_request_count` | Requests served from cache |
| `cache_hit_rate` | Percentage of cached requests |
| `total_cost_usd` | Total cost across all requests |
| `total_gwp_kgco2eq` | Total carbon emissions |
| `emissions_avoided_kgco2eq` | Emissions saved by caching |

### Usage Examples

```python
tracking = client.cumulative_tracking

# Request counts
print(f"Total: {tracking.total_request_count}")
print(f"API: {tracking.api_request_count}")
print(f"Cached: {tracking.cached_request_count}")
print(f"Cache hit rate: {tracking.cache_hit_rate:.1%}")

# Costs and emissions
print(f"Total cost: ${tracking.total_cost_usd:.4f}")
print(f"API emissions: {tracking.api_gwp_kgco2eq:.6f} kgCO2eq")
print(f"Avoided emissions: {tracking.emissions_avoided_kgco2eq:.6f} kgCO2eq")

# Reset tracking
client.reset_cumulative_tracking()
```

---

## Usage

Token usage information.

::: seeds_clients.core.types.Usage
    options:
      show_root_heading: true

### Usage Examples

```python
usage = response.usage

print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")
```

---

## BatchResult

Results from batch processing operations.

::: seeds_clients.core.batch.BatchResult
    options:
      show_root_heading: true

### Usage Examples

```python
result = await client.batch_generate(messages_list, max_concurrent=5)

# Check results
print(f"Successful: {result.successful_count}")
print(f"Failed: {result.failed_count}")

# Access responses
for response in result.responses:
    if response is not None:
        print(response.content)

# Access errors
for idx, error in result.errors.items():
    print(f"Request {idx} failed: {error}")

# Aggregated metrics
print(f"Total tokens: {result.total_tokens}")
print(f"Total cost: ${result.total_cost_usd:.4f}")
print(f"Total carbon: {result.total_gwp_kgco2eq:.6f} kgCO2eq")
```
