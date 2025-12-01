# Configuration

seeds-clients can be configured through constructor arguments, environment variables, or a combination of both.

## Client Configuration

All clients accept these common configuration options:

```python
from seeds_clients import OpenAIClient

client = OpenAIClient(
    # Required
    model="gpt-4.1",                    # Model name
    
    # API Configuration
    api_key="sk-...",                   # API key (or use env var)
    
    # Caching
    cache_dir="./cache",                # Enable caching with this directory
    ttl_hours=24.0,                     # Cache TTL in hours (default: 24)
    
    # Carbon Tracking
    enable_tracking=True,               # Enable EcoLogits tracking (default: True)
    electricity_mix_zone="WOR",         # Electricity zone for carbon calculations
    
    # Default Generation Parameters
    temperature=0.7,                    # Default temperature
    max_tokens=1000,                    # Default max tokens
)
```

## Environment Variables

### API Keys

Set API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GEMINI_API_KEY="..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."
```

### Using .env Files

Create a `.env` file in your project root:

```env
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=sk-or-...

# Optional: Carbon Tracking
ECOLOGITS_ELECTRICITY_MIX=FRA
```

seeds-clients automatically loads `.env` files using python-dotenv.

## Electricity Mix Zones

The `electricity_mix_zone` parameter affects carbon calculations. Common values:

| Zone | Description |
|------|-------------|
| `WOR` | World average (default) |
| `USA` | United States average |
| `FRA` | France (low carbon due to nuclear) |
| `DEU` | Germany |
| `GBR` | United Kingdom |
| `CHN` | China |

Full list available in the [EcoLogits documentation](https://ecologits.ai/).

## Caching Configuration

### Enable Caching

```python
client = OpenAIClient(
    model="gpt-4.1",
    cache_dir="./cache",      # Enable caching
    ttl_hours=24.0,           # Cache entries expire after 24 hours
)
```

### Cache Behavior

- Cache keys are based on: model, messages, and generation parameters
- Different temperatures produce different cache keys
- Images are hashed for cache key generation
- Cache is stored using diskcache for persistence

### Skip Cache

```python
# Skip cache for a single request
response = client.generate(messages, use_cache=False)
```

## Generation Parameters

Override defaults per-request:

```python
response = client.generate(
    messages,
    temperature=0.9,          # Override default temperature
    max_tokens=500,           # Override default max tokens
    top_p=0.95,               # Set top_p
    use_cache=True,           # Use cache (default)
)
```

## Tracking Configuration

### Disable Tracking

```python
client = OpenAIClient(
    model="gpt-4.1",
    enable_tracking=False,    # Disable EcoLogits tracking
)
```

### Access Tracking Data

```python
# Per-request tracking
print(response.tracking.cost_usd)
print(response.tracking.energy_kwh)
print(response.tracking.gwp_kgco2eq)

# Cumulative tracking
print(client.cumulative_tracking.total_cost_usd)
print(client.cumulative_tracking.api_gwp_kgco2eq)
print(client.cumulative_tracking.cache_hit_rate)
```

### Reset Cumulative Tracking

```python
client.reset_cumulative_tracking()
```

## Provider-Specific Configuration

### OpenAI

```python
from seeds_clients import OpenAIClient

client = OpenAIClient(
    model="gpt-4.1",
    api_key="sk-...",           # Or OPENAI_API_KEY env var
    organization="org-...",      # Optional: organization ID
)
```

### Anthropic

```python
from seeds_clients import AnthropicClient

client = AnthropicClient(
    model="claude-sonnet-4-20250514",
    api_key="sk-ant-...",        # Or ANTHROPIC_API_KEY env var
)
```

### Google

```python
from seeds_clients import GoogleClient

client = GoogleClient(
    model="gemini-2.5-flash",
    api_key="...",               # Or GEMINI_API_KEY env var
)
```

### OpenRouter

```python
from seeds_clients import OpenRouterClient

client = OpenRouterClient(
    model="anthropic/claude-3.5-sonnet",  # provider/model format
    api_key="sk-or-...",                  # Or OPENROUTER_API_KEY env var
)
```
