# 🌱 seeds-clients

A modern Python library providing unified LLM clients with integrated carbon impact tracking, cost tracking, and intelligent local caching.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

`seeds-clients` provides a unified interface to multiple LLM providers (OpenAI, Anthropic, Google, Mistral, etc.) with built-in environmental impact tracking and cost monitoring. The library is designed for researchers and developers who want to understand and optimize the carbon footprint of their AI applications.

### Key Features

- 🔌 **Unified API** across multiple LLM providers
- 🌍 **Carbon Tracking** with EcoLogits and CodeCarbon integration
- 💰 **Cost Monitoring** with detailed usage analytics
- 💾 **Smart Caching** using raw API responses for maximum flexibility
- 📊 **BoAmps Reporting** standardized energy consumption reports
- 🖼️ **Multimodal Support** for text and image inputs
- 🔧 **Structured Outputs** with Pydantic model validation
- 📈 **Batch Processing** with parallel execution and aggregated tracking

## 🚀 Quick Start

```python
from seeds_clients import OpenAIClient, Message

# Initialize client with caching and tracking
client = OpenAIClient(
    api_key="your-api-key",
    model="gpt-4o",
    cache_dir="./cache",
    enable_tracking=True,
    electricity_mix_zone="WOR"  # World average (default)
)

# Generate text
messages = [Message(role="user", content="Explain quantum computing")]
response = client.generate(messages)

print(response.content)
print(f"Cost: ${response.tracking.cost_usd:.4f}")
print(f"Carbon: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq")
print(f"Energy: {response.tracking.energy_kwh:.6f} kWh")

# Detailed breakdown available
print(f"Usage phase GWP: {response.tracking.gwp_usage_kgco2eq:.6f} kgCO2eq")
print(f"Embodied phase GWP: {response.tracking.gwp_embodied_kgco2eq:.6f} kgCO2eq")
```

### Structured Output

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

person = client.generate_structured(
    messages=[Message(role="user", content="Extract: Alice is a 28-year-old engineer")],
    response_model=Person
)
```

### Multimodal (Image + Text)

```python
from PIL import Image

messages = [
    Message(
        role="user",
        content=[
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "source": Image.open("photo.jpg")}
        ]
    )
]
response = client.generate(messages)
```

### Async & Batch Processing

```python
import asyncio

async def main():
    async with OpenAIClient(
        api_key="your-api-key",
        model="gpt-4o-mini"
    ) as client:
        # Single async request
        response = await client.agenerate(
            messages=[Message(role="user", content="Hello!")]
        )
        print(response.content)

        # Batch processing with parallel execution
        prompts = [
            [Message(role="user", content="What is 1+1?")],
            [Message(role="user", content="What is 2+2?")],
            [Message(role="user", content="What is 3+3?")],
        ]

        result = await client.batch_generate(
            prompts,
            max_concurrent=3,
            on_progress=lambda done, total, r: print(f"Progress: {done}/{total}")
        )

        print(f"Total cost: ${result.total_cost_usd:.4f}")
        print(f"Total carbon: {result.total_gwp_kgco2eq:.6f} kgCO2eq")

        # Stream results as they complete
        async for idx, response in client.batch_generate_iter(prompts):
            print(f"Result {idx}: {response.content}")

asyncio.run(main())
```

### Export BoAmps Report

```python
# Get standardized energy consumption report
report = client.export_boamps_report(
    output_path="energy_report.json",
    include_calibration=True
)
```

## 📦 Installation

```bash
# Using uv (recommended)
uv pip install seeds-clients

# Using pip
pip install seeds-clients

# With optional dependencies
uv pip install seeds-clients[codecarbon]  # For CodeCarbon support
uv pip install seeds-clients[all]         # All optional dependencies
```

## 🏗️ Architecture

### Core Design Principles

1. **Raw Response Caching**: Store complete API responses, not transformed data
   - Future-proof: Library updates don't invalidate cache
   - Flexible: Re-process cached data with new logic
   - Debuggable: Always have original response for troubleshooting

2. **Provider Agnostic**: Unified interface across all LLM providers
   - Consistent API regardless of backend
   - Easy provider switching without code changes
   - Provider-specific optimizations under the hood

3. **Comprehensive Tracking**: Monitor both environmental and financial costs
   - EcoLogits for model-based carbon estimates
   - CodeCarbon for hardware-measured emissions
   - Detailed cost tracking with per-request granularity
   - BoAmps-compliant reporting for standardization

### Project Structure

```
seeds_clients/
├── core/
│   ├── base_client.py          # Abstract base client
│   ├── types.py                # Shared types (Message, Response, Usage)
│   ├── cache.py                # Cache manager (raw response storage)
│   └── exceptions.py           # Custom exceptions
├── tracking/
│   ├── ecologits_tracker.py    # EcoLogits integration
│   ├── codecarbon_tracker.py   # CodeCarbon integration (optional)
│   ├── cost_tracker.py         # Cost calculation engine
│   └── boamps_reporter.py      # BoAmps format reporter
├── providers/
│   ├── openai.py               # OpenAI client
│   ├── anthropic.py            # Anthropic client
│   ├── google.py               # Google GenAI client
│   ├── mistral.py              # Mistral client
│   └── ...                     # More providers
├── utils/
│   ├── multimodal.py           # Image/text handling
│   ├── pricing.py              # Pricing database
│   └── hash.py                 # Cache key generation
└── __init__.py
```

## 📋 Implementation Plan

### Phase 1: Core Foundation (Week 1-2)

**Goals**: Establish the foundational architecture with caching and type system

- [x] Project setup with `uv` and `pyproject.toml`
- [x] Core type system (`Message`, `Response`, `Usage`, `TrackingData`)
- [x] Cache manager with raw response storage
- [x] Abstract base client with unified interface
- [x] Cache key generation (content-based hashing)
- [x] Basic testing infrastructure

**Deliverables**:
- ✅ Working cache system that stores/retrieves raw API responses
- ✅ Type-safe interfaces for all core components
- ✅ Unit tests with 90%+ coverage

### Phase 2: Provider Implementations (Week 2-3)

**Goals**: Implement clients for major LLM providers

- [x] OpenAI client (GPT-4o, GPT-4, GPT-3.5)
- [x] Cost tracking with pricing database (JSON-based, easily updatable)
- [x] Structured outputs with Pydantic models
- [x] Multimodal support (text + images)
- [ ] Anthropic client (Claude 3.5 Sonnet, Claude 3 Opus)
- [ ] Google GenAI client (Gemini Pro, Gemini Flash)
- [ ] Mistral client (Mistral Large, Mistral Medium)
- [ ] Provider-specific response parsing
- [ ] Error handling and retry logic

**Deliverables**:
- ✅ OpenAI client with automatic cost tracking
- ✅ JSON-based pricing configuration (easily updatable)
- ✅ Structured outputs with response validation
- Integration tests with real API calls (VCR-based)
- Provider-specific documentation

### Phase 3: Tracking & Monitoring (Week 3-4)

**Goals**: Integrate carbon and cost tracking

- [x] EcoLogits integration
  - Automatic impact calculation per request
  - Model-based carbon estimates
  - Electricity mix zone configuration
  - Full metrics extraction (energy, GWP, ADPe, PE)
  - Usage vs embodied phase breakdown
- [ ] CodeCarbon integration (optional)
  - Hardware-measured emissions
  - Server-side tracking support
- [x] Cost tracking engine
  - Pricing database for all providers
  - Per-request cost calculation
  - Cumulative cost aggregation
- [ ] BoAmps reporter
  - Schema-compliant report generation
  - Hardware/software metadata
  - Calibration support

**Deliverables**:
- Working carbon tracking with EcoLogits
- Accurate cost calculation for all providers
- BoAmps JSON export functionality

### Phase 4: Advanced Features (Week 4-5)

**Goals**: Add structured outputs, multimodal, and batch processing

- [x] Structured output support
  - Pydantic model validation
  - Provider-native structured outputs (OpenAI JSON schema)
  - JSON mode fallback
- [x] Multimodal support
  - Image input handling (URL, bytes, PIL Image)
  - Base64 encoding/decoding
  - Multi-image support
- [x] Batch processing
  - Async generation (`agenerate()`)
  - Parallel execution with `asyncio` (`batch_generate()`)
  - Streaming results (`batch_generate_iter()`)
  - Progress tracking callbacks
  - Aggregated metrics (cost, carbon, tokens)
- [x] Advanced caching
  - TTL-based expiration
  - Cache invalidation strategies
  - Cache size management (via diskcache)

**Deliverables**:
- ✅ Structured output with OpenAI (other providers pending)
- ✅ Full multimodal support for OpenAI
- ✅ Efficient batch processing with aggregated tracking

### Phase 5: Testing & Documentation (Week 5-6)

**Goals**: Comprehensive testing and documentation

- [x] Unit tests (target: 90%+ coverage) - Currently at 92%
- [ ] Integration tests with real APIs
- [x] End-to-end examples (basic_usage.py, structured_outputs.py, batch_processing.py)
- [ ] API documentation (auto-generated)
- [ ] User guides for each provider
- [ ] Contributing guidelines
- [ ] Performance benchmarks

**Deliverables**:
- ✅ Full test suite (126 tests passing)
- Complete documentation site
- 10+ working examples

## 🔑 Key Implementation Details

### Raw Response Caching

```python
class CacheManager:
    """Manages caching of raw API responses"""
    
    def get(self, key: str) -> dict | None:
        """Retrieve raw API response from cache"""
        cached = self.cache.get(key)
        if cached and not self._is_expired(cached):
            return cached["raw_response"]
        return None
    
    def set(self, key: str, raw_response: dict, ttl: int | None):
        """Store raw API response with metadata"""
        self.cache.set(key, {
            "raw_response": raw_response,
            "cached_at": time.time(),
            "model": raw_response.get("model"),
            "provider": self.provider_name
        }, expire=ttl)
```

**Benefits**:
- **Forward compatibility**: Library updates don't break cache
- **Flexibility**: Re-process old responses with new logic
- **Debugging**: Always have original data
- **Provider migration**: Same cache works across different parsing

### Cache Key Generation

```python
def _compute_cache_key(
    messages: list[Message], 
    model: str,
    **kwargs
) -> str:
    """Generate deterministic cache key from request parameters"""
    cache_data = {
        "model": model,
        "messages": [
            {
                "role": m.role,
                "content": _hash_content(m.content)  # Hash images
            }
            for m in messages
        ],
        "params": {
            k: v for k, v in kwargs.items() 
            if k in ["temperature", "top_p", "max_tokens"]
        }
    }
    return hashlib.sha256(
        json.dumps(cache_data, sort_keys=True).encode()
    ).hexdigest()
```

### Tracking Architecture

```python
class TrackingData(BaseModel):
    """Unified tracking data structure"""
    
    # Carbon metrics - totals
    energy_kwh: float
    gwp_kgco2eq: float  # Global Warming Potential
    adpe_kgsbeq: float | None = None  # Abiotic Depletion Potential
    pe_mj: float | None = None  # Primary Energy
    
    # Cost metrics
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    
    # Metadata
    provider: str
    model: str
    tracking_method: str  # "ecologits" or "codecarbon" or "none"
    electricity_mix_zone: str | None = None
    
    # Timestamps
    measured_at: datetime
    duration_seconds: float
    
    # Usage phase breakdown (from electricity consumption)
    energy_usage_kwh: float | None = None
    gwp_usage_kgco2eq: float | None = None
    adpe_usage_kgsbeq: float | None = None
    pe_usage_mj: float | None = None
    
    # Embodied phase breakdown (from manufacturing, etc.)
    gwp_embodied_kgco2eq: float | None = None
    adpe_embodied_kgsbeq: float | None = None
    pe_embodied_mj: float | None = None
    
    # Status messages from EcoLogits
    ecologits_warnings: list[str] | None = None
    ecologits_errors: list[str] | None = None
```

## 🌍 Carbon Tracking Methods

### EcoLogits (Recommended)

Model-based carbon estimation using research data:

```python
client = OpenAIClient(
    api_key="...",
    model="gpt-4o",
    enable_tracking=True,
    tracking_method="ecologits",
    electricity_mix_zone="WOR"  # World average, or specific region
)

response = client.generate([Message(role="user", content="Hello!")])

# Access tracking data
print(f"Energy: {response.tracking.energy_kwh} kWh")
print(f"Carbon: {response.tracking.gwp_kgco2eq} kgCO2eq")
print(f"Cost: ${response.tracking.cost_usd}")

# Detailed breakdown
print(f"Usage phase GWP: {response.tracking.gwp_usage_kgco2eq} kgCO2eq")
print(f"Embodied phase GWP: {response.tracking.gwp_embodied_kgco2eq} kgCO2eq")

# Check for any warnings from EcoLogits
if response.tracking.ecologits_warnings:
    print(f"Warnings: {response.tracking.ecologits_warnings}")
```

**Pros**:
- No server instrumentation needed
- Works with any API provider
- Based on peer-reviewed research
- Supports all major models

**Cons**:
- Estimates, not direct measurements
- Limited to known models

### CodeCarbon (Optional)

Hardware-measured emissions for self-hosted models:

```python
client = LocalModelClient(
    model="llama-3.1-70b",
    enable_tracking=True,
    tracking_method="codecarbon",
    codecarbon_server="http://localhost:8000"
)
```

**Pros**:
- Actual hardware measurements
- CPU/GPU/RAM breakdown
- Real-time monitoring

**Cons**:
- Requires server setup
- Only for self-hosted models
- More complex infrastructure

## 📊 BoAmps Reporting

Export standardized energy consumption reports compliant with the [BoAmps format](https://github.com/Boavizta/BoAmps):

```python
report = client.export_boamps_report(
    output_path="energy_report.json",
    task_description="Image classification inference",
    algorithm_name="resnet50",
    dataset_info={
        "name": "imagenet",
        "size": 1000,
        "data_type": "image"
    },
    include_calibration=True
)
```

**Report Structure**:
- Algorithm metadata (name, task, framework)
- Dataset characteristics (type, size, format)
- Hardware configuration (CPU, GPU, RAM)
- Measurements (energy, duration, calibration)
- Environmental context (location, electricity mix)

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export MISTRAL_API_KEY="..."

# Tracking
export ECOLOGITS_ELECTRICITY_MIX="FRA"  # France
export CODECARBON_SERVER_URL="http://localhost:8000"

# Cache
export SEEDS_CACHE_DIR="./cache"
export SEEDS_CACHE_TTL_HOURS="24"
```

### Configuration File

```python
# seeds_config.py
from seeds_clients import Config

config = Config(
    cache_dir="./cache",
    cache_ttl_hours=24,
    enable_tracking=True,
    tracking_method="ecologits",
    electricity_mix_zone="FRA",
    default_temperature=0.7,
    default_max_tokens=1000
)
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/leokeba/seeds-clients.git
cd seeds-clients

# Install with uv
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e ".[dev]"

# Run tests
pytest

# Run linters
ruff check .
mypy .
```

## 📚 Resources

- **EcoLogits**: [https://ecologits.ai/](https://ecologits.ai/)
- **CodeCarbon**: [https://codecarbon.io/](https://codecarbon.io/)
- **BoAmps**: [https://github.com/Boavizta/BoAmps](https://github.com/Boavizta/BoAmps)
- **Boavizta**: [https://boavizta.org/](https://boavizta.org/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [EcoLogits](https://ecologits.ai/) for model-based carbon tracking
- [CodeCarbon](https://codecarbon.io/) for hardware-measured emissions
- [Boavizta](https://boavizta.org/) for the BoAmps reporting standard
- All LLM providers for their APIs

## 📮 Contact

- **Issues**: [GitHub Issues](https://github.com/leokeba/seeds-clients/issues)
- **Discussions**: [GitHub Discussions](https://github.com/leokeba/seeds-clients/discussions)

---

Built with 💚 for sustainable AI
