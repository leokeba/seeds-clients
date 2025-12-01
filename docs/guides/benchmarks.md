# Performance Benchmarks

seeds-clients includes a comprehensive performance analysis tool to measure latency, throughput, caching efficiency, and environmental impact.

## Running Benchmarks

### Quick Test

```bash
uv run python examples/performance_analysis.py --quick
```

### Full Test

```bash
uv run python examples/performance_analysis.py --full
```

### Save Results

```bash
uv run python examples/performance_analysis.py --full --output reports/benchmark.json
```

## Benchmark Categories

### 1. Latency Test

Measures individual request latency:

- **min/max**: Range of response times
- **mean/median**: Average response time
- **p95/p99**: 95th and 99th percentile latencies
- **std_dev**: Consistency of response times

### 2. Throughput Test

Measures concurrent request handling:

- **requests_per_second**: How many requests can be processed
- **tokens_per_second**: Token generation rate
- **total_duration**: Time to complete all requests

### 3. Cache Test

Measures caching efficiency:

- **hit_rate**: Percentage of cache hits
- **avg_cache_lookup_ms**: Time to retrieve from cache
- **hits/misses**: Absolute counts

### 4. Carbon/Cost Test

Measures environmental and financial impact:

- **total_energy_kwh**: Total energy consumed
- **total_gwp_kgco2eq**: Total carbon emissions
- **total_cost_usd**: Total API costs
- **per_request averages**: Impact per request

## Sample Output

```
============================================================
📋 Performance Report: openai/gpt-4.1-mini
   Generated: 2024-12-01T10:00:00
   Test type: quick
============================================================

⏱️  LATENCY
   Mean:    450.2 ms
   Median:  432.1 ms
   P95:     512.3 ms
   P99:     534.7 ms
   Min/Max: 398.2 / 534.7 ms

🚀 THROUGHPUT
   Requests/sec:  4.21
   Tokens/sec:    168.4
   Total time:    2.37 s

💾 CACHE
   Hit rate:      66.7%
   Avg lookup:    1.23 ms
   Hits/Misses:   4 / 2

🌍 ENVIRONMENTAL IMPACT
   Total energy:  0.000012 kWh
   Total carbon:  0.000006 kgCO2eq
   Total cost:    $0.0012
   Avg/request:   0.000006 kgCO2eq, $0.000600

============================================================
```

## Comparing Providers

```bash
# OpenAI direct
uv run python examples/performance_analysis.py --provider openai --output reports/openai.json

# OpenRouter
uv run python examples/performance_analysis.py --provider openrouter --output reports/openrouter.json
```

## Comparing Models

```bash
# GPT-4.1
uv run python examples/performance_analysis.py --model gpt-4.1 --output reports/gpt4.1.json

# GPT-4.1-mini
uv run python examples/performance_analysis.py --model gpt-4.1-mini --output reports/gpt4.1-mini.json
```

## Interpreting Results

### Latency Considerations

- **API latency** is dominated by network round-trip and model inference
- **p95/p99** latencies show worst-case scenarios
- **Caching** eliminates API latency entirely for repeated requests

### Throughput Factors

- **max_concurrent** setting limits parallel requests
- **Rate limits** may affect throughput at scale
- **Token generation** is the main bottleneck for long outputs

### Cache Efficiency

- **100% hit rate** is ideal for development/testing
- **Production** hit rates depend on query diversity
- **Cache lookup** should be <5ms for local disk cache

### Environmental Impact

- **Energy** scales with token count
- **Carbon** depends on electricity mix zone
- **Caching** can reduce emissions by avoiding duplicate calls
