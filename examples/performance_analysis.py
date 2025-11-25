#!/usr/bin/env python3
"""End-to-end performance analysis for seeds-clients.

This script runs comprehensive performance tests and generates detailed reports
on latency, throughput, caching efficiency, and environmental impact.

Usage:
    python examples/performance_analysis.py [--quick] [--full] [--output report.json]

Options:
    --quick     Run minimal tests (default)
    --full      Run comprehensive tests
    --output    Save results to JSON file
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from seeds_clients import Message, OpenAIClient, OpenRouterClient


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""

    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    std_dev_ms: float
    p95_ms: float
    p99_ms: float
    samples: int


@dataclass
class ThroughputStats:
    """Statistics for throughput measurements."""

    requests_per_second: float
    tokens_per_second: float
    total_requests: int
    total_tokens: int
    duration_seconds: float


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    hit_rate: float
    miss_rate: float
    hits: int
    misses: int
    avg_cache_lookup_ms: float


@dataclass
class CarbonStats:
    """Statistics for carbon/environmental impact."""

    total_energy_kwh: float
    total_gwp_kgco2eq: float
    avg_energy_per_request_kwh: float
    avg_gwp_per_request_kgco2eq: float
    total_cost_usd: float
    avg_cost_per_request_usd: float


@dataclass
class PerformanceReport:
    """Complete performance analysis report."""

    timestamp: str
    provider: str
    model: str
    test_type: str

    latency: LatencyStats | None = None
    throughput: ThroughputStats | None = None
    cache: CacheStats | None = None
    carbon: CarbonStats | None = None

    config: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def calculate_latency_stats(latencies_ms: list[float]) -> LatencyStats:
    """Calculate comprehensive latency statistics."""
    if not latencies_ms:
        return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0)

    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)

    return LatencyStats(
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
        mean_ms=statistics.mean(latencies_ms),
        median_ms=statistics.median(latencies_ms),
        std_dev_ms=statistics.stdev(latencies_ms) if n > 1 else 0,
        p95_ms=sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[0],
        p99_ms=sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[0],
        samples=n,
    )


class PerformanceAnalyzer:
    """Performance analysis toolkit for seeds-clients."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4.1-mini",
        verbose: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.verbose = verbose
        self.cache_dir = tempfile.mkdtemp(prefix="seeds_perf_")

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _create_client(self, **kwargs) -> OpenAIClient | OpenRouterClient:
        """Create client based on provider."""
        if self.provider == "openrouter":
            return OpenRouterClient(
                model=self.model,
                cache_dir=self.cache_dir,
                **kwargs,
            )
        return OpenAIClient(
            model=self.model,
            cache_dir=self.cache_dir,
            **kwargs,
        )

    def run_latency_test(
        self,
        num_requests: int = 10,
        prompt: str = "Say 'test'",
    ) -> LatencyStats:
        """
        Measure request latency.

        Args:
            num_requests: Number of requests to make
            prompt: Prompt to use for each request

        Returns:
            LatencyStats with timing information
        """
        self._log(f"\n📊 Running latency test ({num_requests} requests)...")

        client = self._create_client(ttl_hours=0.001)  # Very short TTL to avoid cache
        messages = [Message(role="user", content=prompt)]

        latencies_ms = []
        for i in range(num_requests):
            start = time.perf_counter()
            try:
                client.generate(messages=messages, use_cache=False)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies_ms.append(elapsed_ms)
                self._log(f"  Request {i + 1}/{num_requests}: {elapsed_ms:.1f}ms")
            except Exception as e:
                self._log(f"  Request {i + 1}/{num_requests}: FAILED - {e}")

        stats = calculate_latency_stats(latencies_ms)
        self._log(f"\n  Mean: {stats.mean_ms:.1f}ms, Median: {stats.median_ms:.1f}ms, P95: {stats.p95_ms:.1f}ms")

        return stats

    async def run_throughput_test(
        self,
        num_requests: int = 20,
        max_concurrent: int = 5,
        prompt: str = "Say 'test'",
    ) -> ThroughputStats:
        """
        Measure throughput with concurrent requests.

        Args:
            num_requests: Total number of requests
            max_concurrent: Maximum concurrent requests
            prompt: Prompt to use for each request

        Returns:
            ThroughputStats with throughput information
        """
        self._log(f"\n🚀 Running throughput test ({num_requests} requests, {max_concurrent} concurrent)...")

        client = self._create_client(ttl_hours=0.001)

        messages_list = [
            [Message(role="user", content=f"{prompt} #{i}")]
            for i in range(num_requests)
        ]

        start = time.perf_counter()
        try:
            result = await client.batch_generate(
                messages_list,
                max_concurrent=max_concurrent,
                on_progress=lambda done, total, _: self._log(f"  Progress: {done}/{total}") if done % 5 == 0 else None,
            )

            duration = time.perf_counter() - start
            total_tokens = result.total_tokens

            stats = ThroughputStats(
                requests_per_second=result.successful_count / duration,
                tokens_per_second=total_tokens / duration,
                total_requests=result.successful_count,
                total_tokens=total_tokens,
                duration_seconds=duration,
            )

            self._log(f"\n  Throughput: {stats.requests_per_second:.2f} req/s, {stats.tokens_per_second:.1f} tokens/s")
            self._log(f"  Total duration: {duration:.2f}s")

            return stats
        finally:
            await client.aclose()

    def run_cache_test(
        self,
        num_unique_prompts: int = 5,
        repeats_per_prompt: int = 3,
    ) -> CacheStats:
        """
        Measure cache hit rate and lookup performance.

        Args:
            num_unique_prompts: Number of unique prompts
            repeats_per_prompt: Number of times to repeat each prompt

        Returns:
            CacheStats with cache performance information
        """
        self._log(f"\n💾 Running cache test ({num_unique_prompts} unique prompts, {repeats_per_prompt} repeats each)...")

        client = self._create_client(ttl_hours=24.0)

        prompts = [f"What is {i}+{i}?" for i in range(num_unique_prompts)]

        hits = 0
        misses = 0
        cache_lookup_times_ms = []

        # First pass - all cache misses
        self._log("  First pass (populating cache)...")
        for prompt in prompts:
            messages = [Message(role="user", content=prompt)]
            response = client.generate(messages=messages)
            if response.cached:
                hits += 1
            else:
                misses += 1

        # Subsequent passes - should be cache hits
        self._log("  Subsequent passes (cache lookups)...")
        for repeat in range(repeats_per_prompt - 1):
            for prompt in prompts:
                messages = [Message(role="user", content=prompt)]
                start = time.perf_counter()
                response = client.generate(messages=messages)
                elapsed_ms = (time.perf_counter() - start) * 1000
                cache_lookup_times_ms.append(elapsed_ms)

                if response.cached:
                    hits += 1
                else:
                    misses += 1

        total = hits + misses
        stats = CacheStats(
            hit_rate=hits / total if total > 0 else 0,
            miss_rate=misses / total if total > 0 else 0,
            hits=hits,
            misses=misses,
            avg_cache_lookup_ms=statistics.mean(cache_lookup_times_ms) if cache_lookup_times_ms else 0,
        )

        self._log(f"\n  Hit rate: {stats.hit_rate:.1%}")
        self._log(f"  Avg cache lookup: {stats.avg_cache_lookup_ms:.2f}ms")

        return stats

    def run_carbon_test(
        self,
        num_requests: int = 10,
        prompt: str = "Write a short paragraph about AI.",
    ) -> CarbonStats:
        """
        Measure environmental impact across requests.

        Args:
            num_requests: Number of requests to make
            prompt: Prompt to use (longer = more tokens = more impact)

        Returns:
            CarbonStats with environmental impact information
        """
        self._log(f"\n🌍 Running carbon/environmental impact test ({num_requests} requests)...")

        client = self._create_client(ttl_hours=0.001)
        messages = [Message(role="user", content=prompt)]

        total_energy = 0.0
        total_gwp = 0.0
        total_cost = 0.0
        successful_requests = 0

        for i in range(num_requests):
            try:
                response = client.generate(messages=messages, use_cache=False)
                if response.tracking:
                    total_energy += response.tracking.energy_kwh or 0
                    total_gwp += response.tracking.gwp_kgco2eq or 0
                    total_cost += response.tracking.cost_usd or 0
                    successful_requests += 1
                    self._log(f"  Request {i + 1}: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq, ${response.tracking.cost_usd:.6f}")
            except Exception as e:
                self._log(f"  Request {i + 1}: FAILED - {e}")

        if successful_requests == 0:
            return CarbonStats(0, 0, 0, 0, 0, 0)

        stats = CarbonStats(
            total_energy_kwh=total_energy,
            total_gwp_kgco2eq=total_gwp,
            avg_energy_per_request_kwh=total_energy / successful_requests,
            avg_gwp_per_request_kgco2eq=total_gwp / successful_requests,
            total_cost_usd=total_cost,
            avg_cost_per_request_usd=total_cost / successful_requests,
        )

        self._log(f"\n  Total carbon: {stats.total_gwp_kgco2eq:.6f} kgCO2eq")
        self._log(f"  Total cost: ${stats.total_cost_usd:.4f}")
        self._log(f"  Avg per request: {stats.avg_gwp_per_request_kgco2eq:.6f} kgCO2eq, ${stats.avg_cost_per_request_usd:.6f}")

        return stats

    async def run_full_analysis(
        self,
        quick: bool = True,
    ) -> PerformanceReport:
        """
        Run complete performance analysis.

        Args:
            quick: If True, run minimal tests. If False, run comprehensive tests.

        Returns:
            PerformanceReport with all test results
        """
        self._log(f"\n{'=' * 60}")
        self._log(f"🔬 Performance Analysis: {self.provider}/{self.model}")
        self._log(f"   Mode: {'Quick' if quick else 'Full'}")
        self._log(f"{'=' * 60}")

        # Test parameters based on mode
        if quick:
            latency_requests = 3
            throughput_requests = 5
            throughput_concurrent = 2
            cache_unique = 2
            cache_repeats = 2
            carbon_requests = 2
        else:
            latency_requests = 20
            throughput_requests = 50
            throughput_concurrent = 10
            cache_unique = 10
            cache_repeats = 5
            carbon_requests = 10

        report = PerformanceReport(
            timestamp=datetime.now().isoformat(),
            provider=self.provider,
            model=self.model,
            test_type="quick" if quick else "full",
            config={
                "latency_requests": latency_requests,
                "throughput_requests": throughput_requests,
                "throughput_concurrent": throughput_concurrent,
                "cache_unique": cache_unique,
                "cache_repeats": cache_repeats,
                "carbon_requests": carbon_requests,
            },
        )

        try:
            # Latency test
            report.latency = self.run_latency_test(num_requests=latency_requests)
        except Exception as e:
            report.notes.append(f"Latency test failed: {e}")

        try:
            # Throughput test
            report.throughput = await self.run_throughput_test(
                num_requests=throughput_requests,
                max_concurrent=throughput_concurrent,
            )
        except Exception as e:
            report.notes.append(f"Throughput test failed: {e}")

        try:
            # Cache test
            report.cache = self.run_cache_test(
                num_unique_prompts=cache_unique,
                repeats_per_prompt=cache_repeats,
            )
        except Exception as e:
            report.notes.append(f"Cache test failed: {e}")

        try:
            # Carbon test
            report.carbon = self.run_carbon_test(num_requests=carbon_requests)
        except Exception as e:
            report.notes.append(f"Carbon test failed: {e}")

        return report


def print_report(report: PerformanceReport) -> None:
    """Print a formatted performance report."""
    print(f"\n{'=' * 60}")
    print(f"📋 Performance Report: {report.provider}/{report.model}")
    print(f"   Generated: {report.timestamp}")
    print(f"   Test type: {report.test_type}")
    print(f"{'=' * 60}")

    if report.latency:
        print("\n⏱️  LATENCY")
        print(f"   Mean:    {report.latency.mean_ms:.1f} ms")
        print(f"   Median:  {report.latency.median_ms:.1f} ms")
        print(f"   P95:     {report.latency.p95_ms:.1f} ms")
        print(f"   P99:     {report.latency.p99_ms:.1f} ms")
        print(f"   Min/Max: {report.latency.min_ms:.1f} / {report.latency.max_ms:.1f} ms")

    if report.throughput:
        print("\n🚀 THROUGHPUT")
        print(f"   Requests/sec:  {report.throughput.requests_per_second:.2f}")
        print(f"   Tokens/sec:    {report.throughput.tokens_per_second:.1f}")
        print(f"   Total time:    {report.throughput.duration_seconds:.2f} s")

    if report.cache:
        print("\n💾 CACHE")
        print(f"   Hit rate:      {report.cache.hit_rate:.1%}")
        print(f"   Avg lookup:    {report.cache.avg_cache_lookup_ms:.2f} ms")
        print(f"   Hits/Misses:   {report.cache.hits} / {report.cache.misses}")

    if report.carbon:
        print("\n🌍 ENVIRONMENTAL IMPACT")
        print(f"   Total energy:  {report.carbon.total_energy_kwh:.6f} kWh")
        print(f"   Total carbon:  {report.carbon.total_gwp_kgco2eq:.6f} kgCO2eq")
        print(f"   Total cost:    ${report.carbon.total_cost_usd:.4f}")
        print(f"   Avg/request:   {report.carbon.avg_gwp_per_request_kgco2eq:.6f} kgCO2eq, ${report.carbon.avg_cost_per_request_usd:.6f}")

    if report.notes:
        print("\n📝 NOTES")
        for note in report.notes:
            print(f"   - {note}")

    print(f"\n{'=' * 60}\n")


def report_to_dict(report: PerformanceReport) -> dict[str, Any]:
    """Convert report to dictionary for JSON serialization."""
    result = {
        "timestamp": report.timestamp,
        "provider": report.provider,
        "model": report.model,
        "test_type": report.test_type,
        "config": report.config,
        "notes": report.notes,
    }

    if report.latency:
        result["latency"] = asdict(report.latency)
    if report.throughput:
        result["throughput"] = asdict(report.throughput)
    if report.cache:
        result["cache"] = asdict(report.cache)
    if report.carbon:
        result["carbon"] = asdict(report.carbon)

    return result


async def main():
    """Run performance analysis."""
    parser = argparse.ArgumentParser(description="Run performance analysis for seeds-clients")
    parser.add_argument("--quick", action="store_true", default=True, help="Run quick tests (default)")
    parser.add_argument("--full", action="store_true", help="Run full comprehensive tests")
    parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")
    parser.add_argument("--provider", "-p", type=str, default="openai", choices=["openai", "openrouter"], help="Provider to test")
    parser.add_argument("--model", "-m", type=str, help="Model to use")
    args = parser.parse_args()

    # Determine test mode
    quick_mode = not args.full

    # Check for API keys
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    if args.provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    # Determine model
    if args.model:
        model = args.model
    elif args.provider == "openrouter":
        model = "openai/gpt-4.1-mini"
    else:
        model = "gpt-4.1-mini"

    # Run analysis
    analyzer = PerformanceAnalyzer(
        provider=args.provider,
        model=model,
        verbose=True,
    )

    report = await analyzer.run_full_analysis(quick=quick_mode)
    print_report(report)

    # Save to file if requested
    if args.output:
        # Create directory if it doesn't exist
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_dict = report_to_dict(report)
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"📁 Report saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
