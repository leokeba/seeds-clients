"""Example demonstrating carbon tracking and BoAmps report generation.

This example shows how to:
1. Track carbon emissions for individual requests
2. Track cumulative emissions across multiple requests
3. Compare API vs cached emissions (avoided emissions)
4. Compare emissions across different electricity mix zones
5. Generate BoAmps-compliant energy consumption reports
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from seeds_clients import Message, OpenAIClient
from seeds_clients.tracking.boamps_reporter import BoAmpsReporter, export_boamps_report

# Load environment variables from .env file
load_dotenv()


def single_request_tracking():
    """Demonstrate carbon tracking for a single request."""
    print("\n" + "=" * 60)
    print("1. SINGLE REQUEST TRACKING")
    print("=" * 60)

    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1-mini",
        cache_dir="cache",
        electricity_mix_zone="WOR",  # World average
    )

    response = client.generate(
        messages=[
            Message(role="user", content="Explain what carbon footprint means in 2 sentences.")
        ],
        use_cache=False,  # Force fresh request for accurate tracking
    )

    print(f"\n📝 Response: {response.content[:200]}...")
    print(f"\n📊 Environmental Impact:")
    
    if response.tracking:
        t = response.tracking
        print(f"   ⚡ Total Energy:     {t.energy_kwh:.6f} kWh")
        print(f"   🌍 Total GWP:        {t.gwp_kgco2eq:.6f} kgCO2eq")
        print(f"   💰 Cost:             ${t.cost_usd:.6f}")
        print(f"   📍 Electricity Zone: {t.electricity_mix_zone}")
        print(f"   🔬 Tracking Method:  {t.tracking_method}")
        print(f"   ⏱️  Duration:         {t.duration_seconds:.3f}s")

        # Detailed breakdown
        print(f"\n📈 Phase Breakdown:")
        if t.gwp_usage_kgco2eq is not None:
            print(f"   Usage Phase (electricity):    {t.gwp_usage_kgco2eq:.6f} kgCO2eq")
        if t.gwp_embodied_kgco2eq is not None:
            print(f"   Embodied Phase (manufacturing): {t.gwp_embodied_kgco2eq:.6f} kgCO2eq")

        # Additional metrics
        if t.adpe_kgsbeq is not None:
            print(f"\n♻️  Additional Metrics:")
            print(f"   ADPe (resource depletion): {t.adpe_kgsbeq:.2e} kgSbeq")
        if t.pe_mj is not None:
            print(f"   Primary Energy:            {t.pe_mj:.6f} MJ")

        # Check for warnings
        if t.ecologits_warnings:
            print(f"\n⚠️  Warnings: {t.ecologits_warnings}")


def cumulative_tracking_example():
    """Demonstrate cumulative tracking across multiple requests."""
    print("\n" + "=" * 60)
    print("2. CUMULATIVE TRACKING ACROSS REQUESTS")
    print("=" * 60)

    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1-mini",
        cache_dir="cache",
        electricity_mix_zone="WOR",
    )

    # Reset cumulative tracking to start fresh
    client.reset_cumulative_tracking()

    # Make several requests (some will be cached on repeat runs)
    questions = [
        "What is machine learning?",
        "What is deep learning?",
        "What is machine learning?",  # Duplicate - will use cache
        "What is artificial intelligence?",
        "What is deep learning?",  # Duplicate - will use cache
    ]

    print("\n📤 Making requests...")
    for i, question in enumerate(questions, 1):
        response = client.generate([Message(role="user", content=question)])
        status = "📦 CACHED" if response.cached else "🌐 API"
        print(f"   {i}. {status}: {question[:40]}...")

    # Get cumulative tracking
    tracking = client.cumulative_tracking

    print(f"\n📊 Cumulative Tracking Summary:")
    print(f"   📈 Total Requests:     {tracking.total_request_count}")
    print(f"   🌐 API Requests:       {tracking.api_request_count}")
    print(f"   📦 Cached Requests:    {tracking.cached_request_count}")
    print(f"   🎯 Cache Hit Rate:     {tracking.cache_hit_rate:.1%}")

    print(f"\n🌍 Emissions:")
    print(f"   Total GWP:          {tracking.total_gwp_kgco2eq:.6f} kgCO2eq")
    print(f"   API GWP (actual):   {tracking.api_gwp_kgco2eq:.6f} kgCO2eq")
    print(f"   Cached GWP (est.):  {tracking.cached_gwp_kgco2eq:.6f} kgCO2eq")
    print(f"   🌱 Emissions Avoided: {tracking.emissions_avoided_kgco2eq:.6f} kgCO2eq")

    print(f"\n⚡ Energy:")
    print(f"   Total Energy:       {tracking.total_energy_kwh:.6f} kWh")
    print(f"   API Energy:         {tracking.api_energy_kwh:.6f} kWh")

    print(f"\n💰 Cost:")
    print(f"   Total Cost:         ${tracking.total_cost_usd:.4f}")
    print(f"   API Cost:           ${tracking.api_cost_usd:.4f}")
    print(f"   Cost Avoided:       ${tracking.cached_cost_usd:.4f}")

    print(f"\n📊 Tokens:")
    print(f"   Total Prompt:       {tracking.total_prompt_tokens}")
    print(f"   Total Completion:   {tracking.total_completion_tokens}")


def electricity_mix_comparison():
    """Compare emissions across different electricity mix zones."""
    print("\n" + "=" * 60)
    print("3. ELECTRICITY MIX ZONE COMPARISON")
    print("=" * 60)

    zones = {
        "WOR": "🌍 World Average",
        "FRA": "🇫🇷 France (nuclear heavy)",
        "DEU": "🇩🇪 Germany",
        "USA": "🇺🇸 United States",
        "POL": "🇵🇱 Poland (coal heavy)",
    }

    prompt = "Write a haiku about renewable energy."
    results = {}

    print(f"\n📝 Prompt: '{prompt}'")
    print("\n📊 Emissions by Electricity Mix Zone:")

    for zone, description in zones.items():
        client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4.1-mini",
            electricity_mix_zone=zone,
        )

        response = client.generate(
            [Message(role="user", content=prompt)],
            use_cache=False,
        )

        if response.tracking:
            results[zone] = response.tracking.gwp_kgco2eq
            print(f"   {description}: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq")

    # Show comparison
    if results:
        min_zone = min(results, key=lambda z: results[z])
        max_zone = max(results, key=lambda z: results[z])
        print(f"\n💡 Insight:")
        print(f"   Lowest carbon: {zones[min_zone]} ({results[min_zone]:.6f} kgCO2eq)")
        print(f"   Highest carbon: {zones[max_zone]} ({results[max_zone]:.6f} kgCO2eq)")
        if results[max_zone] > 0 and results[min_zone] > 0:
            ratio = results[max_zone] / results[min_zone]
            print(f"   Difference: {ratio:.1f}x more emissions")


def boamps_report_example():
    """Generate a BoAmps-compliant energy consumption report."""
    print("\n" + "=" * 60)
    print("4. BOAMPS REPORT GENERATION")
    print("=" * 60)

    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1-mini",
        cache_dir="cache",
        electricity_mix_zone="WOR",
    )

    # Reset tracking for clean report
    client.reset_cumulative_tracking()

    # Simulate a workload
    print("\n📤 Simulating workload...")
    tasks = [
        "Summarize the benefits of sustainable AI.",
        "What are the main sources of carbon emissions in computing?",
        "How can machine learning help with climate change?",
        "Explain the concept of green computing.",
        "What is the carbon footprint of training large language models?",
    ]

    for task in tasks:
        response = client.generate([Message(role="user", content=task)])
        status = "📦 CACHED" if response.cached else "🌐 API"
        print(f"   {status}: {task[:50]}...")

    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Method 1: Quick export using convenience function
    print("\n📄 Generating BoAmps Report (Method 1: Quick Export)...")
    report_path = reports_dir / "carbon_report_quick.json"
    
    report = export_boamps_report(
        client=client,
        output_path=report_path,
        publisher_name="Seeds Clients Example",
        task_description="LLM inference for sustainability education",
        include_summary=True,
    )

    # Method 2: Detailed report using BoAmpsReporter class
    print("\n📄 Generating BoAmps Report (Method 2: Detailed)...")
    
    reporter = BoAmpsReporter(
        client=client,
        publisher_name="Research Organization",
        publisher_division="AI Sustainability Team",
        project_name="Green AI Initiative",
        task_description="Evaluating carbon footprint of LLM inference",
        task_family="textGeneration",
        data_type="text",
        infrastructure_type="publicCloud",
        quality="high",
        include_system_info=True,
    )

    detailed_report = reporter.generate_report()
    detailed_path = reports_dir / "carbon_report_detailed.json"
    detailed_report.save(detailed_path)
    print(f"   Saved to: {detailed_path}")

    # Show report contents
    print("\n📋 Report Contents Preview:")
    print(f"   Report ID: {detailed_report.header.reportId}")
    print(f"   Date: {detailed_report.header.reportDatetime}")
    print(f"   Publisher: {detailed_report.header.publisher.name if detailed_report.header.publisher else 'N/A'}")
    print(f"\n   Task:")
    print(f"     - Stage: {detailed_report.task.taskStage}")
    print(f"     - Family: {detailed_report.task.taskFamily}")
    print(f"     - Requests: {detailed_report.task.nbRequest}")
    if detailed_report.task.algorithms:
        print(f"     - Model: {detailed_report.task.algorithms[0].foundationModelName}")
    if detailed_report.task.dataset:
        # BoAmps schema has separate input/output datasets
        for ds in detailed_report.task.dataset:
            print(f"     - {ds.dataUsage.capitalize()} tokens: {ds.dataQuantity}")

    if detailed_report.measures:
        measure = detailed_report.measures[0]
        print(f"\n   Measures:")
        print(f"     - Method: {measure.measurementMethod}")
        print(f"     - Energy: {measure.powerConsumption:.6f} kWh")

    print(f"\n   Environment:")
    print(f"     - Country: {detailed_report.environment.country}")

    if detailed_report.system:
        print(f"\n   System:")
        print(f"     - OS: {detailed_report.system.os}")
    if detailed_report.software:
        print(f"     - Python: {detailed_report.software.version}")


def carbon_budget_example():
    """Demonstrate tracking against a carbon budget."""
    print("\n" + "=" * 60)
    print("5. CARBON BUDGET MONITORING")
    print("=" * 60)

    # Set a carbon budget (in kgCO2eq)
    CARBON_BUDGET = 0.0001  # Very small for demo purposes

    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1-mini",
        cache_dir="cache",
        electricity_mix_zone="WOR",
    )

    client.reset_cumulative_tracking()

    print(f"\n🎯 Carbon Budget: {CARBON_BUDGET:.6f} kgCO2eq")
    print("\n📤 Making requests until budget is exhausted...")

    questions = [
        "What is 2+2?",
        "What is the capital of France?",
        "Explain photosynthesis briefly.",
        "What is the speed of light?",
        "Name three programming languages.",
        "What year did WW2 end?",
        "What is the chemical formula for water?",
        "Who wrote Romeo and Juliet?",
    ]

    for i, question in enumerate(questions, 1):
        response = client.generate([Message(role="user", content=question)])
        
        current_emissions = client.cumulative_tracking.api_gwp_kgco2eq
        budget_used = (current_emissions / CARBON_BUDGET) * 100
        
        status = "📦" if response.cached else "🌐"
        print(f"   {i}. {status} Budget used: {budget_used:.1f}% ({current_emissions:.6f} kgCO2eq)")

        if current_emissions >= CARBON_BUDGET:
            print(f"\n⚠️  Carbon budget exhausted after {i} requests!")
            print(f"   Consider using cached responses or a lower-carbon electricity zone.")
            break

    # Final summary
    tracking = client.cumulative_tracking
    print(f"\n📊 Final Summary:")
    print(f"   Total emissions: {tracking.api_gwp_kgco2eq:.6f} kgCO2eq")
    print(f"   Budget: {CARBON_BUDGET:.6f} kgCO2eq")
    print(f"   Status: {'⚠️ OVER BUDGET' if tracking.api_gwp_kgco2eq > CARBON_BUDGET else '✅ WITHIN BUDGET'}")


def main():
    """Run all carbon tracking examples."""
    print("\n🌱 CARBON TRACKING & REPORT GENERATION EXAMPLES")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    # Run examples
    single_request_tracking()
    cumulative_tracking_example()
    electricity_mix_comparison()
    boamps_report_example()
    carbon_budget_example()

    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)
    print("\n💡 Tips for reducing carbon footprint:")
    print("   • Use caching to avoid redundant API calls")
    print("   • Choose electricity mix zones with cleaner energy (e.g., France)")
    print("   • Use smaller models when possible (e.g., gpt-4.1-mini)")
    print("   • Batch requests to improve efficiency")
    print("   • Monitor cumulative emissions and set budgets")
    print()


if __name__ == "__main__":
    main()
