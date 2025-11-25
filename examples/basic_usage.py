"""Simple example demonstrating seeds-clients basic usage with carbon tracking."""

import os

from dotenv import load_dotenv

from seeds_clients import Message, OpenAIClient

# Load environment variables from .env file
load_dotenv()

# Initialize the client with carbon tracking
client = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1",
    cache_dir="cache",            # Enable caching
    ttl_hours=24.0,               # Cache for 24 hours
    electricity_mix_zone="WOR",   # World average electricity mix (default)
)

# Example 1: Simple text generation with tracking
print("Example 1: Simple text generation with carbon tracking")
print("-" * 50)
response = client.generate(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is 2+2?"),
    ]
)
print(f"Response: {response.content}")
print(f"Tokens used: {response.usage.total_tokens}")
print(f"Cached: {response.cached}")

# Show carbon tracking data
if response.tracking:
    print("\n📊 Carbon & Cost Tracking:")
    print(f"  💰 Cost: ${response.tracking.cost_usd:.6f}")
    print(f"  ⚡ Energy: {response.tracking.energy_kwh:.6f} kWh")
    print(f"  🌍 Carbon: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq")
    print(f"  📍 Electricity mix: {response.tracking.electricity_mix_zone}")
    print(f"  ⏱️ Duration: {response.tracking.duration_seconds:.3f}s")
    print(f"  🔬 Tracking method: {response.tracking.tracking_method}")
print()

# Example 2: Same request uses cache
print("Example 2: Same request uses cache")
print("-" * 50)
response = client.generate(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is 2+2?"),
    ]
)
print(f"Response: {response.content}")
print(f"Cached: {response.cached}")
print()

# Example 3: Custom parameters
print("Example 3: Custom parameters")
print("-" * 50)
response = client.generate(
    messages=[
        Message(role="user", content="Write a haiku about Python programming."),
    ],
    temperature=0.9,  # More creative
    max_tokens=100,
)
print(f"Response: {response.content}")
print(f"Model: {response.model}")

# Show detailed carbon breakdown
if response.tracking:
    print("\n📊 Detailed Environmental Impact:")
    print(f"  ⚡ Total Energy: {response.tracking.energy_kwh:.6f} kWh")
    print(f"  🌍 Total GWP: {response.tracking.gwp_kgco2eq:.6f} kgCO2eq")
    
    if response.tracking.adpe_kgsbeq:
        print(f"  ⛏️ ADPe: {response.tracking.adpe_kgsbeq:.2e} kgSbeq")
    if response.tracking.pe_mj:
        print(f"  🔋 Primary Energy: {response.tracking.pe_mj:.6f} MJ")
    
    # Usage vs Embodied breakdown
    if response.tracking.gwp_usage_kgco2eq:
        print(f"\n  📈 Usage phase (electricity):")
        print(f"     GWP: {response.tracking.gwp_usage_kgco2eq:.6f} kgCO2eq")
    if response.tracking.gwp_embodied_kgco2eq:
        print(f"  🏭 Embodied phase (manufacturing):")
        print(f"     GWP: {response.tracking.gwp_embodied_kgco2eq:.6f} kgCO2eq")
print()

# Example 4: Different electricity mix zone (France - lower carbon)
print("Example 4: Carbon comparison with different electricity mix")
print("-" * 50)

# Create client with French electricity mix (nuclear heavy = low carbon)
client_fra = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1",
    cache_dir="cache",
    electricity_mix_zone="FRA",  # France
)

response_fra = client_fra.generate(
    messages=[
        Message(role="user", content="Hello, world!"),
    ],
    use_cache=False,  # Skip cache to get fresh tracking data
)

print(f"Response: {response_fra.content}")
if response_fra.tracking:
    print(f"\n🇫🇷 France electricity mix:")
    print(f"  ⚡ Energy: {response_fra.tracking.energy_kwh:.6f} kWh")
    print(f"  🌍 Carbon: {response_fra.tracking.gwp_kgco2eq:.6f} kgCO2eq")
print()

# Example 5: Multimodal (image + text)
# Note: Commented out as it requires a valid image URL accessible by OpenAI
# print("Example 5: Multimodal input")
# print("-" * 50)
# response = client.generate(
#     messages=[
#         Message(
#             role="user",
#             content=[
#                 {"type": "text", "text": "What's in this image?"},
#                 {"type": "image", "source": "https://example.com/image.png"},
#             ],
#         ),
#     ]
# )
# print(f"Response: {response.content}")
print("\n✅ All examples completed successfully!")
print()
