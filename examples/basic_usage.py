"""Simple example demonstrating seeds-clients basic usage."""

import os

from dotenv import load_dotenv

from seeds_clients import Message, OpenAIClient

# Load environment variables from .env file
load_dotenv()

# Initialize the client
client = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    cache_dir="cache",  # Enable caching
    ttl_hours=24.0,     # Cache for 24 hours
)

# Example 1: Simple text generation
print("Example 1: Simple text generation")
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
print()

# Example 4: Multimodal (image + text)
# Note: Commented out as it requires a valid image URL accessible by OpenAI
# print("Example 4: Multimodal input")
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
