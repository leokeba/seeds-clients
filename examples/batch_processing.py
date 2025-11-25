"""Example demonstrating async and batch processing capabilities."""

import asyncio
import os

from seeds_clients import Message, OpenAIClient


async def basic_async_example():
    """Basic async generation example."""
    print("\n=== Basic Async Generation ===\n")

    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        cache_dir="cache",
    )

    try:
        # Single async request
        response = await client.agenerate(
            messages=[
                Message(role="user", content="What is the capital of France?")
            ]
        )
        print(f"Response: {response.content}")
        print(f"Tokens: {response.usage.total_tokens}")
        if response.tracking:
            print(f"Cost: ${response.tracking.cost_usd:.4f}")

    finally:
        await client.aclose()


async def batch_processing_example():
    """Batch processing multiple requests in parallel."""
    print("\n=== Batch Processing Example ===\n")

    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        cache_dir="cache",
    )

    # Prepare multiple requests
    questions = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of Portugal?",
    ]

    messages_list = [
        [Message(role="user", content=q)] for q in questions
    ]

    try:
        # Process batch with progress tracking
        print("Processing batch of", len(questions), "requests...")

        result = await client.batch_generate(
            messages_list,
            max_concurrent=3,  # Process 3 at a time
            on_progress=lambda done, total, r: print(f"  Progress: {done}/{total}"),
        )

        print("\n=== Results ===")
        print(f"Successful: {result.successful_count}")
        print(f"Failed: {result.failed_count}")
        print(f"Total tokens: {result.total_tokens}")
        print(f"Total cost: ${result.total_cost_usd:.4f}")
        print(f"Total carbon: {result.total_gwp_kgco2eq:.6f} kgCO2eq")
        print(f"Duration: {result.total_duration_seconds:.2f}s")

        print("\n=== Responses ===")
        for i, response in enumerate(result.responses):
            print(f"{i+1}. {response.content}")

        # Show any errors
        if result.errors:
            print("\n=== Errors ===")
            for idx, error in result.errors:
                print(f"Request {idx} failed: {error}")

    finally:
        await client.aclose()


async def batch_iterator_example():
    """Process results as they complete using async iterator."""
    print("\n=== Batch Iterator Example ===\n")

    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        cache_dir="cache",
    )

    questions = [
        "What is 2+2?",
        "What is 3*3?",
        "What is 4/2?",
    ]

    messages_list = [
        [Message(role="user", content=q)] for q in questions
    ]

    try:
        # Process and handle results as they come in
        print("Processing requests (results streamed as they complete):")

        async for idx, result in client.batch_generate_iter(
            messages_list, max_concurrent=2
        ):
            if isinstance(result, Exception):
                print(f"  Request {idx} FAILED: {result}")
            else:
                print(f"  Request {idx} ({questions[idx]}): {result.content[:50]}...")

    finally:
        await client.aclose()


async def async_context_manager_example():
    """Using async context manager for automatic cleanup."""
    print("\n=== Async Context Manager Example ===\n")

    async with OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    ) as client:
        response = await client.agenerate(
            messages=[Message(role="user", content="Hello!")]
        )
        print(f"Response: {response.content}")

    # Client is automatically closed here


async def main():
    """Run all examples."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Run examples
    await basic_async_example()
    await batch_processing_example()
    await batch_iterator_example()
    await async_context_manager_example()


if __name__ == "__main__":
    asyncio.run(main())
