"""Example: Using structured outputs with OpenAI."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from seeds_clients import Message, OpenAIClient

# Load environment variables from .env file
load_dotenv()


# Define your output structure
class PersonInfo(BaseModel):
    """Structured person information."""
    
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years", ge=0, le=150)
    occupation: str = Field(description="Current occupation")
    location: str = Field(description="City and country")


class ProductReview(BaseModel):
    """Structured product review."""
    
    model_config = ConfigDict(extra="forbid")
    
    product_name: str
    rating: int = Field(ge=1, le=5, description="Rating from 1-5 stars")
    pros: list[str] = Field(description="List of positive aspects")
    cons: list[str] = Field(description="List of negative aspects")
    recommended: bool = Field(description="Whether the product is recommended")


def example_basic_structured_output() -> None:
    """Extract structured data from unstructured text."""
    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1",
    )

    # Unstructured input
    text = """
    John Smith is a 35-year-old software engineer living in San Francisco, USA.
    He has been working in tech for over 10 years.
    """

    # Get structured output
    response = client.generate(
        messages=[
            Message(
                role="system",
                content="Extract person information from the text.",
            ),
            Message(role="user", content=text),
        ],
        response_format=PersonInfo,
    )

    # Access parsed data with type safety
    if response.parsed:
        person = response.parsed
        print(f"Name: {person.name}")
        print(f"Age: {person.age}")
        print(f"Occupation: {person.occupation}")
        print(f"Location: {person.location}")
        print(f"\nRaw JSON: {response.content}")
        print(f"Tokens used: {response.usage.total_tokens}")


def example_complex_nested_structure() -> None:
    """Extract complex nested data structures."""

    class Address(BaseModel):
        model_config = ConfigDict(extra="forbid")
        
        street: str
        city: str
        country: str
        postal_code: str

    class Company(BaseModel):
        model_config = ConfigDict(extra="forbid")
        
        name: str
        industry: str
        founded_year: int
        employees: int
        headquarters: Address

    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1",
    )

    text = """
    Acme Corporation is a leading technology company in the software industry.
    Founded in 2010, the company now employs over 5,000 people worldwide.
    Their headquarters is located at 123 Tech Street, San Francisco, CA 94105, USA.
    """

    response = client.generate(
        messages=[
            Message(role="system", content="Extract company information."),
            Message(role="user", content=text),
        ],
        response_format=Company,
    )

    if response.parsed:
        company = response.parsed
        print(f"Company: {company.name}")
        print(f"Founded: {company.founded_year}")
        print(f"Employees: {company.employees:,}")
        print(f"HQ: {company.headquarters.city}, {company.headquarters.country}")


def example_with_caching() -> None:
    """Structured outputs work with automatic caching."""
    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1",
        cache_dir="cache",
        ttl_hours=24.0,
    )

    text = "This laptop is amazing! Great battery life and fast processor. Only downside is the price."

    messages = [
        Message(role="system", content="Extract product review sentiment."),
        Message(role="user", content=text),
    ]

    # First call - hits API
    print("First call...")
    response1 = client.generate(messages, response_format=ProductReview)
    print(f"Cached: {response1.cached}")
    if response1.parsed:
        print(f"Product: {response1.parsed.product_name}")
        print(f"Rating: {response1.parsed.rating}/5")

    # Second call - uses cache
    print("\nSecond call...")
    response2 = client.generate(messages, response_format=ProductReview)
    print(f"Cached: {response2.cached}")
    if response2.parsed:
        print(f"Rating: {response2.parsed.rating}/5")

    # Same result, no API cost!
    if response1.parsed and response2.parsed:
        assert response1.parsed.product_name == response2.parsed.product_name


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Basic Structured Output")
    print("=" * 60)
    example_basic_structured_output()

    print("\n" + "=" * 60)
    print("Example 2: Complex Nested Structures")
    print("=" * 60)
    example_complex_nested_structure()

    print("\n" + "=" * 60)
    print("Example 3: Structured Outputs with Caching")
    print("=" * 60)
    example_with_caching()
