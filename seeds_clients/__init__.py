"""
seeds-clients: Unified LLM clients with carbon tracking and smart caching.
"""

from seeds_clients.core.exceptions import (
    CacheError,
    ProviderError,
    SeedsClientError,
    ValidationError,
)
from seeds_clients.core.types import Message, Response, TrackingData, Usage
from seeds_clients.providers import (
    AnthropicClient,
    BatchResult,
    GoogleClient,
    ModelGardenClient,
    OpenAIClient,
    OpenRouterClient,
    OpenRouterCostData,
)

__version__ = "0.1.0"

__all__ = [
    # Core types
    "Message",
    "Response",
    "Usage",
    "TrackingData",
    # Batch processing
    "BatchResult",
    # Exceptions
    "SeedsClientError",
    "CacheError",
    "ProviderError",
    "ValidationError",
    # Providers
    "AnthropicClient",
    "GoogleClient",
    "OpenAIClient",
    "OpenRouterClient",
    "OpenRouterCostData",
    "ModelGardenClient",
]
