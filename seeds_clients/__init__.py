"""
seeds-clients: Unified LLM clients with carbon tracking and smart caching.
"""

from seeds_clients.core.exceptions import (
    CacheError,
    ProviderError,
    SeedsClientError,
    ValidationError,
)
from seeds_clients.core.types import CumulativeTracking, Message, Response, TrackingData, Usage
from seeds_clients.providers import (
    AnthropicClient,
    BatchResult,
    GoogleClient,
    ModelGardenClient,
    OpenAIClient,
    OpenRouterClient,
    OpenRouterCostData,
)
from seeds_clients.utils.logging_utils import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    # Core types
    "CumulativeTracking",
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
    "configure_logging",
    "get_logger",
]
