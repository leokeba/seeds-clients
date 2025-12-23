"""Shared test fixtures."""

import os
import shutil
import tempfile
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

# Load environment variables for tests
load_dotenv()


@pytest.fixture
def mock_env_vars() -> Generator[None, None, None]:
    """Mock environment variables."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "test-key",
        "GOOGLE_API_KEY": "test-key",
        "OPENROUTER_API_KEY": "test-key",
        "MODEL_GARDEN_BASE_URL": "http://localhost:8000/v1",
        "MODEL_GARDEN_API_KEY": "test-key",
    }):
        yield


@pytest.fixture
def temp_cache_dir() -> Generator[str, None, None]:
    """Create a temporary directory for caching."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_ecologits() -> Generator[MagicMock, None, None]:
    """Mock EcoLogits to prevent actual calculations during tests."""
    with patch("seeds_clients.tracking.ecologits_tracker.llm_impacts") as mock:
        mock.return_value = MagicMock(
            energy=MagicMock(value=0.001),
            gwp=MagicMock(value=0.0005),
            adpe=None,
            pe=None,
            usage=None,
            embodied=None,
            warnings=None,
            errors=None,
        )
        yield mock
