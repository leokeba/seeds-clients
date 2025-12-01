# Contributing to seeds-clients

Thank you for your interest in contributing to seeds-clients! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Adding a New Provider](#adding-a-new-provider)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/seeds-clients.git
   cd seeds-clients
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/leokeba/seeds-clients.git
   ```

## Development Setup

We use **uv** as our package manager. Install it first if you haven't:

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Setting up the development environment

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies including dev tools
uv sync

# Set up environment variables (copy and edit)
cp .env.example .env  # Add your API keys for integration tests
```

### IDE Setup

For VS Code, we recommend these extensions:
- Python
- ty (astral-sh.ty) - Primary type checker and language server
- Ruff

## Making Changes

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following our code style guidelines

3. **Write tests** for new functionality

4. **Run the test suite** to ensure nothing is broken

## Testing

### Running Tests

```bash
# Run all unit tests
uv run python -m pytest tests/ -v

# Run with coverage report
uv run python -m pytest tests/ --cov=seeds_clients --cov-report=term-missing

# Run a specific test file
uv run python -m pytest tests/test_openai.py -v

# Run a specific test
uv run python -m pytest tests/test_openai.py::TestOpenAIClient::test_generate -v
```

### Integration Tests

Integration tests require API keys and make real API calls:

```bash
# Set up API keys in .env file first
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Run integration tests
uv run python -m pytest tests/ -v -m integration --run-integration
```

### Test Categories

- **Unit tests**: Run by default, no API keys needed (mocked)
- **Integration tests**: Require `--run-integration` flag and API keys

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use `pytest` fixtures for common setup
- Mock external API calls in unit tests
- Use `@pytest.mark.integration` for integration tests

Example test structure:

```python
import pytest
from unittest.mock import Mock, patch

from seeds_clients import OpenAIClient, Message


class TestOpenAIClient:
    """Tests for OpenAI client."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a client with mocked API."""
        with patch("seeds_clients.providers.openai.OpenAI"):
            return OpenAIClient(
                api_key="test-key",
                model="gpt-4.1-mini",
                cache_dir=str(tmp_path),
            )

    def test_generate_returns_response(self, client):
        """Test that generate returns a valid response."""
        # Arrange
        messages = [Message(role="user", content="Hello")]
        
        # Act
        response = client.generate(messages)
        
        # Assert
        assert response.content is not None
```

## Code Style

We use **Ruff** for linting and formatting:

```bash
# Check for lint errors
uv run ruff check seeds_clients/ tests/

# Auto-fix lint errors
uv run ruff check seeds_clients/ tests/ --fix

# Format code
uv run ruff format seeds_clients/ tests/
```

### Type Hints

We use type hints throughout the codebase. Check types with **ty** (our primary type checker):

```bash
# Type check with ty (recommended)
uv run ty check

# Or with mypy (alternative)
uv run mypy seeds_clients/
```

### Style Guidelines

- Use type hints for all function parameters and return values
- Write docstrings for all public classes and functions
- Follow PEP 8 naming conventions
- Keep functions focused and reasonably sized
- Use meaningful variable names

Example:

```python
def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> float:
    """Calculate the cost of an API request.
    
    Args:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        model: The model name used for the request.
        
    Returns:
        Cost in USD.
        
    Raises:
        ValueError: If the model is not found in pricing data.
    """
    pricing = get_pricing(model)
    return (
        prompt_tokens * pricing.prompt_price_per_token +
        completion_tokens * pricing.completion_price_per_token
    )
```

## Submitting Changes

1. **Ensure all tests pass**:
   ```bash
   uv run python -m pytest tests/ -v
   ```

2. **Check for lint errors**:
   ```bash
   uv run ruff check seeds_clients/ tests/
   ```

3. **Check for type errors**:
   ```bash
   uv run ty check
   ```

4. **Commit your changes** with a clear message:
   ```bash
   git add .
   git commit -m "Add feature: description of the feature"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

### Pull Request Guidelines

- Provide a clear description of what the PR does
- Reference any related issues
- Ensure CI checks pass
- Be responsive to review feedback

## Adding a New Provider

To add support for a new LLM provider:

1. **Create a new provider file** in `seeds_clients/providers/`:
   ```python
   # seeds_clients/providers/new_provider.py
   
   from seeds_clients.core.base_client import BaseClient
   from seeds_clients.tracking import EcoLogitsMixin
   
   class NewProviderClient(EcoLogitsMixin, BaseClient):
       """Client for NewProvider API."""
       
       def __init__(
           self,
           api_key: str | None = None,
           model: str = "default-model",
           **kwargs,
       ):
           super().__init__(**kwargs)
           # Initialize provider-specific client
       
       def _get_provider_name(self) -> str:
           return "newprovider"
       
       def _get_ecologits_provider(self) -> str:
           return "newprovider"  # Must match EcoLogits provider name
       
       # Implement required abstract methods...
   ```

2. **Add pricing data** in `seeds_clients/utils/pricing.json`

3. **Export from `__init__.py`** files

4. **Write tests** in `tests/test_new_provider.py`

5. **Add documentation** and examples

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas

Thank you for contributing! 💚
