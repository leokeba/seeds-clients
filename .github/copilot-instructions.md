# Copilot Instructions for seeds-clients

## Project Overview

This is a Python library providing unified LLM clients with carbon tracking, cost monitoring, and smart caching.

## Development Environment

### Package Manager: uv

We use **uv** for all Python-related tasks:

```bash
# Install all dependencies (including dev group)
uv sync --all-groups

# Run Python scripts
uv run python script.py

# Run pytest
uv run python -m pytest tests/

# Run a specific test file
uv run python -m pytest tests/test_openai.py -v

# Run with coverage
uv run python -m pytest tests/ --cov=seeds_clients
```

**Do NOT use:**
- `pip install` - use `uv add` or `uv sync` instead
- `python` directly - use `uv run python` instead
- `pytest` directly - use `uv run python -m pytest` instead

### Dependency Groups

This project uses uv's `[dependency-groups]` feature (PEP 735) instead of `[project.optional-dependencies]` for dev dependencies:

```toml
# In pyproject.toml - this is the correct format:
[dependency-groups]
dev = [
    "pytest>=9.0",
    "ruff>=0.14",
    # ... other dev dependencies
]
```

**Important:** When adding dev dependencies, add them to `[dependency-groups].dev`, NOT to `[project.optional-dependencies].dev`. The latter is for optional runtime dependencies, while dependency-groups is for development tools.

```bash
# Add a dev dependency
uv add --group dev package-name

# Sync all groups including dev
uv sync --all-groups
```

### Environment Variables

API keys are loaded from `.env` file using `python-dotenv`. The `.env` file is gitignored.

## Code Quality Requirements

### Before Committing

**Always run tests and check for errors before committing:**

```bash
# Run all tests
uv run python -m pytest tests/ -v --tb=short

# Check for lint errors (optional but recommended)
uv run ruff check seeds_clients/ tests/

# Check for type errors with ty (recommended)
uv run ty check
```

- All tests must pass
- No new lint errors should be introduced
- Coverage should not decrease significantly
- Type errors should be addressed (ty is the primary type checker)

### Test Categories

- **Unit tests**: Run by default, no API keys needed
- **Integration tests**: Require `--run-integration` flag and API keys

```bash
# Run only unit tests (default)
uv run python -m pytest tests/

# Run integration tests (requires API keys)
uv run python -m pytest tests/ --run-integration
```

## Project Structure

```
seeds_clients/
├── core/           # Base classes, types, exceptions, caching
├── providers/      # OpenAI, OpenRouter implementations
├── tracking/       # EcoLogits carbon tracking
└── utils/          # Pricing utilities
tests/              # Test files (test_*.py)
examples/           # Usage examples
```

## Key Patterns

- All clients extend `BaseClient` and use `EcoLogitsMixin` for carbon tracking
- Structured outputs use Pydantic models with automatic OpenAI schema patching
- Async methods: `agenerate()`, `batch_generate()`, `batch_generate_iter()`
- Response objects include `tracking` data with cost, energy, and carbon metrics

## Default Model for Testing

We use **`gpt-4.1-mini`** as the default model for all testing purposes. This model offers:
- Lower cost for running tests frequently
- Good balance of capability and speed
- Consistent behavior for CI/CD pipelines

When adding new tests or examples, prefer `gpt-4.1-mini` unless testing specific model features.
