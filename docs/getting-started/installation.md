# Installation

## Requirements

- Python 3.11 or higher
- pip or uv package manager

## Basic Installation

Install seeds-clients using pip:

```bash
pip install seeds-clients
```

Or using uv (recommended):

```bash
uv pip install seeds-clients
```

## Optional Dependencies

seeds-clients has optional dependencies for specific features:

### CodeCarbon Support

For hardware-measured carbon emissions:

```bash
pip install seeds-clients[codecarbon]
```

### All Dependencies

Install everything:

```bash
pip install seeds-clients[all]
```

## Development Installation

For contributing or development:

```bash
# Clone the repository
git clone https://github.com/leokeba/seeds-clients.git
cd seeds-clients

# Install with uv
uv venv
source .venv/bin/activate
uv sync
```

## API Keys

seeds-clients requires API keys for the LLM providers you want to use. Set them as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GEMINI_API_KEY="..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."
```

Or create a `.env` file in your project root:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
```

seeds-clients will automatically load environment variables from `.env` files.

## Verifying Installation

Test your installation:

```python
from seeds_clients import OpenAIClient, Message

# This will fail if OPENAI_API_KEY is not set
client = OpenAIClient(model="gpt-4.1-mini")

response = client.generate([
    Message(role="user", content="Hello!")
])
print(response.content)
```
