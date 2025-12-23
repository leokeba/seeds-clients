# AGENTS

## Environment
- Use `uv` for all Python tasks: `uv sync --all-groups`, `uv run python …`, `uv run python -m pytest …`.
- Add dev dependencies in `[dependency-groups].dev` via `uv add --group dev ...`; do not add dev tools to `[project.optional-dependencies].dev`.
- API keys come from `.env` (gitignored).

## Quality Gates
- Run `uv run python -m pytest tests/ -v --tb=short` before submitting; coverage is enabled by default per `pyproject.toml` addopts.
- Prefer `uv run ty check` for types; `mypy`/`pyright` configs exist—run them when relevant to your change.
- Recommended lint: `uv run ruff check seeds_clients/ tests/`.
- Integration tests are marked `integration`; run with `uv run python -m pytest tests/ -m integration` (or `--run-integration`) when API keys are set.

## Design & Safety Principles
- Fail fast and loud: avoid silent error handling, implicit fallbacks, or swallowed exceptions.
- Keep logic modular and composable; prefer small, focused units over large monoliths.
- Favor strong, explicit types to maximize static analysis signal; avoid loose typing and unchecked `Any` where practical.
- Maintain backward-compatible behavior unless explicitly changing the contract.

## Codebase Orientation
- Core clients: `seeds_clients/core`; providers: `seeds_clients/providers`; tracking: `seeds_clients/tracking`; utilities/pricing: `seeds_clients/utils`; examples: `examples/`; tests: `tests/`.
- Clients extend `BaseClient`; carbon tracking via `EcoLogitsMixin`. Structured outputs use Pydantic; async methods include `agenerate`, `batch_generate`, `batch_generate_iter`. Responses expose `tracking` (cost/energy/carbon).

## Testing Defaults
- Prefer `gpt-4.1-mini` for tests and examples unless exercising model-specific behavior.
- Keep new tests fast and offline unless explicitly covering provider API interactions.

## Workflow Notes
- Keep changes minimal and aligned with existing patterns; avoid introducing new tooling without discussion.
- Do not create commits or branches unless requested.
