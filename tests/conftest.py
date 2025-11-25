"""Pytest configuration for seeds-clients tests."""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests that require API keys",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require API keys, skipped by default)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        # --run-integration given: do not skip integration tests
        return

    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
