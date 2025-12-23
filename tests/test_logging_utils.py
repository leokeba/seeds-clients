import importlib
import logging
from io import StringIO

import pytest

from seeds_clients.utils import logging_utils


@pytest.fixture(autouse=True)
def reset_logging(monkeypatch):
    # Ensure env does not affect root level during tests
    monkeypatch.delenv("SEEDS_CLIENTS_LOG_LEVEL", raising=False)

    root_logger = logging.getLogger("seeds_clients")
    original_level = root_logger.level
    original_handlers = list(root_logger.handlers)
    original_propagate = root_logger.propagate

    yield

    root_logger.setLevel(original_level)
    root_logger.handlers[:] = original_handlers
    root_logger.propagate = original_propagate

    # Reset module loggers under seeds_clients namespace
    for name in list(logging.root.manager.loggerDict):
        if str(name).startswith("seeds_clients."):
            module_logger = logging.getLogger(name)
            module_logger.setLevel(logging.NOTSET)
            module_logger.handlers.clear()


def test_to_level_accepts_int_and_str():
    assert logging_utils._to_level(logging.INFO) == logging.INFO
    assert logging_utils._to_level("warning") == logging.WARNING


def test_to_level_invalid_raises_valueerror():
    with pytest.raises(ValueError, match="Invalid log level"):
        logging_utils._to_level("nope")


def test_get_logger_namespaces_and_null_handler():
    logger = logging_utils.get_logger("providers.openai")
    assert logger.name == "seeds_clients.providers.openai"
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


def test_get_logger_defaults_to_root():
    logger = logging_utils.get_logger()
    assert logger.name == "seeds_clients"
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


def test_configure_logging_sets_level_handler_and_formatter():
    handler = logging.StreamHandler(StringIO())
    logger = logging_utils.configure_logging(
        level="debug",
        handler=handler,
        fmt="%(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    assert handler in logger.handlers
    assert logger.level == logging.DEBUG
    assert isinstance(handler.formatter, logging.Formatter)


def test_configure_logging_sets_module_levels_and_propagation():
    logger = logging_utils.configure_logging(
        module_levels={"providers": "WARNING", "seeds_clients.utils": logging.ERROR},
        propagate=False,
    )

    provider_logger = logging.getLogger("seeds_clients.providers")
    utils_logger = logging.getLogger("seeds_clients.utils")

    assert provider_logger.level == logging.WARNING
    assert utils_logger.level == logging.ERROR
    assert logger.propagate is False


def test_env_level_applied_on_import(monkeypatch):
    monkeypatch.setenv("SEEDS_CLIENTS_LOG_LEVEL", "INFO")
    importlib.reload(logging_utils)

    logger = logging.getLogger("seeds_clients")
    assert logger.level == logging.INFO

    # Clear env and reload to avoid leaking state
    monkeypatch.delenv("SEEDS_CLIENTS_LOG_LEVEL", raising=False)
    importlib.reload(logging_utils)
