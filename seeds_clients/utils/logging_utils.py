"""Shared logging helpers for seeds-clients.

Provides per-module configuration and a NullHandler by default so the
library stays quiet until the host application enables logging.
"""

from __future__ import annotations

import logging
import os
from typing import Mapping

_ROOT_LOGGER_NAME = "seeds_clients"


def _to_level(level: int | str) -> int:
    """Convert int or string log level to logging level constant."""
    if isinstance(level, int):
        return level

    value = getattr(logging, str(level).upper(), None)
    if isinstance(value, int):
        return value
    raise ValueError(f"Invalid log level: {level}")


def _ensure_null_handler(logger: logging.Logger) -> None:
    """Attach a NullHandler once to avoid 'No handler' warnings."""
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())


# Initialize root logger with env override and a NullHandler
_root_logger = logging.getLogger(_ROOT_LOGGER_NAME)
_ensure_null_handler(_root_logger)
_env_level = os.getenv("SEEDS_CLIENTS_LOG_LEVEL")
if _env_level:
    _root_logger.setLevel(_to_level(_env_level))


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the seeds_clients namespace.

    Args:
        name: Optional module name. If omitted, returns the package root logger.
    """
    if name:
        full_name = name if name.startswith(_ROOT_LOGGER_NAME) else f"{_ROOT_LOGGER_NAME}.{name}"
    else:
        full_name = _ROOT_LOGGER_NAME

    logger = logging.getLogger(full_name)
    _ensure_null_handler(logger)
    return logger


def configure_logging(
    *,
    level: int | str | None = None,
    module_levels: Mapping[str, int | str] | None = None,
    handler: logging.Handler | None = None,
    fmt: str | None = None,
    datefmt: str | None = None,
    propagate: bool | None = None,
) -> logging.Logger:
    """Configure seeds-clients logging.

    Args:
        level: Root level for seeds_clients (overrides SEEDS_CLIENTS_LOG_LEVEL).
        module_levels: Optional mapping of module name -> level (relative or absolute).
        handler: Optional handler to attach to the root seeds_clients logger.
        fmt: Optional formatter string to apply to the handler.
        datefmt: Optional date format string for the formatter.
        propagate: Whether seeds_clients logs propagate to the root logger.

    Returns:
        The configured root seeds_clients logger.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    _ensure_null_handler(logger)

    if level is not None:
        logger.setLevel(_to_level(level))

    if handler:
        if fmt:
            handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(handler)

    if module_levels:
        for module_name, module_level in module_levels.items():
            full_name = (
                module_name
                if module_name.startswith(_ROOT_LOGGER_NAME)
                else f"{_ROOT_LOGGER_NAME}.{module_name}"
            )
            module_logger = logging.getLogger(full_name)
            module_logger.setLevel(_to_level(module_level))
            _ensure_null_handler(module_logger)

    if propagate is not None:
        logger.propagate = propagate

    return logger
