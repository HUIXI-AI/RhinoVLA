"""Minimal logging helpers for RhinoVLA training modules."""

from __future__ import annotations

import logging
import os


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%m/%d %H:%M:%S"
_HANDLER_ATTR = "_rhinovla_training_handler"


def configure_training_logging(level: int | str | None = None) -> None:
    """Install a small, idempotent root logger configuration for training logs."""

    if level is None:
        level = os.environ.get("RHINOVLA_LOG_LEVEL", "INFO")

    root = logging.getLogger()
    handler = next(
        (
            existing
            for existing in root.handlers
            if getattr(existing, _HANDLER_ATTR, False)
        ),
        None,
    )

    if handler is None:
        handler = logging.StreamHandler()
        setattr(handler, _HANDLER_ATTR, True)
        root.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root.setLevel(level)


def get_training_logger(name: str) -> logging.Logger:
    configure_training_logging()
    return logging.getLogger(name)
