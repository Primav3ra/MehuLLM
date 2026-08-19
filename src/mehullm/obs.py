"""Logging setup and trace correlation."""

from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Any

import structlog

_configured = False


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Console renderer to stderr, JSON lines to log_file. Idempotent."""
    global _configured
    if _configured:
        return

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(processor=structlog.processors.JSONRenderer())
        )
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(level.upper())

    structlog.configure(
        processors=[
            *shared,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def utf8_stdout() -> None:
    """Windows consoles default to cp1252; model output and box-drawing are not."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)


def bind_run(**kw: Any) -> None:
    """Attach trace_id/run_id to every later log line on this task."""
    structlog.contextvars.bind_contextvars(**{k: v for k, v in kw.items() if v is not None})


def redacted_len(value: str) -> str:
    """Describe a secret without printing it."""
    return f"resolved(len={len(value)})" if value else "EMPTY"
