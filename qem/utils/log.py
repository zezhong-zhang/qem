"""Lightweight logging helpers for QEM.

We're a numerical library, not a service that needs JSON logs piped
into ELK. So this module is intentionally thin: a stdlib basicConfig
wrapper plus a tqdm-friendly handler so progress bars and log lines
don't fight for the terminal.

Public surface:
    setup_logging(level="INFO", log_file=None)
    get_logger(name) -> logging.Logger
    temporary_log_level(name, level)  # context manager
"""

from __future__ import annotations

import logging
import logging.handlers
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Route log records through tqdm.write so progress bars stay clean."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
) -> None:
    """Configure the root QEM logger. Idempotent."""
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    qem_logger = logging.getLogger("qem")
    if qem_logger.handlers:
        return

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = TqdmLoggingHandler()
    console.setFormatter(fmt)
    console.setLevel(level)

    qem_logger.setLevel(level)
    qem_logger.addHandler(console)
    qem_logger.propagate = False

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5,
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(level)
        qem_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a stdlib logger. Use this inside qem.* modules."""
    return logging.getLogger(name)


@contextmanager
def temporary_log_level(logger_name: str, level: str):
    """Temporarily change a logger's level for debugging."""
    logger = logging.getLogger(logger_name)
    old_level = logger.level
    logger.setLevel(getattr(logging, level.upper()))
    try:
        yield
    finally:
        logger.setLevel(old_level)


__all__ = ["setup_logging", "get_logger", "temporary_log_level", "TqdmLoggingHandler"]
