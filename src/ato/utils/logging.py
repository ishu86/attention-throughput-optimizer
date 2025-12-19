"""Logging configuration for ATO."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration for ATO.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        format_string: Custom format string. Uses default if not provided.
        log_file: Optional file path to write logs.

    Returns:
        Configured root logger for ATO.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get ATO logger
    logger = logging.getLogger("ato")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "ato") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name. Prepended with 'ato.' if not already.

    Returns:
        Logger instance.
    """
    if not name.startswith("ato"):
        name = f"ato.{name}"
    return logging.getLogger(name)


class LogLevel:
    """Logging level constants for convenience."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
