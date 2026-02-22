"""Central logging for ESG platform: retrieval, LLM, ingestion, auditor, and other operations."""

import logging
import sys
from pathlib import Path

import config

LOG_DIR = config.BASE_DIR
LOG_FILE = getattr(config, "LOG_FILE", LOG_DIR / "esg_platform.log")
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: Path = None,
    to_console: bool = True,
) -> logging.Logger:
    """
    Configure and return the application logger. Writes to esg_platform.log and optionally console.
    """
    log_file = log_file or LOG_FILE
    logger = logging.getLogger("esg_platform")
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # File handler: append to log file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def get_logger(name: str = "esg_platform") -> logging.Logger:
    """Return the application logger (call setup_logging first, or this will set it up)."""
    log = logging.getLogger(name)
    if not log.handlers:
        setup_logging()
    return logging.getLogger("esg_platform")


# Initialize on import so all modules can use get_logger()
_app_logger = setup_logging()
