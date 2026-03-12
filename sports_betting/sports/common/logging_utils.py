"""Logging helpers."""

import logging
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str = "sports_betting.log") -> logging.Logger:
    """Create and configure root logger."""
    logger = logging.getLogger("sports_betting")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    Path("sports_betting/data/outputs").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(f"sports_betting/data/outputs/{log_file}")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
