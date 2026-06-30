"""
logger.py
---------
Centralised logging setup using Python's standard `logging` module.
Writes a timed log file + plain console output (ASCII-safe for Windows).

Usage
-----
    from src.preprocessing.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting pipeline...")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


class _PlainFormatter(logging.Formatter):
    """Simple formatter with level tag, timestamp and module name."""

    def format(self, record: logging.LogRecord) -> str:
        time_str = self.formatTime(record, "%H:%M:%S")
        return f"{time_str} [{record.levelname:<8}] {record.name}: {record.getMessage()}"


def setup_logging(log_dir: Path, level: int = logging.DEBUG) -> None:
    """
    Call once at application startup.  Creates a dated log file and
    configures the root logger.

    Parameters
    ----------
    log_dir : Path
        Directory where log files are written.
    level : int
        Minimum log level (default DEBUG).
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"pipeline_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers already attached (avoid duplicate output on re-runs)
    root.handlers.clear()

    # File handler - plain text, debug level, UTF-8
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(fh)

    # Console handler - ASCII-safe
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_PlainFormatter())
    root.addHandler(ch)

    root.info("Logging initialised -> %s", log_file)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.  Call setup_logging first.

    Parameters
    ----------
    name : str
        Typically __name__ of the calling module.
    """
    return logging.getLogger(name)
