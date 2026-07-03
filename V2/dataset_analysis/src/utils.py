"""
src/utils.py
------------
Shared utility functions used across all analysis modules.

Responsibilities
----------------
- Directory creation helper
- Safe logging setup
- Matplotlib figure saver
- Pandas display helpers
- Timer/profiling context manager
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(
    name: str,
    log_file: Path,
    level: int = logging.DEBUG,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> logging.Logger:
    """
    Create (or retrieve) a named logger that writes to both a file and stdout.

    Parameters
    ----------
    name     : Logger name (typically the module ``__name__``).
    log_file : Absolute path to the log file.
    level    : Logging verbosity (default: DEBUG).
    fmt      : Log format string.

    Returns
    -------
    logging.Logger
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:            # avoid duplicate handlers on re-import
        return logger

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Directory Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs(*dirs: Path) -> None:
    """Create multiple directories (and parents) if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Figure Saver
# ─────────────────────────────────────────────────────────────────────────────

def save_figure(
    fig: plt.Figure,
    path: Path,
    dpi: int = 150,
    tight: bool = True,
) -> None:
    """
    Save a Matplotlib figure to *path*, creating parent directories.

    Parameters
    ----------
    fig   : The figure object to save.
    path  : Destination file path (PNG recommended).
    dpi   : Dots-per-inch for the output image.
    tight : Whether to call ``tight_layout()`` before saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CSV / Table Savers
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_excel(df: pd.DataFrame, path: Path, sheet_name: str = "Sheet1", index: bool = False) -> None:
    """Save a DataFrame to Excel (.xlsx), creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=index, sheet_name=sheet_name, engine="openpyxl")


# ─────────────────────────────────────────────────────────────────────────────
# Timer Context Manager
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Generator[None, None, None]:
    """
    Context manager that logs wall-clock time for a labelled block.

    Usage
    -----
    with timer("Loading data", logger):
        df = pd.read_csv(...)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"[TIMER] {label} completed in {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame Helpers
# ─────────────────────────────────────────────────────────────────────────────

def memory_usage_mb(df: pd.DataFrame) -> float:
    """Return the total memory usage of a DataFrame in megabytes."""
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def human_readable_bytes(num_bytes: int) -> str:
    """Convert a byte count to a human-readable string (KB / MB / GB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024.0  # type: ignore[assignment]
    return f"{num_bytes:.1f} PB"


def classify_dtype(series: pd.Series) -> str:
    """
    Classify a pandas Series into a human-readable type category.

    Returns one of: 'Numeric', 'Categorical', 'Datetime', 'Boolean', 'Text', 'Unknown'.
    """
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "Boolean"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "Datetime"
    if pd.api.types.is_numeric_dtype(dtype):
        return "Numeric"
    if dtype == object:
        # Try to distinguish free-text from categoricals
        n_unique = series.nunique(dropna=True)
        n_non_null = series.count()
        if n_non_null == 0:
            return "Unknown"
        ratio = n_unique / max(n_non_null, 1)
        if ratio > 0.5 or n_unique > 200:
            return "Text"
        return "Categorical"
    if hasattr(dtype, "categories"):
        return "Categorical"
    return "Unknown"


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split *lst* into sub-lists of at most *chunk_size* elements."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_corr(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Compute a correlation matrix, dropping columns that are entirely NaN
    to prevent ValueError in scipy/numpy.
    """
    num_df = df.select_dtypes(include="number").copy()
    # Drop columns with zero standard deviation
    std = num_df.std()
    valid_cols = std[std > 0].index.tolist()
    return num_df[valid_cols].corr(method=method)
