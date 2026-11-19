"""Shared utilities, paths, logging, plotting style, and JSON helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Global reproducibility seed
RANDOM_STATE: int = 42

# Project directory hierarchy
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_F1_DIR: Path = REPORTS_DIR / "figures" / "followup1"
FIGURES_F2_DIR: Path = REPORTS_DIR / "figures" / "followup2"

# Okabe-Ito colorblind-safe publication palette
PALETTE: dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "gray": "#6C757D",
    "dark": "#212529",
}


def ensure_directories() -> None:
    """Ensure all standard project output directories exist."""
    for directory in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        NOTEBOOKS_DIR,
        FIGURES_F1_DIR,
        FIGURES_F2_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "f1_ml", level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a formatted console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def configure_plot_style() -> None:
    """Configure consistent publication-quality matplotlib/seaborn styling."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


def json_default(obj: Any) -> Any:
    """Convert numpy, pandas, and Path objects to JSON-serializable primitives."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if obj is pd.NA:
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: Any, path: Path, indent: int = 2) -> Path:
    """Serialize data cleanly to UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=json_default, ensure_ascii=False)
    return path


def load_json(path: Path) -> Any:
    """Load UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
