"""Shared utilities and project paths for PhishGuard-Lite."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
REPORT_DIR = PROJECT_ROOT / "report"
REPORT_FIGURES_DIR = REPORT_DIR / "figures"


def ensure_directories() -> None:
    """Create expected project output directories if they are missing."""

    for path in [
        DATA_DIR,
        MODEL_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        REPORT_DIR,
        REPORT_FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_url_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Read a URL dataset with columns ``url`` and ``label``.

    Labels must be binary, where 0 means legitimate and 1 means phishing.
    """

    df = pd.read_csv(csv_path)
    required_columns = {"url", "label"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required column(s): {missing_text}")

    df = df[["url", "label"]].copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["url", "label"])
    df = df[df["url"] != ""]
    df["label"] = df["label"].astype(int)

    invalid_labels = sorted(set(df["label"]) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Labels must be 0 or 1. Found: {invalid_labels}")

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")
    return df.reset_index(drop=True)


def write_json(path: str | Path, data: object) -> None:
    """Write JSON with stable formatting."""

    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def safe_divide(numerator: float, denominator: float) -> float:
    """Return 0.0 when the denominator is zero."""

    return float(numerator / denominator) if denominator else 0.0


def feature_columns_from_frame(df: pd.DataFrame, excluded: Iterable[str] = ("url", "label")) -> list[str]:
    """Return numeric feature columns from a dataframe."""

    excluded_set = set(excluded)
    return [column for column in df.columns if column not in excluded_set]
