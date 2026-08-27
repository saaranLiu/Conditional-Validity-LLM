"""Utilities for loading processed ISSP files in Python."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "processed"


def _read_table(path: Path) -> pl.DataFrame:
    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        return pl.read_parquet(parquet_path)
    return pl.read_csv(
        path,
        infer_schema_length=10_000,
        ignore_errors=True,
        null_values=["", "NA", "NaN", "nan"],
    )


def load_survey(*, root: Path = DEFAULT_ROOT) -> pl.DataFrame:
    """Load the ISSP 2017 integrated survey table."""
    for name in ("survey.parquet", "survey.csv"):
        path = root / name
        if path.exists():
            return _read_table(path)
    raise FileNotFoundError(f"No survey table found under {root}")


def load_value_labels(*, root: Path = DEFAULT_ROOT) -> dict:
    path = root / "value_labels.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_variable_labels(*, root: Path = DEFAULT_ROOT) -> dict[str, str]:
    path = root / "variable_labels.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(*, root: Path = DEFAULT_ROOT) -> dict:
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
