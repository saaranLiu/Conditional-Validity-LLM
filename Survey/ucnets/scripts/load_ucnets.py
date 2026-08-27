"""Utilities for loading processed UCNets files in Python."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "processed"


def list_waves(root: Path = DEFAULT_ROOT / "by_wave") -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


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


def load_wave(
    wave: str,
    *,
    root: Path = DEFAULT_ROOT / "by_wave",
    tables: tuple[str, ...] = ("ego", "alters"),
) -> dict[str, pl.DataFrame]:
    wave_dir = root / wave
    if not wave_dir.exists():
        raise FileNotFoundError(f"No exported data for {wave}: {wave_dir}")

    out: dict[str, pl.DataFrame] = {}
    for name in tables:
        path = wave_dir / f"{name}.csv"
        if path.exists():
            out[name] = _read_table(path)
    return out


def load_value_labels(
    wave: str,
    dataset: str,
    *,
    root: Path = DEFAULT_ROOT / "by_wave",
) -> dict:
    path = root / wave / f"value_labels_{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_variable_labels(
    wave: str,
    dataset: str,
    *,
    root: Path = DEFAULT_ROOT / "by_wave",
) -> dict[str, str]:
    """Return variable name -> English question/field description."""
    path = root / wave / f"variable_labels_{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
