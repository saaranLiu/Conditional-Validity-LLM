"""Utilities for loading processed NSHAP files in Python."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "processed"


def list_rounds(root: Path = DEFAULT_ROOT / "by_round") -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("Round"))


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


def load_round(
    round_name: str,
    *,
    root: Path = DEFAULT_ROOT / "by_round",
    tables: tuple[str, ...] | None = None,
) -> dict[str, pl.DataFrame]:
    round_dir = root / round_name
    if not round_dir.exists():
        raise FileNotFoundError(f"No exported data for {round_name}: {round_dir}")

    if tables is None:
        tables = tuple(
            p.stem for p in sorted(round_dir.glob("*.csv")) if not p.stem.startswith(("variable_labels_", "value_labels_"))
        )

    out: dict[str, pl.DataFrame] = {}
    for name in tables:
        path = round_dir / f"{name}.csv"
        if path.exists():
            out[name] = _read_table(path)
    return out


def load_network_round(
    round_name: str,
    *,
    root: Path = DEFAULT_ROOT / "by_round",
) -> dict[str, pl.DataFrame]:
    tables = ("core", "social_networks")
    if round_name == "Round2":
        tables = ("core", "social_networks", "social_networks_update")
    return load_round(round_name, root=root, tables=tables)


def load_value_labels(
    round_name: str,
    dataset: str,
    *,
    root: Path = DEFAULT_ROOT / "by_round",
) -> dict:
    path = root / round_name / f"value_labels_{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
