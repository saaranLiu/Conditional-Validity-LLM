"""Utilities for loading processed GSS year files in Python."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paths import NETWORK_TABLES, network_table_path  # noqa: E402

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "processed"


def list_years(root: Path = DEFAULT_ROOT / "by_year") -> list[int]:
    return sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())


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


def load_year(
    year: int,
    *,
    root: Path = DEFAULT_ROOT / "by_year",
    tables: tuple[str, ...] = ("survey",),
) -> dict[str, pl.DataFrame]:
    year_dir = root / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"No exported data for year {year}: {year_dir}")

    out: dict[str, pl.DataFrame] = {}
    for name in tables:
        if name in NETWORK_TABLES:
            path = network_table_path(year_dir, name)
            legacy = year_dir / f"{name}.csv"
            if path.exists():
                out[name] = _read_table(path)
            elif legacy.exists():
                out[name] = _read_table(legacy)
            continue

        path = year_dir / f"{name}.csv"
        if path.exists():
            out[name] = _read_table(path)
    return out


def load_egor_year(
    year: int,
    *,
    root: Path = DEFAULT_ROOT / "by_year",
) -> dict[str, pl.DataFrame]:
    return load_year(
        year,
        root=root,
        tables=("ego", "alter", "alter_alter"),
    )


def load_network_year(
    year: int,
    *,
    root: Path = DEFAULT_ROOT / "by_year",
) -> dict[str, pl.DataFrame]:
    """Load network tables for a year (ego / alter / alter_alter when present)."""
    year_dir = root / str(year)
    tables = tuple(t for t in NETWORK_TABLES if network_table_path(year_dir, t).exists())
    return load_year(year, root=root, tables=tables)
