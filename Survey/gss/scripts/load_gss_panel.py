"""Utilities for loading processed GSS panel wave files."""

from __future__ import annotations

from pathlib import Path

import polars as pl

DEFAULT_PANEL_ROOT = Path(__file__).resolve().parents[1] / "processed" / "panel"


def list_cohorts(root: Path = DEFAULT_PANEL_ROOT) -> list[int]:
    return sorted(
        int(p.name.replace("cohort_", ""))
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith("cohort_")
    )


def list_waves(cohort: int, *, root: Path = DEFAULT_PANEL_ROOT) -> list[str]:
    cohort_dir = root / f"cohort_{cohort}"
    if not cohort_dir.exists():
        raise FileNotFoundError(f"No panel cohort: {cohort_dir}")
    return sorted(
        p.name
        for p in cohort_dir.iterdir()
        if p.is_dir() and p.name.startswith("wave_")
    )


def load_panel_wave(
    cohort: int,
    wave_dir: str,
    *,
    root: Path = DEFAULT_PANEL_ROOT,
) -> pl.DataFrame:
    path = root / f"cohort_{cohort}" / wave_dir / "survey.parquet"
    csv_path = path.with_suffix(".csv")
    if path.exists():
        return pl.read_parquet(path)
    if csv_path.exists():
        return pl.read_csv(
            csv_path,
            infer_schema_length=10_000,
            ignore_errors=True,
            null_values=["", "NA", "NaN", "nan"],
        )
    raise FileNotFoundError(f"No survey for cohort {cohort} {wave_dir}: {csv_path}")
