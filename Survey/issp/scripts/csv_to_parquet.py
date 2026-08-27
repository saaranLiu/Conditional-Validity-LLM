#!/usr/bin/env python3
"""Convert exported ISSP CSV to Parquet for faster Python loading."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def convert_csv(csv_path: Path) -> Path:
    parquet_path = csv_path.with_suffix(".parquet")
    df = pl.read_csv(
        csv_path,
        infer_schema_length=10_000,
        ignore_errors=True,
        null_values=["", "NA", "NaN", "nan"],
    )
    df.write_parquet(parquet_path)
    return parquet_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Survey/issp/processed"),
        help="Directory containing survey.csv",
    )
    args = parser.parse_args()
    csv_path = args.root / "survey.csv"
    if not csv_path.exists():
        raise SystemExit(f"File not found: {csv_path}")
    parquet_path = convert_csv(csv_path)
    print(f"Wrote {parquet_path}")


if __name__ == "__main__":
    main()
