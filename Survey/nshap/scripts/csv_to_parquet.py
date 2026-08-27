#!/usr/bin/env python3
"""Convert exported NSHAP CSV files to Parquet for faster Python loading."""

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
        default=Path("Survey/nshap/processed/by_round"),
        help="Root directory containing Round* subfolders",
    )
    args = parser.parse_args()
    root: Path = args.root

    if not root.exists():
        raise SystemExit(f"Directory not found: {root}")

    converted = 0
    for round_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for csv_path in sorted(round_dir.glob("*.csv")):
            parquet_path = convert_csv(csv_path)
            print(f"Wrote {parquet_path}")
            converted += 1

    print(f"Converted {converted} CSV file(s).")


if __name__ == "__main__":
    main()
