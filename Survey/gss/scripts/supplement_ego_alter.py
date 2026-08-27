#!/usr/bin/env python3
"""Supplement ego/alter tables for years with partial network modules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from enhance_network import process_year, update_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Survey/gss/processed/by_year"))
    parser.add_argument("--years", type=int, nargs="*", default=[1987, 2024])
    parser.add_argument("--manifest", type=Path, default=Path("Survey/gss/processed/manifest.csv"))
    args = parser.parse_args()

    for year in args.years:
        year_dir = args.root / str(year)
        print(f"Processing {year}...")
        stats = process_year(year_dir)
        print(
            f"  has_ego_alter={stats['has_ego_alter']} "
            f"ego={stats.get('n_ego', 0)} alter={stats.get('n_alter', 0)} "
            f"max_alters={stats.get('max_alters')} bases={stats.get('alter_bases')}"
        )
        update_manifest(args.manifest, year, stats)

    print(f"Updated manifest: {args.manifest}")


if __name__ == "__main__":
    main()
