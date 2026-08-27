#!/usr/bin/env python3
"""Move network tables from by_year/{year}/ into by_year/{year}/network/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from paths import NETWORK_META, NETWORK_TABLES, network_dir

ROOT = Path("Survey/gss/processed")
MANIFEST = ROOT / "manifest.csv"


def migrate_year(year_dir: Path) -> bool:
    net = network_dir(year_dir)
    moved = 0
    for name in NETWORK_TABLES:
        for ext in (".csv", ".parquet"):
            src = year_dir / f"{name}{ext}"
            if not src.exists():
                continue
            net.mkdir(parents=True, exist_ok=True)
            dst = net / src.name
            if dst.exists():
                src.unlink()
            else:
                src.rename(dst)
            moved += 1

    for name in NETWORK_META:
        src = year_dir / name
        if src.exists():
            net.mkdir(parents=True, exist_ok=True)
            dst = net / name
            if dst.exists():
                src.unlink()
            else:
                src.rename(dst)
            moved += 1

    return moved > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "by_year")
    args = parser.parse_args()

    if MANIFEST.exists():
        manifest = pl.read_csv(MANIFEST)
        years = manifest.filter(pl.col("has_ego_alter") == True)["year"].to_list()  # noqa: E712
    else:
        years = []

    if not years:
        years = [
            int(p.name)
            for p in args.root.iterdir()
            if p.is_dir() and p.name.isdigit() and (p / "ego.csv").exists()
        ]

    for year in sorted(years):
        year_dir = args.root / str(year)
        if migrate_year(year_dir):
            print(f"Migrated network files for {year} -> {network_dir(year_dir)}")

    print("Done.")


if __name__ == "__main__":
    main()
