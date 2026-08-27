#!/usr/bin/env python3
"""Rebuild ego/alter(/alter_alter) network tables from the year survey.

Design:
  - Only respondents who entered the network module (`numgiven` non-null).
  - `ego` keeps the **full survey** (all columns) plus network flags.
  - `alter` / `alter_alter` are long-form convenience tables; original wide
    columns remain on `ego`.
  - No redundant aliases (`ego_network`, `survey_network`, `network_wide`).
  - Results are written under by_year/{year}/network/ and mirrored to Survey/gss/network/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from network_common import (  # noqa: E402
    active_alter_bases,
    build_alter_alter_table,
    build_alter_table,
    build_ego_table,
    detect_max_alters,
    has_ego_alter_module,
    present_alter_bases,
)
from paths import network_dir, network_table_path  # noqa: E402

ROOT = SCRIPT_DIR.parent.parent
NETWORK_MIRROR_ROOT = ROOT / "Survey" / "gss" / "network"

# Tables retained under network/ (and Survey/gss/network/).
KEEP_TABLES = ("ego", "alter", "alter_alter")
# Removed as redundant once ego is network-module-only.
REDUNDANT_TABLES = ("ego_network", "survey_network", "network_wide")


def read_survey(year_dir: Path) -> pl.DataFrame:
    parquet_path = year_dir / "survey.parquet"
    csv_path = year_dir / "survey.csv"
    if parquet_path.exists():
        df = pl.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pl.read_csv(
            csv_path,
            infer_schema_length=10_000,
            ignore_errors=True,
            null_values=["", "NA", "NaN", "nan"],
        )
    else:
        raise FileNotFoundError(f"No survey file in {year_dir}")

    if "ego_id" not in df.columns:
        df = df.with_columns(
            ego_id=pl.format("{}_{}", pl.col("year"), pl.col("id").cast(pl.Utf8))
        )
    return df


def write_table(df: pl.DataFrame, path: Path) -> None:
    df.write_csv(path)
    df.write_parquet(path.with_suffix(".parquet"))


def remove_table(net: Path, name: str) -> None:
    for p in (net / f"{name}.csv", net / f"{name}.parquet"):
        if p.exists():
            p.unlink()


def prune_redundant(net: Path) -> None:
    for name in REDUNDANT_TABLES:
        remove_table(net, name)


def sync_network_mirror(year: int, net_dir: Path) -> None:
    """Mirror curated network tables to Survey/gss/network/{year}/."""
    dest = NETWORK_MIRROR_ROOT / str(year)
    dest.mkdir(parents=True, exist_ok=True)

    # Clear stale / redundant files in the mirror.
    for name in (*KEEP_TABLES, *REDUNDANT_TABLES):
        for p in (dest / f"{name}.csv", dest / f"{name}.parquet"):
            if p.exists():
                p.unlink()

    for path in net_dir.iterdir():
        if not path.is_file():
            continue
        # Skip redundant data tables if any remain.
        stem = path.name.split(".")[0]
        if stem in REDUNDANT_TABLES:
            continue
        shutil.copy2(path, dest / path.name)


def process_year(year_dir: Path) -> dict:
    df = read_survey(year_dir)
    max_alters = detect_max_alters(df.columns)
    bases_active = active_alter_bases(df, max_alters)
    bases_all = present_alter_bases(df, max_alters)
    module = has_ego_alter_module(df, max_alters, bases_active)

    net = network_dir(year_dir)
    net.mkdir(parents=True, exist_ok=True)

    if not module:
        prune_redundant(net)
        for name in KEEP_TABLES:
            remove_table(net, name)
        year = int(year_dir.name)
        sync_network_mirror(year, net)
        return {
            "has_ego_alter": False,
            "n_ego": 0,
            "n_alter": 0,
            "n_alter_alter": 0,
            "max_alters": max_alters,
            "alter_bases": bases_all,
        }

    # Network-module respondents only (full survey columns).
    ego_df = build_ego_table(df, max_alters).filter(pl.col("in_network_module"))
    alter_df = build_alter_table(df, max_alters, bases_all)
    alter_alter_df = build_alter_alter_table(df, max_alters)

    write_table(ego_df, network_table_path(year_dir, "ego"))
    write_table(alter_df, network_table_path(year_dir, "alter"))

    aa_path = network_table_path(year_dir, "alter_alter")
    if alter_alter_df.height > 0:
        write_table(alter_alter_df, aa_path)
    else:
        remove_table(net, "alter_alter")

    prune_redundant(net)

    year = int(year_dir.name)
    sync_network_mirror(year, net)

    return {
        "has_ego_alter": True,
        "n_ego": ego_df.height,
        "n_alter": alter_df.height,
        "n_alter_alter": alter_alter_df.height,
        "n_ego_cols": ego_df.width,
        "max_alters": max_alters,
        "alter_bases": bases_all,
    }


def update_manifest(manifest_path: Path, year: int, stats: dict) -> None:
    manifest = pl.read_csv(manifest_path)

    for col, dtype in [
        ("has_ego_alter", pl.Boolean),
        ("n_ego_network", pl.Int64),
        ("max_alters", pl.Int64),
        ("alter_bases", pl.Utf8),
    ]:
        if col not in manifest.columns:
            manifest = manifest.with_columns(pl.lit(None).alias(col).cast(dtype))

    bases_str = ",".join(stats.get("alter_bases", []))
    kin_suffix = ",kin" if stats.get("has_ego_alter") else ""
    alter_bases_out = f"{bases_str}{kin_suffix}" if bases_str else ("kin" if kin_suffix else "")
    n_ego = int(stats.get("n_ego", 0))

    manifest = manifest.with_columns(
        has_ego_alter=pl.when(pl.col("year") == year)
        .then(pl.lit(bool(stats.get("has_ego_alter", False))))
        .otherwise(pl.col("has_ego_alter")),
        n_ego=pl.when(pl.col("year") == year)
        .then(pl.lit(n_ego))
        .otherwise(pl.col("n_ego")),
        # Historical column: same as n_ego now that only module respondents are kept.
        n_ego_network=pl.when(pl.col("year") == year)
        .then(pl.lit(n_ego))
        .otherwise(pl.col("n_ego_network")),
        n_alter=pl.when(pl.col("year") == year)
        .then(pl.lit(int(stats.get("n_alter", 0))))
        .otherwise(pl.col("n_alter")),
        max_alters=pl.when(pl.col("year") == year)
        .then(pl.lit(stats.get("max_alters") or None))
        .otherwise(pl.col("max_alters")),
        alter_bases=pl.when(pl.col("year") == year)
        .then(pl.lit(alter_bases_out or None))
        .otherwise(pl.col("alter_bases")),
    )
    if "n_alter_alter" in manifest.columns:
        manifest = manifest.with_columns(
            n_alter_alter=pl.when(pl.col("year") == year)
            .then(pl.lit(int(stats.get("n_alter_alter", 0))))
            .otherwise(pl.col("n_alter_alter")),
        )
    manifest.write_csv(manifest_path)


def discover_network_years(root: Path, manifest_path: Path) -> list[int]:
    if manifest_path.exists():
        manifest = pl.read_csv(manifest_path)
        if "has_ego_alter" in manifest.columns:
            years = manifest.filter(pl.col("has_ego_alter") == True)["year"].to_list()  # noqa: E712
            if years:
                return sorted(int(y) for y in years)
    return sorted(
        int(p.name)
        for p in root.iterdir()
        if p.is_dir()
        and p.name.isdigit()
        and (
            (p / "survey.csv").exists()
            or (p / "network" / "ego.csv").exists()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Survey/gss/processed/by_year"))
    parser.add_argument("--manifest", type=Path, default=Path("Survey/gss/processed/manifest.csv"))
    parser.add_argument("--years", type=int, nargs="*", default=None)
    args = parser.parse_args()

    years = args.years or discover_network_years(args.root, args.manifest)
    # 2010 network comes from Merged file, not cumulative survey.csv.
    years = [y for y in years if y != 2010]
    summary: dict[str, dict] = {}

    for year in years:
        year_dir = args.root / str(year)
        print(f"Enhancing {year}...")
        stats = process_year(year_dir)
        update_manifest(args.manifest, year, stats)
        summary[str(year)] = stats
        print(
            f"  ego={stats.get('n_ego', 0)}x{stats.get('n_ego_cols', '?')} "
            f"alter={stats.get('n_alter', 0)} "
            f"alter_alter={stats.get('n_alter_alter', 0)} "
            f"bases={stats.get('alter_bases')}"
        )

    report_path = args.manifest.parent / "enhance_report.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nUpdated manifest: {args.manifest}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
