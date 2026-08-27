#!/usr/bin/env python3
"""Update manifest and enhance_report after 2010 merged network export."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

ROOT = Path("Survey/gss/processed")
YEAR_DIR = ROOT / "by_year/2010"
META_PATH = ROOT / "by_year/2010/network/network_source.json"
MANIFEST_PATH = ROOT / "manifest.csv"
REPORT_PATH = ROOT / "enhance_report.json"


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    manifest = pl.read_csv(MANIFEST_PATH)

    bases = meta.get("alter_bases", [])
    bases_str = ",".join(bases) if bases else ""

    manifest = manifest.with_columns(
        has_ego_alter=pl.when(pl.col("year") == 2010)
        .then(pl.lit(True))
        .otherwise(pl.col("has_ego_alter")),
        has_network_module=pl.when(pl.col("year") == 2010)
        .then(pl.lit(False))
        .otherwise(pl.col("has_network_module")),
        n_ego=pl.when(pl.col("year") == 2010)
        .then(pl.lit(int(meta["n_ego"])))
        .otherwise(pl.col("n_ego")),
        n_ego_network=pl.when(pl.col("year") == 2010)
        .then(pl.lit(int(meta["n_ego_network"])))
        .otherwise(pl.col("n_ego_network")),
        n_alter=pl.when(pl.col("year") == 2010)
        .then(pl.lit(int(meta["n_alter"])))
        .otherwise(pl.col("n_alter")),
        n_alter_alter=pl.when(pl.col("year") == 2010)
        .then(pl.lit(int(meta.get("n_alter_alter", 0))))
        .otherwise(pl.col("n_alter_alter")),
        max_alters=pl.when(pl.col("year") == 2010)
        .then(pl.lit(int(meta["max_alters"])))
        .otherwise(pl.col("max_alters")),
        alter_bases=pl.when(pl.col("year") == 2010)
        .then(pl.lit(bases_str))
        .otherwise(pl.col("alter_bases")),
    )
    manifest.write_csv(MANIFEST_PATH)

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else {}
    report["2010"] = {
        "has_ego_alter": True,
        "n_ego": meta["n_ego"],
        "n_ego_network": meta["n_ego_network"],
        "n_alter": meta["n_alter"],
        "max_alters": meta["max_alters"],
        "alter_bases": [b for b in bases if b != "kin"],
        "source": "GSS2010merged_R8.dta",
        "samptype": "2006",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Updated {MANIFEST_PATH}")
    print(f"Updated {REPORT_PATH}")


if __name__ == "__main__":
    main()
