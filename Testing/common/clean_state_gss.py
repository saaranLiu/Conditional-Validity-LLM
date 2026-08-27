#!/usr/bin/env python3
"""Clean Phase 1 GSS network-module years from Survey/gss/network/{year}/ego.csv.

Covers 1985, 1987, 2004, 2010, 2024. Produces cleaned CSV + Task 1/2/3 JSONL under
Survey/ucnets/cleaned/ with dataset ids: gss_1985, gss_1987, gss_2004,
gss_2010_network, gss_2024.

Note: 1985/1987 predate the trust/fair/helpful ballot rotation, so those targets are
simply absent for those years rather than imputed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import clean_state_ucnets as base  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Survey" / "gss" / "cleaned"
NETWORK_ROOT = ROOT / "Survey" / "gss" / "network"

YEARS = (1985, 1987, 2004, 2010, 2024)

# Prefer year survey labels (full codebook); fall back encoding for 2010.
LABEL_CANDIDATES = {
    year: [
        ROOT / "Survey" / "gss" / "processed" / "by_year" / str(year) / "value_labels.json",
        NETWORK_ROOT / str(year) / "value_labels.json",
    ]
    for year in YEARS
}

VOTE_FIELD = {
    1985: "vote84",
    1987: "vote84",
    2004: "vote00",
    2010: "VOTE08",  # case as in Merged export; also try vote08
    2024: "vote20",
}

# Presidential election each year's vote item refers to.
VOTE_KEY = {
    1985: "voting_1984",
    1987: "voting_1984",
    2004: "voting_2000",
    2010: "voting_2008",
    2024: "voting_2020",
}

VOTE_CODE_MAP = {
    # Align to Phase 1 allowed: voted / did not vote / ineligible
    "1": "voted",
    "2": "did not vote",
    "3": "ineligible",
}

DATASET_ID = {
    1985: "gss_1985",
    1987: "gss_1987",
    2004: "gss_2004",
    2010: "gss_2010_network",
    2024: "gss_2024",
}


def read_json_flexible(path: Path) -> dict[str, Any]:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with path.open(encoding=enc, errors="strict") as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    with path.open(encoding="utf-8", errors="replace") as f:
        return json.load(f)


def load_labels(year: int) -> dict[str, Any]:
    for path in LABEL_CANDIDATES[year]:
        if path.exists():
            return read_json_flexible(path)
    raise FileNotFoundError(f"No value_labels.json for year {year}")


def resolve_vote_col(headers: list[str], year: int) -> str | None:
    preferred = VOTE_FIELD[year]
    lower = {h.lower(): h for h in headers}
    if preferred.lower() in lower:
        return lower[preferred.lower()]
    for alt in ("vote08", "vote84", "vote00", "vote20", "vote16", "vote80"):
        if alt in lower:
            return lower[alt]
    return None


def clean_network_year(year: int) -> dict[str, Any]:
    dataset = DATASET_ID[year]
    ego_path = NETWORK_ROOT / str(year) / "ego.csv"
    rows = base.read_csv(ego_path)
    labels = load_labels(year)
    headers = list(rows[0].keys()) if rows else []
    vote_col = resolve_vote_col(headers, year)

    # Reuse GSS anchor/target lists; extras ignored if absent.
    selected = [c for c in (base.GSS_ANCHORS + base.GSS_TARGETS) if c in headers]
    # Always keep id/year if present
    for must in ("id", "year", "ego_id"):
        if must in headers and must not in selected:
            selected.append(must)

    clean_rows: list[dict[str, Any]] = []
    vote_map: dict[str, str] = {}

    for row in rows:
        rid = str(row.get("id") or "").strip()
        out: dict[str, Any] = {
            "respondent_id": rid,
            "dataset": dataset,
            "weight": row.get("wtssall") or row.get("wtss") or row.get("wtcomb") or None,
            "ego_id": row.get("ego_id"),
            "source_year": year,
            "source_path": str(ego_path.relative_to(ROOT)),
        }
        out.update(base.add_labeled_columns("gss", row, selected, labels))
        out.update(
            {
                "age_group_harmonized": base.age_bin(row.get("age")),
                "gender_harmonized": base.clean_text(out.get("sex_label")),
                "race_ethnicity_harmonized": base.race_gss(row, labels),
                "education_harmonized": base.education_gss_degree(row.get("degree")),
                "marital_harmonized": base.clean_text(out.get("marital_label")),
                "work_status_harmonized": base.clean_text(out.get("wrkstat_label")),
                "children_harmonized": base.children_gss(row.get("childs")),
                "health_harmonized": base.clean_text(out.get("health_label")),
            }
        )
        clean_rows.append(out)

        if vote_col and rid:
            code = str(row.get(vote_col, "")).strip()
            label = VOTE_CODE_MAP.get(code)
            if label:
                vote_map[rid] = label

    # Task JSONL (same logic as base.build_gss_jsonl but with dataset id)
    required = [
        "age_group_harmonized",
        "gender_harmonized",
        "race_ethnicity_harmonized",
        "education_harmonized",
        "marital_harmonized",
        "work_status_harmonized",
    ]
    task1: list[dict[str, Any]] = []
    task2: list[dict[str, Any]] = []
    target_fields = [
        "self_rated_health",
        "general_happiness",
        "people_can_be_trusted",
        "people_are_fair",
        "people_are_helpful",
        "party_identity",
        "political_ideology",
        "religion",
        "religious_service_attendance",
        "religious_affiliation_strength",
    ]

    for row in clean_rows:
        if not base.complete_required(row, required):
            continue
        task1.append(
            base.task_record(
                row,
                dataset,
                "s0_baseline_state",
                base.gss_prompt(row, include_health=False),
                base.gss_target(row, include_health=True),
                target_fields,
            )
        )
        if row.get("health_harmonized"):
            task2.append(
                base.task_record(
                    row,
                    dataset,
                    "s1_health_conditioned",
                    base.gss_prompt(row, include_health=True),
                    base.gss_target(row, include_health=False),
                    [f for f in target_fields if f != "self_rated_health"],
                )
            )

    # Task 3
    vote_key = VOTE_KEY[year]
    task3: list[dict[str, Any]] = []
    for row in task1:
        rid = str(row["respondent_id"])
        vote = vote_map.get(rid)
        if not vote:
            continue
        prompt = dict(row["prompt_json"])
        target = dict(row["target_json"])
        for field in ("party_identity", "religion"):
            value = target.get(field)
            if value in (None, ""):
                prompt = None  # type: ignore
                break
            prompt[field] = value
            target.pop(field, None)
        if prompt is None:
            continue
        prompt[vote_key] = vote
        meta = dict(row.get("metadata") or {})
        meta["conditioning_fields"] = ["party_identity", "religion", vote_key]
        meta["source_task"] = "s0_baseline_state"
        meta["voting_source"] = vote_col
        task3.append(
            {
                "respondent_id": row["respondent_id"],
                "dataset": dataset,
                "task": "s2_politics_religion_conditioned",
                "prompt_json": prompt,
                "target_json": target,
                "metadata": meta,
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_csv = OUT_DIR / f"{dataset}_state_clean.csv"
    base.write_csv(clean_csv, clean_rows, base.output_fieldnames(clean_rows))

    files = {
        f"{dataset}_s0.jsonl": task1,
        f"{dataset}_s1.jsonl": task2,
        f"{dataset}_s2.jsonl": task3,
    }
    for name, payload in files.items():
        path = OUT_DIR / name
        with path.open("w", encoding="utf-8") as f:
            for rec in payload:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "dataset": dataset,
        "year": year,
        "n_ego": len(rows),
        "n_clean": len(clean_rows),
        "n_task1": len(task1),
        "n_task2": len(task2),
        "n_task3": len(task3),
        "vote_col": vote_col,
        "n_vote_labels": len(vote_map),
        "clean_csv": str(clean_csv),
    }


def main() -> None:
    summary = []
    for year in YEARS:
        stats = clean_network_year(year)
        summary.append(stats)
        print(
            f"{stats['dataset']}: ego={stats['n_ego']} "
            f"task1={stats['n_task1']} task2={stats['n_task2']} task3={stats['n_task3']} "
            f"vote={stats['vote_col']} (n={stats['n_vote_labels']})"
        )
    (OUT_DIR / "gss_network_clean_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
