#!/usr/bin/env python3
"""Clean UCNets Wave 1 ego-network gold labels for Phase 2.

Builds per-ego:
  - composition_summary (full alter set)
  - core_alters (Top-5 by contact → closeness → confidant order)

Joins Phase 1 true State (S_H) from ucnets_w1_task1.jsonl.
Writes under Survey/ucnets/cleaned/.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Testing" / "scripts"))
from clean_state_ucnets import ucnets_prompt, ucnets_target  # noqa: E402

ALTERS_PATH = (
    ROOT / "Survey" / "ucnets"
    / "processed"
    / "by_wave"
    / "Wave1"
    / "alters.csv"
)
PHASE1_STATE = ROOT / "Survey" / "ucnets" / "cleaned" / "ucnets_w1_state_clean.csv"
OUT_DIR = ROOT / "Survey" / "ucnets" / "cleaned"

CONTACT_MAP = {
    1: "daily",
    2: "weekly",
    3: "monthly",
    4: "several_times_year",
    5: "yearly_or_less",
    6: "never",
}
CONTACT_RANK = {
    "daily": 1,
    "weekly": 2,
    "monthly": 3,
    "several_times_year": 4,
    "yearly_or_less": 5,
    "never": 6,
    "unknown": 7,
}

# Multi-select relationship flags → label; priority when multiple fire.
REL_SPECS: list[tuple[str, str]] = [
    ("C1A_1", "spouse"),
    ("C1A_2", "romantic_partner"),
    ("C1A_4", "child"),
    ("C1A_24", "child"),
    ("C1A_3", "parent"),
    ("C1A_23", "parent"),
    ("C1A_5", "sibling"),
    ("C1A_25", "sibling"),
    ("C1A_26", "sibling"),
    ("C1A_6", "other_relative"),
    ("C1A_10", "housemate"),
    ("C1A_20", "friend"),
    ("C1A_12", "coworker"),
    ("C1A_11", "neighbor"),
    ("C1A_13", "school"),
    ("C1A_14", "church"),
    ("C1A_21", "acquaintance"),
    ("C1A_22", "other"),
]
REL_PRIORITY: dict[str, int] = {}
for _, lab in REL_SPECS:
    if lab not in REL_PRIORITY:
        REL_PRIORITY[lab] = len(REL_PRIORITY)

KIN_LABELS = {
    "spouse",
    "romantic_partner",
    "parent",
    "child",
    "sibling",
    "other_relative",
}

ROLE_SPECS = [
    ("B2A", "companion"),
    ("B4A", "confidant"),
    ("B5A", "advisor"),
    ("B6B", "practical_help"),
    ("B7C", "injury_help"),
    ("B8A", "r_helps_them"),
    ("B9A", "difficult"),
]

GENDER_MAP = {1: "male", 2: "female"}
YESNO_MAP = {1: "yes", 0: "no"}


def _is_present(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and math.isnan(val):
        return False
    if pd.isna(val):
        return False
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return False
    try:
        f = float(s)
        if f in {8, 9, 98, 99}:  # common missing / DK codes when present as numeric
            return False
    except ValueError:
        pass
    return True


def _flag_on(val: Any) -> bool:
    """C1A_* multi-select: treat non-missing / non-zero as selected."""
    if not _is_present(val):
        return False
    try:
        return float(val) != 0.0
    except (TypeError, ValueError):
        return True


def relationship_of(row: pd.Series) -> str:
    hits: list[str] = []
    for col, label in REL_SPECS:
        if col in row.index and _flag_on(row.get(col)):
            hits.append(label)
    if not hits:
        return "other"
    return min(hits, key=lambda x: REL_PRIORITY.get(x, 999))


def support_roles_of(row: pd.Series) -> list[str]:
    roles: list[str] = []
    for col, label in ROLE_SPECS:
        if col in row.index and _is_present(row.get(col)):
            roles.append(label)
    return roles


def contact_frequency_of(row: pd.Series) -> str:
    codes: list[int] = []
    for col in ("D1F", "D1G", "D1H"):
        val = row.get(col)
        if not _is_present(val):
            continue
        try:
            code = int(float(val))
        except (TypeError, ValueError):
            continue
        if code in CONTACT_MAP:
            codes.append(code)
    if not codes:
        return "unknown"
    return CONTACT_MAP[min(codes)]


def gender_of(row: pd.Series) -> str | None:
    val = row.get("N_GENDER")
    if not _is_present(val):
        return None
    try:
        return GENDER_MAP.get(int(float(val)))
    except (TypeError, ValueError):
        return None


def same_race_of(row: pd.Series) -> str | None:
    val = row.get("C2G_NSAMERACE")
    if not _is_present(val):
        return None
    try:
        code = int(float(val))
    except (TypeError, ValueError):
        return None
    if code == 9:
        return None
    return YESNO_MAP.get(code)


def nclose_score(row: pd.Series) -> int:
    """Higher = closer. Especially close (1) preferred."""
    val = row.get("C2B_NCLOSE")
    if not _is_present(val):
        return 0
    try:
        code = int(float(val))
    except (TypeError, ValueError):
        return 0
    if code == 1:
        return 2
    if code == 0:
        return 1
    return 0


def confidant_order(row: pd.Series) -> float:
    """Lower nomination order = better; missing = large."""
    val = row.get("B4A")
    if not _is_present(val):
        return 999.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 999.0


def size_bucket(n: int) -> str:
    if n <= 5:
        return "small"
    if n <= 15:
        return "medium"
    return "large"


def homophily_bucket(share: float | None) -> str | None:
    """3-way bucket for full-network same-race alter share (B5 conditioning)."""
    if share is None:
        return None
    if share < 0.75:
        return "low"
    if share < 1.0:
        return "medium"
    return "high"


def alter_record(row: pd.Series, alter_idx: int) -> dict[str, Any]:
    return {
        "alter_index": alter_idx,
        "relationship": relationship_of(row),
        "support_roles": support_roles_of(row),
        "contact_frequency": contact_frequency_of(row),
        "gender": gender_of(row),
        "same_race_as_ego": same_race_of(row),
        "_contact_rank": CONTACT_RANK[contact_frequency_of(row)],
        "_nclose": nclose_score(row),
        "_confidant_order": confidant_order(row),
    }


def composition_summary(alters: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(alters)
    if n == 0:
        return {
            "num_alters": 0,
            "size_bucket": "small",
            "kin_share": None,
            "friend_share": None,
            "coworker_share": None,
            "confidant_count": 0,
            "companion_count": 0,
            "pct_weekly_plus_contact": None,
        }
    rels = [a["relationship"] for a in alters]
    kin_share = sum(1 for r in rels if r in KIN_LABELS) / n
    friend_share = sum(1 for r in rels if r == "friend") / n
    coworker_share = sum(1 for r in rels if r == "coworker") / n
    confidant_count = sum(1 for a in alters if "confidant" in a["support_roles"])
    companion_count = sum(1 for a in alters if "companion" in a["support_roles"])
    freq_known = [a["contact_frequency"] for a in alters if a["contact_frequency"] != "unknown"]
    if freq_known:
        pct_weekly = sum(1 for f in freq_known if f in {"daily", "weekly"}) / len(freq_known)
    else:
        pct_weekly = None
    race_known = [a["same_race_as_ego"] for a in alters if a["same_race_as_ego"] is not None]
    if race_known:
        same_race_share = sum(1 for r in race_known if r == "yes") / len(race_known)
    else:
        same_race_share = None
    return {
        "num_alters": n,
        "size_bucket": size_bucket(n),
        "kin_share": round(kin_share, 4),
        "friend_share": round(friend_share, 4),
        "coworker_share": round(coworker_share, 4),
        "confidant_count": confidant_count,
        "companion_count": companion_count,
        "pct_weekly_plus_contact": None if pct_weekly is None else round(pct_weekly, 4),
        "same_race_share": None if same_race_share is None else round(same_race_share, 4),
        "same_race_share_coverage": round(len(race_known) / n, 4),
    }


def core_alters_top5(alters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        alters,
        key=lambda a: (a["_contact_rank"], -a["_nclose"], a["_confidant_order"], a["alter_index"]),
    )
    out: list[dict[str, Any]] = []
    for i, a in enumerate(ranked[:5], start=1):
        out.append(
            {
                "rank": i,
                "relationship": a["relationship"],
                "support_roles": a["support_roles"],
                "contact_frequency": a["contact_frequency"]
                if a["contact_frequency"] != "unknown"
                else "yearly_or_less",
                "gender": a["gender"] or "female",
                "same_race_as_ego": a["same_race_as_ego"] or "yes",
            }
        )
    return out


def load_state_by_id() -> dict[str, dict[str, Any]]:
    """Build full S_H from Phase1 state_clean (all alter-overlap egos)."""
    state_df = pd.read_csv(PHASE1_STATE, dtype=str, keep_default_na=False)
    by_id: dict[str, dict[str, Any]] = {}
    for row in state_df.to_dict(orient="records"):
        rid = str(row.get("PRIM_KEY") or row.get("respondent_id") or "")
        if not rid:
            continue
        # Merge demographics + observed state targets as conditioning S_H.
        state = {}
        state.update(ucnets_prompt(row, include_health=False))
        state.update(ucnets_target(row, include_health=True))
        by_id[rid] = {
            "state_json": state,
            "weight": row.get("weight") or row.get("WEIGHT_W1_DEMO"),
        }
    return by_id


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alters_df = pd.read_csv(ALTERS_PATH, low_memory=False)
    state_by_id = load_state_by_id()
    state_ids = set(state_by_id)
    alter_ids = set(alters_df["PRIM_KEY"].astype(str).unique())
    overlap = sorted(state_ids & alter_ids)
    only_state = sorted(state_ids - alter_ids)
    only_alters = sorted(alter_ids - state_ids)

    records: list[dict[str, Any]] = []
    size_counts: Counter[str] = Counter()
    skipped_no_alters = 0

    for pid in overlap:
        ego_alters = alters_df[alters_df["PRIM_KEY"].astype(str) == pid]
        parsed = [alter_record(row, i) for i, (_, row) in enumerate(ego_alters.iterrows())]
        if not parsed:
            skipped_no_alters += 1
            continue
        summary = composition_summary(parsed)
        summary["same_race_share_bucket"] = homophily_bucket(summary["same_race_share"])
        core = core_alters_top5(parsed)
        size_counts[summary["size_bucket"]] += 1
        st = state_by_id[pid]
        records.append(
            {
                "respondent_id": pid,
                "dataset": "ucnets_w1",
                "PRIM_KEY": pid,
                "state_json": st["state_json"],
                "target_json": {
                    "composition_summary": summary,
                    "core_alters": core,
                },
                "metadata": {
                    "weight": st["weight"],
                    "n_alters_raw": len(parsed),
                    "size_bucket": summary["size_bucket"],
                    "contact_freq_coverage": sum(
                        1 for a in parsed if a["contact_frequency"] != "unknown"
                    )
                    / len(parsed),
                },
            }
        )

    # Population priors for B3 (computed on analysis sample; stored separately)
    n_vals = [r["target_json"]["composition_summary"]["num_alters"] for r in records]
    kin_vals = [r["target_json"]["composition_summary"]["kin_share"] for r in records]
    friend_vals = [r["target_json"]["composition_summary"]["friend_share"] for r in records]
    conf_vals = [r["target_json"]["composition_summary"]["confidant_count"] for r in records]
    race_vals = [
        r["target_json"]["composition_summary"]["same_race_share"]
        for r in records
        if r["target_json"]["composition_summary"]["same_race_share"] is not None
    ]
    race_bucket_counts: Counter[str] = Counter(
        r["target_json"]["composition_summary"]["same_race_share_bucket"]
        for r in records
        if r["target_json"]["composition_summary"]["same_race_share_bucket"] is not None
    )
    priors = {
        "n_egos": len(records),
        "mean_num_alters": round(sum(n_vals) / len(n_vals), 2),
        "median_num_alters": float(pd.Series(n_vals).median()),
        "mean_kin_share": round(sum(kin_vals) / len(kin_vals), 3),
        "mean_friend_share": round(sum(friend_vals) / len(friend_vals), 3),
        "median_confidant_count": float(pd.Series(conf_vals).median()),
        "size_bucket_shares": {
            k: round(v / len(records), 3) for k, v in sorted(size_counts.items())
        },
        "mean_same_race_share": round(sum(race_vals) / len(race_vals), 3) if race_vals else None,
        "median_same_race_share": float(pd.Series(race_vals).median()) if race_vals else None,
        "same_race_share_bucket_shares": {
            k: round(v / max(1, sum(race_bucket_counts.values())), 3)
            for k, v in sorted(race_bucket_counts.items())
        },
    }

    write_jsonl(OUT_DIR / "ucnets_w1_network_gold.jsonl", records)

    # Flat CSV for quick inspection
    flat_rows = []
    for r in records:
        s = r["target_json"]["composition_summary"]
        flat_rows.append(
            {
                "respondent_id": r["respondent_id"],
                "num_alters": s["num_alters"],
                "size_bucket": s["size_bucket"],
                "kin_share": s["kin_share"],
                "friend_share": s["friend_share"],
                "coworker_share": s["coworker_share"],
                "confidant_count": s["confidant_count"],
                "companion_count": s["companion_count"],
                "pct_weekly_plus_contact": s["pct_weekly_plus_contact"],
                "same_race_share": s["same_race_share"],
                "same_race_share_bucket": s["same_race_share_bucket"],
                "core1_relationship": r["target_json"]["core_alters"][0]["relationship"]
                if r["target_json"]["core_alters"]
                else None,
            }
        )
    pd.DataFrame(flat_rows).to_csv(OUT_DIR / "ucnets_w1_network_summary.csv", index=False)

    with (OUT_DIR / "ucnets_w1_population_priors.json").open("w", encoding="utf-8") as f:
        json.dump(priors, f, indent=2)

    report = {
        "alters_path": str(ALTERS_PATH),
        "n_alters_rows": int(len(alters_df)),
        "n_alter_egos": len(alter_ids),
        "n_phase1_state_task1": len(state_ids),
        "n_overlap": len(overlap),
        "n_analysis": len(records),
        "only_state": only_state,
        "only_alters": only_alters,
        "skipped_no_alters": skipped_no_alters,
        "size_bucket_counts": dict(size_counts),
        "population_priors": priors,
    }
    with (OUT_DIR / "cleaning_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = OUT_DIR / "cleaning_report.md"
    md.write_text(
        "\n".join(
            [
                "# Phase 2 UCNets Network Cleaning Report",
                "",
                f"- Alter rows: {report['n_alters_rows']}",
                f"- Alter egos: {report['n_alter_egos']}",
                f"- Phase1 Task1 State egos: {report['n_phase1_state_task1']}",
                f"- Overlap: {report['n_overlap']}",
                f"- Analysis N (gold written): {report['n_analysis']}",
                f"- Only in State: {len(only_state)} → {only_state[:5]}",
                f"- Only in Alters: {len(only_alters)} → {only_alters[:5]}",
                f"- Size buckets: {dict(size_counts)}",
                f"- Mean num_alters: {priors['mean_num_alters']}",
                f"- Mean kin_share: {priors['mean_kin_share']}",
                "",
                "Outputs:",
                "- `ucnets_w1_network_gold.jsonl`",
                "- `ucnets_w1_network_summary.csv`",
                "- `ucnets_w1_population_priors.json`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("n_analysis", "n_overlap", "size_bucket_counts", "population_priors")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
