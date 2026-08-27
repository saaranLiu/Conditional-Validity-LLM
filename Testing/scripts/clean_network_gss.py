#!/usr/bin/env python3
"""Clean GSS network-module years into Phase 2 network-stage gold data.

Covers 1985, 1987, 2004, 2010, 2024 from Survey/gss/network/{year}/{ego,alter,alter_alter}.csv
and emits, per dataset id (gss_1985, gss_1987, gss_2004, gss_2010_network, gss_2024):

  {dataset}_network_gold.jsonl     ego State + full alter roster + alter-alter ties
  {dataset}_network_summary.csv    one flat row per ego for quick inspection
  {dataset}_population_priors.json marginals plus the bucket cutoffs used

Design notes that differ from the UCNets Phase 2 cleaner:

* GSS name generators cap at 5 alters (numgiven is topcoded at "6 or more"), so the
  roster is the *complete* nomination list, not a Top-5 subset. Size buckets are
  therefore redefined for the 0-5 range instead of reusing UCNets' 1-30 buckets.
* GSS has no support-role name interpreter, so confidant/companion counts have no
  GSS analogue and are omitted rather than approximated.
* 1985 and 2004 carry the full alter-alter closeness matrix, which yields the
  triadic-closure estimands (density, esp_close_density) that UCNets cannot support.

Continuous estimands get tertile buckets computed from the human distribution of the
same dataset, and the cutoffs are written into the priors file so downstream prompt
construction and evaluation use one definition.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import clean_state_ucnets as base  # noqa: E402
from network_cleaning_common import (  # noqa: E402
    assign_buckets,
    bucket_cutoff_block,
    bucket_share_block,
    code,
    mean_of,
    median_of,
    num,
    read_csv,
    round_or_none,
    share,
    tertile_cuts,
    write_flat_csv,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
NETWORK_ROOT = ROOT / "Survey" / "gss" / "network"
PHASE1_CLEAN = ROOT / "Survey" / "gss" / "cleaned"
OUT_DIR = ROOT / "Survey" / "gss" / "cleaned"

YEARS = (1985, 1987, 2004, 2010, 2024)

DATASET_ID = {
    1985: "gss_1985",
    1987: "gss_1987",
    2004: "gss_2004",
    2010: "gss_2010_network",
    2024: "gss_2024",
}

# Nomination flags are 1 = mentioned, 2 = not mentioned. An alter may carry several;
# collapse to one label with kin taking precedence, matching the UCNets ordering.
RELATION_PRIORITY: list[tuple[str, str]] = [
    ("spouse", "spouse"),
    ("child", "child"),
    ("parent", "parent"),
    ("sibling", "sibling"),
    ("othfam", "other_relative"),
    ("friend", "friend"),
    ("cowork", "coworker"),
    ("neighbr", "neighbor"),
    ("memgrp", "group_member"),
    ("advisor", "advisor"),
    ("other", "other"),
]

TALKTO_LABEL = {1: "almost_daily", 2: "weekly", 3: "monthly", 4: "less_than_monthly"}
KNOWN_LABEL = {1: "lt_3_years", 2: "3_to_6_years", 3: "6_years_plus"}
RCLOSE_LABEL = {1: "especially_close", 2: "not_especially_close"}
ALTER_SEX = {1: "male", 2: "female"}
ALTER_RACE = {1: "asian", 2: "black", 3: "hispanic", 4: "white", 5: "other"}
ALTER_EDUC = {
    0: "1-6 years",
    1: "7-9 years",
    2: "10-12 years",
    3: "hs_grad",
    4: "some_college",
    5: "associate",
    6: "bachelor",
    7: "grad_or_prof",
}
ALTER_RELIG = {1: "protestant", 2: "catholic", 3: "jewish", 4: "none", 5: "other"}
AA_WEIGHT_LABEL = {1: "especially_close", 2: "know_each_other", 3: "total_strangers"}
FRNDKNOW_LABEL = {
    1: "all_know_each_other",
    2: "most_know_each_other",
    3: "few_know_each_other",
    4: "none_know_each_other",
    5: "no_friends",
}
EQCLOSE_LABEL = {1: "yes", 2: "no"}

# Ego race is a 3-category item while alter race distinguishes asian/hispanic. Compare
# on the coarser ego scale so no ego is dropped; hispanic/asian alters of "other" egos
# are therefore treated as matches, which is a known limitation of the GSS coding.
EGO_RACE3 = {1: "white", 2: "black", 3: "other"}
ALTER_RACE_TO_EGO3 = {1: "other", 2: "black", 3: "other", 4: "white", 5: "other"}

AGE_HOMOPHILY_WINDOW = 5
COLLEGE_EGO_YEARS = 16  # ego educ is years of schooling
COLLEGE_ALTER_CODE = 6  # alter educ code 6 = bachelor

# GSS rosters top out at 5, so UCNets' small/medium/large cutoffs do not transfer.
SIZE_BUCKETS = [(2, "small"), (4, "medium")]  # else "large"

# Density-based estimands are only meaningful once a triad exists.
MIN_ALTERS_FOR_DENSITY = 3


def size_bucket(n: int) -> str:
    for upper, label in SIZE_BUCKETS:
        if n <= upper:
            return label
    return "large"


def relationship_of(row: dict[str, str]) -> str | None:
    for field, label in RELATION_PRIORITY:
        if code(row.get(field)) == 1:
            return label
    return None


def build_alter(row: dict[str, str]) -> dict[str, Any]:
    """Alter record; absent items are dropped rather than filled."""
    out: dict[str, Any] = {
        "alter_id": code(row.get("alter_id")),
        "relationship": relationship_of(row),
        "is_kin": {1: "yes", 2: "no"}.get(code(row.get("kin"))),
        "gender": ALTER_SEX.get(code(row.get("sex"))),
        "age": code(row.get("age")),
        "race": ALTER_RACE.get(code(row.get("race"))),
        "education": ALTER_EDUC.get(code(row.get("educ"))),
        "religion": ALTER_RELIG.get(code(row.get("relig"))),
        "contact_frequency": TALKTO_LABEL.get(code(row.get("talkto"))),
        "known_duration": KNOWN_LABEL.get(code(row.get("known"))),
        "closeness_to_ego": RCLOSE_LABEL.get(code(row.get("rclose"))),
    }
    return {k: v for k, v in out.items() if v is not None}


def composition_of(
    ego_row: dict[str, str],
    alters: list[dict[str, str]],
    aa_rows: list[dict[str, str]],
) -> dict[str, Any]:
    n = len(alters)
    ego_race = EGO_RACE3.get(code(ego_row.get("race")))
    ego_sex = code(ego_row.get("sex"))
    ego_age = num(ego_row.get("age"))
    ego_educ = num(ego_row.get("educ"))

    def flag_share(field: str) -> float | None:
        return share(sum(1 for a in alters if code(a.get(field)) == 1), n)

    # Each homophily share is computed over the alters that actually report the item,
    # and the coverage is recorded so partial reporting stays visible downstream.
    race_vals = [ALTER_RACE_TO_EGO3.get(code(a.get("race"))) for a in alters]
    race_vals = [v for v in race_vals if v is not None]
    same_race = (
        share(sum(1 for v in race_vals if v == ego_race), len(race_vals))
        if ego_race and race_vals
        else None
    )

    sex_vals = [code(a.get("sex")) for a in alters]
    sex_vals = [v for v in sex_vals if v in (1, 2)]
    same_sex = (
        share(sum(1 for v in sex_vals if v == ego_sex), len(sex_vals))
        if ego_sex in (1, 2) and sex_vals
        else None
    )

    age_vals = [num(a.get("age")) for a in alters]
    age_vals = [v for v in age_vals if v is not None]
    age_hom = (
        share(sum(1 for v in age_vals if abs(v - ego_age) <= AGE_HOMOPHILY_WINDOW), len(age_vals))
        if ego_age is not None and age_vals
        else None
    )

    educ_vals = [code(a.get("educ")) for a in alters]
    educ_vals = [v for v in educ_vals if v is not None]
    educ_hom = None
    if ego_educ is not None and educ_vals:
        ego_college = ego_educ >= COLLEGE_EGO_YEARS
        educ_hom = share(
            sum(1 for v in educ_vals if (v >= COLLEGE_ALTER_CODE) == ego_college), len(educ_vals)
        )

    talk_vals = [code(a.get("talkto")) for a in alters]
    talk_vals = [v for v in talk_vals if v is not None]
    weekly_plus = share(sum(1 for v in talk_vals if v in (1, 2)), len(talk_vals)) if talk_vals else None

    close_vals = [code(a.get("rclose")) for a in alters]
    close_vals = [v for v in close_vals if v in (1, 2)]
    esp_close_share = share(sum(1 for v in close_vals if v == 1), len(close_vals)) if close_vals else None

    comp: dict[str, Any] = {
        "num_alters": n,
        "size_bucket": size_bucket(n),
        "numgiven": code(ego_row.get("numgiven")),
        "numgiven_is_topcoded": code(ego_row.get("numgiven_is_topcoded")) == 1,
        "kin_share": round_or_none(flag_share("kin")),
        "friend_share": round_or_none(flag_share("friend")),
        "coworker_share": round_or_none(flag_share("cowork")),
        "neighbor_share": round_or_none(flag_share("neighbr")),
        "pct_weekly_plus_contact": round_or_none(weekly_plus),
        "pct_weekly_plus_contact_coverage": round_or_none(share(len(talk_vals), n)),
        "same_race_share": round_or_none(same_race),
        "same_race_share_coverage": round_or_none(share(len(race_vals), n)),
        "same_sex_share": round_or_none(same_sex),
        "same_sex_share_coverage": round_or_none(share(len(sex_vals), n)),
        "age_homophily_share": round_or_none(age_hom),
        "age_homophily_coverage": round_or_none(share(len(age_vals), n)),
        "educ_homophily_share": round_or_none(educ_hom),
        "esp_close_share": round_or_none(esp_close_share),
    }

    # Alter-alter closeness. weight 7 is "refused", so it leaves the denominator too.
    valid = [(r, code(r.get("weight"))) for r in aa_rows]
    valid = [(r, w) for r, w in valid if w in (1, 2, 3)]
    possible = n * (n - 1) // 2
    if valid and n >= 2:
        denom = len(valid)
        comp["density"] = round_or_none(share(sum(1 for _, w in valid if w in (1, 2)), denom))
        comp["esp_close_density"] = round_or_none(share(sum(1 for _, w in valid if w == 1), denom))
        comp["dyad_coverage"] = round_or_none(share(denom, possible))
        comp["n_dyads_observed"] = denom
    else:
        comp["density"] = None
        comp["esp_close_density"] = None
        comp["dyad_coverage"] = None
        comp["n_dyads_observed"] = 0

    ego_reported = {
        "friends_know_each_other": FRNDKNOW_LABEL.get(code(ego_row.get("frndknow"))),
        "feels_equally_close_to_all": EQCLOSE_LABEL.get(code(ego_row.get("eqclose"))),
    }
    ego_reported = {k: v for k, v in ego_reported.items() if v is not None}
    if ego_reported:
        comp["ego_reported"] = ego_reported

    return comp


# Continuous estimands that receive tertile buckets, and whether they are restricted to
# egos large enough for the quantity to be meaningful.
BUCKETED = [
    ("density", True),
    ("esp_close_density", True),
    ("same_race_share", False),
    ("same_sex_share", False),
    ("age_homophily_share", False),
    ("educ_homophily_share", False),
    ("pct_weekly_plus_contact", False),
]


def bucket_spec() -> list[tuple[str, Any]]:
    """`BUCKETED` in the (field, eligibility predicate) form the shared bucketer wants."""
    big_enough = lambda comp: comp["num_alters"] >= MIN_ALTERS_FOR_DENSITY  # noqa: E731
    return [(field, big_enough if needs_triad else None) for field, needs_triad in BUCKETED]


def clean_year(year: int) -> dict[str, Any]:
    dataset = DATASET_ID[year]
    year_dir = NETWORK_ROOT / str(year)
    ego_rows = read_csv(year_dir / "ego.csv")
    alter_rows = read_csv(year_dir / "alter.csv")
    aa_path = year_dir / "alter_alter.csv"
    aa_rows = read_csv(aa_path) if aa_path.exists() else []

    alters_by_ego: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in alter_rows:
        alters_by_ego[row["ego_id"]].append(row)
    for rows in alters_by_ego.values():
        rows.sort(key=lambda r: code(r.get("alter_id")) or 0)

    aa_by_ego: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in aa_rows:
        aa_by_ego[row["ego_id"]].append(row)

    # Reuse the Phase 1 cleaned State so both stages condition on identical ego records.
    state_path = PHASE1_CLEAN / f"{dataset}_state_clean.csv"
    if not state_path.exists():
        raise FileNotFoundError(
            f"missing {state_path}; run Testing/scripts/clean_state_gss.py first"
        )
    state_by_id = {str(r.get("respondent_id")): r for r in read_csv(state_path)}

    records: list[dict[str, Any]] = []
    dropped = Counter()

    for ego_row in ego_rows:
        ego_id = str(ego_row.get("ego_id") or "").strip()
        rid = str(ego_row.get("id") or "").strip()
        alters = alters_by_ego.get(ego_id, [])
        if not alters:
            dropped["no_alters"] += 1
            continue
        state_row = state_by_id.get(rid)
        if state_row is None:
            dropped["no_phase1_state"] += 1
            continue

        state_json = dict(base.gss_prompt(state_row, include_health=False))
        state_json.update(base.gss_target(state_row, include_health=True))
        if not state_json.get("age") or not state_json.get("sex"):
            dropped["incomplete_state"] += 1
            continue

        comp = composition_of(ego_row, alters, aa_by_ego.get(ego_id, []))
        alter_list = [build_alter(a) for a in alters]
        if any(a.get("relationship") is None for a in alter_list):
            dropped["alter_without_relationship"] += 1

        target: dict[str, Any] = {"composition_summary": comp, "alters": alter_list}
        if aa_by_ego.get(ego_id):
            target["alter_alter"] = [
                {
                    "src_id": code(r.get("src_id")),
                    "tgt_id": code(r.get("tgt_id")),
                    "tie_strength": AA_WEIGHT_LABEL.get(code(r.get("weight"))),
                }
                for r in aa_by_ego[ego_id]
                if AA_WEIGHT_LABEL.get(code(r.get("weight"))) is not None
            ]

        records.append(
            {
                "respondent_id": rid,
                "dataset": dataset,
                "ego_id": ego_id,
                "state_json": state_json,
                "target_json": target,
                "metadata": {
                    "source_year": year,
                    "source_path": str(year_dir.relative_to(ROOT)),
                    "weight": state_row.get("wtssall") or None,
                    "has_alter_alter": bool(aa_by_ego.get(ego_id)),
                },
            }
        )

    # Second pass: bucket cutoffs from this dataset's own human distribution.
    cuts, bucket_counts = assign_buckets(records, bucket_spec())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold_path = OUT_DIR / f"{dataset}_network_gold.jsonl"
    write_jsonl(gold_path, records)

    summary_rows = []
    for rec in records:
        comp = rec["target_json"]["composition_summary"]
        flat = {"respondent_id": rec["respondent_id"], "ego_id": rec["ego_id"]}
        flat.update({k: v for k, v in comp.items() if not isinstance(v, dict)})
        ego_reported = comp.get("ego_reported") or {}
        flat.update({f"ego_reported_{k}": v for k, v in ego_reported.items()})
        summary_rows.append(flat)
    summary_path = OUT_DIR / f"{dataset}_network_summary.csv"
    write_flat_csv(summary_path, summary_rows)

    priors = build_priors(dataset, year, records, cuts, bucket_counts)
    priors_path = OUT_DIR / f"{dataset}_population_priors.json"
    priors_path.write_text(json.dumps(priors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "dataset": dataset,
        "year": year,
        "n_ego_rows": len(ego_rows),
        "n_gold": len(records),
        "dropped": dict(dropped),
        "has_alter_alter": bool(aa_rows),
        "bucket_shares": {
            field: {
                str(k): round(v / max(1, sum(counts.values())), 3) for k, v in counts.most_common()
            }
            for field, counts in bucket_counts.items()
        },
        "cuts": {k: (list(v) if v else None) for k, v in cuts.items()},
        "priors": priors,
    }


def build_priors(
    dataset: str,
    year: int,
    records: list[dict[str, Any]],
    cuts: dict[str, tuple[float, float] | None],
    bucket_counts: dict[str, Counter],
) -> dict[str, Any]:
    comps = [rec["target_json"]["composition_summary"] for rec in records]
    n = len(comps)

    def col(field: str) -> list[float]:
        return [c.get(field) for c in comps if c.get(field) is not None]

    priors: dict[str, Any] = {
        "dataset": dataset,
        "source_year": year,
        "n_egos": n,
        "mean_num_alters": mean_of([c["num_alters"] for c in comps]),
        "median_num_alters": median_of([c["num_alters"] for c in comps]),
        "size_bucket_shares": {},
        "min_alters_for_density": MIN_ALTERS_FOR_DENSITY,
        "bucket_cutoffs": {},
        "bucket_shares": {},
    }
    sb = Counter(c["size_bucket"] for c in comps)
    priors["size_bucket_shares"] = {k: round(v / max(1, n), 3) for k, v in sb.most_common()}

    for field in ("kin_share", "friend_share", "coworker_share", "neighbor_share",
                  "pct_weekly_plus_contact", "same_race_share", "same_sex_share",
                  "age_homophily_share", "educ_homophily_share", "esp_close_share",
                  "density", "esp_close_density"):
        vals = col(field)
        if vals:
            priors[f"mean_{field}"] = mean_of(vals)
            priors[f"median_{field}"] = median_of(vals)
            priors[f"n_{field}"] = len(vals)

    for field, _ in BUCKETED:
        cutoff_block = bucket_cutoff_block(cuts.get(field))
        if cutoff_block:
            priors["bucket_cutoffs"][field] = cutoff_block
        share_block = bucket_share_block(bucket_counts[field])
        if share_block:
            priors["bucket_shares"][field] = share_block
    return priors


def write_report(stats: list[dict[str, Any]]) -> None:
    lines = [
        "# GSS Network-Stage Cleaning Report",
        "",
        "Generated by `Testing/scripts/clean_network_gss.py`.",
        "",
        "## Sample",
        "",
        "| dataset | year | ego rows | gold egos | alter-alter | 主要丢弃原因 |",
        "|---|---:|---:|---:|:-:|---|",
    ]
    for s in stats:
        drops = ", ".join(f"{k}={v}" for k, v in sorted(s["dropped"].items())) or "-"
        lines.append(
            f"| `{s['dataset']}` | {s['year']} | {s['n_ego_rows']} | {s['n_gold']} | "
            f"{'yes' if s['has_alter_alter'] else 'no'} | {drops} |"
        )

    lines += [
        "",
        "## 分箱分布（Testing 3 靶量非退化性）",
        "",
        "每个连续估计量按本数据集自身的经验三分位切分；越接近 33/33/33 越适合做显式区间跟随测试。",
        "`density` 与 `esp_close_density` 仅在 alter 数 >= "
        f"{MIN_ALTERS_FOR_DENSITY} 的 ego 上计算。",
        "",
    ]
    for s in stats:
        lines += [f"### `{s['dataset']}`", ""]
        rows = []
        for field, _ in BUCKETED:
            shares = s["priors"]["bucket_shares"].get(field)
            if not shares:
                continue
            labelled = shares.get("n_labelled", 0)
            cut = s["cuts"].get(field)
            if cut:
                spread = (
                    f"{shares.get('low', 0):.0%} / {shares.get('medium', 0):.0%} / "
                    f"{shares.get('high', 0):.0%}"
                )
                rows.append(f"| `{field}` | {labelled} | {spread} | {cut[0]:.3f} / {cut[1]:.3f} |")
            else:
                rows.append(f"| `{field}` | {labelled} | - | 分位退化，未分箱 |")
        if rows:
            lines += ["| 估计量 | n | low / med / high | 切点 |", "|---|---:|---|---|"] + rows
        else:
            lines.append("（该年 alter 表无人口学字段，无可分箱估计量）")
        rejected = [
            field
            for field, _ in BUCKETED
            if s["cuts"].get(field) is None and s["priors"].get(f"n_{field}")
        ]
        if rejected:
            lines += [
                "",
                "分位退化、不可用作区间跟随靶量：" + "、".join(f"`{f}`" for f in rejected) + "。",
            ]
        lines.append("")

    path = OUT_DIR / "gss_network_cleaning_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    stats = []
    for year in YEARS:
        s = clean_year(year)
        stats.append(s)
        print(
            f"{s['dataset']}: gold={s['n_gold']}/{s['n_ego_rows']} "
            f"alter_alter={'yes' if s['has_alter_alter'] else 'no'} dropped={s['dropped']}"
        )
        for field, _ in BUCKETED:
            shares = s["priors"]["bucket_shares"].get(field)
            if shares:
                print(
                    f"    {field:24s} n={shares['n_labelled']:4d} "
                    f"low={shares.get('low', 0):.0%} med={shares.get('medium', 0):.0%} "
                    f"high={shares.get('high', 0):.0%}"
                )
    write_report(stats)
    (OUT_DIR / "gss_network_clean_summary.json").write_text(
        json.dumps(
            [{k: v for k, v in s.items() if k != "priors"} for s in stats], indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
