#!/usr/bin/env python3
"""Clean NSHAP rounds 1-3 into Phase 2 network-stage gold data.

Dataset ids: `nshap_r1`, `nshap_r2`, `nshap_r3`. Emits the standard trio per round under
Survey/ucnets/cleaned/, plus `nshap_cleaning_report.md`.

SUPPLEMENTARY dataset with two jobs:
  1. Large-N closure replication. NSHAP records the full alter-alter contact matrix, so
     density is observed rather than inferred, on 2900-4300 respondents per round --
     an order of magnitude more than any GSS network module.
  2. Population contrast. The sample is adults aged 57+, so the human network is
     smaller and more kin-heavy than UCNets. If a generator returns the same modal
     network here as for UCNets, that is predictor behaviour showing up as insensitivity
     to the population it was asked about.

The UCLA three-item loneliness scale (COMPANION / LEFTOUT / ISOLATED) is carried into
the ego state because it is the same instrument UCNets uses, which makes loneliness the
one conditioning variable comparable across the primary and this supplementary survey.

Coding hazards handled here:
  * The alter-alter matrix is stored as TALKFREQ1..TALKFREQ5 on each alter row, with the
    diagonal blank. TALKFREQ0 is the ego-alter frequency and is a different quantity.
    TALKFREQ6 and TALKFREQ7 are populated even for one-alter rosters, so they cannot be
    the 6th/7th alter columns; both are excluded rather than guessed at.
  * `HEALTH` in round 1 is a better/worse comparison item, NOT self-rated health.
    `PHYSHLTH` (poor..excellent) is the self-rating and is the only one present in all
    three rounds.
  * The relationship variable is `RELAT` in round 1 and `RELAT2` afterwards; the
    loneliness items gain a `2` suffix from round 2 on.
  * Round 3 dropped `HOWCLOSE`, so closeness-based estimands are absent there by design,
    not by cleaning loss.
  * SECTION 1 is the confidant roster (cap 5). Sections 2-4 are separate rosters and are
    excluded; including them would silently inflate network size.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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
    size_bucket,
    write_flat_csv,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "Survey" / "nshap" / "processed" / "by_round"
OUT_DIR = ROOT / "Survey" / "nshap" / "cleaned"

CONFIDANT_SECTION = "1"
MAX_ALTERS = 5
TIE_COLUMNS = [f"TALKFREQ{i}" for i in range(1, MAX_ALTERS + 1)]

ROUNDS: dict[str, dict[str, Any]] = {
    "nshap_r1": {
        "dir": "Round1",
        "year": 2005,
        "relat_col": "RELAT",
        "lonely_cols": ("COMPANION", "LEFTOUT", "ISOLATED"),
        "has_howclose": True,
        "has_roster_w1": False,
    },
    "nshap_r2": {
        "dir": "Round2",
        "year": 2010,
        "relat_col": "RELAT2",
        "lonely_cols": ("COMPANION2", "LEFTOUT2", "ISOLATED2"),
        "has_howclose": True,
        "has_roster_w1": True,
    },
    "nshap_r3": {
        "dir": "Round3",
        "year": 2015,
        "relat_col": "RELAT2",
        "lonely_cols": ("COMPANION2", "LEFTOUT2", "ISOLATED2"),
        "has_howclose": False,
        "has_roster_w1": True,
    },
}

GENDER = {1: "male", 2: "female"}
RACE = {1: "white", 2: "black", 3: "hispanic_non_black", 4: "other"}
EDUCATION = {
    1: "less_than_high_school",
    2: "high_school_or_equivalent",
    3: "some_college_or_associate",
    4: "bachelors_or_more",
}
MARITAL = {
    1: "married",
    2: "living_with_partner",
    3: "separated",
    4: "divorced",
    5: "widowed",
    6: "never_married",
}
SELF_RATED_HEALTH = {1: "poor", 2: "fair", 3: "good", 4: "very_good", 5: "excellent"}
LONELY_ITEM = {1: "hardly_ever", 2: "some_of_the_time", 3: "often"}

RELATIONSHIP = {
    1: "spouse",
    2: "ex_spouse",
    3: "romantic_partner",
    4: "parent",
    5: "parent_in_law",
    6: "child",
    7: "step_child",
    8: "sibling",
    9: "other_relative",
    10: "other_in_law",
    11: "friend",
    12: "neighbor",
    13: "coworker_or_boss",
    14: "clergy",
    15: "therapist",
    16: "social_worker",
    17: "home_health_provider",
    18: "other",
}
KIN_RELATIONS = {
    "spouse",
    "romantic_partner",
    "parent",
    "parent_in_law",
    "child",
    "step_child",
    "sibling",
    "other_relative",
    "other_in_law",
}
PROFESSIONAL_RELATIONS = {"clergy", "therapist", "social_worker", "home_health_provider"}

CONTACT_FREQ = {
    0: "never_spoken",
    1: "less_than_once_a_year",
    2: "once_a_year",
    3: "a_couple_times_a_year",
    4: "once_a_month",
    5: "once_every_two_weeks",
    6: "once_a_week",
    7: "several_times_a_week",
    8: "every_day",
}
FREQ_CODE_OF = {label: c for c, label in CONTACT_FREQ.items()}
CLOSENESS = {1: "not_very_close", 2: "somewhat_close", 3: "very_close", 4: "extremely_close"}
LIVEWITH = {0: "no", 1: "yes", 2: "part_of_the_year"}
LONGKNOW = {1: "less_than_1_year", 2: "1_to_5_years", 3: "6_to_10_years", 4: "more_than_10_years"}

WEEKLY_OR_MORE = 6  # once a week and up on the 0-8 contact scale
SIZE_CUTOFFS = [(2, "small"), (4, "medium")]


def build_state(ego: dict[str, str], spec: dict[str, Any]) -> dict[str, Any]:
    lonely_items = [code(ego.get(c)) for c in spec["lonely_cols"]]
    lonely_present = [v for v in lonely_items if v in (1, 2, 3)]
    state = {
        "age": code(ego.get("AGE")),
        "gender": GENDER.get(code(ego.get("GENDER"))),
        "race": RACE.get(code(ego.get("ETHGRP"))),
        "education": EDUCATION.get(code(ego.get("EDUC"))),
        "marital_status": MARITAL.get(code(ego.get("MARITLST"))),
        "self_rated_health": SELF_RATED_HEALTH.get(code(ego.get("PHYSHLTH"))),
        "lacks_companionship": LONELY_ITEM.get(lonely_items[0]),
        "feels_left_out": LONELY_ITEM.get(lonely_items[1]),
        "feels_isolated": LONELY_ITEM.get(lonely_items[2]),
        # 3-9; only meaningful when all three items are answered
        "loneliness_scale": sum(lonely_present) if len(lonely_present) == 3 else None,
    }
    return {k: v for k, v in state.items() if v is not None}


def build_alter(row: dict[str, str], spec: dict[str, Any], ego_gender: int | None) -> dict[str, Any]:
    alter_gender = code(row.get("NODE_GENDER"))
    out: dict[str, Any] = {
        "alter_id": code(row.get("LINENO")),
        "relationship": RELATIONSHIP.get(code(row.get(spec["relat_col"]))),
        "gender": GENDER.get(alter_gender),
        "age": code(row.get("NODE_AGE")),
        "lives_with_ego": LIVEWITH.get(code(row.get("LIVEWITH"))),
        "contact_frequency": CONTACT_FREQ.get(code(row.get("TALKFREQ0"))),
        "same_gender_as_ego": (
            None
            if alter_gender not in (1, 2) or ego_gender not in (1, 2)
            else ("yes" if alter_gender == ego_gender else "no")
        ),
    }
    if spec["has_howclose"]:
        out["closeness_to_ego"] = CLOSENESS.get(code(row.get("HOWCLOSE")))
    if row.get("LONGKNOW"):
        out["known_for"] = LONGKNOW.get(code(row.get("LONGKNOW")))
    if spec["has_roster_w1"]:
        carried = code(row.get("ROSTER_W1"))
        if carried is not None:
            out["carried_from_round1"] = "yes" if carried else "no"
    return {k: v for k, v in out.items() if v is not None}


def extract_ties(roster: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Read the symmetric alter-alter matrix off the per-alter TALKFREQ1..5 columns.

    Both directions of a pair are recorded; where they disagree (rare) the lower line
    number wins, so the result does not depend on row order.
    """
    line_numbers = {code(r.get("LINENO")) for r in roster}
    values: dict[tuple[int, int], int] = {}
    for row in roster:
        src = code(row.get("LINENO"))
        if src is None:
            continue
        for col_index, col in enumerate(TIE_COLUMNS, start=1):
            if col_index == src or col_index not in line_numbers:
                continue
            value = code(row.get(col))
            if value is None or not 0 <= value <= 8:
                continue
            key = (min(src, col_index), max(src, col_index))
            if key not in values or src < col_index:
                values[key] = value
    return [
        {"src_id": a, "tgt_id": b, "contact_frequency": CONTACT_FREQ.get(v)}
        for (a, b), v in sorted(values.items())
        if CONTACT_FREQ.get(v) is not None
    ]


def composition_of(
    roster: list[dict[str, str]],
    ties: list[dict[str, Any]],
    tie_codes: list[int],
    spec: dict[str, Any],
    ego: dict[str, str],
) -> dict[str, Any]:
    n = len(roster)
    ego_gender = code(ego.get("GENDER"))
    ego_age = code(ego.get("AGE"))

    rels = [RELATIONSHIP.get(code(r.get(spec["relat_col"]))) for r in roster]
    rels_present = [r for r in rels if r is not None]

    def rel_share(names: set[str] | str) -> float | None:
        wanted = {names} if isinstance(names, str) else names
        return share(sum(1 for r in rels_present if r in wanted), len(rels_present))

    genders = [code(r.get("NODE_GENDER")) for r in roster]
    genders = [g for g in genders if g in (1, 2)]
    same_sex = (
        share(sum(1 for g in genders if g == ego_gender), len(genders))
        if ego_gender in (1, 2) and genders
        else None
    )

    freqs = [code(r.get("TALKFREQ0")) for r in roster]
    freqs = [f for f in freqs if f is not None and 0 <= f <= 8]

    coresident = [code(r.get("LIVEWITH")) for r in roster]
    coresident = [c for c in coresident if c is not None]

    alter_ages = [code(r.get("NODE_AGE")) for r in roster]
    alter_ages = [a for a in alter_ages if a is not None and 0 < a < 120]

    comp: dict[str, Any] = {
        "num_alters": n,
        "size_bucket": size_bucket(n, SIZE_CUTOFFS),
        "kin_share": round_or_none(rel_share(KIN_RELATIONS)),
        "friend_share": round_or_none(rel_share("friend")),
        "spouse_present": (
            None if not rels_present else ("yes" if "spouse" in rels_present else "no")
        ),
        "professional_share": round_or_none(rel_share(PROFESSIONAL_RELATIONS)),
        "relationship_coverage": round_or_none(share(len(rels_present), n)),
        "same_sex_share": round_or_none(same_sex),
        "weekly_contact_share": round_or_none(
            share(sum(1 for f in freqs if f >= WEEKLY_OR_MORE), len(freqs))
        ),
        "mean_contact_code": mean_of(freqs),
        "coresident_share": round_or_none(share(sum(1 for c in coresident if c == 1), len(coresident))),
        "alter_age_coverage": round_or_none(share(len(alter_ages), n)),
        "mean_alter_age": mean_of(alter_ages),
        "age_homophily_share": round_or_none(
            share(sum(1 for a in alter_ages if abs(a - ego_age) <= 5), len(alter_ages))
            if ego_age is not None and alter_ages
            else None
        ),
    }

    if spec["has_howclose"]:
        closeness = [code(r.get("HOWCLOSE")) for r in roster]
        closeness = [c for c in closeness if c is not None and 1 <= c <= 4]
        comp["very_close_share"] = round_or_none(
            share(sum(1 for c in closeness if c >= 3), len(closeness))
        )
        comp["mean_closeness_code"] = mean_of(closeness)

    if spec["has_roster_w1"]:
        carried = [code(r.get("ROSTER_W1")) for r in roster]
        carried = [c for c in carried if c is not None]
        comp["carried_from_round1_share"] = round_or_none(
            share(sum(1 for c in carried if c == 1), len(carried))
        )

    possible = n * (n - 1) // 2
    if tie_codes:
        denom = len(tie_codes)
        comp["density"] = round_or_none(share(sum(1 for v in tie_codes if v >= 1), denom))
        comp["density_weekly"] = round_or_none(
            share(sum(1 for v in tie_codes if v >= WEEKLY_OR_MORE), denom)
        )
        comp["dyad_coverage"] = round_or_none(share(denom, possible)) if possible else None
        comp["n_dyads_observed"] = denom
    else:
        comp["density"] = None
        comp["density_weekly"] = None
        comp["dyad_coverage"] = None if possible else 1.0
        comp["n_dyads_observed"] = 0
    return comp


def bucketed_fields(spec: dict[str, Any]) -> list[tuple[str, Any]]:
    """Density needs at least two alters before a dyad exists, so it gates on size."""
    has_dyads = lambda comp: (comp.get("n_dyads_observed") or 0) > 0  # noqa: E731
    fields: list[tuple[str, Any]] = [
        ("density", has_dyads),
        ("density_weekly", has_dyads),
        ("kin_share", None),
        ("same_sex_share", None),
        ("weekly_contact_share", None),
        ("age_homophily_share", None),
    ]
    if spec["has_howclose"]:
        fields.append(("very_close_share", None))
    if spec["has_roster_w1"]:
        fields.append(("carried_from_round1_share", None))
    return fields


def build_priors(
    dataset: str,
    spec: dict[str, Any],
    records: list[dict[str, Any]],
    cuts: dict[str, tuple[float, float] | None],
    counts: dict[str, Counter],
) -> dict[str, Any]:
    comps = [r["target_json"]["composition_summary"] for r in records]
    sizes = [c["num_alters"] for c in comps]
    priors: dict[str, Any] = {
        "dataset": dataset,
        "round": spec["dir"],
        "field_year": spec["year"],
        "n_egos": len(comps),
        "population": "US adults aged 57 and older",
        "roster_design": f"confidant roster, variable size 1-{MAX_ALTERS}, full alter-alter matrix",
        "mean_num_alters": mean_of(sizes),
        "median_num_alters": median_of(sizes),
        "size_distribution": {
            str(k): round(v / len(sizes), 4) for k, v in sorted(Counter(sizes).items())
        },
        "size_bucket_shares": {
            k: round(v / len(comps), 4)
            for k, v in Counter(c["size_bucket"] for c in comps).most_common()
        },
        "bucket_cutoffs": {},
        "bucket_shares": {},
    }
    numeric_fields = [
        "kin_share",
        "friend_share",
        "professional_share",
        "same_sex_share",
        "weekly_contact_share",
        "coresident_share",
        "age_homophily_share",
        "density",
        "density_weekly",
    ]
    if spec["has_howclose"]:
        numeric_fields.append("very_close_share")
    if spec["has_roster_w1"]:
        numeric_fields.append("carried_from_round1_share")
    for field in numeric_fields:
        vals = [c.get(field) for c in comps if c.get(field) is not None]
        if vals:
            priors[f"mean_{field}"] = mean_of(vals)
            priors[f"median_{field}"] = median_of(vals)
            priors[f"n_{field}"] = len(vals)

    for field, _ in bucketed_fields(spec):
        block = bucket_cutoff_block(cuts.get(field))
        if block:
            priors["bucket_cutoffs"][field] = block
        share_block = bucket_share_block(counts[field])
        if share_block:
            priors["bucket_shares"][field] = share_block
    return priors


def clean_round(dataset: str, spec: dict[str, Any]) -> dict[str, Any]:
    round_dir = SRC / spec["dir"]
    egos = {row["ID"]: row for row in read_csv(round_dir / "core.csv")}
    net_rows = read_csv(round_dir / "social_networks.csv")

    rosters: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in net_rows:
        if row.get("SECTION") != CONFIDANT_SECTION:
            continue
        line = code(row.get("LINENO"))
        if line is None or not 1 <= line <= MAX_ALTERS:
            continue
        rosters[row["ID"]].append(row)
    for roster in rosters.values():
        roster.sort(key=lambda r: code(r.get("LINENO")) or 0)

    records: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    for ego_id, roster in rosters.items():
        ego = egos.get(ego_id)
        if ego is None:
            dropped["no_core_record"] += 1
            continue
        state = build_state(ego, spec)
        if state.get("age") is None or state.get("gender") is None:
            dropped["incomplete_state"] += 1
            continue

        ties = extract_ties(roster)
        tie_codes = [FREQ_CODE_OF[t["contact_frequency"]] for t in ties]

        comp = composition_of(roster, ties, tie_codes, spec, ego)
        ego_gender = code(ego.get("GENDER"))
        records.append(
            {
                "respondent_id": f"{dataset}_{ego_id}",
                "dataset": dataset,
                "ego_id": ego_id,
                "state_json": state,
                "target_json": {
                    "composition_summary": comp,
                    "alters": [build_alter(r, spec, ego_gender) for r in roster],
                    "alter_ties": ties,
                },
                "metadata": {
                    "round": spec["dir"],
                    "field_year": spec["year"],
                    "weight": num(ego.get("WEIGHT")),
                    "source_path": str(round_dir.relative_to(ROOT)),
                },
            }
        )

    records.sort(key=lambda r: r["ego_id"])
    cuts, counts = assign_buckets(records, bucketed_fields(spec))

    write_jsonl(OUT_DIR / f"{dataset}_network_gold.jsonl", records)

    summary_rows = []
    for rec in records:
        comp = rec["target_json"]["composition_summary"]
        flat = {"respondent_id": rec["respondent_id"], "ego_id": rec["ego_id"]}
        flat.update({k: v for k, v in comp.items() if not isinstance(v, dict)})
        summary_rows.append(flat)
    write_flat_csv(OUT_DIR / f"{dataset}_network_summary.csv", summary_rows)

    priors = build_priors(dataset, spec, records, cuts, counts)
    (OUT_DIR / f"{dataset}_population_priors.json").write_text(
        json.dumps(priors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return {
        "dataset": dataset,
        "round": spec["dir"],
        "n_rosters": len(rosters),
        "n_gold": len(records),
        "dropped": dict(dropped),
        "cuts": cuts,
        "counts": counts,
        "priors": priors,
    }


def write_report(stats: list[dict[str, Any]]) -> None:
    lines = [
        "# NSHAP 网络阶段清洗报告",
        "",
        "由 `Testing/scripts/clean_network_nshap.py` 生成。",
        "",
        "NSHAP 在本项目中是**补充数据**，承担两件事：一是用每轮数千个 ego 做闭合度的",
        "大样本复核（alter–alter 联系矩阵是实测的，不是推断的）；二是提供 57 岁以上人群",
        "这个与 UCNets 不同的总体，用来检验生成器是否会随被问的人群改变输出。",
        "",
        "## 各轮清洗结果",
        "",
        "| 数据集 | 轮次 | 名单数 | 入库 ego | 丢弃 | 均值名单规模 | 均值密度（任意联系） | 均值密度（每周以上） |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for st in stats:
        p = st["priors"]
        lines.append(
            f"| `{st['dataset']}` | {st['round']} | {st['n_rosters']} | {st['n_gold']} | "
            f"{st['dropped'] or '无'} | {p.get('mean_num_alters')} | {p.get('mean_density')} | "
            f"{p.get('mean_density_weekly')} |"
        )
    lines += [
        "",
        "注意上表两个密度列的差别，它决定了 Testing 3 该用哪一个：**「任意联系」密度贴着",
        "天花板**（均值 0.80–0.85），因为 5 个密友之间几乎都彼此说过话；收紧到「每周以上",
        "联系」才铺得开。前者已被判据拒绝分箱，后者是本数据集实际可用的闭合度靶量。",
        "",
        "## 三分箱靶量（Testing 3 用）",
        "",
        "分箱切点取自各轮自己的人类分布。两种情况会拒绝分箱，下表都标出：分位点相等",
        "（分布过度并列），或任一档占比低于 15%（天花板/地板效应，模型完全错过那一档也",
        "几乎不受惩罚，区间跟随检验失去意义）。拒绝本身是结论，说明该估计量在这一人群里",
        "没有可用的区分度。",
        "",
    ]
    for st in stats:
        lines.append(f"### `{st['dataset']}`（{st['round']}）")
        lines.append("")
        lines.append("| 靶量 | 切点 | low | medium | high | 有标签 ego |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for field, _ in bucketed_fields(ROUNDS[st["dataset"]]):
            cut = st["cuts"].get(field)
            block = st["priors"]["bucket_shares"].get(field)
            if cut and block:
                lines.append(
                    f"| `{field}` | {cut[0]:.3f} / {cut[1]:.3f} | "
                    f"{block.get('low', 0):.1%} | {block.get('medium', 0):.1%} | "
                    f"{block.get('high', 0):.1%} | {block['n_labelled']} |"
                )
            else:
                lines.append(f"| `{field}` | 分位退化，未分箱 | - | - | - | - |")
        lines.append("")

    lines += [
        "## 逐轮口径差异（不是清洗损失）",
        "",
        "- 第 3 轮取消了 `HOWCLOSE`，所以该轮没有亲密度类靶量。",
        "- 关系变量第 1 轮叫 `RELAT`，之后叫 `RELAT2`；孤独三项从第 2 轮起加 `2` 后缀。",
        "- `ROSTER_W1` 只在第 2、3 轮存在，用于记录 alter 是否从第 1 轮延续下来。",
        "  当前论文范围不含 response/dynamics 阶段，这一列先存不用。",
        "- `NODE_AGE` 填充率仅约两成，因此 `age_homophily_share` 的覆盖率一并写入产物，",
        "  不要当成全样本量看。",
        "",
        "## 两个容易踩的坑",
        "",
        "- `HEALTH`（第 1 轮）是「比以前好/差」的比较题，不是自评健康。自评健康用",
        "  `PHYSHLTH`（poor…excellent），三轮齐备，本脚本用的是后者。",
        "- alter–alter 矩阵是每个 alter 行上的 `TALKFREQ1..TALKFREQ5`，对角为空，实测对称。",
        "  `TALKFREQ0` 是 ego 与该 alter 的频率，量不同。`TALKFREQ6`/`TALKFREQ7` 在只有 1 个",
        "  alter 的名单上也有四到六成填充，不可能是第 6、7 个 alter 列，已整列排除。",
        "- 只取 `SECTION == 1`（上限 5 人的密友名单）。第 2–4 节是另外的名单，混进来会",
        "  凭空放大网络规模。",
        "",
    ]
    (OUT_DIR / "nshap_cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = []
    for dataset, spec in ROUNDS.items():
        st = clean_round(dataset, spec)
        stats.append(st)
        print(
            f"{dataset} ({spec['dir']}): gold={st['n_gold']}/{st['n_rosters']} "
            f"dropped={st['dropped'] or '{}'} "
            f"mean_size={st['priors'].get('mean_num_alters')} "
            f"mean_density={st['priors'].get('mean_density')}"
        )
        for field, _ in bucketed_fields(spec):
            cut = st["cuts"].get(field)
            block = st["priors"]["bucket_shares"].get(field)
            if cut and block:
                print(
                    f"    {field:28s} n={block['n_labelled']:5d} "
                    f"low={block.get('low', 0):.0%} med={block.get('medium', 0):.0%} "
                    f"high={block.get('high', 0):.0%}  切点={cut[0]:.3f}/{cut[1]:.3f}"
                )
            else:
                print(f"    {field:28s} 分位退化，未分箱")
    write_report(stats)
    print(f"\n报告: {(OUT_DIR / 'nshap_cleaning_report.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
