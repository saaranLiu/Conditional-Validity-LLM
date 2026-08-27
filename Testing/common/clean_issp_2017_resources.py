#!/usr/bin/env python3
"""Clean ISSP 2017 (Social Networks and Social Resources) into state-stage gold data.

Dataset id: `issp_2017`. Emits under Survey/ucnets/cleaned/:
`issp_2017_resource_gold.jsonl`, `issp_2017_resource_summary.csv`,
`issp_2017_population_priors.json`, `issp_2017_cleaning_report.md`.

SUPPLEMENTARY dataset. It has NO network stage and never will: ISSP 2017 asked no name
generator, so there is no roster, no alter attributes and no alter-alter ties. Nothing
here can be scored against a generated network, and none of the Phase 2 network metrics
apply.

What it does carry that no other survey in this project does is 30 country samples asked
identical questions, which makes it the natural home for the cross-population criterion:
hold the prompt template fixed, vary only the country, and ask whether the generator
moves the way humans do. Human reliance on kin versus institutions differs sharply across
these 30 samples, so a generator that returns one global profile is visibly predicting a
stereotype rather than sampling a population.

The target here is a social-resource profile rather than a network:
  * position generator (Q1) -- for each of 10 occupations, the RELATIONAL TIER at which
    the respondent knows someone. Note this is not a yes/no item: 1 = family, 2 = close
    friend, 3 = other acquaintance, 4 = no one. So it yields both an access count and a
    channel mix.
  * support generator (Q7) -- who they would turn to for five everyday needs.
  * institutional generator (Q8) -- person versus organisation for five larger needs.
  * contact volume (Q19) and role-specific contact frequency (Q21-Q25).

Coding hazards handled here:
  * Every item uses 8/9 for can't-choose / no-answer, and the background variables add
    their own (97/98/99, and 0 for "not available in this country"). Treating any of
    these as data would silently invent respondents with 98 years of schooling.
  * Q21-Q25 carry 95 = "no such person exists" and 96 = "that person lives with me".
    95 must leave the denominator (no relationship to have contact within), while 96 is
    the highest contact level, not missing. Conflating the two would bias contact
    frequency in opposite directions.
  * Bucket cutoffs are POOLED across countries on purpose. Per-country tertiles would
    hide exactly the cross-national spread the dataset is here to expose.
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
    read_csv,
    round_or_none,
    share,
    write_flat_csv,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "Survey" / "issp" / "processed" / "survey.csv"
OUT_DIR = ROOT / "Survey" / "issp" / "cleaned"
DATASET = "issp_2017"
FIELD_YEAR = 2017

COUNTRY_NAMES = {
    "AT": "Austria",
    "AU": "Australia",
    "CH": "Switzerland",
    "CN": "China",
    "CZ": "Czechia",
    "DE": "Germany",
    "DK": "Denmark",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "GB-GBN": "Great Britain",
    "HR": "Croatia",
    "HU": "Hungary",
    "IL": "Israel",
    "IN": "India",
    "IS": "Iceland",
    "JP": "Japan",
    "LT": "Lithuania",
    "MX": "Mexico",
    "NZ": "New Zealand",
    "PH": "Philippines",
    "RU": "Russia",
    "SE": "Sweden",
    "SI": "Slovenia",
    "SK": "Slovakia",
    "SR": "Suriname",
    "TH": "Thailand",
    "TW": "Taiwan",
    "US": "United States",
    "ZA": "South Africa",
}

OCCUPATION_ITEMS = {
    "v1": "bus_or_lorry_driver",
    "v2": "senior_executive_large_company",
    "v3": "home_or_office_cleaner",
    "v4": "hairdresser_or_barber",
    "v5": "human_resource_manager",
    "v6": "lawyer",
    "v7": "car_mechanic",
    "v8": "nurse",
    "v9": "police_officer",
    "v10": "school_teacher",
}
# 1-3 are the tiers at which the respondent knows someone; 4 is genuinely "no one".
ACCESS_TIER = {1: "family_or_relative", 2: "close_friend", 3: "other_acquaintance", 4: "no_one"}

SUPPORT_ITEMS = {
    "v21": "help_around_house_or_garden",
    "v22": "help_when_ill_at_home",
    "v23": "help_when_depressed",
    "v24": "advice_on_family_problems",
    "v25": "company_for_a_social_occasion",
}
SUPPORT_SOURCE = {
    1: "close_family_member",
    2: "more_distant_family_member",
    3: "close_friend",
    4: "neighbour",
    5: "coworker",
    6: "someone_else",
    7: "no_one",
}
SUPPORT_FAMILY = {1, 2}

INSTITUTIONAL_ITEMS = {
    "v26": "borrow_large_sum_of_money",
    "v27": "find_a_job",
    "v28": "official_paperwork",
    "v29": "find_a_place_to_live",
    "v30": "care_if_seriously_ill",
}
INSTITUTIONAL_SOURCE = {
    1: "family_or_close_friends",
    2: "other_persons",
    3: "private_companies",
    4: "public_services",
    5: "nonprofit_or_religious_organisations",
    6: "other_organisations",
    7: "no_person_or_organisation",
}
INSTITUTIONAL_ORGS = {3, 4, 5, 6}
INSTITUTIONAL_PERSONS = {1, 2}

ROLE_CONTACT_ITEMS = {
    "v48": "parent",
    "v49": "sibling",
    "v50": "adult_child",
    "v51": "other_family_member",
    "v52": "close_friend",
}
CONTACT_FREQ = {
    1: "daily",
    2: "several_times_a_week",
    3: "once_a_week",
    4: "two_or_three_times_a_month",
    5: "once_a_month",
    6: "several_times_a_year",
    7: "less_often",
    8: "never",
}
NO_SUCH_PERSON = 95  # leaves the denominator: there is no such relationship to score
LIVES_WITH_RESPONDENT = 96  # highest contact level, NOT missing
WEEKLY_OR_MORE = {1, 2, 3}

CONTACT_VOLUME = {
    1: "0_to_4_people",
    2: "5_to_9",
    3: "10_to_19",
    4: "20_to_49",
    5: "50_to_99",
    6: "100_or_more",
}

GENDER = {1: "male", 2: "female"}
MARITAL = {
    1: "married",
    2: "civil_partnership",
    3: "separated",
    4: "divorced",
    5: "widowed",
    6: "never_married",
}
WORK_STATUS = {
    1: "currently_in_paid_work",
    2: "formerly_in_paid_work",
    3: "never_had_paid_work",
}
URBAN_RURAL = {
    1: "big_city",
    2: "suburb_of_big_city",
    3: "town_or_small_city",
    4: "country_village",
    5: "farm_or_rural_home",
}
RELIGION = {
    0: "none",
    1: "catholic",
    2: "protestant",
    3: "orthodox",
    4: "other_christian",
    5: "jewish",
    6: "islamic",
    7: "buddhist",
    8: "hindu",
    9: "other_asian_religion",
    10: "other_religion",
}
ATTENDANCE = {
    1: "several_times_a_week_or_more",
    2: "once_a_week",
    3: "two_or_three_times_a_month",
    4: "once_a_month",
    5: "several_times_a_year",
    6: "once_a_year",
    7: "less_than_once_a_year",
    8: "never",
}


def coded(row: dict[str, str], col: str, mapping: dict[int, str]) -> str | None:
    """Look up a code, treating anything outside the mapping as missing.

    Every ISSP variable reserves high codes for can't-choose / no-answer / not-available,
    and they differ by variable, so the mapping itself is the whitelist.
    """
    return mapping.get(code(row.get(col)))


def bounded(row: dict[str, str], col: str, low: int, high: int) -> int | None:
    value = code(row.get(col))
    return value if value is not None and low <= value <= high else None


def build_state(row: dict[str, str]) -> dict[str, Any]:
    prefix = row.get("c_alphan", "")
    state = {
        "country": COUNTRY_NAMES.get(prefix, prefix),
        "country_code": prefix,
        "age": bounded(row, "AGE", 15, 105),
        "gender": coded(row, "SEX", GENDER),
        "education_years": bounded(row, "EDUCYRS", 0, 30),
        "marital_status": coded(row, "MARITAL", MARITAL),
        "work_status": coded(row, "WORK", WORK_STATUS),
        "place_of_living": coded(row, "URBRURAL", URBAN_RURAL),
        "religion": coded(row, "RELIGGRP", RELIGION),
        "religious_attendance": coded(row, "ATTEND", ATTENDANCE),
        "social_position_self_rating": bounded(row, "TOPBOT", 1, 10),
        "household_size": bounded(row, "HOMPOP", 1, 20),
        "children_in_household": bounded(row, "HHCHILDR", 0, 15),
    }
    return {k: v for k, v in state.items() if v is not None}


def resource_summary(row: dict[str, str]) -> dict[str, Any]:
    tiers = [code(row.get(c)) for c in OCCUPATION_ITEMS]
    tiers_valid = [t for t in tiers if t in ACCESS_TIER]
    known = [t for t in tiers_valid if t != 4]

    support = [code(row.get(c)) for c in SUPPORT_ITEMS]
    support_valid = [s for s in support if s in SUPPORT_SOURCE]

    inst = [code(row.get(c)) for c in INSTITUTIONAL_ITEMS]
    inst_valid = [s for s in inst if s in INSTITUTIONAL_SOURCE]

    # 96 ("lives with me") is the top of the contact scale; 95 ("no such person") is not
    # a low contact level, it means the relationship does not exist and must not dilute
    # the denominator.
    role_codes = [code(row.get(c)) for c in ROLE_CONTACT_ITEMS]
    role_scored = [c for c in role_codes if c in CONTACT_FREQ or c == LIVES_WITH_RESPONDENT]
    role_weekly = sum(1 for c in role_scored if c == LIVES_WITH_RESPONDENT or c in WEEKLY_OR_MORE)
    role_absent = sum(1 for c in role_codes if c == NO_SUCH_PERSON)

    summary: dict[str, Any] = {
        "occupational_access_count": len(known) if tiers_valid else None,
        "occupational_items_answered": len(tiers_valid),
        "kin_channel_share": round_or_none(share(sum(1 for t in known if t == 1), len(known))),
        "friend_channel_share": round_or_none(share(sum(1 for t in known if t == 2), len(known))),
        "acquaintance_channel_share": round_or_none(
            share(sum(1 for t in known if t == 3), len(known))
        ),
        "support_family_share": round_or_none(
            share(sum(1 for s in support_valid if s in SUPPORT_FAMILY), len(support_valid))
        ),
        "support_friend_share": round_or_none(
            share(sum(1 for s in support_valid if s == 3), len(support_valid))
        ),
        "support_no_one_share": round_or_none(
            share(sum(1 for s in support_valid if s == 7), len(support_valid))
        ),
        "support_items_answered": len(support_valid),
        "institutional_reliance_share": round_or_none(
            share(sum(1 for s in inst_valid if s in INSTITUTIONAL_ORGS), len(inst_valid))
        ),
        "institutional_personal_share": round_or_none(
            share(sum(1 for s in inst_valid if s in INSTITUTIONAL_PERSONS), len(inst_valid))
        ),
        "institutional_no_one_share": round_or_none(
            share(sum(1 for s in inst_valid if s == 7), len(inst_valid))
        ),
        "institutional_items_answered": len(inst_valid),
        "contact_volume_band": coded(row, "v46", CONTACT_VOLUME),
        "weekly_plus_role_contact_share": round_or_none(share(role_weekly, len(role_scored))),
        "role_relationships_absent": role_absent,
        "role_items_scored": len(role_scored),
    }
    # These three shares are zero-inflated: 35-40% of respondents sit at exactly 0, which
    # puts the lower tertile on 0 and leaves the "low" bucket empty. The any/none contrast
    # is the part of them that is actually testable, so it is recorded alongside the share.
    for field in ("kin_channel_share", "friend_channel_share", "institutional_reliance_share"):
        value = summary[field]
        summary[field.replace("_share", "_any")] = (
            None if value is None else ("yes" if value > 0 else "no")
        )
    return summary


def resource_detail(row: dict[str, str]) -> dict[str, Any]:
    return {
        "occupational_access": {
            name: ACCESS_TIER.get(code(row.get(col)))
            for col, name in OCCUPATION_ITEMS.items()
            if ACCESS_TIER.get(code(row.get(col))) is not None
        },
        "support_sources": {
            name: SUPPORT_SOURCE.get(code(row.get(col)))
            for col, name in SUPPORT_ITEMS.items()
            if SUPPORT_SOURCE.get(code(row.get(col))) is not None
        },
        "institutional_sources": {
            name: INSTITUTIONAL_SOURCE.get(code(row.get(col)))
            for col, name in INSTITUTIONAL_ITEMS.items()
            if INSTITUTIONAL_SOURCE.get(code(row.get(col))) is not None
        },
        "role_contact_frequency": {
            name: (
                "lives_with_respondent"
                if code(row.get(col)) == LIVES_WITH_RESPONDENT
                else CONTACT_FREQ.get(code(row.get(col)))
            )
            for col, name in ROLE_CONTACT_ITEMS.items()
            if code(row.get(col)) == LIVES_WITH_RESPONDENT
            or CONTACT_FREQ.get(code(row.get(col))) is not None
        },
    }


BUCKETED: list[tuple[str, Any]] = [
    ("occupational_access_count", None),
    ("kin_channel_share", None),
    ("friend_channel_share", None),
    ("support_family_share", None),
    ("institutional_reliance_share", None),
    ("weekly_plus_role_contact_share", None),
]

CONTINUOUS = [
    "occupational_access_count",
    "kin_channel_share",
    "friend_channel_share",
    "acquaintance_channel_share",
    "support_family_share",
    "support_friend_share",
    "support_no_one_share",
    "institutional_reliance_share",
    "institutional_personal_share",
    "institutional_no_one_share",
    "weekly_plus_role_contact_share",
]


def summary_of(rec: dict[str, Any]) -> dict[str, Any]:
    return rec["target_json"]["resource_summary"]


def build_priors(
    records: list[dict[str, Any]],
    cuts: dict[str, tuple[float, float] | None],
    counts: dict[str, Counter],
) -> dict[str, Any]:
    summaries = [summary_of(r) for r in records]
    priors: dict[str, Any] = {
        "dataset": DATASET,
        "field_year": FIELD_YEAR,
        "n_respondents": len(records),
        "n_country_samples": len({r["state_json"]["country_code"] for r in records}),
        "has_network_stage": False,
        "network_stage_note": (
            "ISSP 2017 asked no name generator. There is no roster, no alter attributes and "
            "no alter-alter ties, so no Phase 2 network metric applies to this dataset."
        ),
        "bucket_cutoffs": {},
        "bucket_shares": {},
        "bucket_cutoffs_note": (
            "Cutoffs are pooled across all 30 country samples, so that per-country bucket "
            "shares stay comparable and the cross-national spread remains visible."
        ),
    }
    for field in CONTINUOUS:
        vals = [s.get(field) for s in summaries if s.get(field) is not None]
        if vals:
            priors[f"mean_{field}"] = mean_of(vals)
            priors[f"median_{field}"] = median_of(vals)
            priors[f"n_{field}"] = len(vals)

    for field, _ in BUCKETED:
        block = bucket_cutoff_block(cuts.get(field))
        if block:
            priors["bucket_cutoffs"][field] = block
        share_block = bucket_share_block(counts[field])
        if share_block:
            priors["bucket_shares"][field] = share_block

    by_country: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[rec["state_json"]["country_code"]].append(rec)
    for ccode, recs in sorted(grouped.items()):
        sums = [summary_of(r) for r in recs]
        entry: dict[str, Any] = {
            "country": COUNTRY_NAMES.get(ccode, ccode),
            "n": len(recs),
        }
        for field in CONTINUOUS:
            vals = [s.get(field) for s in sums if s.get(field) is not None]
            if vals:
                entry[f"mean_{field}"] = mean_of(vals)
        for field, _ in BUCKETED:
            local = Counter(s.get(f"{field}_bucket") for s in sums)
            block = bucket_share_block(local)
            if block:
                entry[f"{field}_buckets"] = block
        by_country[ccode] = entry
    priors["by_country"] = by_country
    return priors


DROP_REASONS = {
    "incomplete_state": "缺年龄或性别",
    "no_country": "无国家标识",
    "no_resource_items": "位置发生器与求助题全部未答",
}


def describe_dropped(dropped: Counter) -> str:
    if not dropped:
        return "无"
    return "；".join(
        f"{DROP_REASONS.get(reason, reason)} {count} 条" for reason, count in dropped.most_common()
    )


def write_report(
    records: list[dict[str, Any]],
    dropped: Counter,
    n_raw: int,
    cuts: dict[str, tuple[float, float] | None],
    priors: dict[str, Any],
) -> None:
    lines = [
        "# ISSP 2017 社会资源清洗报告",
        "",
        "由 `Testing/scripts/clean_issp_2017_resources.py` 生成。",
        "",
        "## 这份数据没有网络阶段",
        "",
        "ISSP 2017 没有提名法（name generator）：不问「请列出与你讨论重要事情的人」，",
        "只问「你认识做某职业的人吗」「遇到某事你会找谁」。因此**没有名单、没有 alter 属性、",
        "也没有 alter–alter 连边**，Phase 2 的任何网络指标都无法套用。这不是清洗损失，",
        "是问卷设计使然。它承担的是跨总体判据：30 个国家样本问同一套题，",
        "把提示模板固定、只改国家，看生成器是否像人类那样移动。",
        "",
        f"- 原始记录：{n_raw}",
        f"- 入库记录：{len(records)}",
        f"- 丢弃：{describe_dropped(dropped)}",
        f"- 国家样本数：{priors['n_country_samples']}",
        "",
        "## 靶量族：社会资源可得性",
        "",
        "| 靶量 | 含义 | 均值 |",
        "| --- | --- | --- |",
    ]
    meaning = {
        "occupational_access_count": "10 个职业中认识人的个数（位置发生器得分）",
        "kin_channel_share": "在认识的职业里，通过家人认识的比例",
        "friend_channel_share": "通过密友认识的比例",
        "acquaintance_channel_share": "通过其他熟人认识的比例",
        "support_family_share": "5 项日常求助中找家人的比例",
        "support_friend_share": "找密友的比例",
        "support_no_one_share": "回答「没有人可找」的比例",
        "institutional_reliance_share": "5 项较大需求中找机构（企业/公共服务/非营利）的比例",
        "institutional_personal_share": "找个人（家人密友/其他人）的比例",
        "institutional_no_one_share": "回答「没有人也没有机构」的比例",
        "weekly_plus_role_contact_share": "父母/兄弟姐妹/成年子女/其他亲属/密友中每周以上联系的比例",
    }
    for field in CONTINUOUS:
        lines.append(
            f"| `{field}` | {meaning[field]} | {priors.get(f'mean_{field}')} |"
        )

    lines += [
        "",
        "## 三分箱（切点在 30 国合并样本上取，不按国家分别取）",
        "",
        "按国家各取分位会把跨国差异抹掉——而跨国差异正是这份数据要暴露的东西。",
        "所以切点统一，各国的桶占比因此可比。",
        "",
        "| 靶量 | 切点 | low | medium | high |",
        "| --- | --- | --- | --- | --- |",
    ]
    for field, _ in BUCKETED:
        cut = cuts.get(field)
        block = priors["bucket_shares"].get(field)
        if cut and block:
            lines.append(
                f"| `{field}` | {cut[0]:.3f} / {cut[1]:.3f} | {block.get('low', 0):.1%} | "
                f"{block.get('medium', 0):.1%} | {block.get('high', 0):.1%} |"
            )
        else:
            lines.append(f"| `{field}` | 零膨胀，未分箱（见下） | - | - | - |")

    lines += [
        "",
        "### 三个靶量为什么不能分箱：零膨胀，不是清洗出错",
        "",
        "`kin_channel_share`、`friend_channel_share`、`institutional_reliance_share` 各有",
        "三成半到四成的受访者取值恰好为 0——一个职业都不是通过家人认识、或完全不找机构。",
        "单一取值占到四成，下三分位点就落在 0 上，low 桶必然为空，三分箱无法成立。",
        "",
        "这类靶量可测的部分是「有/无」而不是「高/中/低」，所以产物里额外给了",
        "`kin_channel_any`、`friend_channel_any`、`institutional_reliance_any` 三个二分标记；",
        "份额本身适合按均值和分布做跨国比较（见上一节的跨国跨度）。",
        "",
    ]
    lines += [
        "",
        "## 跨国差异有多大（跨总体判据的立足点）",
        "",
        "下表按「找家人求助的比例」排序，取两端各 5 国。如果生成器对这 30 国给出的",
        "画像基本相同，就说明它在输出一个全球刻板印象，而不是在为被问的人群抽样。",
        "",
        "| 国家 | n | 找家人求助 | 依赖机构 | 位置发生器得分 |",
        "| --- | --- | --- | --- | --- |",
    ]
    ranked = sorted(
        priors["by_country"].items(),
        key=lambda kv: kv[1].get("mean_support_family_share") or 0,
    )
    for _, entry in ranked[:5] + ranked[-5:]:
        lines.append(
            f"| {entry['country']} | {entry['n']} | "
            f"{entry.get('mean_support_family_share')} | "
            f"{entry.get('mean_institutional_reliance_share')} | "
            f"{entry.get('mean_occupational_access_count')} |"
        )

    lines += [
        "",
        "## 缺失码的处理（不处理就会凭空造出受访者）",
        "",
        "- 各题普遍用 8/9 表示「说不清」「未回答」，背景变量另有 97/98/99，以及 0 表示",
        "  「该国未采集」。若当成数据，会出现「受教育 98 年」这类记录。本脚本以取值映射",
        "  本身作白名单，映射外一律视为缺失。",
        "- Q21–Q25 有两个特殊码必须分开处理：95 =「没有这类亲属/朋友」，应移出分母",
        "  （不存在的关系无从谈联系频率）；96 =「与我同住」，是联系频率的最高档，不是缺失。",
        "  两者混淆会让联系频率朝相反方向偏。",
        "",
        "## 尚未接入生成管线",
        "",
        "Phase 2 的输入构建脚本目前假定数据集有名单（roster），而这份数据没有。",
        "要跑生成实验还需要让配置支持无名单数据集。清洗产物已按统一命名落盘，",
        "接入前不影响其他数据集。",
        "",
    ]
    (OUT_DIR / f"{DATASET}_cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = read_csv(SRC)
    records: list[dict[str, Any]] = []
    dropped: Counter = Counter()

    for index, row in enumerate(rows, start=1):
        state = build_state(row)
        if state.get("age") is None or state.get("gender") is None:
            dropped["incomplete_state"] += 1
            continue
        if not state.get("country_code"):
            dropped["no_country"] += 1
            continue
        summary = resource_summary(row)
        if summary["occupational_items_answered"] == 0 and summary["support_items_answered"] == 0:
            dropped["no_resource_items"] += 1
            continue
        records.append(
            {
                "respondent_id": f"{DATASET}_{index:05d}",
                "dataset": DATASET,
                "state_json": state,
                "target_json": {
                    "resource_summary": summary,
                    "resource_detail": resource_detail(row),
                },
                "metadata": {
                    "field_year": FIELD_YEAR,
                    "country_sample": row.get("c_sample"),
                    "source_path": str(SRC.relative_to(ROOT)),
                },
            }
        )

    cuts, counts = assign_buckets(records, BUCKETED, composition_of=summary_of)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / f"{DATASET}_resource_gold.jsonl", records)

    summary_rows = []
    for rec in records:
        flat = {
            "respondent_id": rec["respondent_id"],
            "country_code": rec["state_json"]["country_code"],
            "country": rec["state_json"]["country"],
        }
        flat.update({k: v for k, v in summary_of(rec).items() if not isinstance(v, dict)})
        summary_rows.append(flat)
    write_flat_csv(OUT_DIR / f"{DATASET}_resource_summary.csv", summary_rows)

    priors = build_priors(records, cuts, counts)
    (OUT_DIR / f"{DATASET}_population_priors.json").write_text(
        json.dumps(priors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(records, dropped, len(rows), cuts, priors)

    print(f"{DATASET}: gold={len(records)}/{len(rows)} dropped={dict(dropped) or '{}'}")
    print(f"    国家样本数={priors['n_country_samples']}")
    for field, _ in BUCKETED:
        cut = cuts.get(field)
        block = priors["bucket_shares"].get(field)
        if cut and block:
            print(
                f"    {field:32s} n={block['n_labelled']:6d} "
                f"low={block.get('low', 0):.0%} med={block.get('medium', 0):.0%} "
                f"high={block.get('high', 0):.0%}  切点={cut[0]:.3f}/{cut[1]:.3f}"
            )
        else:
            print(f"    {field:32s} 分位退化，未分箱")
    ranked = sorted(
        priors["by_country"].items(),
        key=lambda kv: kv[1].get("mean_support_family_share") or 0,
    )
    lo, hi = ranked[0][1], ranked[-1][1]
    print(
        f"    跨国跨度（找家人求助）：{lo['country']} {lo.get('mean_support_family_share')} "
        f"→ {hi['country']} {hi.get('mean_support_family_share')}"
    )


if __name__ == "__main__":
    main()
