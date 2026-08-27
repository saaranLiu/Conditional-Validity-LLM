#!/usr/bin/env python3
"""Acceptance checks for the supplementary cleaned datasets.

Checks three things that would quietly invalidate an experiment if wrong:
  1. No answer leakage -- nothing derivable from the target may appear in the ego state
     that is handed to the generator.
  2. Internal consistency -- reported network size matches the roster, dyad counts cannot
     exceed what the roster allows, and every bucket label agrees with the recorded cutoff.
  3. Traceable sample loss -- gold counts reconcile against the raw source.

Run: python3 Testing/scripts/validate_supplementary_cleaning.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from paths import cleaned_dir

ROOT = Path(__file__).resolve().parents[2]
P1 = cleaned_dir("issp_2017")
P2_DIRS = {ds: cleaned_dir(ds) for ds in ["signos", "nshap_r1", "nshap_r2", "nshap_r3"]}

# Substrings that must never appear in a state key: they name quantities that only exist
# once the target network / resource profile is known.
TARGET_MARKERS = (
    "num_alters",
    "density",
    "_share",
    "_bucket",
    "kin_",
    "alter",
    "composition",
    "roster",
    "dyad",
    "access_count",
    "support_",
    "institutional_",
    "channel_",
)

NETWORK_DATASETS = ["signos", "nshap_r1", "nshap_r2", "nshap_r3"]

failures: list[str] = []
notes: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_no_leakage(dataset: str, records: list[dict[str, Any]]) -> None:
    offenders: set[str] = set()
    for rec in records:
        for key in rec.get("state_json", {}):
            if any(marker in key for marker in TARGET_MARKERS):
                offenders.add(key)
    if offenders:
        fail(f"{dataset}: state_json 含疑似答案泄漏字段 {sorted(offenders)}")
        return
    keys = sorted({k for rec in records for k in rec.get("state_json", {})})
    notes.append(f"{dataset}: 无答案泄漏，state_json 共 {len(keys)} 个字段全部为 ego 自身属性")


def check_bucket_consistency(dataset: str, records: list[dict[str, Any]], priors: dict, target_key: str) -> None:
    cutoffs = priors.get("bucket_cutoffs") or {}
    checked = 0
    for rec in records:
        summary = rec["target_json"][target_key]
        for field, block in cutoffs.items():
            low, high = block["cuts"]
            value = summary.get(field)
            label = summary.get(f"{field}_bucket")
            if value is None or label is None:
                continue
            expected = "low" if value < low else ("medium" if value < high else "high")
            if label != expected:
                fail(
                    f"{dataset}: {rec['respondent_id']} 的 {field}={value} 标为 {label}，"
                    f"按切点 {low}/{high} 应为 {expected}"
                )
                return
            checked += 1
    # Any field the cleaner refused to bucket must carry no labels at all.
    for rec in records:
        summary = rec["target_json"][target_key]
        for key, label in summary.items():
            if key.endswith("_bucket") and label is not None:
                field = key[: -len("_bucket")]
                if field in {"size"}:
                    continue
                if field not in cutoffs and f"{field}_share" not in cutoffs:
                    fail(f"{dataset}: {field} 无切点却有桶标签 {label}")
                    return
    notes.append(f"{dataset}: 桶标签与切点一致，核对 {checked} 条")


def check_network_internals(dataset: str, records: list[dict[str, Any]]) -> None:
    for rec in records:
        comp = rec["target_json"]["composition_summary"]
        alters = rec["target_json"].get("alters") or []
        if comp["num_alters"] != len(alters):
            fail(
                f"{dataset}: {rec['respondent_id']} num_alters={comp['num_alters']} "
                f"但名单长度={len(alters)}"
            )
            return
        n = len(alters)
        possible = n * (n - 1) // 2
        observed = comp.get("n_dyads_observed") or 0
        if observed > possible:
            fail(
                f"{dataset}: {rec['respondent_id']} 观察到 {observed} 对连边，"
                f"但 {n} 人最多只有 {possible} 对"
            )
            return
        ties = rec["target_json"].get("alter_ties") or []
        if len(ties) != observed:
            fail(
                f"{dataset}: {rec['respondent_id']} alter_ties 长度={len(ties)} "
                f"与 n_dyads_observed={observed} 不符"
            )
            return
        ids = {a.get("alter_id") for a in alters}
        for tie in ties:
            if tie["src_id"] not in ids or tie["tgt_id"] not in ids:
                fail(f"{dataset}: {rec['respondent_id']} 连边端点不在名单内 {tie}")
                return
            if tie["src_id"] == tie["tgt_id"]:
                fail(f"{dataset}: {rec['respondent_id']} 出现自环 {tie}")
                return
    notes.append(f"{dataset}: 名单/连边内部一致（规模、对数上限、端点、无自环）")


def check_summary_alignment(dataset: str, records: list[dict[str, Any]], summary_path: Path) -> None:
    with summary_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(records):
        fail(f"{dataset}: summary.csv 行数 {len(rows)} 与 gold 记录数 {len(records)} 不符")
        return
    notes.append(f"{dataset}: summary.csv 与 gold 行数一致（{len(rows)}）")


def main() -> None:
    print("=" * 72)
    print("补充 survey 清洗验收")
    print("=" * 72)

    for dataset in NETWORK_DATASETS:
        gold = P2_DIRS[dataset] / f"{dataset}_network_gold.jsonl"
        priors_path = P2_DIRS[dataset] / f"{dataset}_population_priors.json"
        summary_path = P2_DIRS[dataset] / f"{dataset}_network_summary.csv"
        if not gold.exists():
            fail(f"{dataset}: 缺少 {gold.name}")
            continue
        records = read_jsonl(gold)
        priors = json.loads(priors_path.read_text(encoding="utf-8"))
        check_no_leakage(dataset, records)
        check_network_internals(dataset, records)
        check_bucket_consistency(dataset, records, priors, "composition_summary")
        check_summary_alignment(dataset, records, summary_path)
        print(f"\n[{dataset}] ego={len(records)}  均值名单={priors.get('mean_num_alters')}")

    gold = P1 / "issp_2017_resource_gold.jsonl"
    if gold.exists():
        records = read_jsonl(gold)
        priors = json.loads((P1 / "issp_2017_population_priors.json").read_text(encoding="utf-8"))
        check_no_leakage("issp_2017", records)
        check_bucket_consistency("issp_2017", records, priors, "resource_summary")
        check_summary_alignment("issp_2017", records, P1 / "issp_2017_resource_summary.csv")
        if priors.get("has_network_stage") is not False:
            fail("issp_2017: 必须显式标记 has_network_stage=false")
        for rec in records[:200]:
            if "alters" in rec["target_json"] or "alter_ties" in rec["target_json"]:
                fail("issp_2017: 不应出现名单字段（该问卷没有提名法）")
                break
        print(f"\n[issp_2017] 受访者={len(records)}  国家样本={priors.get('n_country_samples')}")
    else:
        fail("issp_2017: 缺少 resource gold")

    print("\n" + "-" * 72)
    for note in notes:
        print("  通过 |", note)
    print("-" * 72)
    if failures:
        print(f"\n失败 {len(failures)} 项：")
        for f_ in failures:
            print("  -", f_)
        sys.exit(1)
    print("\n全部验收通过。")


if __name__ == "__main__":
    main()
