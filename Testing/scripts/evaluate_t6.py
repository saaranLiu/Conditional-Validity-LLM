#!/usr/bin/env python3
"""T6 cross-population evaluation.

  1. ISSP 2017: 30-country resource means vs human country means
  2. GSS 1985 vs 2004: same name generator, 19 years apart (N0)
  3. NSHAP r1 vs UCNets W1: older confidant net vs Bay Area 30-person net (N0)
  4. GSS 2024: post-cutoff N0 margins (contamination check)

  python Testing/scripts/evaluate_t6.py [--model-slug Qwen]

Writes Testing/T6_cross_population/eval/<model-slug>/t6_summary.json, ...
"""

from __future__ import annotations

import argparse

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paths import arm_input, arm_output_jsonl, cleaned_dir, network_gold, t6_eval_dir

ISSP_NUMERIC = [
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

GSS_FIELDS = [
    "num_alters",
    "kin_share",
    "friend_share",
    "coworker_share",
    "pct_weekly_plus_contact",
]

NSHAP_UCNETS_FIELDS = ["kin_share", "friend_share"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def last_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        rid = str(r.get("respondent_id") or "")
        if rid:
            out[rid] = r
    return out


def schema_ok(rec: dict[str, Any] | None) -> bool:
    if not rec or rec.get("error"):
        return False
    return bool((rec.get("validation") or {}).get("schema_ok"))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx = _rank(xs)
    ry = _rank(ys)
    return pearson(rx, ry)


def _rank(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def mean_or_none(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def span(vals: list[float]) -> float | None:
    return (max(vals) - min(vals)) if vals else None


def issp_eval(model_slug: str = "llama3.1_latest") -> dict[str, Any]:
    inp = last_by_id(read_jsonl(arm_input("issp_2017", "t6_country_resources")))
    out = last_by_id(read_jsonl(arm_output_jsonl("issp_2017", "t6_country_resources", model_slug)))
    priors = json.loads(
        (cleaned_dir("issp_2017") / "issp_2017_population_priors.json").read_text(encoding="utf-8")
    )
    by_country_full = priors.get("by_country") or {}

    grouped: dict[str, dict[str, Any]] = {}
    n_ok = n_err = 0
    for rid, rec in out.items():
        src = inp.get(rid) or {}
        meta = src.get("metadata") or {}
        code = meta.get("country_code") or "UNK"
        name = meta.get("country") or code
        human = ((meta.get("gold_target") or {}).get("resource_summary") or {})
        llm = ((rec.get("prediction_json") or {}).get("resource_summary") or {})
        ok = schema_ok(rec)
        if rec.get("error"):
            n_err += 1
        if ok:
            n_ok += 1
        bucket = grouped.setdefault(
            code,
            {
                "country": name,
                "n_out": 0,
                "n_ok": 0,
                "pairs": defaultdict(list),
            },
        )
        bucket["n_out"] += 1
        if not ok:
            continue
        bucket["n_ok"] += 1
        for f in ISSP_NUMERIC:
            hv = human.get(f)
            lv = llm.get(f)
            if hv is not None and lv is not None:
                bucket["pairs"][f].append((float(hv), float(lv)))

    country_rows: list[dict[str, Any]] = []
    field_stats: dict[str, Any] = {}
    for f in ISSP_NUMERIC:
        h_sub, l_sub, h_full, l_full = [], [], [], []
        for code, bucket in sorted(grouped.items()):
            pairs = bucket["pairs"][f]
            mh = mean_or_none([p[0] for p in pairs])
            ml = mean_or_none([p[1] for p in pairs])
            full = (by_country_full.get(code) or {}).get(f"mean_{f}")
            country_rows.append(
                {
                    "country_code": code,
                    "country": bucket["country"],
                    "field": f,
                    "n_ok": bucket["n_ok"],
                    "n_paired": len(pairs),
                    "n_dropped_unpaired": bucket["n_ok"] - len(pairs),
                    "mean_human_subsample": mh,
                    "mean_llm": ml,
                    "mean_human_full": full,
                }
            )
            if mh is not None and ml is not None:
                h_sub.append(mh)
                l_sub.append(ml)
            if full is not None and ml is not None:
                h_full.append(float(full))
                l_full.append(ml)
        field_stats[f] = {
            "n_countries": len(l_sub),
            "n_paired_countries_subsample": len(l_sub),
            "n_dropped_countries_subsample": len(grouped) - len(l_sub),
            "pearson_subsample": pearson(h_sub, l_sub),
            "spearman_subsample": spearman(h_sub, l_sub),
            "pearson_full": pearson(h_full, l_full),
            "spearman_full": spearman(h_full, l_full),
            "span_human_subsample": span(h_sub),
            "span_llm": span(l_sub),
            "span_human_full": span(h_full) if h_full else None,
            "span_ratio_full": (
                (span(l_full) / span(h_full)) if h_full and span(h_full) not in (None, 0) else None
            ),
            "grand_mean_human_full": mean_or_none(h_full) if h_full else None,
            "grand_mean_llm": mean_or_none(l_full),
            "sign_agree_full": _sign_agree(h_full, l_full),
        }

    r_full = [s["pearson_full"] for s in field_stats.values() if s.get("pearson_full") is not None]
    r_sub = [s["pearson_subsample"] for s in field_stats.values() if s.get("pearson_subsample") is not None]
    return {
        "n_output": len(out),
        "n_schema_ok": n_ok,
        "n_errors": n_err,
        "n_countries": len(grouped),
        "mean_pearson_full": mean_or_none(r_full),
        "mean_pearson_subsample": mean_or_none(r_sub),
        "mean_span_ratio_full": mean_or_none(
            [s["span_ratio_full"] for s in field_stats.values() if s.get("span_ratio_full") is not None]
        ),
        "fields": field_stats,
        "country_rows": country_rows,
    }


def _sign_agree(h: list[float], l: list[float]) -> float | None:
    if len(h) != len(l) or not h:
        return None
    hm, lm = float(np.mean(h)), float(np.mean(l))
    n = sum(1 for a, b in zip(h, l) if (a - hm) * (b - lm) > 0)
    return n / len(h)


def n0_means(dataset: str, fields: list[str], model_slug: str = "llama3.1_latest") -> dict[str, Any]:
    gold = last_by_id(read_jsonl(network_gold(dataset)))
    preds = last_by_id(read_jsonl(arm_output_jsonl(dataset, "n0_baseline_network", model_slug)))
    ids = [rid for rid, rec in preds.items() if schema_ok(rec) and rid in gold]
    out: dict[str, Any] = {
        "n_gold": len(gold),
        "n_schema_ok": sum(1 for r in preds.values() if schema_ok(r)),
        "n_analysis": len(ids),
        "fields": {},
    }
    for f in fields:
        paired: list[tuple[float, float]] = []
        for rid in ids:
            hv_i = ((gold[rid].get("target_json") or {}).get("composition_summary") or {}).get(f)
            lv_i = ((preds[rid].get("prediction_json") or {}).get("composition_summary") or {}).get(f)
            if hv_i is not None and lv_i is not None:
                paired.append((float(hv_i), float(lv_i)))
        hv = [p[0] for p in paired]
        lv = [p[1] for p in paired]
        out["fields"][f] = {
            "mean_human_paired": mean_or_none(hv),
            "mean_llm_paired": mean_or_none(lv),
            "mean_human_analysis": mean_or_none(hv),
            "mean_llm": mean_or_none(lv),
            "n_paired": len(paired),
            "n_dropped_unpaired": len(ids) - len(paired),
            # Compatibility aliases; these now intentionally use the paired estimand.
            "mean_human_all": mean_or_none(hv),
            "n_human_all": len(hv),
            "n_llm": len(lv),
            "_paired_values": paired,
        }
    return out


def bootstrap_contrast(
    pairs_a: list[tuple[float, float]],
    pairs_b: list[tuple[float, float]],
    *,
    seed: int,
    iterations: int = 2000,
) -> dict[str, Any] | None:
    if not pairs_a or not pairs_b or iterations <= 0:
        return None
    a = np.asarray(pairs_a, dtype=float)
    b = np.asarray(pairs_b, dtype=float)
    rng = np.random.default_rng(seed)
    dh: list[float] = []
    dl: list[float] = []
    gap: list[float] = []
    for _ in range(iterations):
        sa = a[rng.integers(0, len(a), size=len(a))]
        sb = b[rng.integers(0, len(b), size=len(b))]
        d_h = float(np.mean(sb[:, 0]) - np.mean(sa[:, 0]))
        d_l = float(np.mean(sb[:, 1]) - np.mean(sa[:, 1]))
        dh.append(d_h)
        dl.append(d_l)
        gap.append(d_l - d_h)

    def ci(vals: list[float]) -> dict[str, float]:
        return {
            "lower": float(np.quantile(vals, 0.025)),
            "upper": float(np.quantile(vals, 0.975)),
        }

    return {
        "method": "ego_bootstrap_percentile",
        "seed": seed,
        "iterations": iterations,
        "ci_level": 0.95,
        "delta_human": ci(dh),
        "delta_llm": ci(dl),
        "delta_llm_minus_delta_human": ci(gap),
    }


def contrast(
    a: dict[str, Any],
    b: dict[str, Any],
    name_a: str,
    name_b: str,
    *,
    bootstrap_seed: int,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    rows = {}
    signs = []
    for field_index, f in enumerate(a["fields"]):
        if f not in b["fields"]:
            continue
        dh = _sub(b["fields"][f]["mean_human_all"], a["fields"][f]["mean_human_all"])
        dl = _sub(b["fields"][f]["mean_llm"], a["fields"][f]["mean_llm"])
        agree = None if dh is None or dl is None else (dh * dl > 0)
        if agree is not None:
            signs.append(agree)
        rows[f] = {
            f"human_{name_a}": a["fields"][f]["mean_human_all"],
            f"human_{name_b}": b["fields"][f]["mean_human_all"],
            f"llm_{name_a}": a["fields"][f]["mean_llm"],
            f"llm_{name_b}": b["fields"][f]["mean_llm"],
            "delta_human": dh,
            "delta_llm": dl,
            "same_sign": agree,
            "compression": (abs(dl) / abs(dh)) if dh not in (None, 0) and dl is not None else None,
            "n_paired_a": a["fields"][f]["n_paired"],
            "n_paired_b": b["fields"][f]["n_paired"],
            "bootstrap_ci": bootstrap_contrast(
                a["fields"][f]["_paired_values"],
                b["fields"][f]["_paired_values"],
                seed=bootstrap_seed + field_index,
                iterations=bootstrap_iterations,
            ),
        }
    return {
        "n_fields": len(rows),
        "n_same_sign": sum(signs),
        "fields": rows,
        "n_a": {"gold": a["n_gold"], "analysis": a["n_analysis"]},
        "n_b": {"gold": b["n_gold"], "analysis": b["n_analysis"]},
    }


def _sub(x: float | None, y: float | None) -> float | None:
    if x is None or y is None:
        return None
    return x - y


def public_n0(result: dict[str, Any]) -> dict[str, Any]:
    """Remove bootstrap-only ego values from the persisted summary."""
    return {
        **result,
        "fields": {
            field: {k: v for k, v in stats.items() if not k.startswith("_")}
            for field, stats in result["fields"].items()
        },
    }


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-slug", default="llama3.1_latest")
    args = parser.parse_args()
    slug = args.model_slug

    issp = issp_eval(slug)
    gss85 = n0_means("gss_1985", GSS_FIELDS, slug)
    gss04 = n0_means("gss_2004", GSS_FIELDS, slug)
    gss24 = n0_means("gss_2024", GSS_FIELDS, slug)
    nshap = n0_means("nshap_r1", NSHAP_UCNETS_FIELDS + ["num_alters", "weekly_contact_share"], slug)
    ucnets = n0_means("ucnets_w1", NSHAP_UCNETS_FIELDS + ["num_alters", "pct_weekly_plus_contact"], slug)
    year = contrast(
        gss85,
        gss04,
        "1985",
        "2004",
        bootstrap_seed=20260820,
    )
    pop = contrast(
        ucnets,
        nshap,
        "ucnets",
        "nshap",
        bootstrap_seed=20261820,
    )

    summary = {
        "model_slug": slug,
        "issp_2017": {k: v for k, v in issp.items() if k != "country_rows"},
        "paired_mean_policy": (
            "Human and LLM means use only egos where both values are non-missing."
        ),
        "gss_1985_n0": public_n0(gss85),
        "gss_2004_n0": public_n0(gss04),
        "gss_2024_n0": public_n0(gss24),
        "nshap_r1_n0": public_n0(nshap),
        "ucnets_w1_n0": public_n0(ucnets),
        "contrast_gss_1985_vs_2004": year,
        "contrast_ucnets_vs_nshap": pop,
    }

    eval_dir = t6_eval_dir() / slug
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "t6_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = eval_dir / "t6_issp_country_means.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "country_code",
                "country",
                "field",
                "n_ok",
                "n_paired",
                "n_dropped_unpaired",
                "mean_human_subsample",
                "mean_human_full",
                "mean_llm",
            ],
        )
        w.writeheader()
        w.writerows(issp["country_rows"])

    lines = [
        "# T6 auto tables",
        "",
        "所有 human/LLM 均值均只使用双方同为非缺失的 paired ego；各字段的 `n_paired`/`n_dropped_unpaired` 见 JSON。",
        "",
        "## ISSP 2017：国家均值是否跟随人类跨国差异",
        "",
        f"schema_ok={issp['n_schema_ok']}/{issp['n_output']}，{issp['n_countries']} 国 × 80。",
        f"人类侧用全样本国家均值（priors）；LLM 侧用 80 人分层子样本。",
        "",
        f"字段平均 Pearson(全样本国家均值, LLM) = **{fmt(issp['mean_pearson_full'])}**；"
        f"平均跨度比 span_L/span_H = **{fmt(issp['mean_span_ratio_full'])}**。",
        "",
        "| 字段 | Pearson 全样本 | Spearman | span_H | span_L | 跨度比 | 同号(相对均值) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for f, s in issp["fields"].items():
        lines.append(
            f"| {f} | {fmt(s['pearson_full'])} | {fmt(s['spearman_full'])} | "
            f"{fmt(s['span_human_full'])} | {fmt(s['span_llm'])} | "
            f"{fmt(s['span_ratio_full'])} | {fmt(s['sign_agree_full'])} |"
        )

    lines += [
        "",
        "## GSS 1985 vs 2004（N0）",
        "",
        f"同号字段 {year['n_same_sign']}/{year['n_fields']}。",
        "",
        "| 字段 | H 1985 | H 2004 | ΔH | L 1985 | L 2004 | ΔL | 同号 |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for f, row in year["fields"].items():
        lines.append(
            f"| {f} | {fmt(row['human_1985'])} | {fmt(row['human_2004'])} | {fmt(row['delta_human'])} | "
            f"{fmt(row['llm_1985'])} | {fmt(row['llm_2004'])} | {fmt(row['delta_llm'])} | "
            f"{fmt(row['same_sign'])} |"
        )

    lines += [
        "",
        "## NSHAP r1 vs UCNets W1（N0）",
        "",
        "可比字段是构成占比；`num_alters` 名单上限不同（5 vs 30），只作方向参考。",
        "",
        "| 字段 | H UCNets | H NSHAP | ΔH | L UCNets | L NSHAP | ΔL | 同号 |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for f, row in pop["fields"].items():
        lines.append(
            f"| {f} | {fmt(row['human_ucnets'])} | {fmt(row['human_nshap'])} | {fmt(row['delta_human'])} | "
            f"{fmt(row['llm_ucnets'])} | {fmt(row['llm_nshap'])} | {fmt(row['delta_llm'])} | "
            f"{fmt(row['same_sign'])} |"
        )

    lines += [
        "",
        "## GSS 2024 N0 边际（污染检查）",
        "",
        "| 字段 | H 1985 | H 2004 | H 2024 | L 1985 | L 2004 | L 2024 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for f in GSS_FIELDS:
        lines.append(
            f"| {f} | {fmt(gss85['fields'][f]['mean_human_all'])} | "
            f"{fmt(gss04['fields'][f]['mean_human_all'])} | "
            f"{fmt(gss24['fields'][f]['mean_human_all'])} | "
            f"{fmt(gss85['fields'][f]['mean_llm'])} | "
            f"{fmt(gss04['fields'][f]['mean_llm'])} | "
            f"{fmt(gss24['fields'][f]['mean_llm'])} |"
        )
    lines.append("")
    (eval_dir / "t6_tables.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "issp_mean_pearson_full": issp["mean_pearson_full"],
                "issp_mean_span_ratio": issp["mean_span_ratio_full"],
                "gss_year_same_sign": f"{year['n_same_sign']}/{year['n_fields']}",
                "nshap_ucnets_same_sign": f"{pop['n_same_sign']}/{pop['n_fields']}",
            },
            indent=2,
        )
    )
    print(f"wrote {eval_dir / 't6_summary.json'}")
    print(f"wrote {csv_path}")
    print(f"wrote {eval_dir / 't6_tables.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
