#!/usr/bin/env python3
"""Phase 1 Eval 2: bivariate association realism.

Compares pairwise association structure in human vs LLM state tables:

  - categorical × categorical: Cramer's V
  - numeric × numeric: Pearson r
  - categorical × numeric: eta-squared

All LLM-minus-human differences use ego-paired bootstrap confidence intervals.
Point-estimate pass compatibility remains |difference| <= delta; outputs also
report whether the CI crosses zero and lies wholly inside [-delta, delta].

Usage:
  python Testing/scripts/evaluate_state_eval2.py --model-slug llama3.1_latest
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from evaluate_state_eval1 import (
    DEFAULT_OUTPUT_DIR,
    TASKS,
    TASK_SHORT,
    extract_human_flat,
    normalize_label,
    read_csv,
    read_jsonl,
)
from paths import arm_predictions_csv, state_eval_dir, state_gold
from state_content_config import DATASET_CONFIGS
from state_prediction_csv import META_COLUMNS, coerce_scalar, filter_valid_prediction_rows


# Prespecified scientifically relevant pairs (anchor/outcome and outcome/outcome).
_GSS_PAIRS: list[tuple[str, str, str]] = [
    # (side_a, side_b, kind) where kind in {cat_cat, num_num, cat_num}
    # side may be prompt.* for anchors or target field names.
    ("prompt.age_group", "general_happiness", "cat_cat"),
    ("prompt.sex", "general_happiness", "cat_cat"),
    ("prompt.degree", "general_happiness", "cat_cat"),
    ("prompt.age_group", "self_rated_health", "cat_cat"),
    ("prompt.degree", "people_can_be_trusted", "cat_cat"),
    ("prompt.sex", "people_can_be_trusted", "cat_cat"),
    ("self_rated_health", "general_happiness", "cat_cat"),
    ("people_can_be_trusted", "people_are_helpful", "cat_cat"),
    ("party_identity", "political_ideology", "cat_cat"),
    ("religion", "religious_service_attendance", "cat_cat"),
    # Task 3: party/religion are observed inputs
    ("prompt.party_identity", "political_ideology", "cat_cat"),
    ("prompt.religion", "religious_service_attendance", "cat_cat"),
    ("prompt.marital_status", "general_happiness", "cat_cat"),
]

PAIR_SPECS: dict[str, list[tuple[str, str, str]]] = {
    "gss_1987": list(_GSS_PAIRS),
    "gss_2010_network": list(_GSS_PAIRS),
    "ucnets_w1": [
        ("prompt.age_group", "days_lonely_past_week", "cat_num"),
        ("prompt.gender", "days_lonely_past_week", "cat_num"),
        ("prompt.education", "days_lonely_past_week", "cat_num"),
        ("prompt.age_group", "self_rated_health", "cat_cat"),
        ("prompt.gender", "self_rated_health", "cat_cat"),
        ("self_rated_health", "days_lonely_past_week", "cat_num"),
        ("self_rated_health", "days_very_happy_past_week", "cat_num"),
        ("days_lonely_past_week", "days_isolated_past_week", "num_num"),
        ("days_very_happy_past_week", "days_lonely_past_week", "num_num"),
        ("relatives_help_serious_problem", "friends_help_serious_problem", "cat_cat"),
        ("prompt.education", "relatives_help_serious_problem", "cat_cat"),
        ("party_identity", "voting_frequency", "cat_cat"),
        # Task 3: party/religion/voting are observed inputs
        ("prompt.party_identity", "self_rated_health", "cat_cat"),
        ("prompt.religion_current", "self_rated_health", "cat_cat"),
    ],
}


def cramers_v(x: list[str], y: list[str]) -> float | None:
    if len(x) < 20:
        return None
    levels_x = sorted(set(x))
    levels_y = sorted(set(y))
    if len(levels_x) < 2 or len(levels_y) < 2:
        return None
    table = np.zeros((len(levels_x), len(levels_y)), dtype=float)
    ix = {v: i for i, v in enumerate(levels_x)}
    iy = {v: i for i, v in enumerate(levels_y)}
    for a, b in zip(x, y):
        table[ix[a], iy[b]] += 1
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    k = min(table.shape) - 1
    if n <= 0 or k <= 0:
        return None
    return float(math.sqrt(chi2 / (n * k)))


def eta_squared(categories: list[str], values: list[float]) -> float | None:
    if len(values) < 20:
        return None
    arr = np.asarray(values, dtype=float)
    grand = arr.mean()
    ss_tot = float(((arr - grand) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    ss_between = 0.0
    for cat in set(categories):
        idx = [i for i, c in enumerate(categories) if c == cat]
        if not idx:
            continue
        group = arr[idx]
        ss_between += len(group) * float((group.mean() - grand) ** 2)
    return float(ss_between / ss_tot)


def paired_bootstrap_difference(
    kind: str,
    ha: list[Any],
    hb: list[Any],
    la: list[Any],
    lb: list[Any],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None, int]:
    """Bootstrap LLM-minus-human association differences by ego row."""
    if n_bootstrap <= 0 or len(ha) < 20:
        return None, None, 0
    n = len(ha)
    diffs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if kind == "cat_cat":
            sh = cramers_v([str(ha[i]) for i in idx], [str(hb[i]) for i in idx])
            sl = cramers_v([str(la[i]) for i in idx], [str(lb[i]) for i in idx])
        elif kind == "num_num":
            sh = float(np.corrcoef(np.asarray(ha, float)[idx], np.asarray(hb, float)[idx])[0, 1])
            sl = float(np.corrcoef(np.asarray(la, float)[idx], np.asarray(lb, float)[idx])[0, 1])
        else:
            sh = eta_squared([str(ha[i]) for i in idx], np.asarray(hb, float)[idx].tolist())
            sl = eta_squared([str(la[i]) for i in idx], np.asarray(lb, float)[idx].tolist())
        if sh is not None and sl is not None and np.isfinite(sh) and np.isfinite(sl):
            diffs.append(float(sl - sh))
    if not diffs:
        return None, None, 0
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), len(diffs)


def get_prompt_value(prompt_json: dict[str, Any], key: str) -> Any:
    # key like age_group after stripping prompt.
    return prompt_json.get(key)


def resolve_pair_values(
    preds: list[dict[str, str]],
    humans: dict[str, dict[str, Any]],
    a_spec: str,
    b_spec: str,
    a_kind: str,
    b_kind: str,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Return human_a, human_b, llm_a, llm_b aligned lists."""
    ha, hb, la, lb = [], [], [], []
    for p in preds:
        hrow = humans.get(str(p["respondent_id"]))
        if not hrow:
            continue
        prompt = hrow.get("prompt_json") or {}
        target = hrow.get("target_json") or {}

        def get_side(spec: str, kind: str, source: str) -> Any | None:
            if spec.startswith("prompt."):
                raw = get_prompt_value(prompt, spec.split(".", 1)[1])
            else:
                if source == "human":
                    raw = extract_human_flat(target, spec)
                else:
                    raw = p.get(spec, "")
                    if raw == "":
                        raw = None
            if raw is None or raw == "":
                return None
            if kind == "num":
                try:
                    return float(coerce_scalar(raw))
                except (TypeError, ValueError):
                    return None
            # categorical
            if isinstance(raw, list):
                return normalize_label(raw)
            return normalize_label(raw)

        # Determine kinds from pair kind string handled by caller via a_kind/b_kind
        hva = get_side(a_spec, a_kind, "human")
        hvb = get_side(b_spec, b_kind, "human")
        lva = get_side(a_spec, a_kind, "llm")
        lvb = get_side(b_spec, b_kind, "llm")
        # For prompt.* sides, LLM and human share the same observed anchor.
        if a_spec.startswith("prompt."):
            lva = hva
        if b_spec.startswith("prompt."):
            lvb = hvb

        if None in (hva, hvb, lva, lvb):
            continue
        ha.append(hva)
        hb.append(hvb)
        la.append(lva)
        lb.append(lvb)
    return ha, hb, la, lb


def evaluate_pair(
    kind: str,
    ha: list[Any],
    hb: list[Any],
    la: list[Any],
    lb: list[Any],
    alpha: float,
    delta_v: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    out: dict[str, Any] = {"n_human": len(ha), "n_llm": len(la), "kind": kind}
    if kind == "cat_cat":
        vh = cramers_v([str(x) for x in ha], [str(x) for x in hb])
        vl = cramers_v([str(x) for x in la], [str(x) for x in lb])
        out.update({"stat_human": vh, "stat_llm": vl, "stat_name": "cramers_v"})
        if vh is None or vl is None:
            out["pass"] = None
            out["abs_diff"] = None
        else:
            signed_diff = vl - vh
            out["signed_diff"] = signed_diff
            out["abs_diff"] = abs(signed_diff)
            out["pass"] = bool(abs(signed_diff) <= delta_v)
    elif kind == "num_num":
        if len(ha) < 20 or len(la) < 20:
            out.update({"stat_human": None, "stat_llm": None, "pass": None})
            return out
        rh = float(np.corrcoef(np.asarray(ha, float), np.asarray(hb, float))[0, 1])
        rl = float(np.corrcoef(np.asarray(la, float), np.asarray(lb, float))[0, 1])
        out.update(
            {
                "stat_human": rh,
                "stat_llm": rl,
                "stat_name": "pearson_r",
                "signed_diff": rl - rh,
                "abs_diff": abs(rh - rl),
                "pvalue": None,
                "pass": bool(abs(rh - rl) <= delta_v),
            }
        )
    elif kind == "cat_num":
        # a categorical, b numeric (by convention in PAIR_SPECS)
        eh = eta_squared([str(x) for x in ha], [float(x) for x in hb])
        el = eta_squared([str(x) for x in la], [float(x) for x in lb])
        out.update({"stat_human": eh, "stat_llm": el, "stat_name": "eta_squared"})
        if eh is None or el is None:
            out["pass"] = None
            out["abs_diff"] = None
        else:
            signed_diff = el - eh
            out["signed_diff"] = signed_diff
            out["abs_diff"] = abs(signed_diff)
            out["pass"] = bool(abs(signed_diff) <= delta_v)
    else:
        out["pass"] = None
    if out.get("signed_diff") is not None:
        lo, hi, n_ok = paired_bootstrap_difference(
            kind, ha, hb, la, lb, n_bootstrap, rng
        )
        out.update(
            {
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "bootstrap_successful": n_ok,
                "ci_crosses_zero": (
                    bool(lo <= 0 <= hi) if lo is not None and hi is not None else None
                ),
                "ci_within_equivalence_threshold": (
                    bool(lo >= -delta_v and hi <= delta_v)
                    if lo is not None and hi is not None
                    else None
                ),
            }
        )
    return out


def evaluate_cell(
    dataset: str,
    task: str,
    pred_path: Path,
    clean_path: Path,
    alpha: float,
    delta_v: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    respondent_ids: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_preds = read_csv(pred_path)
    preds, validity = filter_valid_prediction_rows(raw_preds)
    if respondent_ids is not None:
        preds = [p for p in preds if str(p["respondent_id"]) in respondent_ids]
    humans = {str(r["respondent_id"]): r for r in read_jsonl(clean_path)}
    pairs = PAIR_SPECS.get(dataset, [])

    # Filter pairs whose target fields exist for this task
    available = set(preds[0].keys()) - set(META_COLUMNS) if preds else set()
    # also include prompt fields conceptually
    rows: list[dict[str, Any]] = []
    for a, b, kind in pairs:
        # skip health-involving targets if field absent (task2)
        needed = []
        for spec in (a, b):
            if not spec.startswith("prompt."):
                needed.append(spec)
        if any(f not in available for f in needed):
            continue
        if kind == "cat_cat":
            ak, bk = "cat", "cat"
        elif kind == "num_num":
            ak, bk = "num", "num"
        else:
            ak, bk = "cat", "num"
        ha, hb, la, lb = resolve_pair_values(preds, humans, a, b, ak, bk)
        metrics = evaluate_pair(kind, ha, hb, la, lb, alpha, delta_v, n_bootstrap, rng)
        rows.append(
            {
                "dataset": dataset,
                "task": TASK_SHORT[task],
                "var_a": a,
                "var_b": b,
                **metrics,
            }
        )

    tested = [r for r in rows if r.get("pass") is not None]
    passed = [r for r in tested if r["pass"]]
    diffs = [r["abs_diff"] for r in rows if r.get("abs_diff") is not None]
    cell = {
        "dataset": dataset,
        "task": TASK_SHORT[task],
        "task_full": task,
        **validity,
        "analysis_n": len(preds),
        "n_pairs": len(rows),
        "n_tested": len(tested),
        "n_pass": len(passed),
        "pass_rate": (len(passed) / len(tested)) if tested else None,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
        "alpha": alpha,
        "delta_v": delta_v,
    }
    return cell, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-slug", default="llama3.1_latest")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--delta-v",
        type=float,
        default=0.10,
        help="Max |assoc_H - assoc_L| for Cramer's V / eta-squared pass.",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    args = parser.parse_args()

    out_dir = args.output_dir / "eval2_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cells, all_pairs = [], []
    rng = np.random.default_rng(args.seed)
    for dataset in args.datasets:
        if dataset not in PAIR_SPECS:
            continue
        for task in args.tasks:
            short = TASK_SHORT[task]
            try:
                pred_path = arm_predictions_csv(dataset, task, args.model_slug)
                clean_path = state_gold(dataset, task)
            except KeyError:
                print(f"skip unsupported: {dataset} {task}", file=sys.stderr)
                continue
            if not pred_path.exists() or not clean_path.exists():
                print(f"skip missing: {dataset} {task}", file=sys.stderr)
                continue
            cell, pairs = evaluate_cell(
                dataset,
                task,
                pred_path,
                clean_path,
                args.alpha,
                args.delta_v,
                args.bootstrap,
                rng,
            )
            all_cells.append(cell)
            all_pairs.extend(pairs)
            pr = cell["pass_rate"]
            pr_s = f"{pr:.3f}" if pr is not None else "NA"
            print(
                f"{dataset:10} {short:6} pairs={cell['n_pairs']} "
                f"pass={cell['n_pass']}/{cell['n_tested']} ({pr_s}) "
                f"mean|Δ|={cell['mean_abs_diff']}"
            )

    paired_rows: list[dict[str, Any]] = []
    requested_tasks = set(args.tasks)
    for dataset in args.datasets:
        if dataset not in PAIR_SPECS:
            continue
        for task_b in ("s1_health_conditioned", "s2_politics_religion_conditioned"):
            task_a = "s0_baseline_state"
            try:
                pa = arm_predictions_csv(dataset, task_a, args.model_slug)
                pb = arm_predictions_csv(dataset, task_b, args.model_slug)
            except KeyError:
                continue
            if task_a not in requested_tasks or task_b not in requested_tasks or not pa.exists() or not pb.exists():
                continue
            va, _ = filter_valid_prediction_rows(read_csv(pa))
            vb, _ = filter_valid_prediction_rows(read_csv(pb))
            common_ids = {str(r["respondent_id"]) for r in va} & {
                str(r["respondent_id"]) for r in vb
            }
            _, rows_a = evaluate_cell(
                dataset, task_a, pa, state_gold(dataset, task_a), args.alpha,
                args.delta_v, args.bootstrap, rng, common_ids
            )
            _, rows_b = evaluate_cell(
                dataset, task_b, pb, state_gold(dataset, task_b), args.alpha,
                args.delta_v, args.bootstrap, rng, common_ids
            )
            map_a = {(r["var_a"], r["var_b"], r["kind"]): r for r in rows_a}
            map_b = {(r["var_a"], r["var_b"], r["kind"]): r for r in rows_b}
            for key in sorted(set(map_a) & set(map_b)):
                ra, rb = map_a[key], map_b[key]
                paired_rows.append({
                    "dataset": dataset,
                    "task_a": TASK_SHORT[task_a],
                    "task_b": TASK_SHORT[task_b],
                    "var_a": key[0],
                    "var_b": key[1],
                    "kind": key[2],
                    "common_respondent_n": len(common_ids),
                    "abs_diff_a": ra.get("abs_diff"),
                    "abs_diff_b": rb.get("abs_diff"),
                    "abs_diff_improvement_a_minus_b": (
                        ra["abs_diff"] - rb["abs_diff"]
                        if ra.get("abs_diff") is not None and rb.get("abs_diff") is not None else None
                    ),
                })

    payload = {
        "model_slug": args.model_slug,
        "alpha": args.alpha,
        "delta_v": args.delta_v,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "definition": {
            "cat_cat": "Cramer's V; pass if |V_H-V_L| <= delta_v",
            "num_num": "Pearson r; paired ego bootstrap CI for r_L-r_H",
            "cat_num": "eta-squared; pass if |η²_H-η²_L| <= delta_v",
        },
        "cells": all_cells,
        "pairs": all_pairs,
        "paired_common_sample": paired_rows,
    }
    json_path = out_dir / "eval2_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    pair_cols = [
        "dataset",
        "task",
        "var_a",
        "var_b",
        "kind",
        "stat_name",
        "n_human",
        "n_llm",
        "stat_human",
        "stat_llm",
        "signed_diff",
        "abs_diff",
        "pvalue",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_successful",
        "ci_crosses_zero",
        "ci_within_equivalence_threshold",
        "pass",
    ]
    with (out_dir / "eval2_pair_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pair_cols, extrasaction="ignore")
        w.writeheader()
        for row in all_pairs:
            w.writerow({c: row.get(c) for c in pair_cols})

    cell_cols = [
        "dataset",
        "task",
        "raw_n",
        "valid_n",
        "invalid_n",
        "validity_criterion",
        "analysis_n",
        "n_pairs",
        "n_tested",
        "n_pass",
        "pass_rate",
        "mean_abs_diff",
        "alpha",
        "delta_v",
    ]
    with (out_dir / "eval2_cell_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cell_cols, extrasaction="ignore")
        w.writeheader()
        for row in all_cells:
            w.writerow({c: row.get(c) for c in cell_cols})

    paired_path = out_dir / "eval2_paired_common_sample.csv"
    paired_cols = [
        "dataset", "task_a", "task_b", "var_a", "var_b", "kind",
        "common_respondent_n", "abs_diff_a", "abs_diff_b",
        "abs_diff_improvement_a_minus_b",
    ]
    with paired_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=paired_cols)
        w.writeheader()
        w.writerows(paired_rows)

    print(f"\nWrote {json_path}")
    print(f"Wrote {paired_path}")
    return 0


if __name__ == "__main__":
    # Late import helpers used above — ensure extract_human_flat is exported from eval1
    raise SystemExit(main())
