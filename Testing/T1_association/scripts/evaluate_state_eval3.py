#!/usr/bin/env python3
"""Phase 1 Eval 3: multivariate prediction realism.

Fits the same background-anchor design matrix to human vs LLM outcomes and
compares predictive strength:

  - numeric outcomes: OLS R^2
  - categorical outcomes: binary (modal class vs rest) logistic McFadden
    pseudo-R^2 via L-BFGS (numpy/scipy only)

Human and LLM fits always use the same egos and design rows. Ego-paired
bootstrap confidence intervals are reported for metric_L - metric_H.
Pass if |metric_H - metric_L| <= delta_r2 (default 0.05).

Usage:
  python Testing/scripts/evaluate_state_eval3.py --model-slug llama3.1_latest
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_COMMON = _HERE.parents[1] / "common"
for _p in (_COMMON, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

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


_GSS_ANCHORS = ["age_group", "sex", "degree", "marital_status", "region"]
_GSS_OUTCOMES: list[tuple[str, str]] = [
    ("general_happiness", "cat"),
    ("people_can_be_trusted", "cat"),
    ("self_rated_health", "cat"),
    ("party_identity", "cat"),
    ("political_ideology", "cat"),
    ("religion", "cat"),
    ("religious_service_attendance", "cat"),
    ("people_are_helpful", "cat"),
]

ANCHORS: dict[str, list[str]] = {
    "gss_1987": list(_GSS_ANCHORS),
    "gss_2010_network": list(_GSS_ANCHORS),
    "ucnets_w1": ["age_group", "gender", "education", "marital_status", "race_ethnicity"],
}

OUTCOMES: dict[str, list[tuple[str, str]]] = {
    # (field, kind) kind in {cat, num}
    "gss_1987": list(_GSS_OUTCOMES),
    "gss_2010_network": list(_GSS_OUTCOMES),
    "ucnets_w1": [
        ("self_rated_health", "cat"),
        ("days_lonely_past_week", "num"),
        ("days_isolated_past_week", "num"),
        ("days_very_happy_past_week", "num"),
        ("relatives_help_serious_problem", "cat"),
        ("friends_help_serious_problem", "cat"),
        ("voting_frequency", "cat"),
        ("party_identity", "cat"),
    ],
}


def one_hot(rows: list[dict[str, str]], keys: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build intercept + one-hot design matrix from categorical anchor dicts."""
    level_maps: dict[str, list[str]] = {}
    for key in keys:
        levels = sorted({str(r.get(key, "")) for r in rows if r.get(key) not in (None, "")})
        level_maps[key] = levels
    colnames = ["intercept"]
    for key, levels in level_maps.items():
        for lev in levels[1:]:  # drop first for identifiability
            colnames.append(f"{key}={lev}")
    X = np.zeros((len(rows), len(colnames)), dtype=float)
    X[:, 0] = 1.0
    col = 1
    for key, levels in level_maps.items():
        for lev in levels[1:]:
            for i, r in enumerate(rows):
                if str(r.get(key, "")) == lev:
                    X[i, col] = 1.0
            col += 1
    return X, colnames


def ols_r2(X: np.ndarray, y: np.ndarray) -> float | None:
    if len(y) < 30 or X.shape[1] < 2:
        return None
    if float(np.var(y)) <= 0:
        return 0.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def logistic_mcfadden(X: np.ndarray, y: np.ndarray) -> float | None:
    """McFadden pseudo-R^2 for binary y in {0,1}."""
    y = y.astype(float)
    if len(y) < 30 or y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None

    def nll(beta: np.ndarray) -> float:
        z = X @ beta
        # stable log-likelihood
        return float(-np.sum(y * z - np.logaddexp(0.0, z)))

    def grad(beta: np.ndarray) -> np.ndarray:
        p = expit(X @ beta)
        return X.T @ (p - y)

    x0 = np.zeros(X.shape[1], dtype=float)
    res = minimize(nll, x0, jac=grad, method="L-BFGS-B")
    if not res.success and res.nit < 5:
        return None
    ll_model = -float(res.fun)
    p0 = float(y.mean())
    ll_null = float(np.sum(y * np.log(p0 + 1e-12) + (1 - y) * np.log(1 - p0 + 1e-12)))
    if abs(ll_null) < 1e-12:
        return None
    return float(1.0 - ll_model / ll_null)


def build_aligned_tables(
    preds: list[dict[str, str]],
    humans: dict[str, dict[str, Any]],
    anchors: list[str],
    outcome: str,
    kind: str,
) -> tuple[list[dict[str, str]], list[Any], list[Any]]:
    anchor_rows: list[dict[str, str]] = []
    y_h: list[Any] = []
    y_l: list[Any] = []
    for p in preds:
        h = humans.get(str(p["respondent_id"]))
        if not h:
            continue
        prompt = h.get("prompt_json") or {}
        target = h.get("target_json") or {}
        # anchors must be present
        arow = {}
        ok = True
        for a in anchors:
            v = prompt.get(a)
            if v is None or v == "":
                ok = False
                break
            arow[a] = str(v)
        if not ok:
            continue
        hv = extract_human_flat(target, outcome)
        lv = p.get(outcome, "")
        if lv == "":
            lv = None
        if hv is None or hv == "" or lv is None:
            continue
        if kind == "num":
            try:
                hv = float(coerce_scalar(hv))
                lv = float(coerce_scalar(lv))
            except (TypeError, ValueError):
                continue
        else:
            hv = normalize_label(hv)
            lv = normalize_label(lv)
            if hv is None or lv is None:
                continue
        anchor_rows.append(arow)
        y_h.append(hv)
        y_l.append(lv)
    return anchor_rows, y_h, y_l


def evaluate_outcome(
    anchor_rows: list[dict[str, str]],
    y_h: list[Any],
    y_l: list[Any],
    kind: str,
    delta: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    X, _ = one_hot(anchor_rows, list(anchor_rows[0].keys()) if anchor_rows else [])
    out: dict[str, Any] = {"n": len(anchor_rows), "kind": kind}
    if kind == "num":
        mh = ols_r2(X, np.asarray(y_h, dtype=float))
        ml = ols_r2(X, np.asarray(y_l, dtype=float))
        out.update({"metric_name": "r2", "metric_human": mh, "metric_llm": ml})
    else:
        # modal class vs rest on HUMAN distribution for comparable coding
        from collections import Counter

        mode = Counter(y_h).most_common(1)[0][0]
        yh = np.asarray([1.0 if v == mode else 0.0 for v in y_h])
        yl = np.asarray([1.0 if v == mode else 0.0 for v in y_l])
        mh = logistic_mcfadden(X, yh)
        ml = logistic_mcfadden(X, yl)
        out.update(
            {
                "metric_name": "mcfadden_pseudo_r2",
                "metric_human": mh,
                "metric_llm": ml,
                "binary_positive_class": mode,
            }
        )
    if mh is None or ml is None:
        out["pass"] = None
        out["abs_diff"] = None
        out["signed_diff"] = None
    else:
        signed_diff = ml - mh
        out["signed_diff"] = signed_diff
        out["abs_diff"] = abs(signed_diff)
        out["pass"] = bool(abs(signed_diff) <= delta)
        diffs: list[float] = []
        n = len(anchor_rows)
        for _ in range(max(0, n_bootstrap)):
            idx = rng.integers(0, n, size=n)
            if kind == "num":
                bh = ols_r2(X[idx], np.asarray(y_h, dtype=float)[idx])
                bl = ols_r2(X[idx], np.asarray(y_l, dtype=float)[idx])
            else:
                bh = logistic_mcfadden(X[idx], yh[idx])
                bl = logistic_mcfadden(X[idx], yl[idx])
            if bh is not None and bl is not None and np.isfinite(bh) and np.isfinite(bl):
                diffs.append(float(bl - bh))
        if diffs:
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            out.update(
                {
                    "bootstrap_ci_low": float(lo),
                    "bootstrap_ci_high": float(hi),
                    "bootstrap_successful": len(diffs),
                    "ci_crosses_zero": bool(lo <= 0 <= hi),
                    "ci_within_equivalence_threshold": bool(lo >= -delta and hi <= delta),
                }
            )
        else:
            out.update(
                {
                    "bootstrap_ci_low": None,
                    "bootstrap_ci_high": None,
                    "bootstrap_successful": 0,
                    "ci_crosses_zero": None,
                    "ci_within_equivalence_threshold": None,
                }
            )
    return out


def evaluate_cell(
    dataset: str,
    task: str,
    pred_path: Path,
    clean_path: Path,
    delta: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    respondent_ids: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_preds = read_csv(pred_path)
    preds, validity = filter_valid_prediction_rows(raw_preds)
    if respondent_ids is not None:
        preds = [p for p in preds if str(p["respondent_id"]) in respondent_ids]
    humans = {str(r["respondent_id"]): r for r in read_jsonl(clean_path)}
    available = set(preds[0].keys()) - set(META_COLUMNS) if preds else set()
    anchors = ANCHORS[dataset]
    rows: list[dict[str, Any]] = []
    for outcome, kind in OUTCOMES.get(dataset, []):
        if outcome not in available:
            continue
        anchor_rows, y_h, y_l = build_aligned_tables(preds, humans, anchors, outcome, kind)
        if len(anchor_rows) < 30:
            continue
        metrics = evaluate_outcome(
            anchor_rows, y_h, y_l, kind, delta, n_bootstrap, rng
        )
        rows.append(
            {
                "dataset": dataset,
                "task": TASK_SHORT[task],
                "outcome": outcome,
                "anchors": ",".join(anchors),
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
        "n_outcomes": len(rows),
        "n_tested": len(tested),
        "n_pass": len(passed),
        "pass_rate": (len(passed) / len(tested)) if tested else None,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
        "delta_r2": delta,
    }
    return cell, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-slug", default="llama3.1_latest")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delta-r2", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    args = parser.parse_args()

    out_dir = args.output_dir / "eval3_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cells, all_rows = [], []
    rng = np.random.default_rng(args.seed)
    for dataset in args.datasets:
        if dataset not in ANCHORS:
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
            cell, rows = evaluate_cell(
                dataset,
                task,
                pred_path,
                clean_path,
                args.delta_r2,
                args.bootstrap,
                rng,
            )
            all_cells.append(cell)
            all_rows.extend(rows)
            pr = cell["pass_rate"]
            pr_s = f"{pr:.3f}" if pr is not None else "NA"
            print(
                f"{dataset:10} {short:6} outcomes={cell['n_outcomes']} "
                f"pass={cell['n_pass']}/{cell['n_tested']} ({pr_s}) "
                f"mean|ΔR²|={cell['mean_abs_diff']}"
            )

    paired_rows: list[dict[str, Any]] = []
    requested_tasks = set(args.tasks)
    for dataset in args.datasets:
        if dataset not in ANCHORS:
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
                dataset, task_a, pa, state_gold(dataset, task_a),
                args.delta_r2, args.bootstrap, rng, common_ids
            )
            _, rows_b = evaluate_cell(
                dataset, task_b, pb, state_gold(dataset, task_b),
                args.delta_r2, args.bootstrap, rng, common_ids
            )
            map_a = {r["outcome"]: r for r in rows_a}
            map_b = {r["outcome"]: r for r in rows_b}
            for outcome in sorted(set(map_a) & set(map_b)):
                ra, rb = map_a[outcome], map_b[outcome]
                paired_rows.append({
                    "dataset": dataset,
                    "task_a": TASK_SHORT[task_a],
                    "task_b": TASK_SHORT[task_b],
                    "outcome": outcome,
                    "kind": ra["kind"],
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
        "delta_r2": args.delta_r2,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "definition": {
            "numeric": "OLS R^2 of outcome on one-hot anchors",
            "categorical": "McFadden pseudo-R^2 for (human-modal class vs rest) logistic",
            "pass_rule": f"|metric_H - metric_L| <= {args.delta_r2}",
            "anchors": ANCHORS,
        },
        "cells": all_cells,
        "outcomes": all_rows,
        "paired_common_sample": paired_rows,
    }
    json_path = out_dir / "eval3_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    cols = [
        "dataset",
        "task",
        "outcome",
        "kind",
        "metric_name",
        "n",
        "metric_human",
        "metric_llm",
        "signed_diff",
        "abs_diff",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_successful",
        "ci_crosses_zero",
        "ci_within_equivalence_threshold",
        "pass",
        "binary_positive_class",
        "anchors",
    ]
    with (out_dir / "eval3_outcome_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow({c: row.get(c) for c in cols})

    cell_cols = [
        "dataset",
        "task",
        "raw_n",
        "valid_n",
        "invalid_n",
        "validity_criterion",
        "analysis_n",
        "n_outcomes",
        "n_tested",
        "n_pass",
        "pass_rate",
        "mean_abs_diff",
        "delta_r2",
    ]
    with (out_dir / "eval3_cell_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cell_cols, extrasaction="ignore")
        w.writeheader()
        for row in all_cells:
            w.writerow({c: row.get(c) for c in cell_cols})

    paired_path = out_dir / "eval3_paired_common_sample.csv"
    paired_cols = [
        "dataset", "task_a", "task_b", "outcome", "kind",
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
    raise SystemExit(main())
