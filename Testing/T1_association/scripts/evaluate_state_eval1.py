#!/usr/bin/env python3
"""Phase 1 Eval 1: univariate distributional realism.

Compares LLM-generated state marginals to human target marginals:

  - categorical / array (pipe-joined): chi-square homogeneity + TV / entropy
  - numeric: two-sample K-S + mean/variance ratios + TV on binned values
  - individual accuracy: exact_match; chance-adjusted: Cohen's kappa
    (p_o=exact_match, p_e=sum_k P(k)Q(k) from the two marginals)

Pass rule (default alpha=0.05):
  - categorical: chi-square p >= alpha
  - numeric: K-S p >= alpha

Usage:
  python Testing/scripts/evaluate_state_eval1.py \\
      --model-slug llama3.1_latest
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
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from paths import (
    ROOT,
    STATE_TASKS as TASKS,
    STATE_TASK_SHORT as TASK_SHORT,
    arm_predictions_csv,
    state_eval_dir,
    state_gold,
)
from state_content_config import ContentConfig, DATASET_CONFIGS
from state_prediction_csv import META_COLUMNS, coerce_scalar, filter_valid_prediction_rows

DEFAULT_OUTPUT_DIR = state_eval_dir()
DEFAULT_CLEAN_DIR = ROOT / "Survey" / "ucnets" / "cleaned"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def entropy(counts: Counter[str]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / n
            h -= p * math.log(p, 2)
    return h


def total_variation(c1: Counter[str], c2: Counter[str]) -> float:
    keys = set(c1) | set(c2)
    n1 = sum(c1.values()) or 1
    n2 = sum(c2.values()) or 1
    return 0.5 * sum(abs(c1.get(k, 0) / n1 - c2.get(k, 0) / n2) for k in keys)


def js_divergence(c1: Counter[str], c2: Counter[str]) -> float:
    keys = set(c1) | set(c2)
    n1 = sum(c1.values()) or 1
    n2 = sum(c2.values()) or 1
    p = {k: c1.get(k, 0) / n1 for k in keys}
    q = {k: c2.get(k, 0) / n2 for k in keys}
    m = {k: 0.5 * (p[k] + q[k]) for k in keys}

    def kl(a: dict[str, float], b: dict[str, float]) -> float:
        s = 0.0
        for k in keys:
            if a[k] > 0 and b[k] > 0:
                s += a[k] * math.log(a[k] / b[k], 2)
        return s

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def mode_share(counts: Counter[str]) -> float:
    n = sum(counts.values()) or 1
    return max(counts.values()) / n if counts else 0.0


def cohen_kappa(human_vals: list[str], llm_vals: list[str]) -> float | None:
    """Agreement beyond chance implied by the two marginals P and Q.

    p_o = exact-match rate (individual accuracy)
    p_e = sum_k P(k) Q(k)  (chance agreement if labels drawn independently
          from the observed human / LLM marginals)
    kappa = (p_o - p_e) / (1 - p_e)

    Interprets individual accuracy *in the context of* the overall
    distributions: high exact_match can still yield kappa ≈ 0 when both
    sides collapse onto the same majority class.
    """
    n = len(human_vals)
    if n == 0 or len(llm_vals) != n:
        return None
    p_o = sum(a == b for a, b in zip(human_vals, llm_vals)) / n
    hc = Counter(human_vals)
    lc = Counter(llm_vals)
    keys = set(hc) | set(lc)
    p_e = sum((hc.get(k, 0) / n) * (lc.get(k, 0) / n) for k in keys)
    if p_e >= 1.0 - 1e-15:
        return None
    return float((p_o - p_e) / (1.0 - p_e))


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        parts = sorted(str(coerce_scalar(x)).strip().lower() for x in value if x is not None and str(x).strip() != "")
        return "|".join(parts) if parts else None
    text = str(coerce_scalar(value)).strip()
    if text == "":
        return None
    return text.lower()


def extract_human_flat(target_json: dict[str, Any], field: str) -> Any:
    if field in target_json:
        return target_json[field]
    if field.startswith("personality_"):
        sub = field[len("personality_") :]
        pers = target_json.get("personality")
        if isinstance(pers, dict):
            return pers.get(sub)
    return None


def field_specs(cfg: ContentConfig, task: str) -> dict[str, dict[str, Any]]:
    """Map flattened prediction column -> {kind, parent}."""
    specs: dict[str, dict[str, Any]] = {}
    for key in cfg.output_keys_for_task(task):
        var = cfg.output_variables[key]
        vtype = var.get("type", "static")
        allowed = var.get("allowed")
        if vtype == "nested" and isinstance(allowed, dict):
            for sub in allowed:
                specs[f"{key}_{sub}"] = {"kind": "categorical", "parent": key, "source": "nested"}
        elif vtype == "array":
            specs[key] = {"kind": "categorical", "parent": key, "source": "array"}
        elif isinstance(allowed, dict) and allowed.get("type") == "numeric":
            specs[key] = {"kind": "numeric", "parent": key, "source": "numeric"}
        else:
            specs[key] = {"kind": "categorical", "parent": key, "source": "static"}
    return specs


def paired_values(
    preds: list[dict[str, str]],
    humans: dict[str, dict[str, Any]],
    field: str,
    kind: str,
) -> tuple[list[Any], list[Any]]:
    human_vals: list[Any] = []
    llm_vals: list[Any] = []
    for p in preds:
        hrow = humans.get(str(p["respondent_id"]))
        if not hrow:
            continue
        hv_raw = extract_human_flat(hrow.get("target_json", {}), field)
        lv_raw = p.get(field, "")
        if lv_raw == "":
            lv_raw = None

        if kind == "numeric":
            if hv_raw is None or hv_raw == "" or lv_raw is None:
                continue
            try:
                hv = float(coerce_scalar(hv_raw))
                lv = float(coerce_scalar(lv_raw))
            except (TypeError, ValueError):
                continue
            human_vals.append(hv)
            llm_vals.append(lv)
        else:
            hv = normalize_label(hv_raw)
            lv = normalize_label(lv_raw)
            if hv is None or lv is None:
                continue
            human_vals.append(hv)
            llm_vals.append(lv)
    return human_vals, llm_vals


def chi_square_homogeneity(
    human_vals: list[str],
    llm_vals: list[str],
) -> dict[str, Any]:
    hc = Counter(human_vals)
    lc = Counter(llm_vals)
    keys = sorted(set(hc) | set(lc))
    if len(keys) < 2:
        return {
            "test": "chi_square",
            "statistic": None,
            "pvalue": None,
            "dof": None,
            "pass": None,
            "note": "fewer than 2 categories",
            "simulated": False,
        }

    table = np.array([[hc.get(k, 0) for k in keys], [lc.get(k, 0) for k in keys]], dtype=int)
    col_keep = table.sum(axis=0) > 0
    table = table[:, col_keep]
    if table.shape[1] < 2 or (table.sum(axis=1) == 0).any():
        return {
            "test": "chi_square",
            "statistic": None,
            "pvalue": None,
            "dof": None,
            "pass": None,
            "note": "degenerate contingency table",
            "simulated": False,
        }

    expected = stats.contingency.expected_freq(table.astype(float))
    sparse = bool((expected < 5).mean() > 0.2) or float(expected.min()) < 1.0
    simulated = False
    note = "sparse_expected_lt_5" if sparse else None
    try:
        if sparse:
            result = stats.chi2_contingency(
                table,
                correction=False,
                method=stats.PermutationMethod(n_resamples=5000),
            )
            chi2 = float(getattr(result, "statistic", result[0]))
            p = float(getattr(result, "pvalue", result[1]))
            dof = int(getattr(result, "dof", result[2]))
            simulated = True
            note = "permutation_p_value"
        else:
            chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
            chi2, p, dof = float(chi2), float(p), int(dof)
    except Exception as exc:
        try:
            chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
            chi2, p, dof = float(chi2), float(p), int(dof)
            note = f"asymptotic_fallback:{type(exc).__name__}"
            simulated = False
        except Exception as exc2:
            return {
                "test": "chi_square",
                "statistic": None,
                "pvalue": None,
                "dof": None,
                "pass": None,
                "note": f"chi-square failed: {exc2}",
                "simulated": False,
            }

    return {
        "test": "chi_square",
        "statistic": chi2,
        "pvalue": p,
        "dof": dof,
        "pass": None,
        "note": note,
        "simulated": simulated,
        "n_categories": int(table.shape[1]),
        "min_expected": float(expected.min()) if expected.size else None,
    }


def ks_two_sample(human_vals: list[float], llm_vals: list[float]) -> dict[str, Any]:
    if len(human_vals) < 5 or len(llm_vals) < 5:
        return {
            "test": "ks",
            "statistic": None,
            "pvalue": None,
            "pass": None,
            "note": "n < 5",
        }
    stat, p = stats.ks_2samp(human_vals, llm_vals, alternative="two-sided", method="auto")
    return {
        "test": "ks",
        "statistic": float(stat),
        "pvalue": float(p),
        "pass": None,
        "note": None,
    }


def evaluate_field(
    human_vals: list[Any],
    llm_vals: list[Any],
    kind: str,
    alpha: float,
) -> dict[str, Any]:
    n = len(human_vals)
    out: dict[str, Any] = {"n": n, "kind": kind}

    if kind == "numeric":
        h = np.asarray(human_vals, dtype=float)
        l = np.asarray(llm_vals, dtype=float)
        mean_h = float(h.mean())
        mean_l = float(l.mean())
        var_h = float(h.var(ddof=0))
        var_l = float(l.var(ddof=0))
        # Discrete numeric: also TV / kappa on integer labels
        hs = [str(int(x)) if float(x).is_integer() else str(x) for x in h]
        ls = [str(int(x)) if float(x).is_integer() else str(x) for x in l]
        hc, lc = Counter(hs), Counter(ls)
        eh, el = entropy(hc), entropy(lc)
        out.update(
            {
                "mean_human": mean_h,
                "mean_llm": mean_l,
                "var_human": var_h,
                "var_llm": var_l,
                "var_ratio": (var_l / var_h) if var_h > 0 else None,
                "exact_match": float(np.mean(h == l)),
                "cohen_kappa": cohen_kappa(hs, ls),
                "chance_agreement": sum(
                    (hc.get(k, 0) / n) * (lc.get(k, 0) / n) for k in set(hc) | set(lc)
                )
                if n
                else None,
                "entropy_human": eh,
                "entropy_llm": el,
                "entropy_ratio": (el / eh) if eh > 0 else None,
                "tv": total_variation(hc, lc),
                "js": js_divergence(hc, lc),
                "mode_share_human": mode_share(hc),
                "mode_share_llm": mode_share(lc),
            }
        )
        test = ks_two_sample(list(h), list(l))
    else:
        hs = [str(x) for x in human_vals]
        ls = [str(x) for x in llm_vals]
        hc, lc = Counter(hs), Counter(ls)
        eh, el = entropy(hc), entropy(lc)
        n_safe = n or 1
        out.update(
            {
                "exact_match": sum(a == b for a, b in zip(hs, ls)) / n if n else None,
                "cohen_kappa": cohen_kappa(hs, ls),
                "chance_agreement": sum(
                    (hc.get(k, 0) / n_safe) * (lc.get(k, 0) / n_safe) for k in set(hc) | set(lc)
                )
                if n
                else None,
                "entropy_human": eh,
                "entropy_llm": el,
                "entropy_ratio": (el / eh) if eh > 0 else None,
                "tv": total_variation(hc, lc),
                "js": js_divergence(hc, lc),
                "mode_share_human": mode_share(hc),
                "mode_share_llm": mode_share(lc),
                "n_unique_human": len(hc),
                "n_unique_llm": len(lc),
            }
        )
        test = chi_square_homogeneity(hs, ls)

    if test.get("pvalue") is not None:
        test["pass"] = bool(test["pvalue"] >= alpha)
    out["test"] = test
    out["pass"] = test.get("pass")
    return out


def summarize_cell(field_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tested = [r for r in field_rows if r.get("pass") is not None]
    passed = [r for r in tested if r["pass"]]
    tvs = [r["tv"] for r in field_rows if r.get("tv") is not None]
    jss = [r["js"] for r in field_rows if r.get("js") is not None]
    ers = [r["entropy_ratio"] for r in field_rows if r.get("entropy_ratio") is not None]
    ems = [r["exact_match"] for r in field_rows if r.get("exact_match") is not None]
    kappas = [r["cohen_kappa"] for r in field_rows if r.get("cohen_kappa") is not None]
    pe = [r["chance_agreement"] for r in field_rows if r.get("chance_agreement") is not None]
    return {
        "n_fields": len(field_rows),
        "n_tested": len(tested),
        "n_pass": len(passed),
        "pass_rate": (len(passed) / len(tested)) if tested else None,
        "mean_tv": float(np.mean(tvs)) if tvs else None,
        "median_tv": float(np.median(tvs)) if tvs else None,
        "mean_js": float(np.mean(jss)) if jss else None,
        "mean_entropy_ratio": float(np.mean(ers)) if ers else None,
        "mean_exact_match": float(np.mean(ems)) if ems else None,
        "mean_cohen_kappa": float(np.mean(kappas)) if kappas else None,
        "mean_chance_agreement": float(np.mean(pe)) if pe else None,
        "fields_tv_gt_0_3": sum(1 for t in tvs if t > 0.3),
        "fields_entropy_lt_0_7": sum(1 for e in ers if e < 0.7),
        "fields_kappa_le_0": sum(1 for k in kappas if k <= 0),
    }


def evaluate_cell(
    dataset: str,
    task: str,
    pred_path: Path,
    clean_path: Path,
    alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = ContentConfig.load(dataset)
    specs = field_specs(cfg, task)
    raw_preds = read_csv(pred_path)
    preds, validity = filter_valid_prediction_rows(raw_preds)
    humans = {str(r["respondent_id"]): r for r in read_jsonl(clean_path)}

    field_rows: list[dict[str, Any]] = []

    # Prefer columns present in predictions that we know how to score.
    pred_fields = [k for k in preds[0].keys() if k not in META_COLUMNS] if preds else []
    for field in pred_fields:
        spec = specs.get(field)
        if spec is None:
            # Personality / unexpected flattened cols: treat as categorical if values look non-numeric.
            spec = {"kind": "categorical", "parent": field, "source": "inferred"}
        hv, lv = paired_values(preds, humans, field, spec["kind"])
        if len(hv) < 10:
            continue
        metrics = evaluate_field(hv, lv, spec["kind"], alpha)
        row = {
            "dataset": dataset,
            "task": TASK_SHORT[task],
            "task_full": task,
            "field": field,
            "parent": spec["parent"],
            "source": spec["source"],
            **metrics,
        }
        # Flatten test keys for CSV
        test = metrics["test"]
        row["test_name"] = test.get("test")
        row["statistic"] = test.get("statistic")
        row["pvalue"] = test.get("pvalue")
        row["test_note"] = test.get("note")
        field_rows.append(row)

    cell = {
        "dataset": dataset,
        "task": TASK_SHORT[task],
        "task_full": task,
        "n_pred": validity["raw_n"],
        **validity,
        "schema_ok_rate": (
            validity["valid_n"] / validity["raw_n"] if validity["raw_n"] else None
        ),
        "alpha": alpha,
        **summarize_cell(field_rows),
    }
    return cell, field_rows


def compare_tasks_common_sample(
    dataset: str,
    task_a: str,
    task_b: str,
    model_slug: str,
    alpha: float,
) -> list[dict[str, Any]]:
    """Recompute field metrics on respondents valid in both task arms."""
    raw_a = read_csv(arm_predictions_csv(dataset, task_a, model_slug))
    raw_b = read_csv(arm_predictions_csv(dataset, task_b, model_slug))
    preds_a, _ = filter_valid_prediction_rows(raw_a)
    preds_b, _ = filter_valid_prediction_rows(raw_b)
    by_a = {str(r["respondent_id"]): r for r in preds_a}
    by_b = {str(r["respondent_id"]): r for r in preds_b}
    common_ids = sorted(set(by_a) & set(by_b))
    if not common_ids:
        return []
    common_a = [by_a[rid] for rid in common_ids]
    common_b = [by_b[rid] for rid in common_ids]
    humans_a = {str(r["respondent_id"]): r for r in read_jsonl(state_gold(dataset, task_a))}
    humans_b = {str(r["respondent_id"]): r for r in read_jsonl(state_gold(dataset, task_b))}
    cfg = ContentConfig.load(dataset)
    specs_a = field_specs(cfg, task_a)
    specs_b = field_specs(cfg, task_b)
    fields_a = set(common_a[0]) - set(META_COLUMNS)
    fields_b = set(common_b[0]) - set(META_COLUMNS)
    rows: list[dict[str, Any]] = []
    for field in sorted(fields_a & fields_b):
        spec = specs_a.get(field) or specs_b.get(field)
        if spec is None:
            spec = {"kind": "categorical"}
        kind = spec["kind"]
        ha: list[Any] = []
        la: list[Any] = []
        hb: list[Any] = []
        lb: list[Any] = []
        for row_a, row_b in zip(common_a, common_b):
            one_ha, one_la = paired_values([row_a], humans_a, field, kind)
            one_hb, one_lb = paired_values([row_b], humans_b, field, kind)
            if not one_ha or not one_hb:
                continue
            ha.append(one_ha[0])
            la.append(one_la[0])
            hb.append(one_hb[0])
            lb.append(one_lb[0])
        if len(ha) < 10 or len(hb) < 10:
            continue
        ma = evaluate_field(ha, la, kind, alpha)
        mb = evaluate_field(hb, lb, kind, alpha)
        rows.append(
            {
                "dataset": dataset,
                "task_a": TASK_SHORT[task_a],
                "task_b": TASK_SHORT[task_b],
                "field": field,
                "kind": kind,
                "common_respondent_n": len(common_ids),
                "paired_field_n_a": len(ha),
                "paired_field_n_b": len(hb),
                "tv_a": ma.get("tv"),
                "tv_b": mb.get("tv"),
                "tv_improvement_a_minus_b": ma.get("tv") - mb.get("tv"),
                "exact_match_a": ma.get("exact_match"),
                "exact_match_b": mb.get("exact_match"),
                "exact_match_improvement_b_minus_a": (
                    mb.get("exact_match") - ma.get("exact_match")
                ),
                "kappa_a": ma.get("cohen_kappa"),
                "kappa_b": mb.get("cohen_kappa"),
                "kappa_improvement_b_minus_a": (
                    mb.get("cohen_kappa") - ma.get("cohen_kappa")
                    if ma.get("cohen_kappa") is not None and mb.get("cohen_kappa") is not None
                    else None
                ),
            }
        )
    return rows


def write_field_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "dataset",
        "task",
        "field",
        "kind",
        "source",
        "n",
        "test_name",
        "statistic",
        "pvalue",
        "pass",
        "tv",
        "js",
        "entropy_ratio",
        "exact_match",
        "chance_agreement",
        "cohen_kappa",
        "mode_share_human",
        "mode_share_llm",
        "mean_human",
        "mean_llm",
        "var_ratio",
        "test_note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {c: row.get(c) for c in cols}
            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-slug", default="llama3.1_latest")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    args = parser.parse_args()

    out_dir = args.output_dir / "eval1_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cells: list[dict[str, Any]] = []
    all_fields: list[dict[str, Any]] = []

    for dataset in args.datasets:
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
            cell, fields = evaluate_cell(dataset, task, pred_path, clean_path, args.alpha)
            all_cells.append(cell)
            all_fields.extend(fields)
            pr = cell["pass_rate"]
            pr_s = f"{pr:.3f}" if pr is not None else "NA"
            mk = cell.get("mean_cohen_kappa")
            em = cell.get("mean_exact_match")
            mk_s = f"{mk:.3f}" if mk is not None else "NA"
            em_s = f"{em:.3f}" if em is not None else "NA"
            print(
                f"{dataset:10} {short:6} fields={cell['n_fields']:2d} "
                f"pass={cell['n_pass']}/{cell['n_tested']} ({pr_s}) "
                f"mean_tv={cell['mean_tv']:.3f} er={cell['mean_entropy_ratio']:.3f} "
                f"em={em_s} kappa={mk_s}"
            )

    paired_rows: list[dict[str, Any]] = []
    requested_tasks = set(args.tasks)
    for dataset in args.datasets:
        for task_b in ("s1_health_conditioned", "s2_politics_religion_conditioned"):
            task_a = "s0_baseline_state"
            if task_a not in requested_tasks or task_b not in requested_tasks:
                continue
            try:
                path_a = arm_predictions_csv(dataset, task_a, args.model_slug)
                path_b = arm_predictions_csv(dataset, task_b, args.model_slug)
            except KeyError:
                continue
            if not path_a.exists():
                continue
            if not path_b.exists():
                continue
            paired_rows.extend(
                compare_tasks_common_sample(dataset, task_a, task_b, args.model_slug, args.alpha)
            )

    payload = {
        "model_slug": args.model_slug,
        "alpha": args.alpha,
        "definition": {
            "categorical_test": "chi-square test of homogeneity (2 x K: human vs LLM)",
            "numeric_test": "two-sample Kolmogorov-Smirnov",
            "pass_rule": f"p >= {args.alpha}",
            "diagnostics": [
                "tv",
                "js",
                "entropy_ratio",
                "mode_share",
                "var_ratio",
                "exact_match",
                "chance_agreement",
                "cohen_kappa",
            ],
            "cohen_kappa": (
                "kappa=(p_o-p_e)/(1-p_e); p_o=exact_match; "
                "p_e=sum_k P(k)Q(k) from human/LLM marginals"
            ),
        },
        "cells": all_cells,
        "fields": all_fields,
        "paired_common_sample": paired_rows,
    }

    json_path = out_dir / "eval1_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    write_field_csv(out_dir / "eval1_field_metrics.csv", all_fields)

    cell_csv = out_dir / "eval1_cell_summary.csv"
    cell_cols = [
        "dataset",
        "task",
        "n_pred",
        "raw_n",
        "valid_n",
        "invalid_n",
        "validity_criterion",
        "schema_ok_rate",
        "n_fields",
        "n_tested",
        "n_pass",
        "pass_rate",
        "mean_tv",
        "median_tv",
        "mean_js",
        "mean_entropy_ratio",
        "mean_exact_match",
        "mean_chance_agreement",
        "mean_cohen_kappa",
        "fields_tv_gt_0_3",
        "fields_entropy_lt_0_7",
        "fields_kappa_le_0",
        "alpha",
    ]
    with cell_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cell_cols, extrasaction="ignore")
        writer.writeheader()
        for row in all_cells:
            writer.writerow({c: row.get(c) for c in cell_cols})

    paired_path = out_dir / "eval1_paired_common_sample.csv"
    paired_cols = [
        "dataset",
        "task_a",
        "task_b",
        "field",
        "kind",
        "common_respondent_n",
        "paired_field_n_a",
        "paired_field_n_b",
        "tv_a",
        "tv_b",
        "tv_improvement_a_minus_b",
        "exact_match_a",
        "exact_match_b",
        "exact_match_improvement_b_minus_a",
        "kappa_a",
        "kappa_b",
        "kappa_improvement_b_minus_a",
    ]
    with paired_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=paired_cols, extrasaction="ignore")
        writer.writeheader()
        for row in paired_rows:
            writer.writerow({c: row.get(c) for c in paired_cols})

    print(f"\nWrote {json_path}")
    print(f"Wrote {out_dir / 'eval1_field_metrics.csv'}")
    print(f"Wrote {cell_csv}")
    print(f"Wrote {paired_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
