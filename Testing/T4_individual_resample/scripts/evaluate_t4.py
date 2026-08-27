#!/usr/bin/env python3
"""T4 evaluation: R0 (N|S_H, equals N0) vs R1 (N|S_L) vs N_H.

Default dataset is UCNets Wave 1. Also supports gss_1985 and nshap_r1:

  python Testing/scripts/evaluate_t4.py
  python Testing/scripts/evaluate_t4.py --dataset gss_1985
  python Testing/scripts/evaluate_t4.py --dataset nshap_r1

Writes under Testing/T4_individual_resample/{dataset}/eval/:
  sr1_summary.json, sr1_field_attribution.csv, sr1_tables.md

UCNets curated narrative (do not overwrite):
  Testing/T4_individual_resample/ucnets_w1/eval/T4_Evaluation_Report.md
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


from network_metrics import roster_of
from network_schema import load_config, roster_spec
from paths import arm_input, arm_output_jsonl, network_gold, t4_eval_dir

SHARE_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def last_by_id(rows: list[dict[str, Any]], key: str = "respondent_id") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        rid = str(r.get(key) or "")
        if rid:
            out[rid] = r
    return out


def schema_ok(rec: dict[str, Any] | None) -> bool:
    if not rec or rec.get("error"):
        return False
    pred = rec.get("prediction_json")
    if not isinstance(pred, dict):
        return False
    return bool((rec.get("validation") or {}).get("schema_ok"))


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


def bin_vals(vals: list[float], edges: list[float]) -> Counter[str]:
    labels: list[str] = []
    for v in vals:
        placed = False
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if (lo <= v < hi) or (i == len(edges) - 2 and v == hi):
                labels.append(f"[{lo},{hi})")
                placed = True
                break
        if not placed:
            labels.append("other")
    return Counter(labels)


def size_edges(cfg: dict[str, Any]) -> list[float]:
    bounds = (cfg.get("evaluation") or {}).get("size_bucket_bounds") or {}
    if not bounds:
        return [1, 6, 16, 31]
    cuts: list[float] = []
    for name in ("small", "medium", "large"):
        pair = bounds.get(name)
        if not pair:
            continue
        lo, hi = float(pair[0]), float(pair[1])
        if not cuts:
            cuts.append(lo)
        cuts.append(hi + 1.0)
    return cuts or [1, 6, 16, 31]


def pred_net(rec: dict[str, Any]) -> dict[str, Any]:
    return rec.get("prediction_json") or {}


def gold_net(g: dict[str, Any]) -> dict[str, Any]:
    return g.get("target_json") or {}


def summary_of(net: dict[str, Any]) -> dict[str, Any]:
    return net.get("composition_summary") or {}


def field_dist_metrics(
    human_vals: list[Any],
    llm_vals: list[Any],
    *,
    continuous: bool,
    edges: list[float] | None = None,
) -> dict[str, Any]:
    if continuous:
        hv = [float(x) for x in human_vals if x is not None]
        lv = [float(x) for x in llm_vals if x is not None]
        e = edges or SHARE_EDGES
        ch, cl = bin_vals(hv, e), bin_vals(lv, e)
        mean_h = float(sum(hv) / len(hv)) if hv else None
        mean_l = float(sum(lv) / len(lv)) if lv else None
        var_h = float(sum((x - mean_h) ** 2 for x in hv) / len(hv)) if hv and mean_h is not None else None
        var_l = float(sum((x - mean_l) ** 2 for x in lv) / len(lv)) if lv and mean_l is not None else None
        vm_h = (var_h / mean_h) if (var_h is not None and mean_h not in (None, 0)) else None
        vm_l = (var_l / mean_l) if (var_l is not None and mean_l not in (None, 0)) else None
    else:
        ch = Counter(str(x) for x in human_vals if x is not None)
        cl = Counter(str(x) for x in llm_vals if x is not None)
        mean_h = mean_l = vm_h = vm_l = None

    eh, el = entropy(ch), entropy(cl)
    return {
        "n_h": sum(ch.values()),
        "n_l": sum(cl.values()),
        "tv": total_variation(ch, cl),
        "entropy_ratio": (el / eh) if eh > 0 else None,
        "mean_h": mean_h,
        "mean_l": mean_l,
        "var_mean_h": vm_h,
        "var_mean_l": vm_l,
    }


def eval_fields(cfg: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    ev = cfg.get("evaluation") or {}
    share = list(ev.get("share_fields") or [])
    count = list(ev.get("count_fields") or [])
    alter_schema = (cfg.get("output_schema") or {}).get("core_alter_fields") or {}
    skip = {"alter_id", "rank", "age", "support_roles"}
    roster_cats = [f for f in alter_schema if f not in skip]
    if not roster_cats:
        roster_cats = ["relationship", "contact_frequency", "gender", "same_race_as_ego"]
    return share, count, roster_cats


def aligned_roster_pairs(
    gold_alters: list[dict[str, Any]],
    pred_alters: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Align alters by the schema identity field, never by list position."""
    identity = str(roster_spec(cfg).get("rank_field") or "alter_id")
    gold_by_id = {
        str(a[identity]): a for a in gold_alters if a.get(identity) is not None
    }
    pred_by_id = {
        str(a[identity]): a for a in pred_alters if a.get(identity) is not None
    }
    return [(gold_by_id[k], pred_by_id[k]) for k in sorted(gold_by_id.keys() & pred_by_id.keys())]


def ego_value(state: dict[str, Any], names: list[str]) -> str | None:
    for name in names:
        if state.get(name) is not None:
            return str(state[name])
    return None


def collect_condition(
    ids: list[str],
    gold: dict[str, dict[str, Any]],
    preds: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    share, count, roster_cats = eval_fields(cfg)
    edges_size = size_edges(cfg)
    out: dict[str, Any] = {
        "n": len(ids),
        "fields": {},
        "core": {},
        "homophily": {},
        "roster_alignment": {
            "identity_field": str(roster_spec(cfg).get("rank_field") or "alter_id"),
            "n_matched_pairs": 0,
        },
    }

    for f in count + share:
        hv, lv = [], []
        for rid in ids:
            hv.append(summary_of(gold_net(gold[rid])).get(f))
            lv.append(summary_of(pred_net(preds[rid])).get(f))
        if f == "num_alters":
            edges = edges_size
        elif f.endswith("_count"):
            edges = list(range(0, 11)) + [100]
        else:
            edges = SHARE_EDGES
        out["fields"][f] = field_dist_metrics(hv, lv, continuous=True, edges=edges)

    hv = [summary_of(gold_net(gold[rid])).get("size_bucket") for rid in ids]
    lv = [summary_of(pred_net(preds[rid])).get("size_bucket") for rid in ids]
    if any(x is not None for x in hv + lv):
        out["fields"]["size_bucket"] = field_dist_metrics(hv, lv, continuous=False)

    for f in roster_cats:
        hv, lv = [], []
        for rid in ids:
            g_alters = roster_of(gold_net(gold[rid]), cfg)
            p_alters = roster_of(pred_net(preds[rid]), cfg)
            pairs = aligned_roster_pairs(g_alters, p_alters, cfg)
            for a, b in pairs:
                hv.append(a.get(f))
                lv.append(b.get(f))
        if any(x is not None for x in hv + lv):
            out["core"][f] = field_dist_metrics(hv, lv, continuous=False)

    out["roster_alignment"]["n_matched_pairs"] = sum(
        len(
            aligned_roster_pairs(
                roster_of(gold_net(gold[rid]), cfg),
                roster_of(pred_net(preds[rid]), cfg),
                cfg,
            )
        )
        for rid in ids
    )

    homophily_specs = []
    if "same_race_as_ego" in roster_cats:
        homophily_specs.append(
            (
                "same_race",
                lambda ego, alter: alter.get("same_race_as_ego") == "yes",
                lambda alter: alter.get("same_race_as_ego") in ("yes", "no"),
            )
        )
    # GSS ego race is 3-category; alter race is 5-category. Do not score same_race there.
    if "gender" in roster_cats:
        homophily_specs.append(
            (
                "same_gender",
                lambda ego, alter: str(alter.get("gender")) == ego_value(ego, ["gender", "sex"]),
                lambda alter: alter.get("gender") in ("male", "female"),
            )
        )

    for label, match_fn, known_fn in homophily_specs:
        h_yes = h_n = l_yes = l_n = 0
        for rid in ids:
            ego = gold[rid].get("state_json") or {}
            for a in roster_of(gold_net(gold[rid]), cfg):
                if not known_fn(a):
                    continue
                h_n += 1
                if match_fn(ego, a):
                    h_yes += 1
            for a in roster_of(pred_net(preds[rid]), cfg):
                if not known_fn(a):
                    continue
                l_n += 1
                if match_fn(ego, a):
                    l_yes += 1
        out["homophily"][label] = {
            "h": (h_yes / h_n) if h_n else None,
            "l": (l_yes / l_n) if l_n else None,
            "n_h": h_n,
            "n_l": l_n,
        }

    tvs = [m["tv"] for m in out["fields"].values() if m.get("tv") is not None]
    out["mean_tv"] = float(sum(tvs) / len(tvs)) if tvs else None
    out["mean_tv_scope"] = "composition_summary_only"
    out["mean_tv_fields"] = list(out["fields"])
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def person_level(
    ids: list[str],
    gold: dict[str, dict[str, Any]],
    r0: dict[str, dict[str, Any]],
    r1: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    share, count, _ = eval_fields(cfg)
    cont = [f for f in count + share]
    fields: dict[str, Any] = {}
    corrs = []
    for f in cont:
        x, y, h, absd = [], [], [], []
        for rid in ids:
            a = summary_of(pred_net(r0[rid])).get(f)
            b = summary_of(pred_net(r1[rid])).get(f)
            hv = summary_of(gold_net(gold[rid])).get(f)
            if a is None or b is None:
                continue
            x.append(float(a))
            y.append(float(b))
            absd.append(abs(float(a) - float(b)))
            if hv is not None:
                h.append(float(hv))
        same_sign = None
        if h and len(h) == len(x):
            n_ss = sum(
                1
                for hv, a, b in zip(h, x, y)
                if (a - hv) * (b - hv) > 0
            )
            same_sign = n_ss / len(h)
        r = pearson(x, y)
        if r is not None:
            corrs.append(r)
        fields[f] = {
            "n": len(x),
            "pearson_r0_r1": r,
            "mean_abs_delta": float(np.mean(absd)) if absd else None,
            "median_abs_delta": float(np.median(absd)) if absd else None,
            "same_sign_error_vs_human": same_sign,
        }

    same_bucket = 0
    n_bucket = 0
    for rid in ids:
        a = summary_of(pred_net(r0[rid])).get("size_bucket")
        b = summary_of(pred_net(r1[rid])).get("size_bucket")
        if a is None or b is None:
            continue
        n_bucket += 1
        if a == b:
            same_bucket += 1

    return {
        "size_bucket_agree": (same_bucket / n_bucket) if n_bucket else None,
        "n_size_bucket": n_bucket,
        "mean_pearson_r0_r1": float(np.mean(corrs)) if corrs else None,
        "fields": fields,
    }


def tv_between_preds(
    ids: list[str],
    a: dict[str, dict[str, Any]],
    b: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, float]:
    """TV(R0, R1) on composition fields — how similar the two LLM conditions are."""
    share, count, _ = eval_fields(cfg)
    out: dict[str, float] = {}
    dummy_gold = {
        rid: {"target_json": pred_net(a[rid]), "state_json": {}} for rid in ids
    }
    m = collect_condition(ids, dummy_gold, b, cfg)
    for field, stats in m["fields"].items():
        if stats.get("tv") is not None:
            out[field] = stats["tv"]
    return out


def bootstrap_mean_tv_delta(
    ids: list[str],
    gold: dict[str, dict[str, Any]],
    r0: dict[str, dict[str, Any]],
    r1: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any] | None:
    """Ego bootstrap for the paired R1-R0 composition-only mean-TV difference."""
    if not ids or iterations <= 0:
        return None

    share, count, _ = eval_fields(cfg)
    edges_size = size_edges(cfg)
    encoded: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []

    def label(value: Any, edges: list[float] | None) -> str | None:
        if value is None:
            return None
        if edges is None:
            return str(value)
        return next(iter(bin_vals([float(value)], edges)))

    fields: list[tuple[str, list[float] | None]] = []
    for field in count + share:
        if field == "num_alters":
            edges = edges_size
        elif field.endswith("_count"):
            edges = list(range(0, 11)) + [100]
        else:
            edges = SHARE_EDGES
        fields.append((field, edges))
    if any(
        summary_of(gold_net(gold[rid])).get("size_bucket") is not None
        or summary_of(pred_net(r0[rid])).get("size_bucket") is not None
        or summary_of(pred_net(r1[rid])).get("size_bucket") is not None
        for rid in ids
    ):
        fields.append(("size_bucket", None))

    for field, edges in fields:
        h_labels = [label(summary_of(gold_net(gold[rid])).get(field), edges) for rid in ids]
        r0_labels = [label(summary_of(pred_net(r0[rid])).get(field), edges) for rid in ids]
        r1_labels = [label(summary_of(pred_net(r1[rid])).get(field), edges) for rid in ids]
        levels = sorted({v for v in h_labels + r0_labels + r1_labels if v is not None})
        code = {value: i for i, value in enumerate(levels)}
        arrays = [
            np.asarray([code.get(v, -1) for v in values], dtype=int)
            for values in (h_labels, r0_labels, r1_labels)
        ]
        encoded.append((arrays[0], arrays[1], arrays[2], len(levels)))

    def tv(code_a: np.ndarray, code_b: np.ndarray, n_levels: int, index: np.ndarray) -> float:
        a = code_a[index]
        b = code_b[index]
        ca = np.bincount(a[a >= 0], minlength=n_levels).astype(float)
        cb = np.bincount(b[b >= 0], minlength=n_levels).astype(float)
        ca /= ca.sum() or 1.0
        cb /= cb.sum() or 1.0
        return float(0.5 * np.abs(ca - cb).sum())

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(iterations):
        index = rng.integers(0, len(ids), size=len(ids))
        r0_tvs = [tv(h, p0, n, index) for h, p0, _p1, n in encoded]
        r1_tvs = [tv(h, p1, n, index) for h, _p0, p1, n in encoded]
        draws.append(float(np.mean(r1_tvs) - np.mean(r0_tvs)))
    if not draws:
        return None
    return {
        "method": "paired_ego_bootstrap_percentile",
        "seed": seed,
        "iterations": iterations,
        "n_successful": len(draws),
        "ci_level": 0.95,
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
    }


def selection_diagnostics(
    dataset: str,
    gold_ids: set[str],
    r0_ok_ids: set[str],
    r1_ok_ids: set[str],
) -> dict[str, Any]:
    manifest_path = t4_eval_dir(dataset).parent / "input" / "manifest.json"
    s0_ids: set[str] | None = None
    s0_source: str | None = None
    manifest_n: int | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_n = manifest.get("n_phase1_task1_input")
        source = (manifest.get("sources") or {}).get("phase1_input")
        if source and Path(source).exists():
            s0_source = str(source)
            s0_ids = set(last_by_id(read_jsonl(Path(source))))
        else:
            fallback = arm_input(dataset, "s0_baseline_state")
            if fallback.exists():
                s0_source = str(fallback)
                s0_ids = set(last_by_id(read_jsonl(fallback)))

    common = gold_ids & r0_ok_ids & r1_ok_ids
    if s0_ids is not None:
        common &= s0_ids
    return {
        "n_gold": len(gold_ids),
        "s0_available_input": {
            "n": len(s0_ids) if s0_ids is not None else manifest_n,
            "source": s0_source or (str(manifest_path) if manifest_path.exists() else None),
            "count_source": "file_ids" if s0_ids is not None else ("manifest" if manifest_n is not None else None),
        },
        "n_r0_schema_ok": len(r0_ok_ids),
        "n_r1_schema_ok": len(r1_ok_ids),
        "n_common_intersection": len(common),
        "common_intersection_definition": (
            "gold ∩ S0_available_input ∩ R0_schema_ok ∩ R1_schema_ok"
            if s0_ids is not None
            else "gold ∩ R0_schema_ok ∩ R1_schema_ok (S0 IDs unavailable)"
        ),
        "randomness_limitation": (
            "Each condition has one generation per ego; the observed R1-R0 difference "
            "cannot separate state-input effects from generation randomness."
        ),
    }


def evaluate_dataset(
    dataset: str,
    slug: str,
    *,
    bootstrap_seed: int = 20260820,
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    cfg = load_config(dataset=dataset)
    gold_path = network_gold(dataset)
    r0_path = arm_output_jsonl(dataset, "n0_baseline_network", slug)
    r1_path = arm_output_jsonl(dataset, "r1_llm_state_network", slug)
    if not r1_path.exists():
        raise FileNotFoundError(f"R1 output missing: {r1_path}")
    if not r0_path.exists():
        raise FileNotFoundError(f"R0/N0 output missing: {r0_path}")

    gold = last_by_id(read_jsonl(gold_path))
    r0 = last_by_id(read_jsonl(r0_path))
    r1 = last_by_id(read_jsonl(r1_path))
    gold_ids = set(gold)
    r0_ok_ids = {rid for rid, rec in r0.items() if schema_ok(rec)}
    r1_ok_ids = {rid for rid, rec in r1.items() if schema_ok(rec)}
    diagnostics = selection_diagnostics(dataset, gold_ids, r0_ok_ids, r1_ok_ids)
    ids = sorted(gold_ids & r0_ok_ids & r1_ok_ids)

    m0 = collect_condition(ids, gold, r0, cfg)
    m1 = collect_condition(ids, gold, r1, cfg)
    person = person_level(ids, gold, r0, r1, cfg)
    tv_r0_r1 = tv_between_preds(ids, r0, r1, cfg)

    attribution: dict[str, Any] = {}
    rows_csv: list[dict[str, Any]] = []
    for section, keyset in (("fields", m0["fields"]), ("core", m0["core"])):
        for field in keyset:
            d0 = m0[section][field]["tv"]
            d1 = m1[section][field]["tv"]
            a = (d1 - d0) if (d0 is not None and d1 is not None) else None
            attribution[f"{section}.{field}"] = {
                "tv_r0_SH": d0,
                "tv_r1_SL": d1,
                "A_state_to_net": a,
                "mean_h": m0[section][field].get("mean_h"),
                "mean_r0": m0[section][field].get("mean_l"),
                "mean_r1": m1[section][field].get("mean_l"),
            }
            rows_csv.append(
                {
                    "section": section,
                    "field": field,
                    "tv_r0": d0,
                    "tv_r1": d1,
                    "A_state_to_net": a,
                    "tv_r0_r1": tv_r0_r1.get(field) if section == "fields" else None,
                    "mean_h": m0[section][field].get("mean_h"),
                    "mean_r0": m0[section][field].get("mean_l"),
                    "mean_r1": m1[section][field].get("mean_l"),
                    "entropy_ratio_r0": m0[section][field].get("entropy_ratio"),
                    "entropy_ratio_r1": m1[section][field].get("entropy_ratio"),
                    "pearson_r0_r1": (person["fields"].get(field) or {}).get("pearson_r0_r1"),
                    "mean_abs_delta": (person["fields"].get(field) or {}).get("mean_abs_delta"),
                }
            )

    a_mean = (
        (m1["mean_tv"] - m0["mean_tv"])
        if (m0["mean_tv"] is not None and m1["mean_tv"] is not None)
        else None
    )
    a_mean_ci = bootstrap_mean_tv_delta(
        ids,
        gold,
        r0,
        r1,
        cfg,
        seed=bootstrap_seed,
        iterations=bootstrap_iterations,
    )
    summary = {
        "dataset": dataset,
        "n_analysis": len(ids),
        "n_gold": len(gold),
        "n_r0_schema_ok": sum(1 for r in r0.values() if schema_ok(r)),
        "n_r1_schema_ok": sum(1 for r in r1.values() if schema_ok(r)),
        "selection_diagnostics": diagnostics,
        "mean_tv_scope": "composition_summary_only",
        "mean_tv_fields": m0["mean_tv_fields"],
        "mean_tv_r0_composition": m0["mean_tv"],
        "mean_tv_r1_composition": m1["mean_tv"],
        "A_mean_tv_composition": a_mean,
        "A_mean_tv_composition_bootstrap_ci": a_mean_ci,
        "mean_tv_r0": m0["mean_tv"],
        "mean_tv_r1": m1["mean_tv"],
        "A_mean_tv": a_mean,
        "homophily_r0": m0["homophily"],
        "homophily_r1": m1["homophily"],
        "roster_alignment_r0": m0["roster_alignment"],
        "roster_alignment_r1": m1["roster_alignment"],
        "person_level": person,
        "tv_r0_vs_r1": tv_r0_r1,
        "attribution": attribution,
        "paths": {"r0": str(r0_path), "r1": str(r1_path), "gold": str(gold_path)},
        # aliases kept so UCNets curated report keys still resolve if re-read
        "mean_tv_c1": m0["mean_tv"],
        "mean_tv_c2": m1["mean_tv"],
        "n_c1_schema_ok": sum(1 for r in r0.values() if schema_ok(r)),
        "n_c2_schema_ok": sum(1 for r in r1.values() if schema_ok(r)),
        "homophily_c1": m0["homophily"],
        "homophily_c2": m1["homophily"],
    }

    eval_dir = t4_eval_dir(dataset) / slug
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "sr1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = eval_dir / "sr1_field_attribution.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()) if rows_csv else [])
        if rows_csv:
            w.writeheader()
            w.writerows(rows_csv)

    lines = [
        f"# T4 auto tables ({dataset})",
        "",
        f"**分析 N** = {len(ids)}（R0∩R1 schema_ok）",
        "",
        f"Mean TV 范围：仅 `composition_summary`；字段：`{', '.join(m0['mean_tv_fields'])}`。",
        "",
        "当前每个条件每位 ego 只有一次生成，R1−R0 不能将输入状态效应与生成随机性分离。",
        "",
        "## 总览",
        "",
        "| 指标 | R0 $N_L\\|S_H$ | R1 $N_L\\|S_L$ | $A$=R1−R0 |",
        "|---|---:|---:|---:|",
        f"| Mean TV | {_fmt(m0['mean_tv'])} | {_fmt(m1['mean_tv'])} | {_fmt(a_mean)} |",
        f"| size_bucket 一致率（同人 R0 vs R1） |  |  | {_fmt(person['size_bucket_agree'])} |",
        f"| 连续字段 mean Pearson(R0,R1) |  |  | {_fmt(person['mean_pearson_r0_r1'])} |",
        "",
        "## 分字段 TV 与个体相关",
        "",
        "| 字段 | TV R0 | TV R1 | A | TV(R0,R1) | Pearson(R0,R1) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows_csv:
        lines.append(
            f"| {row['section']}.{row['field']} | {_fmt(row['tv_r0'])} | {_fmt(row['tv_r1'])} | "
            f"{_fmt(row['A_state_to_net'])} | {_fmt(row['tv_r0_r1'])} | {_fmt(row['pearson_r0_r1'])} |"
        )
    lines.extend(["", f"数值：`{eval_dir / 'sr1_summary.json'}`", ""])
    (eval_dir / "sr1_tables.md").write_text("\n".join(lines), encoding="utf-8")
    return summary, eval_dir, csv_path


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", dest="datasets", default=None)
    parser.add_argument(
        "--all-t4",
        action="store_true",
        help="Evaluate ucnets_w1, gss_1985, nshap_r1.",
    )
    parser.add_argument("--slug", default="llama3.1_latest")
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    datasets: list[str] = []
    if args.all_t4:
        datasets.extend(["ucnets_w1", "gss_1985", "nshap_r1"])
    if args.datasets:
        datasets.extend(args.datasets)
    if not datasets:
        datasets = ["ucnets_w1"]

    status = 0
    for dataset in datasets:
        try:
            summary, eval_dir, csv_path = evaluate_dataset(
                dataset,
                args.slug,
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_iterations=args.bootstrap_iterations,
            )
        except FileNotFoundError as exc:
            print(f"{dataset}: skipped ({exc})")
            status = 2
            continue
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "n_analysis": summary["n_analysis"],
                    "mean_tv_r0": summary["mean_tv_r0"],
                    "mean_tv_r1": summary["mean_tv_r1"],
                    "A_mean_tv": summary["A_mean_tv"],
                    "mean_pearson_r0_r1": summary["person_level"]["mean_pearson_r0_r1"],
                    "size_bucket_agree": summary["person_level"]["size_bucket_agree"],
                },
                indent=2,
            )
        )
        print(f"wrote {eval_dir / 'sr1_summary.json'}")
        print(f"wrote {csv_path}")
        print(f"wrote {eval_dir / 'sr1_tables.md'}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
