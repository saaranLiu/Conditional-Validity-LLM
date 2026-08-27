#!/usr/bin/env python3
"""Phase 2 network evaluation.

UCNets (default): Layers 0–4 on N0–N6, written to T1_association/ucnets_w1/eval/.
Supplementary surveys: schema + composition TV + T3 interval following, declared in
each `Testing/configs/network/{dataset}.yaml` under `evaluation`. T3 values are
recomputed from the generated roster / alter pairs, not from composition_summary.

  python evaluate_network.py
  python evaluate_network.py --dataset gss_1985
  python evaluate_network.py --from-gold --all-supplementary
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
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


from network_schema import alter_tie_pair_diagnostics, load_config, roster_spec, top_level_keys
from network_metrics import (
    compute_target,
    gold_bucket,
    interval_following,
    prep_flags,
    prompt_bucket,
    size_bucket_ok,
    t3_targets,
)
from paths import (
    TESTING,
    arm_input,
    arm_output_jsonl,
    network_eval_dir,
    network_gold,
)

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = network_gold("ucnets_w1")
DEFAULT_OUTPUT = network_eval_dir()
SUPPLEMENTARY_DATASETS = ("gss_1985", "gss_2004", "gss_2024", "nshap_r1", "signos")

TASKS = {
    "B1": "n0_baseline_network",
    "B2": "n1_size_bucket",
    "B3": "n2_population_prior",
    "B4": "n3_roster_multistep",
    "B5": "n4_homophily",
    "B6": "n5_size_multistep",
    "B7": "n6_prior_multistep",
}
SUBSAMPLE_TASKS = {"B6", "B7"}
SUBSAMPLE_IDS_PATH = TESTING / "T5_additivity" / "ucnets_w1" / "input" / "n5n6_subsample_ids.jsonl"
COMPOSITION_TV_FIELDS = (
    "kin_share",
    "friend_share",
    "coworker_share",
    "pct_weekly_plus_contact",
    "num_alters",
    "confidant_count",
    "companion_count",
)
CORE_TV_PREFIX = "core_"

KIN = {
    "spouse",
    "romantic_partner",
    "parent",
    "child",
    "sibling",
    "other_relative",
}

FREQ_ORD = {
    "daily": 6,
    "weekly": 5,
    "monthly": 4,
    "several_times_year": 3,
    "yearly_or_less": 2,
    "never": 1,
}

REL_STRENGTH = {
    "spouse": 6,
    "romantic_partner": 5,
    "child": 5,
    "parent": 4,
    "sibling": 4,
    "other_relative": 3,
    "friend": 3,
    "housemate": 2,
    "coworker": 2,
    "neighbor": 1,
    "school": 2,
    "church": 2,
    "acquaintance": 1,
    "other": 1,
}


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
        rid = str(r.get(key) or r.get("input_id") or "")
        if rid:
            out[rid] = r
    return out


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


def bin_share(vals: list[float], edges: list[float]) -> Counter[str]:
    labels = []
    for v in vals:
        placed = False
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v == edges[i + 1]):
                labels.append(f"[{edges[i]:.2f},{edges[i+1]:.2f}]")
                placed = True
                break
        if not placed:
            labels.append("other")
    return Counter(labels)


def eta_squared(groups: dict[str, list[float]]) -> float | None:
    xs = [(g, v) for g, vals in groups.items() for v in vals if v is not None]
    if len(xs) < 5 or len(groups) < 2:
        return None
    overall = [v for _, v in xs]
    grand = float(np.mean(overall))
    ss_b = 0.0
    ss_t = sum((v - grand) ** 2 for v in overall)
    if ss_t <= 0:
        return 0.0
    for g, vals in groups.items():
        if not vals:
            continue
        m = float(np.mean(vals))
        ss_b += len(vals) * (m - grand) ** 2
    return float(ss_b / ss_t)


def ols_r2(X: np.ndarray, y: np.ndarray) -> float | None:
    if X.ndim != 2 or len(y) < X.shape[1] + 2:
        return None
    y = y.astype(float)
    if np.allclose(y, y[0]):
        return 0.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def blau(counts: Counter[str]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    mat = np.zeros((k, k), dtype=float)
    for a, b in pairs:
        mat[idx[a], idx[b]] += 1
    n = mat.sum()
    if n <= 0:
        return None
    po = float(np.trace(mat) / n)
    pe = float(np.sum(mat.sum(0) * mat.sum(1)) / (n * n))
    if abs(1 - pe) < 1e-12:
        return 0.0
    return (po - pe) / (1 - pe)


def load_gold(dataset: str = "ucnets_w1") -> dict[str, dict[str, Any]]:
    by = {}
    for r in read_jsonl(network_gold(dataset)):
        by[str(r["respondent_id"])] = r
    return by


def extract_llm_pred(rec: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if rec.get("error"):
        return None
    pred = rec.get("prediction_json")
    if not isinstance(pred, dict):
        return None
    if cfg is None:
        if "composition_summary" not in pred or "core_alters" not in pred:
            return None
        return pred
    needed = top_level_keys(cfg)
    if any(k not in pred for k in needed):
        return None
    return pred


def is_analysis_row(rec: dict[str, Any], require_schema_ok: bool, cfg: dict[str, Any] | None = None) -> bool:
    if rec.get("error"):
        return False
    if extract_llm_pred(rec, cfg) is None:
        return False
    if require_schema_ok:
        return bool((rec.get("validation") or {}).get("schema_ok"))
    return True


def layer0(recs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(recs)
    schema_ok = sum(1 for r in recs if (r.get("validation") or {}).get("schema_ok"))
    errors = sum(1 for r in recs if r.get("error"))
    spouse_ok = 0
    size_consistent = 0
    checked = 0
    for r in recs:
        pred = extract_llm_pred(r)
        if pred is None:
            continue
        checked += 1
        core = pred.get("core_alters") or []
        n_spouse = sum(1 for a in core if isinstance(a, dict) and a.get("relationship") == "spouse")
        if n_spouse <= 1:
            spouse_ok += 1
        cs = pred.get("composition_summary") or {}
        num = cs.get("num_alters")
        bucket = cs.get("size_bucket")
        ok = False
        if isinstance(num, (int, float)) and isinstance(bucket, str) and bucket in {"small", "medium", "large"}:
            n_i = int(num)
            if bucket == "small":
                ok = 1 <= n_i <= 5
            elif bucket == "medium":
                ok = 6 <= n_i <= 15
            else:
                ok = 16 <= n_i <= 30
        if ok:
            size_consistent += 1
    return {
        "n_output_rows": n,
        "n_with_prediction": checked,
        "n_errors": errors,
        "schema_ok": schema_ok,
        "schema_ok_rate": round(schema_ok / n, 4) if n else None,
        "spouse_le1_rate": round(spouse_ok / checked, 4) if checked else None,
        "size_bucket_consistent_rate": round(size_consistent / checked, 4) if checked else None,
    }


def composition_metrics(
    human_rows: list[dict[str, Any]],
    llm_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Paired by order: same respondent lists already aligned."""
    share_fields = ["kin_share", "friend_share", "coworker_share", "pct_weekly_plus_contact"]
    count_fields = ["num_alters", "confidant_count", "companion_count"]
    out: dict[str, Any] = {"fields": {}, "size_bucket": {}, "dispersion": {}}

    h_bucket = Counter()
    l_bucket = Counter()
    for h, l in zip(human_rows, llm_rows):
        h_bucket[h["composition_summary"].get("size_bucket", "na")] += 1
        l_bucket[l["composition_summary"].get("size_bucket", "na")] += 1
    out["size_bucket"] = {
        "tv": round(total_variation(h_bucket, l_bucket), 4),
        "js": round(js_divergence(h_bucket, l_bucket), 4),
        "entropy_ratio": round(entropy(l_bucket) / entropy(h_bucket), 4) if entropy(h_bucket) > 0 else None,
        "human": dict(h_bucket),
        "llm": dict(l_bucket),
    }

    for field in share_fields:
        hv = [float(h["composition_summary"][field]) for h in human_rows if h["composition_summary"].get(field) is not None]
        lv = [float(l["composition_summary"][field]) for l in llm_rows if l["composition_summary"].get(field) is not None]
        # align length by using paired where both present
        h_pairs, l_pairs = [], []
        for h, l in zip(human_rows, llm_rows):
            a = h["composition_summary"].get(field)
            b = l["composition_summary"].get(field)
            if a is None or b is None:
                continue
            h_pairs.append(float(a))
            l_pairs.append(float(b))
        if len(h_pairs) < 10:
            continue
        edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
        hc, lc = bin_share(h_pairs, edges), bin_share(l_pairs, edges)
        out["fields"][field] = {
            "n": len(h_pairs),
            "mean_human": round(float(np.mean(h_pairs)), 4),
            "mean_llm": round(float(np.mean(l_pairs)), 4),
            "median_human": round(float(np.median(h_pairs)), 4),
            "median_llm": round(float(np.median(l_pairs)), 4),
            "tv_binned": round(total_variation(hc, lc), 4),
            "js_binned": round(js_divergence(hc, lc), 4),
            "entropy_ratio_binned": round(entropy(lc) / entropy(hc), 4) if entropy(hc) > 0 else None,
            "ks_stat": round(float(stats.ks_2samp(h_pairs, l_pairs).statistic), 4),
            "ks_p": float(stats.ks_2samp(h_pairs, l_pairs).pvalue),
        }

    for field in count_fields:
        h_pairs, l_pairs = [], []
        for h, l in zip(human_rows, llm_rows):
            a = h["composition_summary"].get(field)
            b = l["composition_summary"].get(field)
            if a is None or b is None:
                continue
            h_pairs.append(float(a))
            l_pairs.append(float(b))
        if len(h_pairs) < 10:
            continue
        hc = Counter(str(int(v)) for v in h_pairs)
        lc = Counter(str(int(v)) for v in l_pairs)
        out["fields"][field] = {
            "n": len(h_pairs),
            "mean_human": round(float(np.mean(h_pairs)), 4),
            "mean_llm": round(float(np.mean(l_pairs)), 4),
            "median_human": round(float(np.median(h_pairs)), 4),
            "median_llm": round(float(np.median(l_pairs)), 4),
            "var_human": round(float(np.var(h_pairs, ddof=1)), 4),
            "var_llm": round(float(np.var(l_pairs, ddof=1)), 4),
            "var_mean_ratio_human": round(float(np.var(h_pairs, ddof=1) / max(np.mean(h_pairs), 1e-9)), 4),
            "var_mean_ratio_llm": round(float(np.var(l_pairs, ddof=1) / max(np.mean(l_pairs), 1e-9)), 4),
            "tv": round(total_variation(hc, lc), 4),
            "entropy_ratio": round(entropy(lc) / entropy(hc), 4) if entropy(hc) > 0 else None,
            "ks_stat": round(float(stats.ks_2samp(h_pairs, l_pairs).statistic), 4),
            "ks_p": float(stats.ks_2samp(h_pairs, l_pairs).pvalue),
        }

    # core alter categorical marginals (pool ranks)
    for field in ["relationship", "contact_frequency", "gender", "same_race_as_ego"]:
        hp, lp = [], []
        pairs = []
        for h, l in zip(human_rows, llm_rows):
            ha = {a["rank"]: a for a in h.get("core_alters") or [] if isinstance(a, dict)}
            la = {a["rank"]: a for a in l.get("core_alters") or [] if isinstance(a, dict)}
            for rank in range(1, 6):
                if rank in ha and rank in la and ha[rank].get(field) is not None and la[rank].get(field) is not None:
                    hv, lv = str(ha[rank][field]), str(la[rank][field])
                    hp.append(hv)
                    lp.append(lv)
                    pairs.append((hv, lv))
        if len(hp) < 10:
            continue
        hc, lc = Counter(hp), Counter(lp)
        out["fields"][f"core_{field}"] = {
            "n": len(hp),
            "tv": round(total_variation(hc, lc), 4),
            "js": round(js_divergence(hc, lc), 4),
            "entropy_ratio": round(entropy(lc) / entropy(hc), 4) if entropy(hc) > 0 else None,
            "kappa": None if cohen_kappa(pairs) is None else round(cohen_kappa(pairs), 4),
            "top_human": hc.most_common(5),
            "top_llm": lc.most_common(5),
        }
    return out


def layer2(
    gold_by: dict[str, dict[str, Any]],
    ids: list[str],
    llm_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def groups_for(outcome: str, anchor: str) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        gh: dict[str, list[float]] = defaultdict(list)
        gl: dict[str, list[float]] = defaultdict(list)
        for rid in ids:
            g = gold_by[rid]
            pred = extract_llm_pred(llm_by_id[rid])
            if pred is None:
                continue
            a = (g.get("state_json") or {}).get(anchor)
            if a is None:
                continue
            a = str(a)
            hv = g["target_json"]["composition_summary"].get(outcome)
            lv = pred["composition_summary"].get(outcome)
            if hv is None or lv is None:
                continue
            gh[a].append(float(hv))
            gl[a].append(float(lv))
        return gh, gl

    assoc = []
    for outcome, anchor in [
        ("kin_share", "age_group"),
        ("kin_share", "marital_status"),
        ("num_alters", "education"),
        ("num_alters", "self_rated_health"),
        ("friend_share", "age_group"),
        ("friend_share", "marital_status"),
    ]:
        gh, gl = groups_for(outcome, anchor)
        eh, el = eta_squared(gh), eta_squared(gl)
        assoc.append(
            {
                "outcome": outcome,
                "anchor": anchor,
                "n_paired_egos": sum(len(v) for v in gh.values()),
                "eta2_human": None if eh is None else round(eh, 4),
                "eta2_llm": None if el is None else round(el, 4),
                "abs_delta": None if eh is None or el is None else round(abs(el - eh), 4),
            }
        )

    # Homophily D_h on core alters: same gender / same race
    def dh_gender(side: str) -> float:
        same = total = 0
        for rid in ids:
            g = gold_by[rid]
            ego_g = (g.get("state_json") or {}).get("gender")
            if ego_g is None:
                continue
            if side == "human":
                alters = g["target_json"].get("core_alters") or []
            else:
                pred = extract_llm_pred(llm_by_id[rid])
                if pred is None:
                    continue
                alters = pred.get("core_alters") or []
            for a in alters:
                if not isinstance(a, dict) or a.get("gender") is None:
                    continue
                total += 1
                if str(a["gender"]) == str(ego_g):
                    same += 1
        return same / total if total else float("nan")

    def dh_race(side: str) -> float:
        # human/LLM both have same_race_as_ego yes/no on alters
        yes = total = 0
        for rid in ids:
            if side == "human":
                alters = gold_by[rid]["target_json"].get("core_alters") or []
            else:
                pred = extract_llm_pred(llm_by_id[rid])
                if pred is None:
                    continue
                alters = pred.get("core_alters") or []
            for a in alters:
                if not isinstance(a, dict) or a.get("same_race_as_ego") is None:
                    continue
                total += 1
                if str(a["same_race_as_ego"]) == "yes":
                    yes += 1
        return yes / total if total else float("nan")

    # Role co-occurrence: P(confidant | kin) vs P(confidant | friend)
    def role_cond(side: str) -> dict[str, float]:
        kin_c = kin_n = fri_c = fri_n = 0
        for rid in ids:
            if side == "human":
                alters = gold_by[rid]["target_json"].get("core_alters") or []
            else:
                pred = extract_llm_pred(llm_by_id[rid])
                if pred is None:
                    continue
                alters = pred.get("core_alters") or []
            for a in alters:
                if not isinstance(a, dict):
                    continue
                rel = a.get("relationship")
                roles = a.get("support_roles") or []
                if rel in KIN:
                    kin_n += 1
                    if "confidant" in roles:
                        kin_c += 1
                if rel == "friend":
                    fri_n += 1
                    if "confidant" in roles:
                        fri_c += 1
        return {
            "p_confidant_given_kin": kin_c / kin_n if kin_n else float("nan"),
            "p_confidant_given_friend": fri_c / fri_n if fri_n else float("nan"),
            "n_kin": kin_n,
            "n_friend": fri_n,
        }

    hg, lg = dh_gender("human"), dh_gender("llm")
    hr, lr = dh_race("human"), dh_race("llm")
    return {
        "associations": assoc,
        "homophily": {
            "same_gender_human": round(hg, 4) if hg == hg else None,
            "same_gender_llm": round(lg, 4) if lg == lg else None,
            "same_gender_delta": round(lg - hg, 4) if hg == hg and lg == lg else None,
            "same_race_human": round(hr, 4) if hr == hr else None,
            "same_race_llm": round(lr, 4) if lr == lr else None,
            "same_race_delta": round(lr - hr, 4) if hr == hr and lr == lr else None,
        },
        "role_cooccurrence": {
            "human": {k: (round(v, 4) if isinstance(v, float) and v == v else v) for k, v in role_cond("human").items()},
            "llm": {k: (round(v, 4) if isinstance(v, float) and v == v else v) for k, v in role_cond("llm").items()},
        },
    }


def layer3(
    gold_by: dict[str, dict[str, Any]],
    ids: list[str],
    llm_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    anchors = ["age_group", "gender", "education", "marital_status", "self_rated_health"]
    # build design matrix rows
    rows_meta = []
    for rid in ids:
        st = gold_by[rid].get("state_json") or {}
        rows_meta.append({a: str(st.get(a)) if st.get(a) is not None else "NA" for a in anchors})

    # one-hot
    levels: dict[str, list[str]] = {}
    for a in anchors:
        levels[a] = sorted({r[a] for r in rows_meta})
    X_list = []
    for r in rows_meta:
        vec = [1.0]  # intercept
        for a in anchors:
            for lev in levels[a][1:]:  # drop first
                vec.append(1.0 if r[a] == lev else 0.0)
        X_list.append(vec)
    X = np.asarray(X_list, dtype=float)

    outcomes = ["num_alters", "kin_share", "friend_share"]
    pred_rows = []
    for outcome in outcomes:
        y_h, y_l = [], []
        keep = []
        for i, rid in enumerate(ids):
            hv = gold_by[rid]["target_json"]["composition_summary"].get(outcome)
            pred = extract_llm_pred(llm_by_id[rid])
            lv = None if pred is None else pred["composition_summary"].get(outcome)
            if hv is None or lv is None:
                continue
            y_h.append(float(hv))
            y_l.append(float(lv))
            keep.append(i)
        if len(keep) < 30:
            continue
        Xk = X[keep]
        rh = ols_r2(Xk, np.asarray(y_h))
        rl = ols_r2(Xk, np.asarray(y_l))
        pred_rows.append(
            {
                "outcome": outcome,
                "n": len(keep),
                "r2_human": None if rh is None else round(rh, 4),
                "r2_llm": None if rl is None else round(rl, 4),
                "abs_delta_r2": None if rh is None or rl is None else round(abs(rl - rh), 4),
            }
        )

    # Blau on core relationship distributions (pooled)
    hc: Counter[str] = Counter()
    lc: Counter[str] = Counter()
    per_h, per_l = [], []
    for rid in ids:
        alters_h = gold_by[rid]["target_json"].get("core_alters") or []
        pred = extract_llm_pred(llm_by_id[rid])
        if pred is None:
            continue
        alters_l = pred.get("core_alters") or []
        ch = Counter(str(a.get("relationship")) for a in alters_h if isinstance(a, dict))
        cl = Counter(str(a.get("relationship")) for a in alters_l if isinstance(a, dict))
        hc.update(ch)
        lc.update(cl)
        per_h.append(blau(ch))
        per_l.append(blau(cl))

    return {
        "predictability": pred_rows,
        "blau_pooled_human": round(blau(hc), 4),
        "blau_pooled_llm": round(blau(lc), 4),
        "blau_mean_ego_human": round(float(np.mean(per_h)), 4) if per_h else None,
        "blau_mean_ego_llm": round(float(np.mean(per_l)), 4) if per_l else None,
    }


def layer4(
    gold_by: dict[str, dict[str, Any]],
    ids: list[str],
    llm_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    # Role multiplicity
    def role_mult(side: str) -> list[float]:
        vals = []
        for rid in ids:
            if side == "human":
                alters = gold_by[rid]["target_json"].get("core_alters") or []
            else:
                pred = extract_llm_pred(llm_by_id[rid])
                if pred is None:
                    continue
                alters = pred.get("core_alters") or []
            for a in alters:
                if isinstance(a, dict):
                    vals.append(float(len(a.get("support_roles") or [])))
        return vals

    hm, lm = role_mult("human"), role_mult("llm")

    # Strength gradient: Kendall tau between relationship strength and contact ordinal
    def kendall(side: str) -> float | None:
        xs, ys = [], []
        for rid in ids:
            if side == "human":
                alters = gold_by[rid]["target_json"].get("core_alters") or []
            else:
                pred = extract_llm_pred(llm_by_id[rid])
                if pred is None:
                    continue
                alters = pred.get("core_alters") or []
            for a in alters:
                if not isinstance(a, dict):
                    continue
                rel = a.get("relationship")
                freq = a.get("contact_frequency")
                if rel not in REL_STRENGTH or freq not in FREQ_ORD:
                    continue
                xs.append(REL_STRENGTH[rel])
                ys.append(FREQ_ORD[freq])
        if len(xs) < 30:
            return None
        tau, _ = stats.kendalltau(xs, ys)
        return float(tau)

    # kin_share by marital_status
    def kin_by_marital(side: str) -> dict[str, float]:
        groups: dict[str, list[float]] = defaultdict(list)
        for rid in ids:
            m = (gold_by[rid].get("state_json") or {}).get("marital_status")
            if m is None:
                continue
            if side == "human":
                v = gold_by[rid]["target_json"]["composition_summary"].get("kin_share")
            else:
                pred = extract_llm_pred(llm_by_id[rid])
                if pred is None:
                    continue
                v = pred["composition_summary"].get("kin_share")
            if v is not None:
                groups[str(m)].append(float(v))
        return {k: round(float(np.mean(vs)), 4) for k, vs in sorted(groups.items()) if vs}

    th, tl = kendall("human"), kendall("llm")
    return {
        "mean_roles_per_alter_human": round(float(np.mean(hm)), 4) if hm else None,
        "mean_roles_per_alter_llm": round(float(np.mean(lm)), 4) if lm else None,
        "roles_delta": round(float(np.mean(lm) - np.mean(hm)), 4) if hm and lm else None,
        "kendall_strength_contact_human": None if th is None else round(th, 4),
        "kendall_strength_contact_llm": None if tl is None else round(tl, 4),
        "kin_share_by_marital_human": kin_by_marital("human"),
        "kin_share_by_marital_llm": kin_by_marital("llm"),
    }


def evaluate_task(
    short: str,
    task: str,
    gold_by: dict[str, dict[str, Any]],
    out_dir: Path,
    require_schema_ok: bool,
    model_slug: str,
    restrict_ids: set[str] | None = None,
) -> dict[str, Any]:
    path = arm_output_jsonl("ucnets_w1", task, model_slug)
    recs = list(last_by_id(read_jsonl(path)).values()) if path.exists() else []
    if restrict_ids is not None:
        recs = [r for r in recs if str(r.get("respondent_id")) in restrict_ids]
    l0 = layer0(recs)

    # analysis ids: exclude errors; optionally require schema_ok; must be in gold
    analysis_recs = [r for r in recs if is_analysis_row(r, require_schema_ok) and str(r.get("respondent_id")) in gold_by]
    # For B4, also explicitly track excluded errors
    excluded_errors = [r for r in recs if r.get("error")]
    ids = [str(r["respondent_id"]) for r in analysis_recs]
    llm_by = {str(r["respondent_id"]): r for r in analysis_recs}

    human_rows = [gold_by[i]["target_json"] for i in ids]
    llm_rows = [extract_llm_pred(llm_by[i]) for i in ids]
    llm_rows = [x for x in llm_rows if x is not None]
    # re-sync ids if any None slipped
    ids2, human2, llm2 = [], [], []
    for i in ids:
        pred = extract_llm_pred(llm_by[i])
        if pred is None:
            continue
        ids2.append(i)
        human2.append(gold_by[i]["target_json"])
        llm2.append(pred)

    l1 = composition_metrics(human2, llm2)
    l2 = layer2(gold_by, ids2, llm_by)
    l3 = layer3(gold_by, ids2, llm_by)
    l4 = layer4(gold_by, ids2, llm_by)

    # Keep the historical mean_tv key, but define it explicitly as the fixed
    # composition-only estimand used by T5. Core-alter TVs are summarized apart.
    composition_tvs = []
    core_tvs = []
    composition_ents = []
    core_ents = []
    for k, v in l1.get("fields", {}).items():
        tv = None
        if "tv_binned" in v:
            tv = v["tv_binned"]
        elif "tv" in v:
            tv = v["tv"]
        ent = None
        if v.get("entropy_ratio_binned") is not None:
            ent = v["entropy_ratio_binned"]
        elif v.get("entropy_ratio") is not None:
            ent = v["entropy_ratio"]
        if k in COMPOSITION_TV_FIELDS:
            if tv is not None:
                composition_tvs.append(tv)
            if ent is not None:
                composition_ents.append(ent)
        elif k.startswith(CORE_TV_PREFIX):
            if tv is not None:
                core_tvs.append(tv)
            if ent is not None:
                core_ents.append(ent)

    return {
        "task_short": short,
        "task": task,
        "n_gold": len(gold_by),
        "n_output_unique": len(recs),
        "n_analysis": len(ids2),
        "n_excluded_errors": len(excluded_errors),
        "excluded_error_reasons": dict(Counter(str(r.get("error")) for r in excluded_errors)),
        "require_schema_ok": require_schema_ok,
        "layer0": l0,
        "layer1": l1,
        "layer1_summary": {
            "mean_tv": round(float(np.mean(composition_tvs)), 4) if composition_tvs else None,
            "mean_tv_definition": "fixed_composition_fields",
            "mean_tv_composition": round(float(np.mean(composition_tvs)), 4) if composition_tvs else None,
            "mean_tv_core": round(float(np.mean(core_tvs)), 4) if core_tvs else None,
            "composition_fields_requested": list(COMPOSITION_TV_FIELDS),
            "composition_fields_used": [
                k for k in COMPOSITION_TV_FIELDS if k in l1.get("fields", {})
            ],
            "core_fields_used": [
                k for k in l1.get("fields", {}) if k.startswith(CORE_TV_PREFIX)
            ],
            "mean_entropy_ratio": (
                round(float(np.mean(composition_ents)), 4) if composition_ents else None
            ),
            "mean_entropy_ratio_composition": (
                round(float(np.mean(composition_ents)), 4) if composition_ents else None
            ),
            "mean_entropy_ratio_core": (
                round(float(np.mean(core_ents)), 4) if core_ents else None
            ),
            "size_bucket_tv": l1.get("size_bucket", {}).get("tv"),
        },
        "layer2": l2,
        "layer3": l3,
        "layer4": l4,
    }


def _get_path(d: dict[str, Any], *path: str):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


DELTA_KEYS = [
    ("mean_tv", ("layer1_summary", "mean_tv")),
    ("mean_tv_composition", ("layer1_summary", "mean_tv_composition")),
    ("mean_tv_core", ("layer1_summary", "mean_tv_core")),
    ("mean_entropy_ratio", ("layer1_summary", "mean_entropy_ratio")),
    ("friend_share_mean_llm", ("layer1", "fields", "friend_share", "mean_llm")),
    ("kin_share_mean_llm", ("layer1", "fields", "kin_share", "mean_llm")),
    ("num_alters_mean_llm", ("layer1", "fields", "num_alters", "mean_llm")),
    ("num_alters_var_mean_ratio_llm", ("layer1", "fields", "num_alters", "var_mean_ratio_llm")),
    ("blau_mean_ego_llm", ("layer3", "blau_mean_ego_llm")),
    ("mean_roles_llm", ("layer4", "mean_roles_per_alter_llm")),
    ("same_race_llm", ("layer2", "homophily", "same_race_llm")),
    ("same_gender_llm", ("layer2", "homophily", "same_gender_llm")),
]


def b1_b4_delta(
    b1: dict[str, Any], comparison: dict[str, Any], comparison_short: str = "B4"
) -> dict[str, Any]:
    out = {}
    for name, path in DELTA_KEYS:
        v1, v4 = _get_path(b1, *path), _get_path(comparison, *path)
        out[name] = {
            "B1": v1,
            comparison_short: v4,
            f"{comparison_short}_minus_B1": (
                None if v1 is None or v4 is None else round(float(v4) - float(v1), 4)
            ),
        }
    return out


def multi_task_table(results_by_short: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Rows = DELTA_KEYS metric, columns = every task present in results_by_short."""
    shorts = list(results_by_short.keys())
    out: dict[str, Any] = {}
    for name, path in DELTA_KEYS:
        out[name] = {short: _get_path(results_by_short[short], *path) for short in shorts}
    return out


def common_arm_evaluation(
    shorts: list[str],
    gold_by: dict[str, dict[str, Any]],
    require_schema_ok: bool,
    model_slug: str,
    candidate_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Evaluate exactly the schema-valid ego intersection needed by these arms."""
    arm_ids: dict[str, set[str]] = {}
    for short in shorts:
        task = TASKS[short]
        path = arm_output_jsonl("ucnets_w1", task, model_slug)
        if not path.exists():
            return None
        recs = list(last_by_id(read_jsonl(path)).values())
        ids = {
            str(r.get("respondent_id"))
            for r in recs
            if str(r.get("respondent_id")) in gold_by
            and is_analysis_row(r, require_schema_ok)
        }
        if candidate_ids is not None:
            ids &= candidate_ids
        arm_ids[short] = ids
    common_ids = set.intersection(*(arm_ids[s] for s in shorts))
    per_task = {
        short: evaluate_task(
            short,
            TASKS[short],
            gold_by,
            Path("."),
            require_schema_ok,
            model_slug,
            restrict_ids=common_ids,
        )
        for short in shorts
    }
    return {
        "required_arms": shorts,
        "n_candidate_ids": len(candidate_ids) if candidate_ids is not None else len(gold_by),
        "n_common_schema_ok": len(common_ids),
        "n_schema_ok_by_arm": {short: len(arm_ids[short]) for short in shorts},
        "per_task": per_task,
        "metric_table": multi_task_table(per_task),
    }


def same_race_focus(results_by_short: dict[str, dict[str, Any]], human_value: float | None = None) -> dict[str, Any]:
    """B5-focused: is the homophily gap fixed when explicitly conditioned?"""
    out = {}
    for short, res in results_by_short.items():
        llm = _get_path(res, "layer2", "homophily", "same_race_llm")
        human = _get_path(res, "layer2", "homophily", "same_race_human")
        delta = _get_path(res, "layer2", "homophily", "same_race_delta")
        out[short] = {"human": human, "llm": llm, "delta_llm_minus_human": delta}
    return out


def layer0_generic(
    recs: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    n = len(recs)
    schema_ok = sum(1 for r in recs if (r.get("validation") or {}).get("schema_ok"))
    errors = sum(1 for r in recs if r.get("error"))
    bounds = (cfg.get("evaluation") or {}).get("size_bucket_bounds")
    roster_key = roster_spec(cfg).get("key", "alters")
    checked = 0
    size_ok = 0
    size_checked = 0
    pair_checked = 0
    pair_complete = 0
    pair_anomaly_egos = 0
    pair_anomaly_totals: Counter[str] = Counter()
    for r in recs:
        pred = extract_llm_pred(r, cfg)
        if pred is None:
            continue
        checked += 1
        cs = pred.get("composition_summary") or {}
        ok = size_bucket_ok(cs.get("num_alters"), cs.get("size_bucket"), bounds)
        if ok is not None:
            size_checked += 1
            if ok:
                size_ok += 1
        pair_diag = alter_tie_pair_diagnostics(pred, cfg)
        if pair_diag is not None:
            pair_checked += 1
            if pair_diag["complete_c_n_2"]:
                pair_complete += 1
            else:
                pair_anomaly_egos += 1
            for key in (
                "missing_pairs",
                "duplicate_rows",
                "self_pairs",
                "unknown_endpoint_rows",
                "malformed_rows",
            ):
                pair_anomaly_totals[key] += int(pair_diag[key])
    return {
        "n_output_rows": n,
        "n_with_prediction": checked,
        "n_errors": errors,
        "schema_ok": schema_ok,
        "schema_ok_rate": round(schema_ok / n, 4) if n else None,
        "roster_key": roster_key,
        "size_bucket_consistent_rate": round(size_ok / size_checked, 4) if size_checked else None,
        "alter_tie_pair_completeness": {
            "n_checked": pair_checked,
            "n_complete_c_n_2": pair_complete,
            "n_anomalous_egos": pair_anomaly_egos,
            "complete_rate": round(pair_complete / pair_checked, 4) if pair_checked else None,
            "anomaly_counts": dict(pair_anomaly_totals),
        },
    }


def composition_metrics_generic(
    human_rows: list[dict[str, Any]],
    llm_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    ev = cfg.get("evaluation") or {}
    share_fields = list(ev.get("share_fields") or [])
    count_fields = list(ev.get("count_fields") or [])
    out: dict[str, Any] = {"fields": {}, "size_bucket": {}}

    h_bucket = Counter()
    l_bucket = Counter()
    for h, l in zip(human_rows, llm_rows):
        h_bucket[str((h.get("composition_summary") or {}).get("size_bucket", "na"))] += 1
        l_bucket[str((l.get("composition_summary") or {}).get("size_bucket", "na"))] += 1
    if any(k != "na" for k in list(h_bucket) + list(l_bucket)):
        out["size_bucket"] = {
            "tv": round(total_variation(h_bucket, l_bucket), 4),
            "human": dict(h_bucket),
            "llm": dict(l_bucket),
        }

    def _pairs(field: str) -> tuple[list[float], list[float]]:
        hp, lp = [], []
        for h, l in zip(human_rows, llm_rows):
            a = (h.get("composition_summary") or {}).get(field)
            b = (l.get("composition_summary") or {}).get(field)
            if a is None or b is None:
                continue
            hp.append(float(a))
            lp.append(float(b))
        return hp, lp

    for field in share_fields:
        h_pairs, l_pairs = _pairs(field)
        if len(h_pairs) < 10:
            continue
        edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
        hc, lc = bin_share(h_pairs, edges), bin_share(l_pairs, edges)
        out["fields"][field] = {
            "n": len(h_pairs),
            "mean_human": round(float(np.mean(h_pairs)), 4),
            "mean_llm": round(float(np.mean(l_pairs)), 4),
            "tv_binned": round(total_variation(hc, lc), 4),
            "entropy_ratio_binned": round(entropy(lc) / entropy(hc), 4) if entropy(hc) > 0 else None,
        }

    for field in count_fields:
        h_pairs, l_pairs = _pairs(field)
        if len(h_pairs) < 10:
            continue
        hc = Counter(str(int(v)) for v in h_pairs)
        lc = Counter(str(int(v)) for v in l_pairs)
        out["fields"][field] = {
            "n": len(h_pairs),
            "mean_human": round(float(np.mean(h_pairs)), 4),
            "mean_llm": round(float(np.mean(l_pairs)), 4),
            "tv": round(total_variation(hc, lc), 4),
            "entropy_ratio": round(entropy(lc) / entropy(hc), 4) if entropy(hc) > 0 else None,
        }
    return out


def association_generic(
    gold_by: dict[str, dict[str, Any]],
    ids: list[str],
    llm_by: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    pairs = (cfg.get("evaluation") or {}).get("associations") or []
    rows = []
    for spec in pairs:
        outcome, anchor = spec["outcome"], spec["anchor"]
        gh: dict[str, list[float]] = defaultdict(list)
        gl: dict[str, list[float]] = defaultdict(list)
        for rid in ids:
            g = gold_by[rid]
            pred = extract_llm_pred(llm_by[rid], cfg)
            if pred is None:
                continue
            a = (g.get("state_json") or {}).get(anchor)
            if a is None:
                continue
            a = str(a)
            hv = g["target_json"]["composition_summary"].get(outcome)
            lv = (pred.get("composition_summary") or {}).get(outcome)
            if hv is None or lv is None:
                continue
            gh[a].append(float(hv))
            gl[a].append(float(lv))
        eh, el = eta_squared(gh), eta_squared(gl)
        rows.append(
            {
                "outcome": outcome,
                "anchor": anchor,
                "n_paired_egos": sum(len(v) for v in gh.values()),
                "eta2_human": None if eh is None else round(eh, 4),
                "eta2_llm": None if el is None else round(el, 4),
                "abs_delta": None if eh is None or el is None else round(abs(el - eh), 4),
            }
        )
    return rows


def t3_for_task(
    dataset: str,
    task: str,
    cfg: dict[str, Any],
    gold_by: dict[str, dict[str, Any]],
    recs: list[dict[str, Any]],
    require_schema_ok: bool,
    use_prompt_bucket: bool = True,
) -> dict[str, Any] | None:
    spec = t3_targets(cfg).get(task)
    if spec is None:
        return None
    input_by = {}
    in_path = arm_input(dataset, task)
    if in_path.exists():
        for row in read_jsonl(in_path):
            input_by[str(row.get("respondent_id"))] = row

    llm_by = last_by_id(recs)
    pairs = []
    n_no_bucket = 0
    for rid, gold in gold_by.items():
        rec = llm_by.get(rid)
        if rec is None or not is_analysis_row(rec, require_schema_ok, cfg):
            continue
        pred = extract_llm_pred(rec, cfg)
        bucket = gold_bucket(gold, spec)
        if use_prompt_bucket:
            bucket = prompt_bucket(input_by.get(rid), spec) or bucket
        if bucket is None:
            n_no_bucket += 1
            continue
        hv = compute_target(gold.get("target_json"), gold.get("state_json"), cfg, spec)
        lv = compute_target(pred, gold.get("state_json"), cfg, spec)
        pairs.append((bucket, hv, lv))
    out = interval_following(pairs)
    out["task"] = task
    out["gold_field"] = spec.get("gold_field")
    out["prompt_key"] = spec.get("prompt_key")
    out["n_dropped_empty_bucket"] = n_no_bucket
    out["bucket_source"] = "prompt_then_gold_fallback" if use_prompt_bucket else "gold"
    if dataset == "ucnets_w1" and task == "n4_homophily":
        out["estimand_scope_note"] = (
            "Condition bucket describes the full-network same-race target, while the "
            "scored LLM target is recomputed from the fixed core roster."
        )
        out["full_network_bucket_vs_core_roster_target_mismatch"] = True
    return out


def evaluate_task_generic(
    dataset: str,
    task: str,
    cfg: dict[str, Any],
    gold_by: dict[str, dict[str, Any]],
    recs: list[dict[str, Any]],
    require_schema_ok: bool,
    from_gold: bool = False,
) -> dict[str, Any]:
    l0 = layer0_generic(recs, cfg)
    analysis_recs = [
        r for r in recs if is_analysis_row(r, require_schema_ok, cfg) and str(r.get("respondent_id")) in gold_by
    ]
    excluded_errors = [r for r in recs if r.get("error")]
    llm_by = {str(r["respondent_id"]): r for r in analysis_recs}
    ids, human2, llm2 = [], [], []
    for r in analysis_recs:
        rid = str(r["respondent_id"])
        pred = extract_llm_pred(r, cfg)
        if pred is None:
            continue
        ids.append(rid)
        human2.append(gold_by[rid]["target_json"])
        llm2.append(pred)
    l1 = composition_metrics_generic(human2, llm2, cfg)
    tvs = []
    fields_used = []
    for field, v in l1.get("fields", {}).items():
        if "tv_binned" in v:
            tvs.append(v["tv_binned"])
            fields_used.append(field)
        elif "tv" in v:
            tvs.append(v["tv"])
            fields_used.append(field)
    result: dict[str, Any] = {
        "task": task,
        "n_gold": len(gold_by),
        "n_output_unique": len(recs),
        "n_analysis": len(ids),
        "n_excluded_errors": len(excluded_errors),
        "layer0": l0,
        "layer1": l1,
        "layer1_summary": {
            "mean_tv": round(float(np.mean(tvs)), 4) if tvs else None,
            "mean_tv_definition": "configured_composition_fields",
            "mean_tv_composition": round(float(np.mean(tvs)), 4) if tvs else None,
            "mean_tv_core": None,
            "composition_fields_used": fields_used,
        },
        "associations": association_generic(gold_by, ids, llm_by, cfg),
    }
    t3 = t3_for_task(
        dataset, task, cfg, gold_by, recs, require_schema_ok, use_prompt_bucket=not from_gold
    )
    if t3 is not None:
        result["interval_following"] = t3
    return result


def recs_from_gold(gold_by: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Smoke: treat human target_json as a schema_ok prediction."""
    recs = []
    for rid, g in gold_by.items():
        pred = dict(g.get("target_json") or {})
        if "alter_ties" not in pred and "alter_alter" in pred:
            pred["alter_ties"] = pred["alter_alter"]
        for key in top_level_keys(cfg):
            if key not in pred:
                pred[key] = [] if key != "composition_summary" else {}
        recs.append(
            {
                "respondent_id": rid,
                "prediction_json": pred,
                "validation": {"schema_ok": True},
            }
        )
    return recs


def eval_dir_for(dataset: str, from_gold: bool) -> Path:
    if dataset == "ucnets_w1" and not from_gold:
        return network_eval_dir()
    d = TESTING / "T3_interval_following" / dataset / "eval"
    if from_gold:
        d = d / "smoke_from_gold"
    return d


def evaluate_dataset(
    dataset: str,
    model_slug: str,
    require_schema_ok: bool,
    from_gold: bool,
) -> dict[str, Any]:
    cfg = load_config(dataset=dataset)
    gold_by = load_gold(dataset)
    tasks = list((cfg.get("tasks") or {}).keys())
    results: dict[str, Any] = {
        "meta": {
            "dataset": dataset,
            "model_slug": model_slug,
            "from_gold": from_gold,
            "require_schema_ok": require_schema_ok,
            "prep_flags": prep_flags(cfg),
        },
        "tasks": {},
    }
    for task in tasks:
        if from_gold:
            recs = recs_from_gold(gold_by, cfg)
        else:
            path = arm_output_jsonl(dataset, task, model_slug)
            if not path.exists():
                results["tasks"][task] = {"skipped": True, "reason": f"no output at {path}"}
                continue
            recs = list(last_by_id(read_jsonl(path)).values())
        results["tasks"][task] = evaluate_task_generic(
            dataset, task, cfg, gold_by, recs, require_schema_ok, from_gold=from_gold
        )
    return results


def write_eval(results: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "network_eval_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_json}")
    csv_path = out_dir / "t3_interval_following.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("dataset,task,bucket,n_egos,mean_human,mean_llm,gap,span_human,span_llm,eta2_llm,llm_monotonic\n")
        dataset = results.get("meta", {}).get("dataset", "")
        for task, res in (results.get("tasks") or {}).items():
            t3 = res.get("interval_following") if isinstance(res, dict) else None
            if not t3:
                continue
            for row in t3.get("by_bucket") or []:
                f.write(
                    f"{dataset},{task},{row['bucket']},{row['n_egos']},"
                    f"{row.get('mean_human','')},{row.get('mean_llm','')},{row.get('gap','')},"
                    f"{t3.get('span_human','')},{t3.get('span_llm','')},"
                    f"{t3.get('eta2_llm','')},{t3.get('llm_means_monotonic','')}\n"
                )
    print(f"wrote {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-slug", default="llama3.1_latest")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset", action="append", dest="datasets", default=None)
    parser.add_argument(
        "--all-supplementary",
        action="store_true",
        help="Evaluate gss_1985/2004/2024, nshap_r1, signos.",
    )
    parser.add_argument(
        "--from-gold",
        action="store_true",
        help="Smoke test: score human gold as if it were the LLM prediction.",
    )
    parser.add_argument(
        "--require-schema-ok",
        action="store_true",
        default=True,
        help="Restrict analysis to schema_ok predictions (default True).",
    )
    parser.add_argument("--no-require-schema-ok", action="store_true")
    args = parser.parse_args()
    require_schema_ok = not args.no_require_schema_ok

    datasets: list[str] = []
    if args.all_supplementary:
        datasets.extend(SUPPLEMENTARY_DATASETS)
    if args.datasets:
        datasets.extend(args.datasets)
    if not datasets:
        datasets = ["ucnets_w1"]

    # UCNets full Layer 0–4 report stays on the original path unless --from-gold.
    if datasets == ["ucnets_w1"] and not args.from_gold:
        gold_by = load_gold("ucnets_w1")
        eval_dir = args.output_dir or network_eval_dir()
        eval_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for short, task in TASKS.items():
            out_path = arm_output_jsonl("ucnets_w1", task, args.model_slug)
            if not out_path.exists():
                print(f"{short}: skipped (no output file yet at {out_path})")
                continue
            results[short] = evaluate_task(
                short, task, gold_by, Path("."), require_schema_ok, args.model_slug
            )
            print(
                f"{short}: analysis={results[short]['n_analysis']} "
                f"errors_excluded={results[short]['n_excluded_errors']} "
                f"mean_tv={results[short]['layer1_summary']['mean_tv']} "
                f"ent_ratio={results[short]['layer1_summary']['mean_entropy_ratio']}"
            )

        results["cross_arm_samples"] = {}
        n0_n3 = common_arm_evaluation(
            ["B1", "B2", "B3", "B4"], gold_by, require_schema_ok, args.model_slug
        )
        if n0_n3 is not None:
            results["n0_n3_common"] = n0_n3
            results["cross_arm_samples"]["n0_n3"] = {
                k: n0_n3[k]
                for k in ("required_arms", "n_candidate_ids", "n_common_schema_ok", "n_schema_ok_by_arm")
            }

        for right in ("B2", "B3", "B4", "B5"):
            comparison = common_arm_evaluation(
                ["B1", right], gold_by, require_schema_ok, args.model_slug
            )
            if comparison is None:
                continue
            key = f"B1_vs_{right}"
            results["cross_arm_samples"][key] = {
                k: comparison[k]
                for k in ("required_arms", "n_candidate_ids", "n_common_schema_ok", "n_schema_ok_by_arm")
            }
            results[key] = b1_b4_delta(
                comparison["per_task"]["B1"], comparison["per_task"][right], right
            )

        homophily_group = common_arm_evaluation(
            ["B1", "B4", "B5"], gold_by, require_schema_ok, args.model_slug
        )
        if homophily_group is not None:
            results["same_race_focus"] = same_race_focus(homophily_group["per_task"])
            results["cross_arm_samples"]["same_race_focus"] = {
                k: homophily_group[k]
                for k in ("required_arms", "n_candidate_ids", "n_common_schema_ok", "n_schema_ok_by_arm")
            }

        if SUBSAMPLE_IDS_PATH.exists() and any(s in results for s in ("B6", "B7")):
            sub_ids = {str(r["respondent_id"]) for r in read_jsonl(SUBSAMPLE_IDS_PATH)}
            subset_results = {}
            for short in ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]:
                task = TASKS[short]
                out_path = arm_output_jsonl("ucnets_w1", task, args.model_slug)
                if not out_path.exists():
                    continue
                subset_results[short] = evaluate_task(
                    short,
                    task,
                    gold_by,
                    Path("."),
                    require_schema_ok,
                    args.model_slug,
                    restrict_ids=sub_ids,
                )
            size_group = common_arm_evaluation(
                ["B1", "B2", "B4", "B6"],
                gold_by,
                require_schema_ok,
                args.model_slug,
                candidate_ids=sub_ids,
            )
            prior_group = common_arm_evaluation(
                ["B1", "B3", "B4", "B7"],
                gold_by,
                require_schema_ok,
                args.model_slug,
                candidate_ids=sub_ids,
            )
            results["subset_400"] = {
                "n_ids": len(sub_ids),
                "per_task": subset_results,
                "size_arm_required_arms": ["B1", "B2", "B4", "B6"],
                "prior_arm_required_arms": ["B1", "B3", "B4", "B7"],
                "size_arm_n_common_schema_ok": (
                    size_group["n_common_schema_ok"] if size_group else None
                ),
                "prior_arm_n_common_schema_ok": (
                    prior_group["n_common_schema_ok"] if prior_group else None
                ),
                "size_arm_n_schema_ok_by_arm": (
                    size_group["n_schema_ok_by_arm"] if size_group else {}
                ),
                "prior_arm_n_schema_ok_by_arm": (
                    prior_group["n_schema_ok_by_arm"] if prior_group else {}
                ),
                "combo_table_size_arm": size_group["metric_table"] if size_group else {},
                "combo_table_prior_arm": prior_group["metric_table"] if prior_group else {},
            }

        cfg = load_config(dataset="ucnets_w1")
        n4_recs = []
        n4_path = arm_output_jsonl("ucnets_w1", "n4_homophily", args.model_slug)
        if n4_path.exists():
            n4_recs = list(last_by_id(read_jsonl(n4_path)).values())
            results["interval_following"] = t3_for_task(
                "ucnets_w1", "n4_homophily", cfg, gold_by, n4_recs, require_schema_ok
            )
        results["meta"] = {
            "model_slug": args.model_slug,
            "n_gold": len(gold_by),
            "require_schema_ok": require_schema_ok,
            "prep_flags": prep_flags(cfg),
            "n4_estimand_scope": {
                "condition_bucket": "full_network_same_race_share",
                "scored_target": "fixed_core_roster_same_race_share",
                "mismatch_flag": True,
                "generation_input_or_bucket_changed": False,
            },
            "note": (
                "B4/B6/B7 generation errors excluded from analysis; schema_ok filter applied when enabled. "
                "Each cross-arm comparison is recomputed on only its required arms' common schema_ok egos. "
                "T5 size and prior four-arm intersections are separate; no N0-N6 global intersection is used."
            ),
        }

        out_json = eval_dir / "phase2_eval_summary.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"wrote {out_json}")

        flat_path = eval_dir / "phase2_eval_layer1_fields.csv"
        with flat_path.open("w", encoding="utf-8") as f:
            f.write("task,field,n,mean_human,mean_llm,tv,entropy_ratio,kappa\n")
            for short in ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]:
                if short not in results:
                    continue
                fields = results[short]["layer1"].get("fields", {})
                for field, v in fields.items():
                    tv = v.get("tv_binned", v.get("tv"))
                    er = v.get("entropy_ratio_binned", v.get("entropy_ratio"))
                    f.write(
                        f"{short},{field},{v.get('n')},{v.get('mean_human','')},{v.get('mean_llm','')},"
                        f"{tv},{er},{v.get('kappa','')}\n"
                    )
        print(f"wrote {flat_path}")
        return 0

    status = 0
    for dataset in datasets:
        results = evaluate_dataset(dataset, args.model_slug, require_schema_ok, args.from_gold)
        out_dir = args.output_dir or eval_dir_for(dataset, args.from_gold)
        write_eval(results, out_dir)
        for task, res in results["tasks"].items():
            if res.get("skipped"):
                print(f"{dataset} {task}: skipped ({res.get('reason')})")
                continue
            t3 = res.get("interval_following")
            extra = ""
            if t3:
                extra = (
                    f" t3_n={t3.get('n_scored')} span_h={t3.get('span_human')} "
                    f"span_l={t3.get('span_llm')} eta2_l={t3.get('eta2_llm')}"
                )
            print(f"{dataset} {task}: analysis={res.get('n_analysis')}{extra}")
        flags = results["meta"]["prep_flags"]
        print(
            f"{dataset} prep: num_predict_min={flags.get('num_predict_min')} "
            f"n4={flags.get('n4')}"
        )
        if args.from_gold:
            for task, res in results["tasks"].items():
                t3 = res.get("interval_following")
                if not t3:
                    continue
                if t3.get("span_llm") != t3.get("span_human") or t3.get("eta2_llm") != t3.get("eta2_human"):
                    print(f"WARNING {dataset} {task}: from-gold human/LLM T3 stats differ")
                    status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
