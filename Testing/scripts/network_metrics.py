#!/usr/bin/env python3
"""Survey-agnostic network metrics for T3 interval following (and shared helpers).

Estimands are declared under `evaluation.t3_targets` in each
`Testing/configs/network/{dataset}.yaml`. LLM values are always recomputed from the
generated roster / alter-pair list — never from a self-reported composition_summary
number — so following is scored on the same object the model named.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from network_schema import roster_spec

BUCKET_ORDER = ("low", "medium", "high")


def roster_of(net: dict[str, Any] | None, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(net, dict):
        return []
    key = roster_spec(cfg).get("key", "alters")
    for candidate in (key, "alters", "core_alters"):
        val = net.get(candidate)
        if isinstance(val, list):
            return [a for a in val if isinstance(a, dict)]
    return []


def ties_of(net: dict[str, Any] | None) -> list[dict[str, Any]]:
    """LLM schema uses alter_ties; GSS gold still stores alter_alter."""
    if not isinstance(net, dict):
        return []
    for key in ("alter_ties", "alter_alter"):
        val = net.get(key)
        if isinstance(val, list):
            return [t for t in val if isinstance(t, dict)]
    return []


def _share(matches: int, total: int) -> float | None:
    return None if total <= 0 else matches / total


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def flag_share(items: list[dict[str, Any]], field: str, positive: list[str]) -> float | None:
    pos = {str(v) for v in positive}
    known = [it for it in items if it.get(field) is not None]
    return _share(sum(1 for it in known if str(it.get(field)) in pos), len(known))


def ego_match_share(
    alters: list[dict[str, Any]],
    field: str,
    state: dict[str, Any] | None,
    ego_fields: list[str],
) -> float | None:
    ego_val = None
    for name in ego_fields:
        if isinstance(state, dict) and state.get(name) is not None:
            ego_val = str(state[name])
            break
    if ego_val is None:
        return None
    known = [a for a in alters if a.get(field) is not None]
    return _share(sum(1 for a in known if str(a.get(field)) == ego_val), len(known))


def pair_share(ties: list[dict[str, Any]], field: str, positive: list[str]) -> float | None:
    """Share over unique unordered alter pairs.

    Generated tie lists can repeat the same pair in either direction.  Counting
    those rows would give duplicated pairs extra weight, so retain the first
    observed value for each unordered pair.
    """
    pos = {str(v) for v in positive}
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for tie in ties:
        src, tgt = tie.get("src_id"), tie.get("tgt_id")
        if src is None or tgt is None or str(src) == str(tgt):
            continue
        pair = tuple(sorted((str(src), str(tgt))))
        unique.setdefault(pair, tie)
    known = [t for t in unique.values() if t.get(field) is not None]
    return _share(sum(1 for t in known if str(t.get(field)) in pos), len(known))


def compute_target(
    net: dict[str, Any] | None,
    state: dict[str, Any] | None,
    cfg: dict[str, Any],
    spec: dict[str, Any],
) -> float | None:
    compute = spec.get("compute") or {}
    kind = compute.get("kind")
    if kind == "flag_share":
        return flag_share(roster_of(net, cfg), compute["field"], compute.get("positive") or ["yes"])
    if kind == "ego_match":
        return ego_match_share(
            roster_of(net, cfg),
            compute["field"],
            state,
            compute.get("ego_fields") or ["gender", "sex"],
        )
    if kind == "pair_share":
        return pair_share(ties_of(net), compute["field"], compute.get("positive") or [])
    if kind == "composition":
        field = compute.get("field") or spec.get("gold_field")
        if not isinstance(net, dict):
            return None
        val = (net.get("composition_summary") or {}).get(field)
        return None if val is None else float(val)
    raise ValueError(f"unknown compute.kind: {kind}")


def gold_bucket(gold: dict[str, Any], spec: dict[str, Any]) -> str | None:
    field = spec.get("bucket_field")
    if not field:
        return None
    val = (gold.get("target_json") or {}).get("composition_summary", {}).get(field)
    return None if val is None else str(val)


def prompt_bucket(input_row: dict[str, Any] | None, spec: dict[str, Any]) -> str | None:
    key = spec.get("prompt_key")
    if not key or not input_row:
        return None
    val = (input_row.get("prompt_json") or {}).get(key)
    return None if val is None else str(val)


def eta_squared(groups: dict[str, list[float]]) -> float | None:
    xs = [(g, v) for g, vals in groups.items() for v in vals if v is not None]
    if len(xs) < 5 or len({g for g, _ in xs}) < 2:
        return None
    overall = [v for _, v in xs]
    grand = float(np.mean(overall))
    ss_t = sum((v - grand) ** 2 for v in overall)
    if ss_t <= 0:
        return 0.0
    ss_b = 0.0
    for vals in groups.values():
        if not vals:
            continue
        m = float(np.mean(vals))
        ss_b += len(vals) * (m - grand) ** 2
    return float(ss_b / ss_t)


def monotonic_means(means: dict[str, float | None]) -> bool | None:
    seq = [means.get(b) for b in BUCKET_ORDER]
    if any(v is None for v in seq):
        return None
    return seq[0] <= seq[1] <= seq[2]  # type: ignore[operator]


def interval_following(
    pairs: list[tuple[str, float | None, float | None]],
) -> dict[str, Any]:
    """Score paired egos only: both human and LLM targets must be observed."""
    by_h: dict[str, list[float]] = defaultdict(list)
    by_l: dict[str, list[float]] = defaultdict(list)
    n_egos: dict[str, int] = {b: 0 for b in BUCKET_ORDER}
    n_dropped_unpaired_target = 0
    for bucket, hv, lv in pairs:
        if bucket not in n_egos:
            continue
        if hv is None or lv is None:
            n_dropped_unpaired_target += 1
            continue
        n_egos[bucket] += 1
        by_h[bucket].append(float(hv))
        by_l[bucket].append(float(lv))

    rows = []
    means_h: dict[str, float | None] = {}
    means_l: dict[str, float | None] = {}
    for bucket in BUCKET_ORDER:
        hv = by_h.get(bucket) or []
        lv = by_l.get(bucket) or []
        mh = float(np.mean(hv)) if hv else None
        ml = float(np.mean(lv)) if lv else None
        means_h[bucket] = mh
        means_l[bucket] = ml
        rows.append(
            {
                "bucket": bucket,
                "n_egos": n_egos[bucket],
                "n_human": len(hv),
                "n_llm": len(lv),
                "mean_human": None if mh is None else round(mh, 4),
                "mean_llm": None if ml is None else round(ml, 4),
                "gap": None if mh is None or ml is None else round(ml - mh, 4),
            }
        )

    def _span(means: dict[str, float | None]) -> float | None:
        vals = [v for v in means.values() if v is not None]
        return None if len(vals) < 2 else round(max(vals) - min(vals), 4)

    return {
        "n_scored": sum(n_egos.values()),
        "n_paired_targets": sum(n_egos.values()),
        "n_dropped_unpaired_target": n_dropped_unpaired_target,
        "by_bucket": rows,
        "span_human": _span(means_h),
        "span_llm": _span(means_l),
        "eta2_human": None if eta_squared(by_h) is None else round(eta_squared(by_h), 4),
        "eta2_llm": None if eta_squared(by_l) is None else round(eta_squared(by_l), 4),
        "llm_means_monotonic": monotonic_means(means_l),
        "human_means_monotonic": monotonic_means(means_h),
    }


def size_bucket_ok(num: Any, bucket: Any, bounds: dict[str, list[int]] | None) -> bool | None:
    if not bounds or not isinstance(bucket, str) or bucket not in bounds:
        return None
    try:
        n_i = int(num)
    except (TypeError, ValueError):
        return False
    lo, hi = bounds[bucket]
    return lo <= n_i <= hi


def t3_targets(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return dict((cfg.get("evaluation") or {}).get("t3_targets") or {})


def prep_flags(cfg: dict[str, Any]) -> dict[str, Any]:
    """Checklist: N4 empty-bucket drop, Signos output-length floor."""
    n4 = []
    for task, spec in (cfg.get("tasks") or {}).items():
        spec = spec or {}
        if not (task.startswith("n4") or spec.get("include_same_race_bucket") or spec.get("condition_bucket")):
            continue
        n4.append(
            {
                "task": task,
                "require_condition_bucket": bool(spec.get("require_condition_bucket")),
                "has_condition_bucket": bool(spec.get("condition_bucket"))
                or bool(spec.get("include_same_race_bucket")),
                "prompt_key": ((spec.get("condition_bucket") or {}).get("prompt_key")),
            }
        )
    return {
        "dataset": cfg.get("dataset"),
        "num_predict_min": (cfg.get("generation") or {}).get("num_predict_min"),
        "n4": n4,
    }
