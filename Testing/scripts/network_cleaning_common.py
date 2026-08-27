#!/usr/bin/env python3
"""Shared helpers for network-stage cleaning across surveys.

Every survey needs the same three things and differs only in coding:
  * tolerant numeric parsing of survey exports,
  * shares computed over the alters that actually report an item, with the coverage
    recorded so partial reporting stays visible downstream,
  * tertile bucket cutoffs derived from that survey's own human distribution.

The tertile helper refuses to bucket a tied distribution rather than emitting three
labels that cannot be distinguished. That refusal is a finding, not an error: it is how
we learned same_race_share is degenerate in UCNets and in most GSS years.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

MIN_N_FOR_TERTILES = 30
MIN_DISTINCT_FOR_TERTILES = 3

# A genuine tertile split puts ~33% in each bucket. Ties drag it away from that, and a
# bucket thin enough to fall under this share makes the interval-following test
# meaningless: the model can miss that bucket entirely and barely be penalised. Estimands
# that cannot clear this are reported as unbucketable rather than split anyway.
MIN_BUCKET_SHARE = 0.15


def num(value: Any) -> float | None:
    text = str(value if value is not None else "").strip()
    if text == "" or text in {".", "NA", "NaN", "nan", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def code(value: Any) -> int | None:
    parsed = num(value)
    return int(parsed) if parsed is not None else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def share(matches: int, total: int) -> float | None:
    return None if total == 0 else matches / total


def mean_of(values: list[Any]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def median_of(values: list[Any]) -> float | None:
    clean = sorted(v for v in values if v is not None)
    return round(clean[len(clean) // 2], 4) if clean else None


def tertile_cuts(values: list[float]) -> tuple[float, float] | None:
    """Lower/upper cut of the empirical tertiles; None if the distribution is too tied."""
    clean = sorted(v for v in values if v is not None)
    if len(clean) < MIN_N_FOR_TERTILES or len(set(clean)) < MIN_DISTINCT_FOR_TERTILES:
        return None
    lower = clean[len(clean) // 3]
    upper = clean[(2 * len(clean)) // 3]
    if lower == upper:
        return None
    return lower, upper


def bucket_of(value: float | None, cuts: tuple[float, float] | None) -> str | None:
    if value is None or cuts is None:
        return None
    lower, upper = cuts
    if value < lower:
        return "low"
    return "medium" if value < upper else "high"


def size_bucket(n: int, cutoffs: list[tuple[int, str]], top_label: str = "large") -> str:
    """`cutoffs` is an ascending list of (inclusive upper bound, label) pairs."""
    for upper, label in cutoffs:
        if n <= upper:
            return label
    return top_label


def homophily_share(
    ego_value: Any, alter_values: list[Any], match=lambda a, b: a == b
) -> tuple[float | None, int]:
    """Share of reporting alters matching the ego, plus how many reported."""
    present = [v for v in alter_values if v is not None]
    if ego_value is None or not present:
        return None, len(present)
    return share(sum(1 for v in present if match(v, ego_value)), len(present)), len(present)


def assign_buckets(
    records: list[dict[str, Any]],
    fields: list[tuple[str, Any]],
    composition_of=lambda rec: rec["target_json"]["composition_summary"],
) -> tuple[dict[str, tuple[float, float] | None], dict[str, Counter]]:
    """Two-pass bucketing: cutoffs from the population, then per-ego labels.

    `fields` pairs a composition key with an eligibility predicate over the composition
    dict (use `None` for "always eligible"), so estimands that only make sense above a
    size threshold are excluded from both the cutoffs and the labels.
    """
    cuts: dict[str, tuple[float, float] | None] = {}
    for field, eligible in fields:
        vals = []
        for rec in records:
            comp = composition_of(rec)
            if eligible is not None and not eligible(comp):
                continue
            value = comp.get(field)
            if value is not None:
                vals.append(value)
        cuts[field] = tertile_cuts(vals)

    counts = _label_all(records, fields, cuts, composition_of)

    # A ceiling-bound estimand can produce unequal cutoffs and still leave a bucket
    # nearly empty (NSHAP's any-contact density puts 4% in "medium"). Drop those and
    # relabel, so the artifacts never carry a split we would not test on.
    demoted = [
        field
        for field, _ in fields
        if cuts.get(field) is not None and _thinnest_bucket_share(counts[field]) < MIN_BUCKET_SHARE
    ]
    if demoted:
        for field in demoted:
            cuts[field] = None
        counts = _label_all(records, fields, cuts, composition_of)
    return cuts, counts


def _label_all(records, fields, cuts, composition_of) -> dict[str, Counter]:
    counts: dict[str, Counter] = {field: Counter() for field, _ in fields}
    for rec in records:
        comp = composition_of(rec)
        for field, eligible in fields:
            ok = eligible is None or eligible(comp)
            label = bucket_of(comp.get(field), cuts[field]) if ok else None
            comp[f"{field}_bucket"] = label
            counts[field][label] += 1
    return counts


def _thinnest_bucket_share(counts: Counter) -> float:
    labelled = sum(v for k, v in counts.items() if k is not None)
    if not labelled:
        return 0.0
    return min(counts.get(label, 0) / labelled for label in ("low", "medium", "high"))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bucket_cutoff_block(cuts: tuple[float, float] | None) -> dict[str, Any] | None:
    if not cuts:
        return None
    lower, upper = cuts
    return {
        "low": f"< {round(lower, 4)}",
        "medium": f"{round(lower, 4)} - {round(upper, 4)}",
        "high": f">= {round(upper, 4)}",
        "cuts": [round(lower, 4), round(upper, 4)],
    }


def bucket_share_block(counts: Counter) -> dict[str, Any] | None:
    labelled = sum(v for k, v in counts.items() if k is not None)
    if not labelled:
        return None
    block = {k: round(v / labelled, 3) for k, v in counts.most_common() if k is not None}
    block["n_labelled"] = labelled
    return block
