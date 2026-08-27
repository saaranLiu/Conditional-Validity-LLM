#!/usr/bin/env python3
"""Build cleaned Task 3 JSONL from Task 1 rows (UCNets only).

Task 3 conditions on observed party / religion / voting fields (moved from
target_json into prompt_json). No Task 3b.

  UCNets: party_identity, religion_current, religion_raised, voting_frequency

GSS network Task 3 rows are built by ``clean_state_gss.py``
(``gss_1987`` / ``gss_2010_network``), not this script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "Survey" / "ucnets" / "cleaned"

TASK = "s2_politics_religion_conditioned"

UCNETS_CONDITIONING = [
    "party_identity",
    "religion_current",
    "religion_raised",
    "voting_frequency",
]

FIELD_ORDER = [
    "respondent_id",
    "dataset",
    "task",
    "prompt_json",
    "target_json",
    "metadata",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            ordered = {k: row[k] for k in FIELD_ORDER if k in row}
            for k, v in row.items():
                if k not in ordered:
                    ordered[k] = v
            f.write(json.dumps(ordered, ensure_ascii=False) + "\n")


def move_fields(
    prompt: dict[str, Any],
    target: dict[str, Any],
    fields: list[str],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    prompt = dict(prompt)
    target = dict(target)
    for field in fields:
        value = target.get(field)
        if value in (None, ""):
            return None
        prompt[field] = value
        target.pop(field, None)
    return prompt, target


def build_ucnets() -> list[dict[str, Any]]:
    src = read_jsonl(CLEAN_DIR / "ucnets_w1_s0.jsonl")
    out: list[dict[str, Any]] = []
    for row in src:
        moved = move_fields(row["prompt_json"], row["target_json"], UCNETS_CONDITIONING)
        if moved is None:
            continue
        prompt, target = moved
        meta = dict(row.get("metadata") or {})
        meta["conditioning_fields"] = list(UCNETS_CONDITIONING)
        meta["source_task"] = "s0_baseline_state"
        out.append(
            {
                "respondent_id": row["respondent_id"],
                "dataset": "ucnets_w1",
                "task": TASK,
                "prompt_json": prompt,
                "target_json": target,
                "metadata": meta,
            }
        )
    return out


def main() -> None:
    ucnets = build_ucnets()
    uc_path = CLEAN_DIR / "ucnets_w1_s2.jsonl"
    write_jsonl(uc_path, ucnets)
    print(f"Wrote {uc_path} ({len(ucnets)} rows)")


if __name__ == "__main__":
    main()
