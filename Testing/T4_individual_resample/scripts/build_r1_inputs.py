#!/usr/bin/env python3
"""Build T4 R1 inputs: X ∪ S_L → N0-style network prompts.

R1 = full LLM chain for network estimands:
  S0 S_L + background X  →  N0 network generation

Default dataset is ucnets_w1. GSS 1985 uses the same construction against its
own N0 schema (alters + alter_ties, no trust items in S_L).
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
from pathlib import Path
from typing import Any

from paths import TESTING, arm_input, arm_output_jsonl, dataset_testing_dir, model_slug, network_gold
from build_network_inputs import assert_no_leak, build_oneshot_prompt
from network_schema import allowed_bundle, load_config, roster_spec, top_level_keys

TASK = "r1_llm_state_network"
PHASE2_TASK_FOR_PROMPT = "n0_baseline_network"

DAY_KEYS = {
    "days_very_happy_past_week",
    "days_isolated_past_week",
    "days_lonely_past_week",
}

LONELY_CODE = {
    "hardly_ever": 1,
    "some_of_the_time": 2,
    "often": 3,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def last_by_id(rows: list[dict[str, Any]], key: str = "respondent_id") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        rid = str(r.get(key) or "")
        if rid:
            out[rid] = r
    return out


def coerce_state_values(s_l: dict[str, Any]) -> dict[str, Any]:
    """Match N0 prompt style (day counts as strings)."""
    out: dict[str, Any] = {}
    for k, v in s_l.items():
        if k in DAY_KEYS and v is not None and not isinstance(v, str):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def enrich_sl_for_network(dataset: str, s_l: dict[str, Any]) -> dict[str, Any]:
    """Fill derived state fields that N0 prompts carry but S0 does not generate."""
    out = coerce_state_values(s_l)
    if dataset.startswith("nshap"):
        codes = [
            LONELY_CODE.get(out.get(k))
            for k in ("lacks_companionship", "feels_left_out", "feels_isolated")
        ]
        if all(c is not None for c in codes):
            out["loneliness_scale"] = sum(codes)
        age = out.get("age")
        if isinstance(age, str) and age.isdigit():
            out["age"] = int(age)
    return out


def is_sl_ok(rec: dict[str, Any]) -> bool:
    if rec.get("error"):
        return False
    if not rec.get("prediction_json"):
        return False
    return bool((rec.get("validation") or {}).get("schema_ok"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ucnets_w1")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (same as generation --model). Derives output slug when set.",
    )
    parser.add_argument(
        "--model-slug",
        default=None,
        help="Output slug for S0/R1 dirs. Defaults to model_slug(--model) or llama3.1_latest.",
    )
    args = parser.parse_args()
    dataset = args.dataset
    if args.model_slug:
        slug = args.model_slug
    elif args.model:
        slug = model_slug(args.model)
    else:
        slug = "llama3.1_latest"

    phase1_input = arm_input(dataset, "s0_baseline_state")
    phase1_output = arm_output_jsonl(dataset, "s0_baseline_state", slug)
    if not phase1_output.exists():
        raise FileNotFoundError(
            f"S0 output missing for slug={slug}: {phase1_output}\n"
            "Pass --model (generation name) or --model-slug (output dir name)."
        )
    phase2_gold = network_gold(dataset)
    phase2_config = TESTING / "configs" / "network" / f"{dataset}.yaml"
    out_dir = dataset_testing_dir(dataset, TASK) / "input" / slug

    cfg = load_config(phase2_config)
    cfg = dict(cfg)
    cfg["tasks"] = dict(cfg["tasks"])
    cfg["tasks"][TASK] = dict(cfg["tasks"][PHASE2_TASK_FOR_PROMPT])

    gold_by = last_by_id(read_jsonl(phase2_gold))
    x_by = last_by_id(read_jsonl(phase1_input))
    sl_by = last_by_id(read_jsonl(phase1_output))

    eligible = sorted(
        rid
        for rid in gold_by
        if rid in x_by and rid in sl_by and is_sl_ok(sl_by[rid])
    )
    skipped = {
        "not_in_phase1_input": sorted(set(gold_by) - set(x_by)),
        "phase1_sl_missing_or_bad": sorted(
            rid for rid in (set(gold_by) & set(x_by)) if rid not in sl_by or not is_sl_ok(sl_by[rid])
        ),
    }

    allowed = allowed_bundle(cfg)
    roster_key = roster_spec(cfg)["key"]
    rows: list[dict[str, Any]] = []
    for idx, rid in enumerate(eligible, start=1):
        gold = gold_by[rid]
        x = dict(x_by[rid]["prompt_json"])
        s_l = enrich_sl_for_network(dataset, dict(sl_by[rid]["prediction_json"]))
        prompt_json = {**x, **s_l}
        assert_no_leak(cfg, prompt_json, PHASE2_TASK_FOR_PROMPT)

        composition = (gold.get("target_json") or {}).get("composition_summary") or {}
        rec: dict[str, Any] = {
            "input_id": f"{dataset}_{TASK}_{idx:05d}",
            "respondent_id": rid,
            "dataset": dataset,
            "task": TASK,
            "prompt_condition": "format_only",
            "generation_mode": "oneshot",
            "state_source": "phase1_task1_S_L",
            "prompt_json": prompt_json,
            "prompt_text": build_oneshot_prompt(cfg, TASK, prompt_json),
            "metadata": {
                "weight": (gold.get("metadata") or {}).get("weight"),
                "size_bucket": composition.get("size_bucket"),
                "n_alters_raw": (gold.get("metadata") or {}).get("n_alters_raw"),
                "gold_target": gold["target_json"],
                "phase1_input_id": x_by[rid].get("input_id"),
                "phase1_run_id": sl_by[rid].get("run_id"),
                "s_h_available": True,
                "condition": "C2_full_llm",
            },
            "expected_output_keys": top_level_keys(cfg),
            "allowed_values": {
                "composition_summary": allowed["composition_summary"],
                roster_key: f"list[{'core_alter' if roster_key == 'core_alters' else 'alter'}]",
                "core_alter": allowed["core_alter"],
            },
        }
        rows.append(rec)

    out_path = out_dir / "r1_input.jsonl"
    write_jsonl(out_path, rows)

    c1_ref = {
        "condition": "R0_human_state",
        "network_task": "n0_baseline_network",
        "output_jsonl": str(arm_output_jsonl(dataset, "n0_baseline_network", slug)),
        "gold": str(phase2_gold),
    }

    manifest = {
        "dataset": dataset,
        "model_slug": slug,
        "task": TASK,
        "phase2_prompt_template": PHASE2_TASK_FOR_PROMPT,
        "n_gold": len(gold_by),
        "n_phase1_task1_input": len(x_by),
        "n_phase1_sl_ok": sum(1 for r in sl_by.values() if is_sl_ok(r)),
        "n_c2": len(rows),
        "n_skipped_not_in_phase1_input": len(skipped["not_in_phase1_input"]),
        "n_skipped_bad_sl": len(skipped["phase1_sl_missing_or_bad"]),
        "skipped_preview": {
            "not_in_phase1_input": skipped["not_in_phase1_input"][:10],
            "phase1_sl_missing_or_bad": skipped["phase1_sl_missing_or_bad"][:10],
        },
        "c1_reference": c1_ref,
        "sources": {
            "phase1_input": str(phase1_input),
            "phase1_output": str(phase1_output),
            "phase2_gold": str(phase2_gold),
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: manifest[k] for k in manifest if not k.startswith("skipped")}, indent=2))
    print(f"wrote {out_path} n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
