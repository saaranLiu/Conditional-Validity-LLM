#!/usr/bin/env python3
"""Post-hoc normalize Phase 1 LLM outputs and refresh validation + CSV.

Usage:
  python Testing/scripts/normalize_state_outputs.py \\
      --model-slug llama3.1_latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from state_prediction_csv import write_predictions_csv
from state_value_normalize import normalize_prediction

# Reuse validation from generation script.
from paths import TESTING, arm_input, iter_primary_inputs
from run_state_generation import validate_prediction


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = TESTING
DEFAULT_INPUT_DIR = TESTING


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_allowed_from_input(input_dir: Path, source_name: str | None, dataset: str, task: str) -> dict[str, Any]:
    candidates: list[Path] = []
    if source_name:
        candidates.append(input_dir / source_name)
    # Fallbacks by naming convention
    short = {
        "s0_baseline_state": "s0",
        "s0b_anti_typology": "s0b",
        "s1_health_conditioned": "s1",
        "s1b_anti_typology": "s1b",
        "s2_politics_religion_conditioned": "s2",
    }.get(task)
    if short:
        candidates.append(input_dir / f"{dataset}_{short}_input.jsonl")
        candidates.append(input_dir / f"{dataset}_{task}_input.jsonl")
    for path in candidates:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        allowed = row.get("allowed_values")
                        if isinstance(allowed, dict):
                            return allowed
    return {}


def process_file(path: Path, input_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        return {"path": str(path), "n": 0}

    sample = rows[0]
    allowed = sample.get("allowed_values")
    if not isinstance(allowed, dict) or not allowed:
        allowed = load_allowed_from_input(
            input_dir,
            sample.get("source_input_file"),
            str(sample.get("dataset", "")),
            str(sample.get("task", "")),
        )
    if not allowed:
        raise RuntimeError(f"Could not load allowed_values for {path}")

    n_changed = 0
    n_schema_before = 0
    n_schema_after = 0
    n_parse = 0
    change_rules: dict[str, int] = {}

    new_rows: list[dict[str, Any]] = []
    for rec in rows:
        expected = rec.get("expected_output_keys") or list(allowed.keys())
        pred = rec.get("prediction_json")
        before = validate_prediction(pred, expected, allowed)
        if before.get("schema_ok"):
            n_schema_before += 1
        if before.get("parsed") or pred is not None:
            n_parse += 1

        new_pred, changes = normalize_prediction(pred if isinstance(pred, dict) else None)
        if changes:
            n_changed += 1
            for ch in changes:
                change_rules[ch["rule"]] = change_rules.get(ch["rule"], 0) + 1

        after = validate_prediction(new_pred, expected, allowed)
        if after.get("schema_ok"):
            n_schema_after += 1

        new_rec = dict(rec)
        new_rec["prediction_json"] = new_pred
        new_rec["validation_before_normalize"] = before
        new_rec["validation"] = after
        new_rec["normalization"] = {
            "applied": bool(changes),
            "n_changes": len(changes),
            "changes": changes,
        }
        new_rows.append(new_rec)

    stats = {
        "path": str(path),
        "n": len(rows),
        "rows_changed": n_changed,
        "schema_ok_before": n_schema_before,
        "schema_ok_after": n_schema_after,
        "parse_ok": n_parse,
        "change_rules": change_rules,
    }

    if dry_run:
        return stats

    with path.open("w", encoding="utf-8") as f:
        for rec in new_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    csv_path = path.with_name(path.name.replace("_output.jsonl", "_predictions.csv"))
    write_predictions_csv(
        new_rows,
        csv_path,
        expected_keys=sample.get("expected_output_keys") or list(allowed.keys()),
        allowed_values=allowed,
    )
    stats["csv"] = str(csv_path)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-slug", default="llama3.1_latest")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--task", action="append", dest="tasks")
    args = parser.parse_args()

    model_dir = args.output_dir / args.model_slug
    files = sorted(p for p in TESTING.rglob("*_output.jsonl") if not p.is_symlink())
    if args.datasets:
        files = [p for p in files if any(ds in p.name for ds in args.datasets)]
    if args.tasks:
        files = [p for p in files if p.parent.name in args.tasks]

    if not files:
        print("No output jsonl files found", file=sys.stderr)
        return 1

    all_stats = []
    for path in files:
        stats = process_file(path, args.input_dir, dry_run=args.dry_run)
        all_stats.append(stats)
        print(
            f"{path.parent.name}/{path.name}: n={stats['n']} "
            f"changed={stats['rows_changed']} "
            f"schema {stats['schema_ok_before']}->{stats['schema_ok_after']} "
            f"rules={stats['change_rules']}"
        )

    summary = {
        "model_slug": args.model_slug,
        "dry_run": args.dry_run,
        "files": all_stats,
        "totals": {
            "n": sum(s["n"] for s in all_stats),
            "rows_changed": sum(s["rows_changed"] for s in all_stats),
            "schema_ok_before": sum(s["schema_ok_before"] for s in all_stats),
            "schema_ok_after": sum(s["schema_ok_after"] for s in all_stats),
        },
    }
    out_summary = model_dir / "normalization_summary.json"
    if not args.dry_run:
        with out_summary.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_summary}")
    print("TOTALS", summary["totals"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
