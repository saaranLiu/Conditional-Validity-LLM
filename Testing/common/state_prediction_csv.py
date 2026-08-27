#!/usr/bin/env python3
"""Flatten Phase 1 prediction JSONL records into analysis-ready CSV.

Rules:
  - Keep identity / run metadata columns.
  - Expand nested objects as ``parent_child`` columns (e.g. personality_reserved).
  - Encode arrays as sorted pipe-joined strings (e.g. cell_call|email|text).
  - Coerce numeric-looking day counts to int when possible.
  - Missing / unparsed predictions leave output fields blank.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


META_COLUMNS = [
    "input_id",
    "respondent_id",
    "dataset",
    "task",
    "prompt_condition",
    "model",
    "run_id",
    "parsed",
    "exact_keys",
    "schema_ok",
    "error",
]


def parse_bool(value: Any) -> bool | None:
    """Parse CSV-style booleans without treating non-empty strings as true."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def filter_valid_prediction_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep evaluable generations and report the filtering basis.

    ``parsed`` means that generation yielded a JSON object. ``schema_ok`` is
    stricter: parsing succeeded, output keys exactly match the requested keys,
    and every value satisfies its configured schema. Current CSVs contain both,
    so schema_ok is authoritative. For legacy CSVs without schema_ok, parsed
    plus an empty error field is the most reliable available fallback.
    """
    raw_n = len(rows)
    columns = set(rows[0]) if rows else set()
    if "schema_ok" in columns:
        criterion = "schema_ok"
        valid = [row for row in rows if parse_bool(row.get("schema_ok")) is True]
    elif "parsed" in columns:
        criterion = "parsed_and_no_error_fallback"
        valid = [
            row
            for row in rows
            if parse_bool(row.get("parsed")) is True and not str(row.get("error", "")).strip()
        ]
    else:
        criterion = "nonempty_respondent_and_no_error_fallback"
        valid = [
            row
            for row in rows
            if str(row.get("respondent_id", "")).strip() and not str(row.get("error", "")).strip()
        ]
    stats = {
        "raw_n": raw_n,
        "valid_n": len(valid),
        "invalid_n": raw_n - len(valid),
        "validity_criterion": criterion,
    }
    return valid, stats


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def coerce_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        try:
            num = float(text)
        except ValueError:
            return value
        if num.is_integer():
            return int(num)
        return num
    return value


def nested_subkeys(allowed_values: dict[str, Any] | None, key: str) -> list[str]:
    if not allowed_values:
        return []
    spec = allowed_values.get(key)
    if isinstance(spec, dict) and spec.get("type") != "numeric":
        return list(spec.keys())
    return []


def flatten_prediction(
    prediction: dict[str, Any] | None,
    expected_keys: list[str] | None = None,
    allowed_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten prediction_json into a single-level dict for CSV."""
    flat: dict[str, Any] = {}
    pred = prediction or {}
    keys = list(expected_keys) if expected_keys else list(pred.keys())
    for key in pred:
        if key not in keys:
            keys.append(key)

    for key in keys:
        value = pred.get(key, None)
        subkeys = nested_subkeys(allowed_values, key)

        if subkeys or isinstance(value, dict):
            # Nested object → parent_child columns (stable order from allowed schema).
            order = list(subkeys)
            if isinstance(value, dict):
                for sub_key in value:
                    if sub_key not in order:
                        order.append(sub_key)
            elif value is None and not order:
                flat[key] = ""
                continue
            for sub_key in order:
                if isinstance(value, dict) and sub_key in value:
                    flat[f"{key}_{sub_key}"] = coerce_scalar(value[sub_key])
                else:
                    flat[f"{key}_{sub_key}"] = ""
            continue

        if key not in pred:
            flat[key] = ""
            continue

        if isinstance(value, list):
            items = [str(coerce_scalar(v)) for v in value]
            flat[key] = "|".join(sorted(items))
        else:
            flat[key] = coerce_scalar(value)
    return flat


def record_to_csv_row(
    record: dict[str, Any],
    expected_keys: list[str] | None = None,
    allowed_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = record.get("validation") or {}
    row: dict[str, Any] = {
        "input_id": record.get("input_id", ""),
        "respondent_id": record.get("respondent_id", ""),
        "dataset": record.get("dataset", ""),
        "task": record.get("task", ""),
        "prompt_condition": record.get("prompt_condition", ""),
        "model": record.get("model", ""),
        "run_id": record.get("run_id", ""),
        "parsed": validation.get("parsed", ""),
        "exact_keys": validation.get("exact_keys", ""),
        "schema_ok": validation.get("schema_ok", ""),
        "error": record.get("error") or "",
    }
    keys = expected_keys or record.get("expected_output_keys")
    allowed = allowed_values or record.get("allowed_values")
    row.update(flatten_prediction(record.get("prediction_json"), keys, allowed))
    return row


def collect_fieldnames(
    rows: list[dict[str, Any]],
    preferred_output_order: list[str] | None = None,
) -> list[str]:
    seen = set(META_COLUMNS)
    extras: list[str] = []
    if preferred_output_order:
        for key in preferred_output_order:
            if key not in seen:
                extras.append(key)
                seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                extras.append(key)
                seen.add(key)
    return META_COLUMNS + extras


def preferred_flat_columns(
    expected_keys: list[str] | None,
    allowed_values: dict[str, Any] | None,
) -> list[str]:
    if not expected_keys:
        return []
    cols: list[str] = []
    for key in expected_keys:
        subkeys = nested_subkeys(allowed_values, key)
        if subkeys:
            cols.extend(f"{key}_{sub}" for sub in subkeys)
        else:
            cols.append(key)
    return cols


def write_predictions_csv(
    records: list[dict[str, Any]],
    csv_path: Path,
    expected_keys: list[str] | None = None,
    allowed_values: dict[str, Any] | None = None,
) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [
        record_to_csv_row(rec, expected_keys=expected_keys, allowed_values=allowed_values)
        for rec in records
    ]
    preferred = preferred_flat_columns(expected_keys, allowed_values)
    fieldnames = collect_fieldnames(flat_rows, preferred_output_order=preferred)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in flat_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return csv_path


def export_jsonl_to_csv(
    jsonl_path: Path,
    csv_path: Path | None = None,
    expected_keys: list[str] | None = None,
    allowed_values: dict[str, Any] | None = None,
) -> Path:
    records = read_jsonl(jsonl_path)
    if expected_keys is None and records:
        expected_keys = records[0].get("expected_output_keys")
    if allowed_values is None and records:
        allowed_values = records[0].get("allowed_values")
    out = csv_path or jsonl_path.with_name(jsonl_path.name.replace("_output.jsonl", "_predictions.csv"))
    if out == jsonl_path:
        out = jsonl_path.with_suffix(".csv")
    return write_predictions_csv(
        records,
        out,
        expected_keys=expected_keys,
        allowed_values=allowed_values,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, nargs="+", help="Prediction JSONL file(s)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory for CSV outputs (default: beside each JSONL)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluation" / "input",
        help="Used to recover expected_output_keys / allowed_values via source_input_file",
    )
    args = parser.parse_args()

    for path in args.jsonl:
        records = read_jsonl(path)
        expected_keys = records[0].get("expected_output_keys") if records else None
        allowed_values = records[0].get("allowed_values") if records else None
        if (expected_keys is None or allowed_values is None) and records:
            src = records[0].get("source_input_file")
            if src:
                src_path = args.input_dir / src
                if src_path.exists():
                    src_row = read_jsonl(src_path)[0]
                    expected_keys = expected_keys or src_row.get("expected_output_keys")
                    allowed_values = allowed_values or src_row.get("allowed_values")
        if args.out_dir is not None:
            csv_path = args.out_dir / path.name.replace("_output.jsonl", "_predictions.csv")
        else:
            csv_path = path.with_name(path.name.replace("_output.jsonl", "_predictions.csv"))
            if csv_path == path:
                csv_path = path.with_suffix(".csv")
        out = write_predictions_csv(
            records,
            csv_path,
            expected_keys=expected_keys,
            allowed_values=allowed_values,
        )
        print(f"{path} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
