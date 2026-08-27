#!/usr/bin/env python3
"""Run Phase 1 state-stage prompts against an Ollama model.

Small-sample or full-file generation. Outputs are organized as:
  evaluation/output/<model_slug>/<task>/<dataset>_output.jsonl
  evaluation/output/<model_slug>/<task>/<dataset>_predictions.csv
Logs go to:
  evaluation/output/logs/<model_slug>/<run_id>.log
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from state_prediction_csv import write_predictions_csv
from state_value_normalize import normalize_prediction


from llm_client import DEFAULT_OLLAMA_HOST, LLMConfig, add_llm_cli_args, llm_config_from_args, llm_generate
from paths import LOGS, TESTING, arm_output_dir, discover_input_files as _discover, model_slug


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = TESTING
DEFAULT_OUTPUT_DIR = LOGS / "state"


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _validate_value(value: Any, allowed: Any) -> Any | None:
    """Return an error payload if value is invalid, else None."""
    if allowed is None:
        return None

    # Numeric range: {"type": "numeric", "min": ..., "max": ...}
    if isinstance(allowed, dict) and allowed.get("type") == "numeric":
        try:
            num = float(value)
        except (TypeError, ValueError):
            return {"got": value, "expected": "numeric", "spec": allowed}
        lo = allowed.get("min")
        hi = allowed.get("max")
        if lo is not None and num < float(lo):
            return {"got": value, "expected_min": lo}
        if hi is not None and num > float(hi):
            return {"got": value, "expected_max": hi}
        return None

    # Nested object: {"field": [allowed...], ...}
    if isinstance(allowed, dict) and "type" not in allowed:
        if not isinstance(value, dict):
            return {"got": value, "expected": "object", "keys": list(allowed.keys())}
        nested_errors: dict[str, Any] = {}
        for sub_key, sub_allowed in allowed.items():
            if sub_key not in value:
                nested_errors[sub_key] = {"missing": True}
                continue
            err = _validate_value(value[sub_key], sub_allowed)
            if err is not None:
                nested_errors[sub_key] = err
        extra = [k for k in value if k not in allowed]
        if extra:
            nested_errors["_extra_keys"] = extra
        return nested_errors or None

    # Categorical list (scalar or multi-select array)
    if isinstance(allowed, list):
        if isinstance(value, list):
            bad = [v for v in value if v not in allowed]
            if bad:
                return {"got": value, "bad_items": bad, "allowed": allowed}
            return None
        if value not in allowed:
            return {"got": value, "allowed": allowed}
        return None

    return None


def validate_prediction(
    pred: dict[str, Any] | None,
    expected_keys: list[str],
    allowed_values: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "parsed": pred is not None,
        "exact_keys": False,
        "missing_keys": [],
        "extra_keys": [],
        "invalid_values": {},
    }
    if pred is None:
        result["missing_keys"] = list(expected_keys)
        result["schema_ok"] = False
        return result

    missing = [k for k in expected_keys if k not in pred]
    extra = [k for k in pred if k not in expected_keys]
    result["missing_keys"] = missing
    result["extra_keys"] = extra
    result["exact_keys"] = not missing and not extra

    invalid: dict[str, Any] = {}
    for key in expected_keys:
        if key not in pred:
            continue
        err = _validate_value(pred[key], allowed_values.get(key))
        if err is not None:
            invalid[key] = err
    result["invalid_values"] = invalid
    result["schema_ok"] = result["exact_keys"] and not invalid
    return result


def discover_input_files(input_dir: Path, datasets: list[str] | None, tasks: list[str] | None) -> list[Path]:
    return _discover(input_dir, datasets, tasks, lambda p: read_jsonl(p, limit=1))


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase1_ollama")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_existing_by_input_id(path: Path) -> dict[str, dict[str, Any]]:
    """Load prior output records keyed by input_id (last write wins)."""
    if not path.exists():
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            iid = rec.get("input_id")
            if iid:
                by_id[str(iid)] = rec
    return by_id


def process_one_row(
    *,
    index: int,
    total: int,
    row: dict[str, Any],
    dataset: str,
    task: str,
    path_name: str,
    model: str,
    run_id: str,
    llm: LLMConfig,
    temperature: float,
    num_predict: int,
    timeout_s: float,
    num_ctx: int,
) -> dict[str, Any]:
    input_id = row.get("input_id", f"row_{index}")
    expected_keys = row.get("expected_output_keys", [])
    allowed_values = row.get("allowed_values", {})
    record: dict[str, Any] = {
        "input_id": input_id,
        "respondent_id": row.get("respondent_id"),
        "dataset": dataset,
        "task": task,
        "prompt_condition": row.get("prompt_condition"),
        "model": model,
        "run_id": run_id,
        "source_input_file": path_name,
        "expected_output_keys": expected_keys,
        "raw_response": None,
        "prediction_json": None,
        "validation": None,
        "error": None,
        "timing": {},
    }
    try:
        resp = llm_generate(
            llm=llm,
            model=model,
            prompt=row["prompt_text"],
            temperature=temperature,
            num_predict=num_predict,
            timeout_s=timeout_s,
            num_ctx=num_ctx,
        )
        raw = resp.get("response", "")
        pred = extract_json_object(raw)
        pred, _norm_changes = normalize_prediction(pred)
        validation = validate_prediction(pred, expected_keys, allowed_values)
        record["raw_response"] = raw
        record["prediction_json"] = pred
        record["validation"] = validation
        if _norm_changes:
            record["normalization"] = {
                "applied": True,
                "n_changes": len(_norm_changes),
                "changes": _norm_changes,
            }
        record["timing"] = {
            "client_elapsed_s": resp.get("_client_elapsed_s"),
            "total_duration_ns": resp.get("total_duration"),
            "eval_count": resp.get("eval_count"),
            "prompt_eval_count": resp.get("prompt_eval_count"),
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError, RuntimeError) as exc:
        record["error"] = str(exc)
    record["_index"] = index
    record["_total"] = total
    return record


def is_successful_record(rec: dict[str, Any]) -> bool:
    """Resume skips rows that already parsed (even if schema_ok is false)."""
    if rec.get("error"):
        return False
    validation = rec.get("validation") or {}
    return bool(validation.get("parsed")) or rec.get("prediction_json") is not None


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    add_llm_cli_args(parser)
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Rows per input file (default=3). Use <=0 for full file.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=4096,
        help="Ollama context window (lower helps fit NUM_PARALLEL on one GPU).",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel Ollama requests (ThreadPool). Default=1.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip input_ids that already have a parsed prediction in the output jsonl.",
    )
    parser.add_argument("--dataset", action="append", dest="datasets", help="Filter dataset (repeatable)")
    parser.add_argument("--task", action="append", dest="tasks", help="Filter task name (repeatable)")
    parser.add_argument("--run-id", default=None, help="Optional run id used in log filename")
    args = parser.parse_args()
    llm = llm_config_from_args(args)

    workers = max(1, int(args.workers))
    slug = model_slug(args.model)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.output_dir / "logs" / slug / f"{run_id}.log"
    logger = setup_logging(log_path)
    row_limit = None if args.limit is None or args.limit <= 0 else args.limit

    input_files = discover_input_files(args.input_dir, args.datasets, args.tasks)
    if not input_files:
        logger.error("No input files found under %s", args.input_dir)
        return 1

    logger.info(
        "model=%s slug=%s run_id=%s workers=%d resume=%s num_ctx=%d",
        args.model,
        slug,
        run_id,
        workers,
        args.resume,
        args.num_ctx,
    )
    logger.info(
        "input_dir=%s output_dir=%s backend=%s host=%s api_base=%s limit=%s",
        args.input_dir,
        args.output_dir,
        llm.backend,
        llm.host,
        llm.api_base,
        row_limit if row_limit is not None else "full",
    )
    logger.info("files=%s", [p.name for p in input_files])

    summary: dict[str, Any] = {
        "model": args.model,
        "model_slug": slug,
        "run_id": run_id,
        "limit": row_limit,
        "workers": workers,
        "resume": bool(args.resume),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "tasks": {},
        "totals": {"n": 0, "ok": 0, "parse_ok": 0, "schema_ok": 0, "errors": 0, "skipped": 0},
    }
    write_lock = threading.Lock()

    for path in input_files:
        rows = read_jsonl(path, limit=row_limit)
        if not rows:
            logger.warning("empty file: %s", path.name)
            continue
        dataset = rows[0]["dataset"]
        task = rows[0]["task"]
        out_dir = arm_output_dir(dataset, task, slug)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset}_output.jsonl"
        meta_path = out_dir / f"{dataset}_run_meta.json"

        existing = load_existing_by_input_id(out_path) if args.resume else {}
        pending: list[tuple[int, dict[str, Any]]] = []
        outputs_by_id: dict[str, dict[str, Any]] = {}

        for i, row in enumerate(rows, start=1):
            input_id = str(row.get("input_id", f"row_{i}"))
            if args.resume and input_id in existing and is_successful_record(existing[input_id]):
                outputs_by_id[input_id] = existing[input_id]
                summary["totals"]["skipped"] += 1
                continue
            pending.append((i, row))

        logger.info(
            "=== %s | %s | n=%d pending=%d skipped=%d workers=%d | %s ===",
            dataset,
            task,
            len(rows),
            len(pending),
            len(rows) - len(pending),
            workers,
            path.name,
        )
        task_key = f"{dataset}::{task}"
        task_stats = {
            "n": 0,
            "ok": 0,
            "parse_ok": 0,
            "schema_ok": 0,
            "errors": 0,
            "skipped": len(rows) - len(pending),
            "output": str(out_path),
            "workers": workers,
        }

        # Preserve already-done rows; append newly generated rows.
        mode = "a" if args.resume and out_path.exists() and pending else "w"
        if not args.resume or not existing:
            mode = "w"
            # Fresh write of any pre-seeded outputs happens after generation via rewrite.
        if args.resume and existing and pending:
            # Rewrite file with successful existing rows first (stable prefix), then append new.
            with out_path.open("w", encoding="utf-8") as out_f:
                for i, row in enumerate(rows, start=1):
                    input_id = str(row.get("input_id", f"row_{i}"))
                    if input_id in outputs_by_id:
                        out_f.write(json.dumps(outputs_by_id[input_id], ensure_ascii=False) + "\n")
            mode = "a"
        elif not pending and args.resume and existing:
            # Nothing to do; rewrite ordered file from existing.
            with out_path.open("w", encoding="utf-8") as out_f:
                for i, row in enumerate(rows, start=1):
                    input_id = str(row.get("input_id", f"row_{i}"))
                    if input_id in outputs_by_id:
                        out_f.write(json.dumps(outputs_by_id[input_id], ensure_ascii=False) + "\n")
        elif not args.resume:
            mode = "w"

        def _account(record: dict[str, Any]) -> None:
            task_stats["n"] += 1
            summary["totals"]["n"] += 1
            validation = record.get("validation") or {}
            if record.get("error"):
                task_stats["errors"] += 1
                summary["totals"]["errors"] += 1
                return
            task_stats["ok"] += 1
            summary["totals"]["ok"] += 1
            if record.get("prediction_json") is not None or validation.get("parsed"):
                task_stats["parse_ok"] += 1
                summary["totals"]["parse_ok"] += 1
            if validation.get("schema_ok"):
                task_stats["schema_ok"] += 1
                summary["totals"]["schema_ok"] += 1

        if pending:
            with out_path.open(mode, encoding="utf-8") as out_f:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(
                            process_one_row,
                            index=i,
                            total=len(rows),
                            row=row,
                            dataset=dataset,
                            task=task,
                            path_name=path.name,
                            model=args.model,
                            run_id=run_id,
                            llm=llm,
                            temperature=args.temperature,
                            num_predict=args.num_predict,
                            timeout_s=args.timeout,
                            num_ctx=args.num_ctx,
                        ): i
                        for i, row in pending
                    }
                    done_count = 0
                    for fut in as_completed(futures):
                        record = fut.result()
                        input_id = str(record["input_id"])
                        # Strip internal bookkeeping before persist.
                        index = record.pop("_index", None)
                        total = record.pop("_total", len(rows))
                        with write_lock:
                            outputs_by_id[input_id] = record
                            _account(record)
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            out_f.flush()
                            done_count += 1
                            validation = record.get("validation") or {}
                            logger.info(
                                "[%s/%s] done=%d/%d %s parse=%s schema=%s err=%s elapsed=%.1fs",
                                index,
                                total,
                                done_count,
                                len(pending),
                                input_id,
                                validation.get("parsed"),
                                validation.get("schema_ok"),
                                bool(record.get("error")),
                                (record.get("timing") or {}).get("client_elapsed_s") or 0.0,
                            )

        # Rewrite in input order for a clean artifact, then CSV.
        ordered: list[dict[str, Any]] = []
        with out_path.open("w", encoding="utf-8") as out_f:
            for i, row in enumerate(rows, start=1):
                input_id = str(row.get("input_id", f"row_{i}"))
                rec = outputs_by_id.get(input_id)
                if rec is None:
                    continue
                ordered.append(rec)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Count skipped rows into task n for meta completeness.
        if args.resume:
            for rec in ordered:
                if rec.get("run_id") != run_id and is_successful_record(rec):
                    # Prior-run successes were not accounted in this run's n/ok.
                    pass

        csv_path = out_dir / f"{dataset}_predictions.csv"
        write_predictions_csv(
            ordered,
            csv_path,
            expected_keys=rows[0].get("expected_output_keys"),
            allowed_values=rows[0].get("allowed_values"),
        )
        task_stats["csv"] = str(csv_path)
        task_stats["n_written"] = len(ordered)

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "run_id": run_id,
                    "dataset": dataset,
                    "task": task,
                    "source_input_file": path.name,
                    "limit": row_limit,
                    "workers": workers,
                    "resume": bool(args.resume),
                    "n_written": len(ordered),
                    "output_jsonl": str(out_path),
                    "output_csv": str(csv_path),
                    "stats": task_stats,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        summary["tasks"][task_key] = task_stats
        logger.info(
            "finished %s -> %s | csv=%s | parse_ok=%d/%d schema_ok=%d errors=%d skipped=%d",
            task_key,
            out_path,
            csv_path,
            task_stats["parse_ok"],
            task_stats["n"],
            task_stats["schema_ok"],
            task_stats["errors"],
            task_stats["skipped"],
        )

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary_path = args.output_dir / "logs" / slug / f"{run_id}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("summary written to %s", summary_path)
    logger.info("TOTALS %s", summary["totals"])
    return 0 if summary["totals"]["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(run())
