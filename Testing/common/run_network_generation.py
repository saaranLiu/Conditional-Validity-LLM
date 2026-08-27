#!/usr/bin/env python3
"""Run Phase 2 network-generation prompts against an Ollama model.

Outputs:
  evaluation/output/<model_slug>/<task>/<dataset>_output.jsonl
Logs:
  evaluation/output/logs/<model_slug>/<run_id>.log

B4 uses a roster-style multi-step process (5 LLM calls / person):
  1) important-matters names
  2) role nominations (+ optional new people)
  3) same_race_as_ego
  4) contact_frequency
  5) composition_summary for full network (fair B1 comparison)
Then code assembles core_alters Top-5 + composition_summary.
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

from llm_client import DEFAULT_OLLAMA_HOST, LLMConfig, add_llm_cli_args, llm_config_from_args, llm_generate
from network_schema import (
    allowed_bundle,
    extract_json_any,
    extract_json_object,
    load_config,
    roster_spec,
    validate_network_prediction,
)
from paths import LOGS, TESTING, arm_output_dir, discover_input_files as _discover, model_slug

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = TESTING
DEFAULT_OUTPUT_DIR = LOGS / "network"

CONTACT_RANK = {
    "daily": 1,
    "weekly": 2,
    "monthly": 3,
    "several_times_year": 4,
    "yearly_or_less": 5,
    "never": 6,
}


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



def discover_input_files(input_dir: Path, datasets: list[str] | None, tasks: list[str] | None) -> list[Path]:
    return _discover(input_dir, datasets, tasks, lambda p: read_jsonl(p, limit=1))


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase2_ollama")
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


def is_successful_record(rec: dict[str, Any]) -> bool:
    if rec.get("error"):
        return False
    validation = rec.get("validation") or {}
    return bool(validation.get("parsed")) or rec.get("prediction_json") is not None


def json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _call(
    llm: LLMConfig,
    model: str,
    prompt: str,
    temperature: float,
    num_predict: int,
    timeout_s: float,
    num_ctx: int,
) -> tuple[str, Any, float]:
    resp = llm_generate(
        llm=llm,
        model=model,
        prompt=prompt,
        temperature=temperature,
        num_predict=num_predict,
        timeout_s=timeout_s,
        num_ctx=num_ctx,
    )
    raw = resp.get("response", "") or ""
    parsed = extract_json_any(raw)
    return raw, parsed, float(resp.get("_client_elapsed_s") or 0.0)


def assemble_b4_prediction(
    people: dict[str, dict[str, Any]],
    role_map: dict[str, list[str]],
    composition: dict[str, Any] | None,
    allowed: dict[str, Any],
) -> dict[str, Any] | None:
    """Build final schema from B4 intermediate state."""
    if not people:
        return None

    # Attach roles
    for label, person in people.items():
        roles: list[str] = []
        for role, labels in role_map.items():
            if label in labels and role in allowed["support_roles"]:
                roles.append(role)
        person["support_roles"] = roles

    # Rank: more roles, then more frequent contact
    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, int, str]:
        label, p = item
        freq = p.get("contact_frequency") or "yearly_or_less"
        return (-len(p.get("support_roles") or []), CONTACT_RANK.get(freq, 9), label)

    ranked = [p for _, p in sorted(people.items(), key=sort_key)]
    core: list[dict[str, Any]] = []
    for i, p in enumerate(ranked[:5], start=1):
        rel = p.get("relationship")
        if rel not in allowed["relationship"]:
            rel = "friend"
        freq = p.get("contact_frequency")
        if freq not in allowed["contact_frequency"]:
            freq = "weekly"
        gender = p.get("gender")
        if gender not in allowed["gender"]:
            gender = "female"
        race = p.get("same_race_as_ego")
        if race not in allowed["same_race_as_ego"]:
            race = "yes"
        roles = [r for r in (p.get("support_roles") or []) if r in allowed["support_roles"]]
        core.append(
            {
                "rank": i,
                "relationship": rel,
                "support_roles": roles,
                "contact_frequency": freq,
                "gender": gender,
                "same_race_as_ego": race,
            }
        )
    # Pad to 5 if model returned fewer people
    while len(core) < 5:
        core.append(
            {
                "rank": len(core) + 1,
                "relationship": "friend",
                "support_roles": ["companion"],
                "contact_frequency": "monthly",
                "gender": "female",
                "same_race_as_ego": "yes",
            }
        )

    if not isinstance(composition, dict):
        # Fallback estimate from named people only
        n = max(len(people), 5)
        kin = {"spouse", "romantic_partner", "parent", "child", "sibling", "other_relative"}
        rels = [p.get("relationship") for p in people.values()]
        composition = {
            "num_alters": min(30, max(1, n)),
            "size_bucket": "small" if n <= 5 else ("medium" if n <= 15 else "large"),
            "kin_share": round(sum(1 for r in rels if r in kin) / max(1, len(rels)), 4),
            "friend_share": round(sum(1 for r in rels if r == "friend") / max(1, len(rels)), 4),
            "coworker_share": round(sum(1 for r in rels if r == "coworker") / max(1, len(rels)), 4),
            "confidant_count": sum(1 for p in people.values() if "confidant" in (p.get("support_roles") or [])),
            "companion_count": sum(1 for p in people.values() if "companion" in (p.get("support_roles") or [])),
            "pct_weekly_plus_contact": round(
                sum(1 for p in people.values() if p.get("contact_frequency") in {"daily", "weekly"})
                / max(1, len(people)),
                4,
            ),
        }

    return {"composition_summary": composition, "core_alters": core}


def run_b4_multistep(
    *,
    state: dict[str, Any],
    cfg: dict[str, Any],
    llm: LLMConfig,
    model: str,
    temperature: float,
    num_predict: int,
    timeout_s: float,
    num_ctx: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    """Returns (prediction, step_traces, error)."""
    allowed = allowed_bundle(cfg)
    traces: list[dict[str, Any]] = []
    opening = f"{cfg['prompt']['opening'].strip()} {cfg['context'].strip()}"
    year = cfg.get("population_context", {}).get("reference_year")
    year_line = (
        f"The data represent a cross-sectional snapshot collected in the year {year}.\n"
        if year is not None
        else ""
    )
    state_block = json_block(state)

    # B6/B7: same informational conditioning as B2/B3, injected into the multistep
    # process rather than a single oneshot call. `state` already carries the extra
    # key(s) (size_bucket / population_network_priors) via build_network_inputs.
    condition_lines = []
    if "size_bucket" in state:
        condition_lines.append(
            "Observed conditioning attribute: this person's TRUE network size_bucket is "
            f"'{state['size_bucket']}' (small=1-5, medium=6-15, large=16-30). "
            "The people you name plus your Step 5 num_alters estimate must be consistent with this bucket."
        )
    if "population_network_priors" in state:
        condition_lines.append(
            "Observed conditioning attribute: population_network_priors for this sample are "
            f"{json_block(state['population_network_priors'])}. "
            "Treat these as distributional anchors, not as this person's exact values."
        )
    condition_block = ("\n" + "\n".join(condition_lines) + "\n") if condition_lines else ""

    # Step 1
    prompt1 = (
        f"{opening}\n{year_line}\n"
        f"Conditioned on this person's observed state:\n{state_block}\n"
        f"{condition_block}\n"
        "Step 1: List up to 5 people this person would discuss important personal matters with.\n"
        "Return ONLY a JSON array. Each element must be an object with keys:\n"
        f'  label (e.g. \"P1\"), relationship (one of {allowed["relationship"]}), '
        f'gender (one of {allowed["gender"]}).\n'
        "Do NOT include contact_frequency or support_roles yet.\n"
        "JSON array:"
    )
    raw1, parsed1, t1 = _call(llm, model, prompt1, temperature, num_predict, timeout_s, num_ctx)
    traces.append({"step": 1, "raw_response": raw1, "parsed": parsed1, "elapsed_s": t1})
    if not isinstance(parsed1, list) or not parsed1:
        return None, traces, "b4_step1_parse_failed"

    people: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(parsed1[:5], start=1):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or f"P{i}")
        people[label] = {
            "label": label,
            "relationship": item.get("relationship"),
            "gender": item.get("gender"),
        }
    if not people:
        return None, traces, "b4_step1_empty_people"

    # Step 2
    prompt2 = (
        f"{opening}\n{year_line}\n"
        f"State:\n{state_block}\n\n"
        f"People already named:\n{json_block(list(people.values()))}\n\n"
        "Step 2: For each support function, list which people provide it. "
        "You may reuse existing labels and/or add new people.\n"
        "Return ONLY a JSON object with keys:\n"
        '  "companionship": [labels],\n'
        '  "practical_help": [labels],\n'
        '  "confidant": [labels],\n'
        '  "advice": [labels],\n'
        '  "new_people": [{"label","relationship","gender"}]\n'
        f"relationship must be one of {allowed['relationship']}; gender one of {allowed['gender']}.\n"
        "JSON object:"
    )
    raw2, parsed2, t2 = _call(llm, model, prompt2, temperature, num_predict, timeout_s, num_ctx)
    traces.append({"step": 2, "raw_response": raw2, "parsed": parsed2, "elapsed_s": t2})
    role_map: dict[str, list[str]] = {
        "companion": [],
        "practical_help": [],
        "confidant": [],
        "advisor": [],
    }
    if isinstance(parsed2, dict):
        for src, dst in (
            ("companionship", "companion"),
            ("practical_help", "practical_help"),
            ("confidant", "confidant"),
            ("advice", "advisor"),
        ):
            vals = parsed2.get(src) or []
            if isinstance(vals, list):
                role_map[dst] = [str(v) for v in vals]
        for np in parsed2.get("new_people") or []:
            if not isinstance(np, dict):
                continue
            label = str(np.get("label") or f"P{len(people)+1}")
            if label not in people:
                people[label] = {
                    "label": label,
                    "relationship": np.get("relationship"),
                    "gender": np.get("gender"),
                }

    labels = list(people.keys())
    # Step 3
    prompt3 = (
        f"{opening}\n"
        f"For each person below, give same_race_as_ego as yes or no.\n"
        f"People:\n{json_block(list(people.values()))}\n\n"
        "Return ONLY a JSON object mapping label -> \"yes\" or \"no\".\n"
        "JSON object:"
    )
    raw3, parsed3, t3 = _call(llm, model, prompt3, temperature, min(num_predict, 1024), timeout_s, num_ctx)
    traces.append({"step": 3, "raw_response": raw3, "parsed": parsed3, "elapsed_s": t3})
    if isinstance(parsed3, dict):
        for lab, val in parsed3.items():
            if lab in people and str(val) in allowed["same_race_as_ego"]:
                people[lab]["same_race_as_ego"] = str(val)

    # Step 4
    prompt4 = (
        f"{opening}\n"
        f"For each person below, give contact_frequency.\n"
        f"Allowed: {allowed['contact_frequency']}.\n"
        f"People:\n{json_block(list(people.values()))}\n\n"
        "Return ONLY a JSON object mapping label -> contact_frequency.\n"
        "JSON object:"
    )
    raw4, parsed4, t4 = _call(llm, model, prompt4, temperature, min(num_predict, 1024), timeout_s, num_ctx)
    traces.append({"step": 4, "raw_response": raw4, "parsed": parsed4, "elapsed_s": t4})
    if isinstance(parsed4, dict):
        for lab, val in parsed4.items():
            if lab in people and str(val) in allowed["contact_frequency"]:
                people[lab]["contact_frequency"] = str(val)

    # Step 5 — full-network composition_summary
    prompt5 = (
        f"{opening}\n{year_line}\n"
        f"State:\n{state_block}\n"
        f"{condition_block}\n"
        f"Named core people so far:\n{json_block(list(people.values()))}\n\n"
        "Step 5: Estimate this person's FULL ego-network composition_summary "
        "(not only the named people).\n"
        "Return ONLY a JSON object with keys:\n"
        "num_alters (1-30), size_bucket (small|medium|large),\n"
        "kin_share, friend_share, coworker_share (0-1),\n"
        "confidant_count, companion_count (integers),\n"
        "pct_weekly_plus_contact (0-1).\n"
        "JSON object:"
    )
    raw5, parsed5, t5 = _call(llm, model, prompt5, temperature, min(num_predict, 1024), timeout_s, num_ctx)
    traces.append({"step": 5, "raw_response": raw5, "parsed": parsed5, "elapsed_s": t5})
    composition = parsed5 if isinstance(parsed5, dict) else None

    pred = assemble_b4_prediction(people, role_map, composition, allowed)
    if pred is None:
        return None, traces, "b4_assemble_failed"
    return pred, traces, None


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
    cfg: dict[str, Any],
) -> dict[str, Any]:
    input_id = row.get("input_id", f"row_{index}")
    record: dict[str, Any] = {
        "input_id": input_id,
        "respondent_id": row.get("respondent_id"),
        "dataset": dataset,
        "task": task,
        "prompt_condition": row.get("prompt_condition"),
        "generation_mode": row.get("generation_mode", "oneshot"),
        "model": model,
        "run_id": run_id,
        "source_input_file": path_name,
        "expected_output_keys": row.get("expected_output_keys", []),
        "raw_response": None,
        "prediction_json": None,
        "validation": None,
        "error": None,
        "timing": {},
        "b4_steps": None,
    }
    try:
        if row.get("generation_mode") == "multistep_roster":
            pred, traces, err = run_b4_multistep(
                state=row.get("prompt_json") or {},
                cfg=cfg,
                llm=llm,
                model=model,
                temperature=temperature,
                num_predict=num_predict,
                timeout_s=timeout_s,
                num_ctx=num_ctx,
            )
            record["b4_steps"] = [
                {
                    "step": t["step"],
                    "elapsed_s": t["elapsed_s"],
                    "parsed_ok": t["parsed"] is not None,
                    "raw_response": t["raw_response"],
                }
                for t in traces
            ]
            record["raw_response"] = json.dumps(
                {"n_steps": len(traces), "final_assembled": True}, ensure_ascii=False
            )
            record["timing"] = {
                "client_elapsed_s": sum(t["elapsed_s"] for t in traces),
                "n_llm_calls": len(traces),
            }
            if err:
                record["error"] = err
            record["prediction_json"] = pred
            record["validation"] = validate_network_prediction(pred, cfg)
        else:
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
            record["raw_response"] = raw
            record["prediction_json"] = pred
            record["validation"] = validate_network_prediction(pred, cfg)
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


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    add_llm_cli_args(parser)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="force one content config; default resolves configs/content/<dataset>.yaml",
    )
    parser.add_argument("--limit", type=int, default=3, help="Rows per file; <=0 = full")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    llm = llm_config_from_args(args)

    # Each input file carries its own dataset, and surveys differ in vocabulary and
    # roster shape, so the content config is resolved per dataset unless overridden.
    config_cache: dict[str, dict[str, Any]] = {}

    def config_for(dataset: str) -> dict[str, Any]:
        key = "__override__" if args.config is not None else dataset
        if key not in config_cache:
            config_cache[key] = (
                load_config(args.config) if args.config is not None else load_config(dataset=dataset)
            )
        return config_cache[key]

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
        "model=%s slug=%s run_id=%s workers=%d resume=%s backend=%s host=%s api_base=%s",
        args.model,
        slug,
        run_id,
        workers,
        args.resume,
        llm.backend,
        llm.host,
        llm.api_base,
    )
    logger.info("input_files=%s", [p.name for p in input_files])

    write_lock = threading.Lock()

    for path in input_files:
        rows = read_jsonl(path, limit=row_limit)
        if not rows:
            continue
        dataset = rows[0]["dataset"]
        task = rows[0]["task"]
        cfg = config_for(dataset)
        # A long roster needs a bigger output budget than the default, and overrunning it
        # fails silently: the JSON is cut mid-array, so the row is logged as unparseable
        # rather than as truncated. Signos asks for 30 alters x 7 fields (~2000 tokens) and
        # lost 5 of 8 pilot rows at the 2048 default. Let the config raise the floor so the
        # dataset carries its own requirement instead of relying on the caller's flags.
        num_predict = args.num_predict
        floor = (cfg.get("generation") or {}).get("num_predict_min")
        if floor and num_predict < int(floor):
            logger.info(
                "dataset=%s raising num_predict %d -> %d (config generation.num_predict_min)",
                dataset,
                num_predict,
                int(floor),
            )
            num_predict = int(floor)
        out_dir = arm_output_dir(dataset, task, slug)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset}_output.jsonl"

        existing = load_existing_by_input_id(out_path) if args.resume else {}
        todo: list[tuple[int, dict[str, Any]]] = []
        for i, row in enumerate(rows):
            iid = str(row.get("input_id", f"row_{i}"))
            if args.resume and iid in existing and is_successful_record(existing[iid]):
                continue
            todo.append((i, row))

        logger.info(
            "file=%s task=%s n=%d todo=%d resume_skip=%d -> %s",
            path.name,
            task,
            len(rows),
            len(todo),
            len(rows) - len(todo),
            out_path,
        )
        if not todo:
            continue

        done = 0
        errors = 0
        schema_ok = 0

        def _handle(rec: dict[str, Any]) -> None:
            nonlocal done, errors, schema_ok
            rec.pop("_index", None)
            rec.pop("_total", None)
            with write_lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done += 1
                if rec.get("error"):
                    errors += 1
                elif (rec.get("validation") or {}).get("schema_ok"):
                    schema_ok += 1
                if done % 10 == 0 or done == len(todo):
                    logger.info(
                        "progress task=%s done=%d/%d schema_ok=%d errors=%d",
                        task,
                        done,
                        len(todo),
                        schema_ok,
                        errors,
                    )

        if workers == 1:
            for i, row in todo:
                rec = process_one_row(
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
                    num_predict=num_predict,
                    timeout_s=args.timeout,
                    num_ctx=args.num_ctx,
                    cfg=cfg,
                )
                _handle(rec)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [
                    ex.submit(
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
                        num_predict=num_predict,
                        timeout_s=args.timeout,
                        num_ctx=args.num_ctx,
                        cfg=cfg,
                    )
                    for i, row in todo
                ]
                for fut in as_completed(futs):
                    _handle(fut.result())

        logger.info(
            "finished task=%s done=%d schema_ok=%d errors=%d out=%s",
            task,
            done,
            schema_ok,
            errors,
            out_path,
        )

    logger.info("all done run_id=%s", run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
