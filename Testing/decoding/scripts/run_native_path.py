#!/usr/bin/env python3
"""Matched native-path decoding and full-JSON counterfactual experiment.

This redesign keeps model, chat template, source prompt, ego sample, condition
intervention, and temperature aligned. It compares:

1. constrained sequence log-probabilities at native JSON value slots; and
2. targets recomputed from full sampled JSON networks.

Legacy forced-choice files are archived under Testing/decoding/temp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
DECODING = ROOT / "Testing" / "decoding"
DEFAULT_MODEL = ROOT / "Env" / "models" / "Meta-Llama-3.1-8B-Instruct"

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "sex_homophily": {
        "source": ROOT
        / "Testing/T3_interval_following/gss_1985/input/n4s_input.jsonl",
        "bucket_key": "same_sex_share_bucket",
        "conditions": ("low", "medium", "high"),
        "full_reps": 3,
        "max_new_tokens": 1000,
    },
    "closure": {
        "source": ROOT
        / "Testing/T3_interval_following/gss_1985/input/n4c_input.jsonl",
        "bucket_key": "alter_closure_bucket",
        "conditions": ("low", "medium", "high"),
        "full_reps": 3,
        "max_new_tokens": 1100,
    },
    "size_bucket": {
        "source": ROOT
        / "Testing/T2_information_use/ucnets_w1/input/n1_input.jsonl",
        "bucket_key": "size_bucket",
        "conditions": ("small", "medium", "large"),
        "full_reps": 1,
        "max_new_tokens": 1400,
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_sample(
    rows: list[dict[str, Any]], max_egos: int, seed: int
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row["respondent_id"]))
    if max_egos <= 0 or len(ordered) <= max_egos:
        return ordered
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(ordered)), max_egos))
    return [ordered[index] for index in indices]


def select_shard(
    rows: list[dict[str, Any]], shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )
    if num_shards == 1:
        return rows
    selected = []
    for row in rows:
        # Keep all three conditions for one ego/task on the same GPU so paired
        # contrasts remain local to a shard.
        identity = f"{row['experiment']}|{row['respondent_id']}"
        value = int(hashlib.sha1(identity.encode()).hexdigest(), 16)
        if value % num_shards == shard_index:
            selected.append(row)
    return selected


def counterfactual_prompt(
    row: dict[str, Any], bucket_key: str, condition: str
) -> tuple[dict[str, Any], str]:
    prompt_json = dict(row["prompt_json"])
    source = prompt_json[bucket_key]
    prompt_json[bucket_key] = condition
    old_fragment = f'"{bucket_key}": "{source}"'
    new_fragment = f'"{bucket_key}": "{condition}"'
    prompt_text = row["prompt_text"]
    count = prompt_text.count(old_fragment)
    if count != 1:
        raise ValueError(
            f"Expected one bucket fragment for {row['input_id']}, found {count}: "
            f"{old_fragment}"
        )
    return prompt_json, prompt_text.replace(old_fragment, new_fragment, 1)


def prepare_inputs(run_name: str, max_egos: int, seed: int) -> Path:
    destination = DECODING / "input" / run_name / "counterfactual_inputs.jsonl"
    if destination.exists():
        print(f"[prepare] reuse {destination}", flush=True)
        return destination

    sex_rows = read_jsonl(EXPERIMENTS["sex_homophily"]["source"])
    closure_rows = read_jsonl(EXPERIMENTS["closure"]["source"])
    sex_by_id = {str(row["respondent_id"]): row for row in sex_rows}

    # GSS tasks intentionally share exactly the same ego panel. Closure eligibility is
    # the narrower population because a closure target requires at least one dyad.
    shared_closure = stable_sample(closure_rows, max_egos, seed)
    shared_ids = [str(row["respondent_id"]) for row in shared_closure]
    if any(respondent not in sex_by_id for respondent in shared_ids):
        raise RuntimeError("Closure panel is not contained in the sex-homophily source")

    selected: dict[str, list[dict[str, Any]]] = {
        "sex_homophily": [sex_by_id[respondent] for respondent in shared_ids],
        "closure": shared_closure,
        "size_bucket": stable_sample(
            read_jsonl(EXPERIMENTS["size_bucket"]["source"]), max_egos, seed
        ),
    }

    output: list[dict[str, Any]] = []
    for experiment, source_rows in selected.items():
        spec = EXPERIMENTS[experiment]
        for row in source_rows:
            source_bucket = row["prompt_json"][spec["bucket_key"]]
            for condition in spec["conditions"]:
                prompt_json, prompt_text = counterfactual_prompt(
                    row, spec["bucket_key"], condition
                )
                identity = (
                    f"{experiment}|{row['respondent_id']}|{condition}|{seed}"
                )
                input_id = hashlib.sha1(identity.encode()).hexdigest()[:16]
                output.append(
                    {
                        "redesign_input_id": f"{experiment}_{input_id}",
                        "experiment": experiment,
                        "respondent_id": str(row["respondent_id"]),
                        "source_input_id": row["input_id"],
                        "source_bucket": source_bucket,
                        "counterfactual_bucket": condition,
                        "prompt_json": prompt_json,
                        "prompt_text": prompt_text,
                        "gold_target": (row.get("metadata") or {}).get("gold_target"),
                    }
                )
    write_jsonl(destination, output)
    print(
        f"[prepare] wrote {len(output)} rows; shared GSS egos={len(shared_ids)} "
        f"to {destination}",
        flush=True,
    )
    return destination


def render_prompt(tokenizer: Any, prompt_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def sex_slot_prefix(row: dict[str, Any], position: int) -> str:
    prior = []
    fixed_genders = ("male", "female", "male", "female", "male")
    for index in range(1, position):
        prior.append(
            {
                "alter_id": index,
                "relationship": "friend",
                "gender": fixed_genders[index - 1],
                "age": 40,
                "race": "white",
                "contact_frequency": "weekly",
            }
        )
    prefix = (
        '{\n  "composition_summary": {"num_alters": 5, "size_bucket": "large", '
        '"kin_share": 0.2, "friend_share": 0.6, "coworker_share": 0.0, '
        '"pct_weekly_plus_contact": 0.8},\n  "alters": [\n'
    )
    if prior:
        prefix += ",\n".join("    " + json.dumps(value) for value in prior) + ",\n"
    prefix += (
        f'    {{"alter_id": {position}, "relationship": "friend", "gender": "'
    )
    return render_prompt(row["_tokenizer"], row["prompt_text"]) + prefix


def closure_slot_prefix(row: dict[str, Any], position: int) -> str:
    alters = [
        {
            "alter_id": index,
            "relationship": "friend",
            "gender": "male" if index % 2 else "female",
            "age": 34 + index,
            "race": "white",
            "contact_frequency": "weekly",
        }
        for index in range(1, 6)
    ]
    pairs = [(left, right) for left in range(1, 6) for right in range(left + 1, 6)]
    fixed = ("know_each_other", "total_strangers")
    prior = [
        {
            "src_id": left,
            "tgt_id": right,
            "tie_strength": fixed[index % 2],
        }
        for index, (left, right) in enumerate(pairs[: position - 1])
    ]
    left, right = pairs[position - 1]
    prefix = (
        '{\n  "composition_summary": {"num_alters": 5, "size_bucket": "large", '
        '"kin_share": 0.2, "friend_share": 0.6, "coworker_share": 0.0, '
        '"pct_weekly_plus_contact": 0.8},\n  "alters": '
        + json.dumps(alters)
        + ',\n  "alter_ties": [\n'
    )
    if prior:
        prefix += ",\n".join("    " + json.dumps(value) for value in prior) + ",\n"
    prefix += (
        f'    {{"src_id": {left}, "tgt_id": {right}, "tie_strength": "'
    )
    return render_prompt(row["_tokenizer"], row["prompt_text"]) + prefix


def size_slot_prefix(row: dict[str, Any]) -> str:
    return (
        render_prompt(row["_tokenizer"], row["prompt_text"])
        + '{\n  "composition_summary": {\n    "num_alters": '
    )


def native_decisions(
    rows: list[dict[str, Any]], tokenizer: Any
) -> Iterable[dict[str, Any]]:
    for source in rows:
        row = dict(source)
        row["_tokenizer"] = tokenizer
        common = {
            "redesign_input_id": row["redesign_input_id"],
            "experiment": row["experiment"],
            "respondent_id": row["respondent_id"],
            "source_bucket": row["source_bucket"],
            "counterfactual_bucket": row["counterfactual_bucket"],
        }
        if row["experiment"] == "sex_homophily":
            ego_sex = row["prompt_json"]["sex"]
            for position in range(1, 6):
                yield {
                    **common,
                    "decision_id": f"{row['redesign_input_id']}_gender_{position}",
                    "slot": f"alters[{position - 1}].gender",
                    "prefix": sex_slot_prefix(row, position),
                    "candidates": ("male", "female"),
                    "candidate_suffix": '"',
                    "ego_sex": ego_sex,
                }
        elif row["experiment"] == "closure":
            for position in range(1, 11):
                yield {
                    **common,
                    "decision_id": f"{row['redesign_input_id']}_tie_{position}",
                    "slot": f"alter_ties[{position - 1}].tie_strength",
                    "prefix": closure_slot_prefix(row, position),
                    "candidates": (
                        "especially_close",
                        "know_each_other",
                        "total_strangers",
                    ),
                    "candidate_suffix": '"',
                }
        else:
            yield {
                **common,
                "decision_id": f"{row['redesign_input_id']}_num_alters",
                "slot": "composition_summary.num_alters",
                "prefix": size_slot_prefix(row),
                "candidates": tuple(str(value) for value in range(1, 31)),
                "candidate_suffix": ",",
            }


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def candidate_logprobs(
    model: Any,
    tokenizer: Any,
    items: list[tuple[list[int], str]],
    temperature: float,
    max_length: int,
) -> list[float]:
    sequences: list[list[int]] = []
    prefix_lengths: list[int] = []
    for prefix_ids, candidate in items:
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
        full = prefix_ids + candidate_ids
        if len(full) > max_length:
            raise ValueError(f"Native sequence length {len(full)} exceeds {max_length}")
        sequences.append(full)
        prefix_lengths.append(len(prefix_ids))

    width = max(len(sequence) for sequence in sequences)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full(
        (len(sequences), width), pad_id, dtype=torch.long, device=model.device
    )
    attention = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        input_ids[index, : len(sequence)] = torch.tensor(
            sequence, dtype=torch.long, device=model.device
        )
        attention[index, : len(sequence)] = 1

    with torch.inference_mode():
        # Calling the full CausalLM head would materialize
        # [batch, sequence, 128k-vocabulary] logits even though only the few
        # candidate positions are needed. Run the decoder once, then apply the
        # LM head only at those positions.
        hidden = model.model(
            input_ids=input_ids, attention_mask=attention, use_cache=False
        ).last_hidden_state
    output: list[float] = []
    for index, sequence in enumerate(sequences):
        start = prefix_lengths[index]
        stop = len(sequence)
        token_hidden = hidden[index, start - 1 : stop - 1]
        with torch.inference_mode():
            token_logits = model.lm_head(token_hidden).float() / temperature
        targets = input_ids[index, start:stop]
        selected = torch.log_softmax(token_logits, dim=-1).gather(
            1, targets.unsqueeze(1)
        )
        output.append(float(selected.sum().item()))
    return output


def run_native(
    rows: list[dict[str, Any]],
    destination: Path,
    model: Any,
    tokenizer: Any,
    temperature: float,
    batch_size: int,
    decision_chunk: int,
    max_length: int,
) -> None:
    done = (
        {row["decision_id"] for row in read_jsonl(destination)}
        if destination.exists()
        else set()
    )
    decisions = [
        decision
        for decision in native_decisions(rows, tokenizer)
        if decision["decision_id"] not in done
    ]
    print(
        f"[native] completed={len(done)} pending_decisions={len(decisions)}",
        flush=True,
    )
    started = time.time()
    completed = 0
    for decision_group in chunks(decisions, decision_chunk):
        prefix_ids = [
            tokenizer.encode(decision["prefix"], add_special_tokens=False)
            for decision in decision_group
        ]
        flat: list[tuple[int, list[int], str, str]] = []
        for decision_index, (decision, ids) in enumerate(
            zip(decision_group, prefix_ids)
        ):
            suffix = decision["candidate_suffix"]
            for candidate in decision["candidates"]:
                flat.append((decision_index, ids, candidate, candidate + suffix))

        scores: dict[int, dict[str, float]] = defaultdict(dict)
        for item_group in chunks(flat, batch_size):
            values = candidate_logprobs(
                model,
                tokenizer,
                [(item[1], item[3]) for item in item_group],
                temperature,
                max_length,
            )
            for item, value in zip(item_group, values):
                scores[item[0]][item[2]] = value

        output = []
        for index, decision in enumerate(decision_group):
            logps = scores[index]
            center = max(logps.values())
            denominator = sum(math.exp(value - center) for value in logps.values())
            probabilities = {
                candidate: math.exp(value - center) / denominator
                for candidate, value in logps.items()
            }
            record = {
                key: value
                for key, value in decision.items()
                if key not in {"prefix", "candidates", "candidate_suffix", "ego_sex"}
            }
            record["candidate_sequence_logprobs"] = logps
            record["candidate_probs_constrained"] = probabilities
            record["temperature"] = temperature
            if decision["experiment"] == "sex_homophily":
                record["primary_probability"] = probabilities[decision["ego_sex"]]
                record["primary_name"] = "p_same_gender"
            elif decision["experiment"] == "closure":
                record["primary_probability"] = probabilities["especially_close"]
                record["primary_name"] = "p_especially_close"
            else:
                bucket_probs = {
                    "small": sum(
                        probabilities[str(value)] for value in range(1, 6)
                    ),
                    "medium": sum(
                        probabilities[str(value)] for value in range(6, 16)
                    ),
                    "large": sum(
                        probabilities[str(value)] for value in range(16, 31)
                    ),
                }
                record["bucket_probs"] = bucket_probs
                record["p_condition_consistent_size"] = bucket_probs[
                    decision["counterfactual_bucket"]
                ]
                record["primary_probability"] = (
                    bucket_probs["medium"] + 2 * bucket_probs["large"]
                )
                record["primary_name"] = "expected_size_bucket_index"
            output.append(record)
        append_jsonl(destination, output)
        completed += len(output)
        elapsed = time.time() - started
        print(
            f"[native] wrote={completed}/{len(decisions)} "
            f"elapsed_min={elapsed / 60:.1f}",
            flush=True,
        )


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def full_target(
    experiment: str,
    prediction: dict[str, Any] | None,
    prompt_json: dict[str, Any],
    condition: str,
) -> tuple[float | None, bool | None]:
    if prediction is None:
        return None, None
    if experiment == "sex_homophily":
        alters = prediction.get("alters")
        if not isinstance(alters, list):
            return None, None
        values = [
            alter.get("gender")
            for alter in alters
            if isinstance(alter, dict) and alter.get("gender") in {"male", "female"}
        ]
        if not values:
            return None, None
        target = sum(value == prompt_json["sex"] for value in values) / len(values)
        bounds = {
            "low": lambda x: x < 0.50,
            "medium": lambda x: 0.50 <= x < 0.75,
            "high": lambda x: x >= 0.75,
        }
        return target, bounds[condition](target)
    if experiment == "closure":
        ties = prediction.get("alter_ties")
        if not isinstance(ties, list):
            return None, None
        values = [
            tie.get("tie_strength")
            for tie in ties
            if isinstance(tie, dict)
            and tie.get("tie_strength")
            in {"especially_close", "know_each_other", "total_strangers"}
        ]
        if not values:
            return None, None
        target = sum(value == "especially_close" for value in values) / len(values)
        bounds = {
            "low": lambda x: x < 0.17,
            "medium": lambda x: 0.17 <= x < 0.50,
            "high": lambda x: x >= 0.50,
        }
        return target, bounds[condition](target)
    summary = prediction.get("composition_summary")
    if not isinstance(summary, dict):
        return None, None
    value = summary.get("num_alters")
    if not isinstance(value, (int, float)):
        return None, None
    target = float(value)
    bounds = {
        "small": lambda x: 1 <= x <= 5,
        "medium": lambda x: 6 <= x <= 15,
        "large": lambda x: 16 <= x <= 30,
    }
    return target, bounds[condition](target)


def run_full_generation(
    rows: list[dict[str, Any]],
    destination: Path,
    model: Any,
    tokenizer: Any,
    temperature: float,
    batch_size: int,
    t3_reps: int,
    size_reps: int,
    max_length: int,
    seed: int,
) -> None:
    done = (
        {row["sample_id"] for row in read_jsonl(destination)}
        if destination.exists()
        else set()
    )
    jobs: list[dict[str, Any]] = []
    for row in rows:
        reps = size_reps if row["experiment"] == "size_bucket" else t3_reps
        for replicate in range(reps):
            sample_id = f"{row['redesign_input_id']}_rep{replicate}"
            if sample_id not in done:
                jobs.append({**row, "sample_id": sample_id, "replicate": replicate})
    print(
        f"[generate] completed={len(done)} pending_samples={len(jobs)}",
        flush=True,
    )
    tokenizer.padding_side = "left"
    started = time.time()
    completed = 0
    # Keep batches homogeneous so each task receives its intended output budget.
    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        by_experiment[job["experiment"]].append(job)
    for experiment in EXPERIMENTS:
        experiment_jobs = by_experiment.get(experiment, [])
        max_new_tokens = EXPERIMENTS[experiment]["max_new_tokens"]
        for group_index, group in enumerate(chunks(experiment_jobs, batch_size)):
            rendered = [
                render_prompt(tokenizer, job["prompt_text"]) for job in group
            ]
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            ).to(model.device)
            torch.manual_seed(seed + group_index + 100_000 * list(EXPERIMENTS).index(experiment))
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=1.0,
                    top_k=0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            prompt_width = encoded.input_ids.shape[1]
            responses = tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
            output = []
            for job, response in zip(group, responses):
                prediction = extract_json_object(response)
                target, condition_ok = full_target(
                    experiment,
                    prediction,
                    job["prompt_json"],
                    job["counterfactual_bucket"],
                )
                output.append(
                    {
                        "sample_id": job["sample_id"],
                        "redesign_input_id": job["redesign_input_id"],
                        "experiment": experiment,
                        "respondent_id": job["respondent_id"],
                        "source_bucket": job["source_bucket"],
                        "counterfactual_bucket": job["counterfactual_bucket"],
                        "replicate": job["replicate"],
                        "temperature": temperature,
                        "raw_response": response,
                        "prediction_json": prediction,
                        "parsed_ok": prediction is not None,
                        "recomputed_target": target,
                        "condition_ok": condition_ok,
                    }
                )
            append_jsonl(destination, output)
            completed += len(output)
            elapsed = time.time() - started
            print(
                f"[generate:{experiment}] wrote={completed}/{len(jobs)} "
                f"elapsed_min={elapsed / 60:.1f}",
                flush=True,
            )


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    draws = [
        statistics.mean(rng.choice(values) for _ in values) for _ in range(2000)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def summarize(run_name: str) -> Path:
    run_dir = DECODING / "output" / run_name
    native_path = run_dir / "native_slot_scores.jsonl"
    full_path = run_dir / "full_json_samples.jsonl"
    output: dict[str, Any] = {}
    native_rows = read_jsonl(native_path) if native_path.exists() else []
    full_rows = read_jsonl(full_path) if full_path.exists() else []

    for experiment, spec in EXPERIMENTS.items():
        result: dict[str, Any] = {}
        relevant_native = [
            row for row in native_rows if row["experiment"] == experiment
        ]
        if relevant_native:
            ego_cells: dict[tuple[str, str], list[float]] = defaultdict(list)
            for row in relevant_native:
                ego_cells[
                    (row["respondent_id"], row["counterfactual_bucket"])
                ].append(float(row["primary_probability"]))
            native_cell = {
                key: statistics.mean(values) for key, values in ego_cells.items()
            }
            result["native_slot"] = {
                "n_decisions": len(relevant_native),
                "condition_means": {
                    condition: statistics.mean(
                        value
                        for (respondent, bucket), value in native_cell.items()
                        if bucket == condition
                    )
                    for condition in spec["conditions"]
                },
            }
            low, high = spec["conditions"][0], spec["conditions"][-1]
            respondents = sorted(
                {
                    respondent
                    for respondent, bucket in native_cell
                    if (respondent, low) in native_cell
                    and (respondent, high) in native_cell
                }
            )
            differences = [
                native_cell[(respondent, high)] - native_cell[(respondent, low)]
                for respondent in respondents
            ]
            result["native_slot"]["paired_last_minus_first"] = {
                "n": len(differences),
                "estimate": statistics.mean(differences),
                "bootstrap_ci_95": paired_bootstrap(
                    differences, 20260825 + list(EXPERIMENTS).index(experiment)
                ),
            }

        relevant_full = [
            row for row in full_rows if row["experiment"] == experiment
        ]
        if relevant_full:
            result["full_json"] = {"n_samples": len(relevant_full)}
            condition_summary = {}
            ego_cells_full: dict[tuple[str, str], list[float]] = defaultdict(list)
            for condition in spec["conditions"]:
                group = [
                    row
                    for row in relevant_full
                    if row["counterfactual_bucket"] == condition
                ]
                valid = [
                    float(row["recomputed_target"])
                    for row in group
                    if row.get("recomputed_target") is not None
                ]
                condition_summary[condition] = {
                    "n": len(group),
                    "parse_rate": sum(bool(row.get("parsed_ok")) for row in group)
                    / len(group),
                    "target_valid_rate": len(valid) / len(group),
                    "mean_recomputed_target": statistics.mean(valid)
                    if valid
                    else None,
                    "condition_compliance_rate": statistics.mean(
                        float(row["condition_ok"])
                        for row in group
                        if row.get("condition_ok") is not None
                    )
                    if any(row.get("condition_ok") is not None for row in group)
                    else None,
                }
                for row in group:
                    if row.get("recomputed_target") is not None:
                        ego_cells_full[
                            (row["respondent_id"], condition)
                        ].append(float(row["recomputed_target"]))
            result["full_json"]["condition_summary"] = condition_summary
            full_cell = {
                key: statistics.mean(values)
                for key, values in ego_cells_full.items()
            }
            low, high = spec["conditions"][0], spec["conditions"][-1]
            respondents = sorted(
                {
                    respondent
                    for respondent, bucket in full_cell
                    if (respondent, low) in full_cell
                    and (respondent, high) in full_cell
                }
            )
            differences = [
                full_cell[(respondent, high)] - full_cell[(respondent, low)]
                for respondent in respondents
            ]
            if differences:
                result["full_json"]["paired_last_minus_first"] = {
                    "n": len(differences),
                    "estimate": statistics.mean(differences),
                    "bootstrap_ci_95": paired_bootstrap(
                        differences, 20260925 + list(EXPERIMENTS).index(experiment)
                    ),
                }
        output[experiment] = result

    destination = run_dir / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[summarize] {destination}", flush=True)
    return destination


def load_model(model_path: Path) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="main")
    parser.add_argument(
        "--input-run-name",
        default=None,
        help="Reuse inputs from this run name; defaults to --run-name",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-egos", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--native-batch-size", type=int, default=24)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--decision-chunk", type=int, default=192)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--t3-reps", type=int, default=3)
    parser.add_argument("--size-reps", type=int, default=1)
    parser.add_argument(
        "--stages",
        default="prepare,native,generate,summarize",
        help="Comma-separated: prepare,native,generate,summarize",
    )
    args = parser.parse_args()
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown = set(stages) - {"prepare", "native", "generate", "summarize"}
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")

    input_run_name = args.input_run_name or args.run_name
    input_path = prepare_inputs(input_run_name, args.max_egos, args.seed)
    rows = select_shard(
        read_jsonl(input_path), args.shard_index, args.num_shards
    )
    print(
        f"[shard] index={args.shard_index}/{args.num_shards} "
        f"counterfactual_rows={len(rows)}",
        flush=True,
    )
    tokenizer = model = None
    if any(stage in stages for stage in ("native", "generate")):
        tokenizer, model = load_model(args.model)
        print(
            f"[model] {args.model} dtype={model.dtype} device={model.device}",
            flush=True,
        )

    run_dir = DECODING / "output" / args.run_name
    if "native" in stages:
        run_native(
            rows,
            run_dir / "native_slot_scores.jsonl",
            model,
            tokenizer,
            args.temperature,
            args.native_batch_size,
            args.decision_chunk,
            args.max_length,
        )
    if "generate" in stages:
        run_full_generation(
            rows,
            run_dir / "full_json_samples.jsonl",
            model,
            tokenizer,
            args.temperature,
            args.generation_batch_size,
            args.t3_reps,
            args.size_reps,
            args.max_length,
            args.seed,
        )
    if "summarize" in stages:
        summarize(args.run_name)


if __name__ == "__main__":
    main()
