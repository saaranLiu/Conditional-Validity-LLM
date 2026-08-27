#!/usr/bin/env python3
"""Build LLM input JSONL files for Phase 1 state-stage experiments.

Schema, descriptions, allowed values, and prompt wording are loaded from
Testing/configs/state/*.yaml. Per-respondent conditioning values
still come from cleaned task JSONL files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paths import arm_input
from state_content_config import ContentConfig, DATASET_CONFIGS


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "Testing" / "T1_association" / "state_input_manifest.json"

INPUT_FIELD_ORDER = [
    "input_id",
    "respondent_id",
    "dataset",
    "task",
    "prompt_condition",
    "prompt_json",
    "prompt_text",
    "metadata",
    "expected_output_keys",
    "allowed_values",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def order_record(row: dict[str, Any], key_order: list[str]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in key_order:
        if key in row:
            ordered[key] = row[key]
    for key, value in row.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def write_jsonl(path: Path, rows: list[dict[str, Any]], key_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = order_record(row, key_order) if key_order else row
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def build_prompt_text(
    config: ContentConfig,
    task: str,
    prompt_json: dict[str, Any],
) -> str:
    output_keys = config.output_keys_for_task(task)
    allowed_values = config.allowed_values_for_task(task)
    descriptions = config.descriptions_for_task(task)
    schema = config.expected_schema(task)
    opening = config.prompt_opening()
    rules = config.prompt_rules(task)

    lines = [
        f"{opening} {config.context}",
    ]
    if config.reference_year is not None:
        lines.append(
            f"The data represent a cross-sectional snapshot collected in the year {config.reference_year}."
        )
    lines.extend(
        [
            "",
            f"Task: {config.task_instruction(task)}",
            "",
            "Conditioned on the following input attributes:",
            json_block(prompt_json),
            "",
            "Return output strictly as JSON.",
            "Do not include explanations, markdown fences, comments, or additional keys.",
            "The JSON object must contain EXACTLY these top-level keys:",
            json_block(output_keys),
            "",
            "Rules:",
        ]
    )
    lines.extend(f"{index}. {rule}" for index, rule in enumerate(rules, start=1))
    lines.extend(
        [
            "",
            "Expected JSON shape:",
            json_block(schema),
            "",
            "Variable descriptions:",
            json_block(descriptions),
            "",
            "Allowed values:",
            json_block(allowed_values),
        ]
    )
    return "\n".join(lines)


def build_input_records(config: ContentConfig, task: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    output_keys = config.output_keys_for_task(task)
    allowed_values = config.allowed_values_for_task(task)
    warnings: list[str] = []
    input_rows = []

    for index, row in enumerate(rows, start=1):
        prompt_json = row["prompt_json"]
        row_warnings = config.validate_prompt_json(prompt_json, task)
        for message in row_warnings:
            warnings.append(f"{config.dataset} {task} respondent {row['respondent_id']}: {message}")

        input_rows.append(
            {
                "input_id": f"{config.dataset}_{task}_{index:05d}",
                "respondent_id": row["respondent_id"],
                "dataset": config.dataset,
                "task": task,
                "prompt_condition": config.prompt_condition(task),
                "prompt_json": prompt_json,
                "prompt_text": build_prompt_text(config, task, prompt_json),
                "metadata": {
                    "source_weight": row.get("metadata", {}).get("weight"),
                    "sensitive_fields_included": row.get("metadata", {}).get("sensitive_fields_included"),
                    "target_available_in_cleaned_data": True,
                    "content_config": str(DATASET_CONFIGS[config.dataset].relative_to(ROOT)),
                },
                "expected_output_keys": output_keys,
                "allowed_values": allowed_values,
            }
        )
    return input_rows, warnings


def write_manifest(outputs: dict[str, int], configs: dict[str, str]) -> None:
    manifest = {
        "purpose": "LLM model input files for Phase 1 state-stage experiments.",
        "important_note": (
            "These files do not contain human target_json answers; targets remain in "
            "Survey/ucnets/cleaned for later evaluation."
        ),
        "prompt_conditions": {
            "s0_baseline_state": "format_only",
            "s0b_anti_typology": "format_anti_typology",
            "s1_health_conditioned": "format_only",
            "s1b_anti_typology": "format_anti_typology",
            "s2_politics_religion_conditioned": "format_only",
        },
        "experiment_tiers": {
            "baseline": {
                "description": "Primary experiment with format rules 1-2 only.",
                "tasks": [
                    "s0_baseline_state",
                    "s1_health_conditioned",
                ],
            },
            "supplementary": {
                "description": "Supplementary experiment adding anti-typology rules 3-4.",
                "tasks": [
                    "s0b_anti_typology",
                    "s1b_anti_typology",
                ],
            },
            "task3_conditioning": {
                "description": "Party/religion/voting conditioned state completion (no 3b).",
                "tasks": [
                    "s2_politics_religion_conditioned",
                ],
            },
        },
        "content_configs": configs,
        "files": outputs,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", dest="datasets", help="Build only these datasets")
    parser.add_argument("--task", action="append", dest="tasks", help="Build only these tasks")
    args = parser.parse_args()

    datasets = args.datasets or list(DATASET_CONFIGS)
    unknown = [d for d in datasets if d not in DATASET_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}")

    outputs: dict[str, int] = {}
    all_warnings: list[str] = []

    for dataset in datasets:
        config = ContentConfig.load(dataset)
        tasks = args.tasks or list(config.tasks)
        for task in tasks:
            if task not in config.tasks:
                raise SystemExit(f"{dataset} has no task {task}")
            source = config.task_file(task)
            rows = read_jsonl(source)
            input_rows, warnings = build_input_records(config, task, rows)
            all_warnings.extend(warnings)
            out_jsonl = arm_input(dataset, task)
            write_jsonl(out_jsonl, input_rows, INPUT_FIELD_ORDER)
            outputs[str(out_jsonl.relative_to(ROOT))] = len(input_rows)

    if all_warnings:
        if args.datasets or args.tasks:
            warning_path = ROOT / "Testing" / "T1_association" / datasets[0] / "eval" / "schema_validation_warnings.txt"
        else:
            warning_path = ROOT / "Testing" / "T1_association" / "eval" / "schema_validation_warnings.txt"
        warning_path.parent.mkdir(parents=True, exist_ok=True)
        warning_path.write_text("\n".join(all_warnings) + "\n", encoding="utf-8")
        print(f"Wrote {len(all_warnings)} schema validation warnings to {warning_path}")

    # A filtered rebuild must not overwrite the global T1 manifest with a subset.
    if not args.datasets and not args.tasks:
        write_manifest(outputs, {d: str(p.relative_to(ROOT)) for d, p in DATASET_CONFIGS.items()})
    for name, count in outputs.items():
        print(f"{name}: {count}")
    print("Wrote model input files under Testing/")


if __name__ == "__main__":
    main()
