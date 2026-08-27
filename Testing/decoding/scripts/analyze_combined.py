#!/usr/bin/env python3
"""Pool four decoding shards and produce ego-paired combined diagnostics."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "Testing" / "decoding" / "output" / "main"
INPUT = (
    ROOT
    / "Testing"
    / "decoding"
    / "input"
    / "main"
    / "counterfactual_inputs.jsonl"
)
CONDITIONS = {
    "sex_homophily": ("low", "medium", "high"),
    "closure": ("low", "medium", "high"),
    "size_bucket": ("small", "medium", "large"),
}
GOLD_FIELD = {
    "sex_homophily": "same_sex_share",
    "closure": "esp_close_density",
    "size_bucket": "num_alters",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    draws = [
        statistics.mean(rng.choice(values) for _ in values) for _ in range(3000)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def correlations(left: list[float], right: list[float]) -> dict[str, float | None]:
    if len(left) < 3 or statistics.pstdev(left) == 0 or statistics.pstdev(right) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(pearsonr(left, right).statistic),
        "spearman": float(spearmanr(left, right).statistic),
    }


def structure_complete(experiment: str, prediction: dict[str, Any] | None) -> bool:
    if not isinstance(prediction, dict):
        return False
    summary = prediction.get("composition_summary")
    if not isinstance(summary, dict):
        return False
    if experiment == "size_bucket":
        value = summary.get("num_alters")
        core = prediction.get("core_alters")
        if not isinstance(value, (int, float)) or not 1 <= float(value) <= 30:
            return False
        if not isinstance(core, list) or len(core) != 5:
            return False
        return {
            alter.get("rank")
            for alter in core
            if isinstance(alter, dict)
        } == {1, 2, 3, 4, 5}

    alters = prediction.get("alters")
    ties = prediction.get("alter_ties")
    value = summary.get("num_alters")
    if not isinstance(value, (int, float)) or not isinstance(alters, list):
        return False
    if int(value) != len(alters) or not 1 <= len(alters) <= 5:
        return False
    alter_ids = {
        alter.get("alter_id")
        for alter in alters
        if isinstance(alter, dict)
    }
    if len(alter_ids) != len(alters) or None in alter_ids:
        return False
    if not isinstance(ties, list):
        return False
    observed = {
        tuple(sorted((tie.get("src_id"), tie.get("tgt_id"))))
        for tie in ties
        if isinstance(tie, dict)
        and tie.get("src_id") in alter_ids
        and tie.get("tgt_id") in alter_ids
        and tie.get("src_id") != tie.get("tgt_id")
    }
    expected = {
        tuple(sorted((left, right)))
        for left in alter_ids
        for right in alter_ids
        if left != right
    }
    return len(ties) == len(observed) and observed == expected


def contrast(
    cells: dict[tuple[str, str], float],
    left: str,
    right: str,
    seed: int,
) -> dict[str, Any]:
    respondents = sorted(
        respondent
        for respondent, condition in cells
        if condition == left
        and (respondent, right) in cells
    )
    values = [
        cells[(respondent, right)] - cells[(respondent, left)]
        for respondent in respondents
    ]
    return {
        "n": len(values),
        "estimate": statistics.mean(values),
        "bootstrap_ci_95": bootstrap_mean(values, seed),
        "sd": statistics.stdev(values),
        "positive_share": sum(value > 0 for value in values) / len(values),
    }


def main() -> None:
    native_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    for shard in range(4):
        directory = RUN / f"shard_{shard}"
        native_rows.extend(read_jsonl(directory / "native_slot_scores.jsonl"))
        full_rows.extend(read_jsonl(directory / "full_json_samples.jsonl"))

    input_rows = read_jsonl(INPUT)
    output: dict[str, Any] = {
        "run_counts": {
            "counterfactual_inputs": len(input_rows),
            "native_decisions": len(native_rows),
            "full_json_samples": len(full_rows),
        },
        "experiments": {},
    }

    for experiment_index, (experiment, conditions) in enumerate(CONDITIONS.items()):
        native = [row for row in native_rows if row["experiment"] == experiment]
        full = [row for row in full_rows if row["experiment"] == experiment]

        native_group: dict[tuple[str, str], list[float]] = defaultdict(list)
        native_aligned: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in native:
            key = (row["respondent_id"], row["counterfactual_bucket"])
            native_group[key].append(float(row["primary_probability"]))
            if experiment == "size_bucket":
                native_aligned[key].append(
                    float(row["p_condition_consistent_size"])
                )
        native_cells = {
            key: statistics.mean(values) for key, values in native_group.items()
        }

        full_group: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in full:
            if row.get("recomputed_target") is not None:
                full_group[
                    (row["respondent_id"], row["counterfactual_bucket"])
                ].append(float(row["recomputed_target"]))
        full_cells = {
            key: statistics.mean(values) for key, values in full_group.items()
        }
        complete_full = [
            row
            for row in full
            if structure_complete(experiment, row.get("prediction_json"))
            and row.get("recomputed_target") is not None
        ]
        complete_group: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in complete_full:
            complete_group[
                (row["respondent_id"], row["counterfactual_bucket"])
            ].append(float(row["recomputed_target"]))
        complete_cells = {
            key: statistics.mean(values) for key, values in complete_group.items()
        }

        result: dict[str, Any] = {
            "n_egos": len({row["respondent_id"] for row in native}),
            "n_native_decisions": len(native),
            "n_full_json_samples": len(full),
            "native_slot": {
                "condition_summary": {
                    condition: {
                        "mean": statistics.mean(
                            value
                            for (respondent, bucket), value in native_cells.items()
                            if bucket == condition
                        ),
                        "sd_across_egos": statistics.stdev(
                            value
                            for (respondent, bucket), value in native_cells.items()
                            if bucket == condition
                        ),
                    }
                    for condition in conditions
                }
            },
            "full_json": {
                "condition_summary": {},
            },
        }

        if experiment == "size_bucket":
            aligned_cells = {
                key: statistics.mean(values)
                for key, values in native_aligned.items()
            }
            for condition in conditions:
                result["native_slot"]["condition_summary"][condition][
                    "mean_probability_in_input_bucket"
                ] = statistics.mean(
                    value
                    for (respondent, bucket), value in aligned_cells.items()
                    if bucket == condition
                )

        for condition in conditions:
            group = [
                row for row in full if row["counterfactual_bucket"] == condition
            ]
            valid = [
                float(row["recomputed_target"])
                for row in group
                if row.get("recomputed_target") is not None
            ]
            compliance = [
                float(row["condition_ok"])
                for row in group
                if row.get("condition_ok") is not None
            ]
            cell_values = [
                value
                for (respondent, bucket), value in full_cells.items()
                if bucket == condition
            ]
            result["full_json"]["condition_summary"][condition] = {
                "n_samples": len(group),
                "parse_rate": sum(bool(row["parsed_ok"]) for row in group)
                / len(group),
                "structure_complete_rate": sum(
                    structure_complete(experiment, row.get("prediction_json"))
                    for row in group
                )
                / len(group),
                "target_valid_rate": len(valid) / len(group),
                "mean_recomputed_target": statistics.mean(valid),
                "sd_across_ego_means": statistics.stdev(cell_values),
                "condition_compliance_rate": statistics.mean(compliance),
            }
        result["full_json"]["structure_complete_sensitivity"] = {
            "condition_means": {
                condition: statistics.mean(
                    value
                    for (respondent, bucket), value in complete_cells.items()
                    if bucket == condition
                )
                for condition in conditions
            },
            "paired_last_minus_first": contrast(
                complete_cells,
                conditions[0],
                conditions[-1],
                20261225 + experiment_index,
            ),
        }

        for layer_name, cells in (
            ("native_slot", native_cells),
            ("full_json", full_cells),
        ):
            contrasts = {}
            for adjacent_index, (left, right) in enumerate(
                zip(conditions, conditions[1:])
            ):
                contrasts[f"{right}_minus_{left}"] = contrast(
                    cells,
                    left,
                    right,
                    20260825
                    + 100 * experiment_index
                    + 10 * (layer_name == "full_json")
                    + adjacent_index,
                )
            contrasts[f"{conditions[-1]}_minus_{conditions[0]}"] = contrast(
                cells,
                conditions[0],
                conditions[-1],
                20260925
                + 100 * experiment_index
                + 10 * (layer_name == "full_json"),
            )
            result[layer_name]["paired_contrasts"] = contrasts
            respondents = sorted(
                {
                    respondent
                    for respondent, condition in cells
                    if all((respondent, value) in cells for value in conditions)
                }
            )
            result[layer_name]["strict_monotonic_ego_share"] = sum(
                all(
                    cells[(respondent, right)] > cells[(respondent, left)]
                    for left, right in zip(conditions, conditions[1:])
                )
                for respondent in respondents
            ) / len(respondents)

        if experiment != "size_bucket":
            common = sorted(set(native_cells) & set(full_cells))
            result["native_full_association"] = {
                "all_ego_conditions": correlations(
                    [native_cells[key] for key in common],
                    [full_cells[key] for key in common],
                ),
                "within_condition": {
                    condition: correlations(
                        [
                            native_cells[key]
                            for key in common
                            if key[1] == condition
                        ],
                        [
                            full_cells[key]
                            for key in common
                            if key[1] == condition
                        ],
                    )
                    for condition in conditions
                },
            }
            common_egos = sorted(
                {
                    respondent
                    for respondent, condition in common
                    if (respondent, conditions[0]) in native_cells
                    and (respondent, conditions[-1]) in native_cells
                    and (respondent, conditions[0]) in full_cells
                    and (respondent, conditions[-1]) in full_cells
                }
            )
            result["native_full_association"]["endpoint_delta"] = correlations(
                [
                    native_cells[(respondent, conditions[-1])]
                    - native_cells[(respondent, conditions[0])]
                    for respondent in common_egos
                ],
                [
                    full_cells[(respondent, conditions[-1])]
                    - full_cells[(respondent, conditions[0])]
                    for respondent in common_egos
                ],
            )

            within_cell_sd = [
                statistics.stdev(values)
                for values in full_group.values()
                if len(values) > 1
            ]
            result["full_json"]["mean_within_ego_condition_sampling_sd"] = (
                statistics.mean(within_cell_sd)
            )

        factual: dict[str, list[float]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        for row in input_rows:
            if row["experiment"] != experiment:
                continue
            key = (experiment, row["respondent_id"])
            if key in seen:
                continue
            seen.add(key)
            composition = (row.get("gold_target") or {}).get(
                "composition_summary", {}
            )
            value = composition.get(GOLD_FIELD[experiment])
            if value is not None:
                factual[row["source_bucket"]].append(float(value))
        result["human_factual_reference"] = {
            condition: {
                "n": len(factual[condition]),
                "mean": statistics.mean(factual[condition]),
            }
            for condition in conditions
        }
        output["experiments"][experiment] = result

    destination = RUN / "combined_analysis.json"
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
