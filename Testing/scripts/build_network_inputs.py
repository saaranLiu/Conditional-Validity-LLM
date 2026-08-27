#!/usr/bin/env python3
"""Build Phase 2 network-generation input JSONL for one survey.

UCNets Wave 1 is the primary dataset and runs the full B1-B7 arm set (no B1b).
Supplementary surveys declare only the arms their data can support, so the arm list,
the value vocabularies, the roster shape and the conditioning prose are all read from
`configs/content/{dataset}.yaml` rather than assumed.

    python build_network_inputs.py                 # ucnets_w1
    python build_network_inputs.py --dataset gss_1985
    python build_network_inputs.py --dataset gss_1985 --out-dir /tmp/check

UCNets prompt text is frozen because its generations are already collected: the
UCNets-shaped defaults below are what the config falls back to, so regenerating
ucnets_w1 inputs must stay byte-identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from network_schema import allowed_bundle, dataset_id, load_config, roster_spec, top_level_keys
from paths import arm_input, cleaned_dir, network_gold

ROOT = Path(__file__).resolve().parents[2]

SUBSAMPLE_SEED = 20260806

# Composition numbers and rosters must never reach the prompt. Coarse bucket fields are
# the explicitly-allowed conditioning channels and are checked separately below.
BASE_LEAK_KEYS = {
    "num_alters",
    "kin_share",
    "friend_share",
    "coworker_share",
    "confidant_count",
    "companion_count",
    "pct_weekly_plus_contact",
    "same_race_share",
    "core_alters",
    "alters",
    "composition_summary",
    "density",
    "esp_close_density",
    "same_sex_share",
    "age_homophily_share",
    "resource_summary",
    "occupational_access_count",
    "kin_channel_share",
    "support_family_share",
    "institutional_reliance_share",
    "weekly_plus_role_contact_share",
}

# UCNets defaults. A survey whose coding differs overrides these in its own config.
UCNETS_PRIORS_FIELDS = [
    "mean_num_alters",
    "median_num_alters",
    "mean_kin_share",
    "mean_friend_share",
    "median_confidant_count",
    "size_bucket_shares",
]
UCNETS_SIZE_RULE = (
    "- size_bucket is an observed conditioning attribute; num_alters must fall in that bucket "
    "(small=1-5, medium=6-15, large=16-30)."
)
UCNETS_PRIORS_RULE = (
    "- population_network_priors are sample-level anchors; do not force every person to the mean."
)
UCNETS_BUCKET_CONDITION = {
    "source": "same_race_share_bucket",
    "prompt_key": "same_race_share_bucket",
    "fallback": "high",
    "rule": (
        "- same_race_share_bucket is an observed conditioning attribute describing the share of this "
        "person's alters who share their own race/ethnicity (low<0.75, medium 0.75-0.99, high=1.0). "
        "The same_race_as_ego values among core_alters should be consistent with this bucket."
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def arm_stem(task: str) -> str:
    """n4_homophily -> b5"""
    return task.split("_", 1)[0]


def task_subsample_n(cfg: dict[str, Any], task: str) -> int | None:
    return (cfg["tasks"][task] or {}).get("subsample_n")


def task_subsample_per_country(cfg: dict[str, Any], task: str) -> int | None:
    return (cfg["tasks"][task] or {}).get("subsample_per_country")


def stratified_country_ids(gold_rows: list[dict[str, Any]], n_per: int, rng) -> set[str]:
    from collections import defaultdict

    by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in gold_rows:
        code = (g.get("state_json") or {}).get("country_code") or "UNK"
        by_country[code].append(g)
    chosen: set[str] = set()
    for code in sorted(by_country):
        group = sorted(by_country[code], key=lambda g: g["respondent_id"])
        rng.shuffle(group)
        chosen.update(g["respondent_id"] for g in group[:n_per])
    return chosen


def task_requires_bucket(cfg: dict[str, Any], task: str) -> bool:
    """Whether an arm should drop respondents whose conditioning bucket is undefined.

    Off by default, so `fallback` keeps its existing behaviour everywhere it is already
    relied on. Turn it on when the fallback would be a fiction rather than a default:
    closure is undefined for a one-alter network, and handing such a respondent an
    invented "medium" closure asks the model to satisfy a constraint that cannot hold,
    while leaving nothing in the gold to score the answer against.
    """
    return bool((cfg["tasks"][task] or {}).get("require_condition_bucket"))


def bucket_condition(cfg: dict[str, Any], task: str) -> dict[str, Any] | None:
    """Which observed bucket this arm conditions on, if any."""
    task_cfg = cfg["tasks"][task] or {}
    declared = task_cfg.get("condition_bucket")
    if declared:
        return declared
    if task_cfg.get("include_same_race_bucket"):
        return UCNETS_BUCKET_CONDITION
    return None


def schema_descriptions(cfg: dict[str, Any]) -> str:
    """Per-survey field description block.

    A survey supplies `prompt.schema_description_lines` to describe its own coding.
    Without it we render the UCNets wording, which is frozen.
    """
    allowed = allowed_bundle(cfg)
    override = (cfg.get("prompt") or {}).get("schema_description_lines")
    if override:
        vocab = {k: v for k, v in allowed.items() if isinstance(v, list)}
        lines = ["Field descriptions and allowed values:"]
        for line in override:
            lines.append(line.format(**{k: str(v) for k, v in vocab.items()}))
        return "\n".join(lines)

    roster = roster_spec(cfg)
    comp = allowed["composition_summary"]
    n_max = (comp.get("num_alters") or {}).get("max", 30)
    lines = [
        "Field descriptions and allowed values:",
        f"- composition_summary.num_alters: integer network size in [1, {n_max}]",
        f"- composition_summary.size_bucket: one of {allowed['size_bucket']}",
        "- composition_summary.kin_share / friend_share / coworker_share: proportions in [0, 1]",
        "- composition_summary.confidant_count / companion_count: non-negative integers",
        "- composition_summary.pct_weekly_plus_contact: proportion with daily/weekly contact in [0, 1]",
        f"- {roster['key']}: array of exactly {roster['size']} objects",
        f"  - {roster['rank_field']}: 1..{roster['size']}",
        f"  - relationship: one of {allowed['relationship']}",
        f"  - support_roles: array subset of {allowed['support_roles']}",
        f"  - contact_frequency: one of {allowed['contact_frequency']}",
        f"  - gender: one of {allowed['gender']}",
        f"  - same_race_as_ego: one of {allowed['same_race_as_ego']}",
    ]
    return "\n".join(lines)


def build_oneshot_prompt(cfg: dict[str, Any], task: str, prompt_json: dict[str, Any]) -> str:
    task_cfg = cfg["tasks"][task]
    opening = cfg["prompt"]["opening"].strip()
    year = cfg.get("population_context", {}).get("reference_year")
    rules = list(cfg["prompt"]["baseline_rules"])
    context = cfg["context"].strip()
    if "{country}" in context:
        context = context.format(country=prompt_json.get("country", "this country"))
    lines = [f"{opening} {context}"]
    if year is not None:
        lines.append(f"The data represent a cross-sectional snapshot collected in the year {year}.")
    lines.extend(
        [
            "",
            f"Task: {task_cfg['instruction'].strip()}",
            "",
            "Conditioned on the following input attributes:",
            json_block(prompt_json),
            "",
            "Return output strictly as JSON.",
            "Do not include explanations, markdown fences, comments, or additional keys.",
            "The JSON object must contain EXACTLY these top-level keys:",
            json_block(top_level_keys(cfg)),
            "",
            schema_descriptions(cfg),
            "",
            "Rules:",
        ]
    )
    for rule in rules:
        lines.append(f"- {rule}")
    if task_cfg.get("include_size_bucket"):
        lines.append(cfg.get("size_bucket_rule") or UCNETS_SIZE_RULE)
    if task_cfg.get("include_population_priors"):
        lines.append(cfg.get("population_priors_rule") or UCNETS_PRIORS_RULE)
    condition = bucket_condition(cfg, task)
    if condition:
        lines.append(condition["rule"])
    lines.append("")
    lines.append("JSON output:")
    return "\n".join(lines)


def assert_no_leak(cfg: dict[str, Any], prompt_json: dict[str, Any], task: str) -> None:
    condition = bucket_condition(cfg, task)
    condition_key = condition["prompt_key"] if condition else None
    leak_keys = BASE_LEAK_KEYS | set(cfg.get("extra_leak_keys") or [])
    # The conditioning bucket is coarse and intended; its raw value never is.
    leak_keys.discard(condition_key)
    for k in prompt_json:
        if k in leak_keys:
            raise ValueError(f"Leakage key {k} in prompt_json for {task}")

    prior_tasks = {t for t, tc in cfg["tasks"].items() if (tc or {}).get("include_population_priors")}
    if "population_network_priors" in prompt_json and task not in prior_tasks:
        raise ValueError(f"population priors not declared for {task}")

    size_tasks = {t for t, tc in cfg["tasks"].items() if (tc or {}).get("include_size_bucket")}
    if "size_bucket" in prompt_json and task not in size_tasks:
        raise ValueError(f"size_bucket condition not declared for {task}")

    for other in cfg["tasks"]:
        other_condition = bucket_condition(cfg, other)
        if not other_condition:
            continue
        key = other_condition["prompt_key"]
        if key in prompt_json and key != condition_key:
            raise ValueError(f"{key} condition only allowed for {other}, got {task}")


def build_record(
    *,
    idx: int,
    gold: dict[str, Any],
    task: str,
    cfg: dict[str, Any],
    priors: dict[str, Any],
) -> dict[str, Any]:
    dataset = dataset_id(cfg)
    task_cfg = cfg["tasks"][task]
    target = gold.get("target_json") or {}
    composition = target.get("composition_summary") or {}
    prompt_json = dict(gold["state_json"])
    for key in cfg.get("prompt_exclude_keys") or []:
        prompt_json.pop(key, None)

    if task_cfg.get("include_size_bucket"):
        prompt_json["size_bucket"] = composition["size_bucket"]

    condition = bucket_condition(cfg, task)
    if condition:
        bucket = composition.get(condition["source"])
        if bucket is None:
            bucket = condition.get("fallback")
        prompt_json[condition["prompt_key"]] = bucket

    if task_cfg.get("include_population_priors"):
        fields = cfg.get("priors_prompt_fields") or UCNETS_PRIORS_FIELDS
        prompt_json["population_network_priors"] = {f: priors[f] for f in fields}

    assert_no_leak(cfg, prompt_json, task)
    mode = task_cfg.get("generation_mode", "oneshot")
    allowed = allowed_bundle(cfg)
    roster = roster_spec(cfg)
    roster_key = roster.get("key")

    if roster.get("mode") == "none":
        allowed_values = {"resource_summary": allowed["resource_summary"]}
    else:
        allowed_values = {
            "composition_summary": allowed["composition_summary"],
            roster_key: f"list[{'core_alter' if roster_key == 'core_alters' else 'alter'}]",
            "core_alter": allowed["core_alter"],
        }

    rec: dict[str, Any] = {
        "input_id": f"{dataset}_{task}_{idx:05d}",
        "respondent_id": gold["respondent_id"],
        "dataset": dataset,
        "task": task,
        "prompt_condition": task_cfg.get("prompt_condition", "format_only"),
        "generation_mode": mode,
        "prompt_json": prompt_json,
        "prompt_text": build_oneshot_prompt(cfg, task, prompt_json) if mode == "oneshot" else None,
        "metadata": {
            "weight": (gold.get("metadata") or {}).get("weight"),
            "size_bucket": composition.get("size_bucket"),
            "n_alters_raw": (gold.get("metadata") or {}).get("n_alters_raw"),
            "country": (gold.get("state_json") or {}).get("country"),
            "country_code": (gold.get("state_json") or {}).get("country_code"),
            "gold_target": gold["target_json"],
        },
        "expected_output_keys": top_level_keys(cfg),
        "allowed_values": allowed_values,
    }
    if mode == "multistep_roster":
        rec["multistep"] = {
            "n_calls_planned": 5,
            "note": (
                "Steps 1-4 build named alters; step 5 asks composition_summary for fair B1 comparison. "
                "Runner assembles final schema."
            ),
        }
        # Minimal prompt_text for logging / oneshot fallback inspection
        rec["prompt_text"] = (
            f"{cfg['prompt']['opening'].strip()} {cfg['context'].strip()}\n"
            f"Task: roster-style multi-step network generation (B4).\n"
            f"State:\n{json_block(prompt_json)}\n"
        )
    return rec


def build_n0_remainder(dataset: str, out_dir: Path | None) -> int:
    """N0 prompts for gold respondents not already in the T3 subsample."""
    cfg = load_config(dataset=dataset)
    dataset = dataset_id(cfg)
    if "n0_baseline_network" not in cfg["tasks"]:
        raise SystemExit(f"{dataset} has no n0_baseline_network task")
    gold_rows = read_jsonl(network_gold(dataset))
    priors_path = cleaned_dir(dataset) / f"{dataset}_population_priors.json"
    with priors_path.open(encoding="utf-8") as f:
        priors = json.load(f)

    input_dir = arm_input(dataset, "n0_baseline_network").parent
    sub_path = input_dir / "n0n4c_subsample_ids.jsonl"
    if not sub_path.exists():
        raise SystemExit(f"missing subsample id file: {sub_path}")
    have = {row["respondent_id"] for row in read_jsonl(sub_path)}
    leftover = [g for g in gold_rows if g["respondent_id"] not in have]
    rows = [
        build_record(idx=i, gold=g, task="n0_baseline_network", cfg=cfg, priors=priors)
        for i, g in enumerate(leftover, start=1)
    ]
    # Distinct input_ids so resume does not collide with the original 00001..01000.
    for rec in rows:
        rec["input_id"] = rec["input_id"].replace(
            "n0_baseline_network_", "n0_baseline_network_rem_"
        )
    dest = (Path(out_dir) if out_dir is not None else input_dir) / "n0_remainder_input.jsonl"
    write_jsonl(dest, rows)
    print(json.dumps({"wrote": str(dest), "n_remainder": len(rows), "n_subsample": len(have)}))
    return 0


def main() -> int:
    import random

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ucnets_w1", help="dataset id (config stem)")
    parser.add_argument("--out-dir", type=Path, default=None, help="override output directory")
    parser.add_argument(
        "--remainder",
        action="store_true",
        help="Write n0 inputs for gold ids not in the existing subsample (does not rewrite n0/n4c).",
    )
    args = parser.parse_args()

    if args.remainder:
        return build_n0_remainder(args.dataset, args.out_dir)

    cfg = load_config(dataset=args.dataset)
    dataset = dataset_id(cfg)
    clean = cleaned_dir(dataset)
    out_dir = args.out_dir

    gold_rows = read_jsonl(network_gold(dataset))
    with (clean / f"{dataset}_population_priors.json").open(encoding="utf-8") as f:
        priors = json.load(f)

    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(SUBSAMPLE_SEED)
    # Fixed shared subsample so multistep combo arms are evaluated on the *same*
    # respondents (paired comparison against the full-sample arms restricted to it).
    subsample_ids = None
    counts = {}
    subsample_manifest = {}
    excluded: dict[str, Any] = {}
    for task in cfg["tasks"]:
        n_sub = task_subsample_n(cfg, task)
        n_per_country = task_subsample_per_country(cfg, task)
        rows_src = gold_rows
        if n_per_country is not None:
            if subsample_ids is None:
                subsample_ids = stratified_country_ids(gold_rows, int(n_per_country), rng)
            rows_src = [g for g in gold_rows if g["respondent_id"] in subsample_ids]
            subsample_manifest[task] = {
                "n": len(rows_src),
                "seed": SUBSAMPLE_SEED,
                "per_country": int(n_per_country),
            }
        elif n_sub is not None:
            if subsample_ids is None:
                shuffled = sorted(gold_rows, key=lambda g: g["respondent_id"])
                rng.shuffle(shuffled)
                subsample_ids = {g["respondent_id"] for g in shuffled[:n_sub]}
            rows_src = [g for g in gold_rows if g["respondent_id"] in subsample_ids]
            subsample_manifest[task] = {"n": len(rows_src), "seed": SUBSAMPLE_SEED}
        condition = bucket_condition(cfg, task)
        if condition and task_requires_bucket(cfg, task):
            before = len(rows_src)
            rows_src = [
                g
                for g in rows_src
                if g["target_json"]["composition_summary"].get(condition["source"]) is not None
            ]
            excluded[task] = {
                "n_excluded": before - len(rows_src),
                "reason": f"{condition['source']} 无定义（该 ego 不满足该估计量的前提）",
            }
        rows = [
            build_record(idx=i, gold=g, task=task, cfg=cfg, priors=priors)
            for i, g in enumerate(rows_src, start=1)
        ]
        if out_dir is not None:
            dest = Path(out_dir) / f"{dataset}_{arm_stem(task)}_input.jsonl"
        else:
            dest = arm_input(dataset, task)
        write_jsonl(dest, rows)
        counts[task] = len(rows)
        print(f"wrote {dest} n={len(rows)}")

    if subsample_ids is not None:
        stems = sorted({arm_stem(t) for t in subsample_manifest})
        sub_name = f"{dataset}_{''.join(stems)}_subsample_ids.jsonl"
        if out_dir is not None:
            sub_path = Path(out_dir) / sub_name
        else:
            first_task = next(iter(subsample_manifest))
            sub_path = arm_input(dataset, first_task).parent / f"{''.join(stems)}_subsample_ids.jsonl"
        write_jsonl(
            sub_path,
            [{"respondent_id": rid} for rid in sorted(subsample_ids)],
        )

    manifest = {
        "dataset": dataset,
        "tasks": counts,
        "skipped_tasks": list(cfg.get("skipped_tasks") or []),
        "n_gold": len(gold_rows),
        "priors": priors,
        "subsample_tasks": subsample_manifest,
        "excluded_undefined_bucket": excluded,
    }
    if out_dir is not None:
        manifest_path = Path(out_dir) / f"{dataset}_manifest.json"
    else:
        first_task = next(iter(cfg["tasks"]))
        manifest_path = arm_input(dataset, first_task).parent / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
