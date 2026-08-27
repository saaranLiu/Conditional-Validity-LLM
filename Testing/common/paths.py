"""Canonical paths and arm-name mapping after the Survey/Testing reorg."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTING = ROOT / "Testing"
SURVEY = ROOT / "Survey"
COMMON = TESTING / "common"
SCRIPTS = COMMON  # backward-compatible alias
STATE_CONFIG_DIR = TESTING / "configs" / "state"
NETWORK_CONFIG_DIR = TESTING / "configs" / "network"
LOGS = TESTING / "logs"

DATASET_SURVEY = {
    "ucnets_w1": "ucnets",
    "gss_1985": "gss",
    "gss_1987": "gss",
    "gss_2004": "gss",
    "gss_2010_network": "gss",
    "gss_2024": "gss",
    "nshap_r1": "nshap",
    "nshap_r2": "nshap",
    "nshap_r3": "nshap",
    "signos": "signos",
    "issp_2017": "issp",
}

SURVEY_DIR = {
    "ucnets": SURVEY / "ucnets",
    "gss": SURVEY / "gss",
    "nshap": SURVEY / "nshap",
    "signos": SURVEY / "signos",
    "issp": SURVEY / "issp",
    "add_health": SURVEY / "add_health",
}

# Full config keys, new names only.
OLD_TO_NEW = {
    "task1_baseline_state_imputation": "s0_baseline_state",
    "task1b_anti_typology_ablation": "s0b_anti_typology",
    "task2_health_conditioned_state_completion": "s1_health_conditioned",
    "task2b_anti_typology_ablation": "s1b_anti_typology",
    "task3_politics_religion_conditioned_state_completion": "s2_politics_religion_conditioned",
    "b1_baseline_network_generation": "n0_baseline_network",
    "b1b_anti_typology_network_generation": "n0b_anti_typology",
    "b2_size_bucket_conditioned_network_generation": "n1_size_bucket",
    "b3_population_prior_conditioned_network_generation": "n2_population_prior",
    "b4_roster_multistep_network_generation": "n3_roster_multistep",
    "b5_homophily_conditioned_network_generation": "n4_homophily",
    "b5_contact_conditioned_network_generation": "n4_contact",
    "b5c_closure_conditioned_network_generation": "n4c_closure",
    "b5s_sex_homophily_conditioned_network_generation": "n4s_sex_homophily",
    "b5o_origin_homophily_conditioned_network_generation": "n4o_same_origin",
    "b5r_transnational_conditioned_network_generation": "n4r_resides_spain",
    "b6_size_multistep_network_generation": "n5_size_multistep",
    "b7_prior_multistep_network_generation": "n6_prior_multistep",
    "c2_full_llm_network_generation": "r1_llm_state_network",
}
NEW_TO_OLD = {v: k for k, v in OLD_TO_NEW.items()}

INPUT_STEM = {
    "s0_baseline_state": "s0",
    "s0b_anti_typology": "s0b",
    "s1_health_conditioned": "s1",
    "s1b_anti_typology": "s1b",
    "s2_politics_religion_conditioned": "s2",
    "n0_baseline_network": "n0",
    "n1_size_bucket": "n1",
    "n2_population_prior": "n2",
    "n3_roster_multistep": "n3",
    "n4_homophily": "n4",
    "n4_contact": "n4",
    "n4c_closure": "n4c",
    "n4s_sex_homophily": "n4s",
    "n4o_same_origin": "n4o",
    "n4r_resides_spain": "n4r",
    "n5_size_multistep": "n5",
    "n6_prior_multistep": "n6",
    "r1_llm_state_network": "r1",
    "t6_country_resources": "t6",
}

GOLD_STEM = {
    "s0_baseline_state": "s0",
    "s0b_anti_typology": "s0",
    "s1_health_conditioned": "s1",
    "s1b_anti_typology": "s1",
    "s2_politics_religion_conditioned": "s2",
}

STATE_TASKS = [
    "s0_baseline_state",
    "s0b_anti_typology",
    "s1_health_conditioned",
    "s1b_anti_typology",
    "s2_politics_religion_conditioned",
]
STATE_TASK_SHORT = {
    "s0_baseline_state": "s0",
    "s0b_anti_typology": "s0b",
    "s1_health_conditioned": "s1",
    "s1b_anti_typology": "s1b",
    "s2_politics_religion_conditioned": "s2",
}

NETWORK_TASKS_SHORT = {
    "N0": "n0_baseline_network",
    "N1": "n1_size_bucket",
    "N2": "n2_population_prior",
    "N3": "n3_roster_multistep",
    "N4": "n4_homophily",
    "N5": "n5_size_multistep",
    "N6": "n6_prior_multistep",
}

# (dataset, task) -> primary Testing folder. Task must be the new slug.
PRIMARY: dict[tuple[str, str], str] = {}


def _p(dataset: str, task: str, testing: str) -> None:
    PRIMARY[(dataset, task)] = testing


for _ds in ("ucnets_w1", "gss_1987", "gss_2010_network"):
    _p(_ds, "s0_baseline_state", "T1_association")
    _p(_ds, "s0b_anti_typology", "T1_association")
    _p(_ds, "s1_health_conditioned", "T2_information_use")
    _p(_ds, "s1b_anti_typology", "T2_information_use")
    _p(_ds, "s2_politics_religion_conditioned", "T2_information_use")

_p("ucnets_w1", "n0_baseline_network", "T1_association")
_p("ucnets_w1", "n1_size_bucket", "T2_information_use")
_p("ucnets_w1", "n2_population_prior", "T2_information_use")
_p("ucnets_w1", "n3_roster_multistep", "T5_additivity")
_p("ucnets_w1", "n4_homophily", "T3_interval_following")
_p("ucnets_w1", "n5_size_multistep", "T5_additivity")
_p("ucnets_w1", "n6_prior_multistep", "T5_additivity")
_p("ucnets_w1", "r1_llm_state_network", "T4_individual_resample")
_p("gss_1985", "s0_baseline_state", "T1_association")
_p("gss_1985", "r1_llm_state_network", "T4_individual_resample")
_p("nshap_r1", "s0_baseline_state", "T1_association")
_p("nshap_r1", "r1_llm_state_network", "T4_individual_resample")

for _ds in ("gss_1985", "gss_2004"):
    _p(_ds, "n0_baseline_network", "T3_interval_following")
    _p(_ds, "n4c_closure", "T3_interval_following")
    _p(_ds, "n4s_sex_homophily", "T3_interval_following")

_p("gss_2024", "n0_baseline_network", "T3_interval_following")
_p("gss_2024", "n4_contact", "T3_interval_following")
_p("nshap_r1", "n0_baseline_network", "T3_interval_following")
_p("nshap_r1", "n4c_closure", "T3_interval_following")
_p("signos", "n0_baseline_network", "T3_interval_following")
_p("signos", "n4o_same_origin", "T3_interval_following")
_p("signos", "n4r_resides_spain", "T3_interval_following")
_p("issp_2017", "t6_country_resources", "T6_cross_population")


def canon_task(task: str) -> str:
    return OLD_TO_NEW.get(task, task)


def survey_for(dataset: str) -> str:
    return DATASET_SURVEY[dataset]


def survey_root(dataset: str) -> Path:
    return SURVEY / survey_for(dataset)


def cleaned_dir(dataset: str) -> Path:
    return survey_root(dataset) / "cleaned"


def state_gold(dataset: str, task: str) -> Path:
    task = canon_task(task)
    stem = GOLD_STEM[task]
    return cleaned_dir(dataset) / f"{dataset}_{stem}.jsonl"


def network_gold(dataset: str) -> Path:
    if dataset == "gss_2010_network":
        return cleaned_dir(dataset) / "gss_2010_network_network_gold.jsonl"
    if dataset == "issp_2017":
        return cleaned_dir(dataset) / "issp_2017_resource_gold.jsonl"
    return cleaned_dir(dataset) / f"{dataset}_network_gold.jsonl"


def primary_testing(dataset: str, task: str) -> str:
    task = canon_task(task)
    return PRIMARY[(dataset, task)]


def dataset_testing_dir(dataset: str, task: str) -> Path:
    return TESTING / primary_testing(dataset, task) / dataset


def model_slug(model: str) -> str:
    """Filesystem slug for generation output dirs (derived from ``--model`` name)."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model.strip()).strip("_")


def arm_input(dataset: str, task: str) -> Path:
    task = canon_task(task)
    stem = INPUT_STEM[task]
    return dataset_testing_dir(dataset, task) / "input" / f"{stem}_input.jsonl"


def arm_output_dir(dataset: str, task: str, model_slug: str = "llama3.1_latest") -> Path:
    task = canon_task(task)
    return dataset_testing_dir(dataset, task) / "output" / model_slug / task


def arm_output_jsonl(dataset: str, task: str, model_slug: str = "llama3.1_latest") -> Path:
    return arm_output_dir(dataset, task, model_slug) / f"{dataset}_output.jsonl"


def arm_predictions_csv(dataset: str, task: str, model_slug: str = "llama3.1_latest") -> Path:
    return arm_output_dir(dataset, task, model_slug) / f"{dataset}_predictions.csv"


def state_eval_dir(model_slug: str = "llama3.1_latest") -> Path:
    return TESTING / "T1_association" / "eval"


def network_eval_dir() -> Path:
    return TESTING / "T1_association" / "ucnets_w1" / "eval"


def t4_eval_dir(dataset: str = "ucnets_w1") -> Path:
    return TESTING / "T4_individual_resample" / dataset / "eval"


def t6_eval_dir() -> Path:
    return TESTING / "T6_cross_population" / "eval"


def iter_primary_inputs() -> list[Path]:
    files: list[Path] = []
    for path in TESTING.rglob("*_input.jsonl"):
        if path.is_symlink():
            continue
        files.append(path)
    return sorted(files)


def discover_input_files(
    input_dir: Path | None,
    datasets: list[str] | None,
    tasks: list[str] | None,
    read_first_row,
) -> list[Path]:
    """If input_dir is Testing or None, scan primary (non-symlink) inputs."""
    if input_dir is None or input_dir == TESTING or input_dir.name == "Testing":
        candidates = iter_primary_inputs()
    else:
        candidates = sorted(input_dir.glob("*_input.jsonl"))
        if not candidates:
            candidates = sorted(p for p in input_dir.rglob("*_input.jsonl") if not p.is_symlink())
    selected: list[Path] = []
    want_tasks = {canon_task(t) for t in tasks} if tasks else None
    for path in candidates:
        row = read_first_row(path)
        if not row:
            continue
        dataset = row[0]["dataset"]
        task = canon_task(row[0]["task"])
        if datasets and dataset not in datasets:
            continue
        if want_tasks and task not in want_tasks:
            continue
        selected.append(path)
    return selected
