"""Load and validate Phase 1 SSDataBench-style content YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "Testing" / "configs" / "state"

DATASET_CONFIGS = {
    "gss_1985": CONFIG_DIR / "gss_1985.yaml",
    "gss_1987": CONFIG_DIR / "gss_1987.yaml",
    "gss_2010_network": CONFIG_DIR / "gss_2010_network.yaml",
    "nshap_r1": CONFIG_DIR / "nshap_r1.yaml",
    "ucnets_w1": CONFIG_DIR / "ucnets_w1.yaml",
}

TASK_OUTPUT_STEMS = {
    "s0_baseline_state": "s0",
    "s0b_anti_typology": "s0b",
    "s1_health_conditioned": "s1",
    "s1b_anti_typology": "s1b",
    "s2_politics_religion_conditioned": "s2",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


class ContentConfig:
    def __init__(self, dataset: str, data: dict[str, Any]):
        self.dataset = dataset
        self.data = data
        self.context = str(data.get("context", "")).strip()
        self.population_context = data.get("population_context", {})
        self.prompt = data.get("prompt", {})
        self.tasks = data.get("tasks", {})
        self.input_variables = data.get("input_variables", {})
        self.output_variables = data.get("output_variables", {})
        self.original_data = data.get("original_data", {})

    @classmethod
    def load(cls, dataset: str) -> ContentConfig:
        path = DATASET_CONFIGS[dataset]
        return cls(dataset, load_yaml(path))

    @property
    def reference_year(self) -> int | None:
        year = self.population_context.get("reference_year")
        return int(year) if year is not None else None

    def task_config(self, task: str) -> dict[str, Any]:
        if task not in self.tasks:
            raise KeyError(f"Unknown task '{task}' in {self.dataset} config")
        return self.tasks[task]

    def prompt_condition(self, task: str) -> str:
        return self.task_config(task).get("prompt_condition", "format_only")

    def task_instruction(self, task: str) -> str:
        return str(self.task_config(task)["instruction"]).strip()

    def task_file(self, task: str) -> Path:
        task_cfg = self.task_config(task)
        source_task = task_cfg.get("reuse_task_file", task)
        rel = self.original_data["task_files"][source_task]
        return resolve_path(rel)

    def output_stem(self, task: str) -> str:
        return f"{self.dataset}_{TASK_OUTPUT_STEMS[task]}"

    def conditioning_fields_in_input(self, task: str) -> list[str]:
        """Observed state fields moved into the prompt for conditioned tasks."""
        task_cfg = self.task_config(task)
        fields = list(task_cfg.get("conditioning_fields_in_input", []))
        if task_cfg.get("include_health_in_input"):
            if "self_rated_health" not in fields:
                fields.append("self_rated_health")
        return fields

    def output_keys_for_task(self, task: str) -> list[str]:
        task_cfg = self.task_config(task)
        keys = list(self.output_variables.keys())
        exclude = set(task_cfg.get("exclude_output_keys", []))
        if not task_cfg.get("include_health_in_output", True):
            exclude.add("self_rated_health")
        # Conditioning fields shown in the prompt must not be requested as outputs.
        exclude.update(self.conditioning_fields_in_input(task))
        return [key for key in keys if key not in exclude]

    def allowed_values_for_task(self, task: str) -> dict[str, Any]:
        allowed: dict[str, Any] = {}
        for key in self.output_keys_for_task(task):
            spec = self.output_variables[key].get("allowed")
            if spec is None:
                raise KeyError(f"Missing allowed values for output variable '{key}'")
            allowed[key] = spec
        return allowed

    def descriptions_for_task(self, task: str) -> dict[str, str]:
        return {
            key: self.output_variables[key].get("description", key.replace("_", " "))
            for key in self.output_keys_for_task(task)
        }

    def variable_type(self, key: str) -> str:
        return self.output_variables[key].get("type", "static")

    def expected_schema(self, task: str) -> dict[str, Any]:
        schema: dict[str, Any] = {}
        for key in self.output_keys_for_task(task):
            var_type = self.variable_type(key)
            if var_type == "nested":
                allowed = self.allowed_values_for_task(task)[key]
                schema[key] = {subkey: "one allowed string value" for subkey in allowed}
            elif var_type == "array":
                schema[key] = "array of zero or more allowed strings"
            else:
                schema[key] = "one allowed value"
        return schema

    def prompt_rules(self, task: str) -> list[str]:
        rule_sets = self.task_config(task).get("rule_sets", ["baseline"])
        rules: list[str] = []
        if "baseline" in rule_sets:
            rules.extend(self.prompt.get("baseline_rules", []))
        if "anti_typology" in rule_sets:
            rules.extend(self.prompt.get("anti_typology_rules", []))
        if not rules:
            rules = [
                "Use ONLY the allowed values listed below for each field.",
                "If a field expects an array, return an array using zero or more allowed values.",
            ]
        return [str(rule) for rule in rules]

    def prompt_opening(self) -> str:
        opening = self.prompt.get(
            "opening",
            "You are simulating one *randomly selected* synthetic individual sampled from",
        )
        return str(opening).strip()

    def _allowed_spec_for_prompt_key(self, key: str) -> Any | None:
        if key in self.input_variables:
            return self.input_variables[key].get("allowed")
        if key in self.output_variables:
            return self.output_variables[key].get("allowed")
        return None

    def validate_prompt_json(self, prompt_json: dict[str, Any], task: str) -> list[str]:
        """Return validation warnings for values outside YAML input schema."""
        warnings: list[str] = []
        allowed_keys = set(self.input_variables.keys())
        allowed_keys.update(self.conditioning_fields_in_input(task))

        for key in prompt_json:
            if key not in allowed_keys:
                warnings.append(f"unexpected input key '{key}'")

        for key, value in prompt_json.items():
            spec = self._allowed_spec_for_prompt_key(key)
            if spec is None:
                continue

            if isinstance(spec, dict) and spec.get("type") == "free_text":
                continue
            if isinstance(spec, dict) and spec.get("type") == "numeric":
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    warnings.append(f"{key}: non-numeric value {value!r}")
                    continue
                if num < spec["min"] or num > spec["max"]:
                    warnings.append(f"{key}: value {value!r} outside [{spec['min']}, {spec['max']}]")
                continue
            if isinstance(spec, list) and str(value) not in {str(v) for v in spec}:
                warnings.append(f"{key}: value {value!r} not in allowed list")

        return warnings
