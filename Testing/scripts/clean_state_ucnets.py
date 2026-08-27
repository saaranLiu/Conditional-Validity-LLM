#!/usr/bin/env python3
"""Clean Phase 1 state-stage data for UCNets Wave 1.

GSS robustness samples use network-module ego tables and are cleaned by
``clean_state_gss.py`` (``gss_1987``, ``gss_2010_network``).

The script intentionally uses only Python's standard library so the cleaning
pipeline is reproducible in environments where pandas/numpy are unavailable.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Survey" / "ucnets" / "cleaned"

UCNETS_SOURCE = (
    ROOT / "Survey" / "ucnets"
    / "processed"
    / "by_wave"
    / "Wave1"
    / "ego.csv"
)
UCNETS_VALUE_LABELS = UCNETS_SOURCE.with_name("value_labels_ego.json")
UCNETS_VARIABLE_LABELS = UCNETS_SOURCE.with_name("variable_labels_ego.json")

GSS_SOURCE = ROOT / "Survey" / "gss" / "processed" / "by_year" / "2010" / "survey.csv"
GSS_VALUE_LABELS = GSS_SOURCE.with_name("value_labels.json")
GSS_VARIABLE_LABELS = GSS_SOURCE.with_name("variable_labels.json")


UCNETS_ANCHORS = [
    "PRIM_KEY",
    "AGEGROUP",
    "AGE",
    "GENDER",
    "K1",
    "A21",
    "A2A",
    "A3A",
    "A12_COUNT",
    "A17A",
    "A18A",
    "K5",
    *[f"K6S{i}" for i in range(1, 7)],
    "K6",
    "RACECATS1",
    "WEIGHT_W1_DEMO",
]
UCNETS_TARGETS = [
    "G1",
    "H10",
    "H12",
    "H14",
    "E1A",
    "E2A",
    "E3",
    "E4",
    "E5",
    *[f"J1{letter}" for letter in "ABCDEFGHIJ"],
    "I1",
    "I2A",
    *[f"I2AS{i}" for i in range(1, 10)],
    "I3A",
    *[f"I3AS{i}" for i in range(1, 10)],
    "K10A",
    "K10B",
    "K15",
    "K16",
]

GSS_ANCHORS = [
    "year",
    "id",
    "age",
    "sex",
    "race",
    "hispanic",
    "racecen1",
    "racecen2",
    "racecen3",
    "educ",
    "degree",
    "marital",
    "childs",
    "wrkstat",
    "region",
    "size",
    "xnorcsiz",
    "srcbelt",
    "wtss",
    "wtssnr",
    "wtssall",
]
GSS_TARGETS = [
    "health",
    "happy",
    "trust",
    "fair",
    "helpful",
    "partyid",
    "polviews",
    "relig",
    "attend",
    "reliten",
]

I2_METHODS = {
    "I2AS1": "letters",
    "I2AS2": "landline",
    "I2AS3": "cell_call",
    "I2AS4": "text",
    "I2AS5": "email",
    "I2AS6": "social_media",
    "I2AS7": "instant_message",
    "I2AS8": "video",
    "I2AS9": "none",
}
I3_METHODS = {
    "I3AS1": "letters",
    "I3AS2": "landline",
    "I3AS3": "cell_call",
    "I3AS4": "text",
    "I3AS5": "email",
    "I3AS6": "social_media",
    "I3AS7": "instant_message",
    "I3AS8": "video",
    "I3AS9": "none",
}
PERSONALITY_FIELDS = {
    "J1A": "reserved",
    "J1B": "sociable",
    "J1C": "trusting",
    "J1D": "finds_fault_with_others",
    "J1E": "handles_stress_well",
    "J1F": "active_imagination",
    "J1G": "few_artistic_interests",
    "J1H": "lazy",
    "J1I": "thorough",
    "J1J": "nervous_easily",
}

MISSING_STRINGS = {
    "",
    ".",
    ".d",
    ".e",
    ".i",
    ".n",
    ".s",
    "na",
    "nan",
    "null",
    "none",
    "[masked by icpsr]",
}

MANUAL_VALUE_LABELS = {
    "ucnets": {
        "A21": {
            "1": "full time",
            "2": "part time",
            "3": "going to school",
            "4": "unemployed and looking",
            "5": "unemployed and not looking",
            "6": "retired",
            "7": "taking care of home or family",
            "8": "else",
        },
        "E2A": {
            "1": "definitely have",
            "2": "probably have",
            "3": "might have",
            "4": "probably do not have",
            "5": "other",
        },
    },
    "gss": {},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", errors="replace") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


TASK_FIELD_ORDER = [
    "respondent_id",
    "dataset",
    "task",
    "prompt_json",
    "target_json",
    "metadata",
]


def order_record(row: dict[str, Any], key_order: list[str]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in key_order:
        if key in row:
            ordered[key] = row[key]
    for key, value in row.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def write_jsonl(path: Path, records: list[dict[str, Any]], key_order: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            payload = order_record(record, key_order) if key_order else record
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def is_missing_raw(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in MISSING_STRINGS


def strip_leading_code(raw_value: str, label: str) -> str:
    raw = raw_value.strip()
    text = re.sub(r"\s+", " ", str(label).strip())
    if raw and text.startswith(raw + " "):
        text = text[len(raw) :].strip()
    return text


def normalize_label(raw_value: str, label: str | None, dataset: str) -> str | None:
    if is_missing_raw(raw_value):
        return None
    if label is None:
        return str(raw_value).strip()
    cleaned = strip_leading_code(str(raw_value), label)
    cleaned_lower = cleaned.lower()
    if dataset == "ucnets":
        if "don't know" in cleaned_lower or "dont know" in cleaned_lower:
            return None
        if "volunteered" in cleaned_lower or "~" in cleaned_lower:
            return None
    return cleaned.strip() or None


def value_label(dataset: str, field: str, value: str, labels: dict[str, Any]) -> str | None:
    if is_missing_raw(value):
        return None
    key = str(value).strip()
    if dataset == "ucnets":
        raw_label = labels.get(field, {}).get(key) or MANUAL_VALUE_LABELS["ucnets"].get(field, {}).get(key)
    else:
        raw_label = labels.get(field, {}).get("values", {}).get(key) or MANUAL_VALUE_LABELS["gss"].get(field, {}).get(key)
    return normalize_label(key, raw_label, dataset) if raw_label is not None else key


def to_int(value: Any) -> int | None:
    if is_missing_raw(value):
        return None
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    return text.lower() if text else None


def compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value not in (None, "", [], {})}


def selected_yes_fields(row: dict[str, str], fields: dict[str, str]) -> list[str]:
    selected = []
    for field, label in fields.items():
        if str(row.get(field, "")).strip() == "1":
            selected.append(label)
    return selected


def age_bin(age: Any) -> str | None:
    value = to_int(age)
    if value is None:
        return None
    if value < 21:
        return "under_21"
    if value <= 30:
        return "21_30"
    if value <= 49:
        return "31_49"
    if value <= 70:
        return "50_70"
    return "71_plus"


def education_ucnets(value: Any) -> str | None:
    code = to_int(value)
    if code in (1, 2):
        return "less_than_high_school"
    if code in (3, 4):
        return "high_school_or_ged"
    if code in (5, 6):
        return "some_college_or_associate"
    if code == 7:
        return "bachelors"
    if code in (8, 9):
        return "graduate_or_professional"
    return None


def education_gss_degree(value: Any) -> str | None:
    code = to_int(value)
    mapping = {
        0: "less_than_high_school",
        1: "high_school",
        2: "associate_or_junior_college",
        3: "bachelors",
        4: "graduate",
    }
    return mapping.get(code)


def children_ucnets(row: dict[str, str]) -> str | None:
    under_18 = to_int(row.get("A17A"))
    adult = to_int(row.get("A18A"))
    yes_values = {1}
    no_values = {2}
    if under_18 in yes_values or adult in yes_values:
        return "has_children"
    if under_18 in no_values and adult in no_values:
        return "no_children"
    return None


def children_gss(value: Any) -> str | None:
    count = to_int(value)
    if count is None:
        return None
    if count == 0:
        return "no_children"
    return "has_children"


def race_ucnets(row: dict[str, str], labels: dict[str, Any]) -> str | None:
    for field in ("RACECATS1", "K6"):
        label = value_label("ucnets", field, row.get(field, ""), labels)
        if label:
            return clean_text(label)
    selected = []
    for field in [f"K6S{i}" for i in range(1, 7)]:
        if str(row.get(field, "")).strip() == "1":
            selected.append(field.lower())
    return ";".join(selected) if selected else None


def race_gss(row: dict[str, str], labels: dict[str, Any]) -> str | None:
    hispanic = value_label("gss", "hispanic", row.get("hispanic", ""), labels)
    if hispanic and hispanic != "not hispanic":
        return "hispanic"
    race_detail = value_label("gss", "racecen1", row.get("racecen1", ""), labels)
    if race_detail:
        return clean_text(race_detail)
    race = value_label("gss", "race", row.get("race", ""), labels)
    return clean_text(race)


def add_labeled_columns(
    dataset: str,
    row: dict[str, str],
    fields: list[str],
    labels: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in fields:
        raw_value = row.get(field, "")
        output[field] = None if is_missing_raw(raw_value) else raw_value
        output[f"{field}_label"] = value_label(dataset, field, raw_value, labels)
    return output


def clean_ucnets(rows: list[dict[str, str]], labels: dict[str, Any]) -> list[dict[str, Any]]:
    clean_rows = []
    selected = UCNETS_ANCHORS + UCNETS_TARGETS
    for row in rows:
        out = {
            "respondent_id": row.get("PRIM_KEY"),
            "dataset": "ucnets_w1",
            "weight": row.get("WEIGHT_W1_DEMO") or None,
        }
        out.update(add_labeled_columns("ucnets", row, selected, labels))
        out.update(
            {
                "age_group_harmonized": clean_text(out.get("AGEGROUP_label")) or age_bin(row.get("AGE")),
                "gender_harmonized": clean_text(out.get("GENDER_label")),
                "race_ethnicity_harmonized": race_ucnets(row, labels),
                "education_harmonized": education_ucnets(row.get("K1")),
                "marital_harmonized": clean_text(out.get("A2A_label")),
                "work_status_harmonized": clean_text(out.get("A21_label")),
                "children_harmonized": children_ucnets(row),
                "health_harmonized": clean_text(out.get("G1_label")),
            }
        )
        clean_rows.append(out)
    return clean_rows


def clean_gss(rows: list[dict[str, str]], labels: dict[str, Any]) -> list[dict[str, Any]]:
    clean_rows = []
    selected = GSS_ANCHORS + GSS_TARGETS
    for row in rows:
        out = {
            "respondent_id": row.get("id"),
            "dataset": "gss_2010",
            "weight": row.get("wtssall") or row.get("wtss") or None,
        }
        out.update(add_labeled_columns("gss", row, selected, labels))
        out.update(
            {
                "age_group_harmonized": age_bin(row.get("age")),
                "gender_harmonized": clean_text(out.get("sex_label")),
                "race_ethnicity_harmonized": race_gss(row, labels),
                "education_harmonized": education_gss_degree(row.get("degree")),
                "marital_harmonized": clean_text(out.get("marital_label")),
                "work_status_harmonized": clean_text(out.get("wrkstat_label")),
                "children_harmonized": children_gss(row.get("childs")),
                "health_harmonized": clean_text(out.get("health_label")),
            }
        )
        clean_rows.append(out)
    return clean_rows


def ucnets_prompt(row: dict[str, Any], include_health: bool = False) -> dict[str, Any]:
    prompt = compact_dict(
        {
            "age_group": row.get("age_group_harmonized"),
            "age": row.get("AGE"),
            "gender": row.get("gender_harmonized"),
            "race_ethnicity": row.get("race_ethnicity_harmonized"),
            "education": row.get("education_harmonized") or clean_text(row.get("K1_label")),
            "work_status": row.get("work_status_harmonized"),
            "marital_status": row.get("marital_harmonized"),
            "romantic_relationship": clean_text(row.get("A3A_label")),
            "household_other_people_count": row.get("A12_COUNT"),
            "children_status": row.get("children_harmonized"),
        }
    )
    if include_health:
        prompt["self_rated_health"] = row.get("health_harmonized")
    return compact_dict(prompt)


def ucnets_target(row: dict[str, Any], include_health: bool = True) -> dict[str, Any]:
    target = {
        "days_very_happy_past_week": row.get("H10"),
        "days_isolated_past_week": row.get("H12"),
        "days_lonely_past_week": row.get("H14"),
        "relatives_help_serious_problem": clean_text(row.get("E1A_label")),
        "friends_help_serious_problem": clean_text(row.get("E2A_label")),
        "wants_more_people_for_personal_concerns": clean_text(row.get("E3_label")),
        "wants_more_people_for_good_time": clean_text(row.get("E4_label")),
        "wants_more_people_for_practical_help": clean_text(row.get("E5_label")),
        "personality": compact_dict(
            {
                label: clean_text(row.get(f"{field}_label"))
                for field, label in PERSONALITY_FIELDS.items()
            }
        ),
        "ever_goes_online": clean_text(row.get("I1_label")),
        "family_contact_methods_weekly": selected_yes_fields(row, I2_METHODS),
        "friend_contact_methods_weekly": selected_yes_fields(row, I3_METHODS),
        "voting_frequency": clean_text(row.get("K10A_label")),
        "party_identity": clean_text(row.get("K10B_label")),
        "religion_current": clean_text(row.get("K15_label")),
        "religion_raised": clean_text(row.get("K16_label")),
    }
    if include_health:
        target["self_rated_health"] = row.get("health_harmonized")
    return compact_dict(target)


def gss_prompt(row: dict[str, Any], include_health: bool = False) -> dict[str, Any]:
    prompt = compact_dict(
        {
            "age": row.get("age"),
            "age_group": row.get("age_group_harmonized"),
            "sex": row.get("gender_harmonized"),
            "race_ethnicity": row.get("race_ethnicity_harmonized"),
            "education_years": row.get("educ"),
            "degree": row.get("education_harmonized") or clean_text(row.get("degree_label")),
            "marital_status": row.get("marital_harmonized"),
            "children_count": row.get("childs"),
            "children_status": row.get("children_harmonized"),
            "work_status": row.get("work_status_harmonized"),
            "region": clean_text(row.get("region_label")),
        }
    )
    if include_health:
        prompt["self_rated_health"] = row.get("health_harmonized")
    return compact_dict(prompt)


def gss_target(row: dict[str, Any], include_health: bool = True) -> dict[str, Any]:
    target = {
        "general_happiness": clean_text(row.get("happy_label")),
        "people_can_be_trusted": clean_text(row.get("trust_label")),
        "people_are_fair": clean_text(row.get("fair_label")),
        "people_are_helpful": clean_text(row.get("helpful_label")),
        "party_identity": clean_text(row.get("partyid_label")),
        "political_ideology": clean_text(row.get("polviews_label")),
        "religion": clean_text(row.get("relig_label")),
        "religious_service_attendance": clean_text(row.get("attend_label")),
        "religious_affiliation_strength": clean_text(row.get("reliten_label")),
    }
    if include_health:
        target["self_rated_health"] = row.get("health_harmonized")
    return compact_dict(target)


def complete_required(row: dict[str, Any], required: list[str]) -> bool:
    return all(row.get(field) not in (None, "") for field in required)


def task_record(
    row: dict[str, Any],
    dataset: str,
    task: str,
    prompt: dict[str, Any],
    target: dict[str, Any],
    target_fields: list[str],
) -> dict[str, Any]:
    missing_targets = [field for field in target_fields if field not in target]
    return {
        "respondent_id": row.get("respondent_id"),
        "dataset": dataset,
        "task": task,
        "prompt_json": prompt,
        "target_json": target,
        "metadata": {
            "weight": row.get("weight"),
            "sensitive_fields_included": "race_ethnicity" in prompt,
            "missing_target_fields": missing_targets,
        },
    }


def build_ucnets_jsonl(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = [
        "age_group_harmonized",
        "gender_harmonized",
        "education_harmonized",
        "marital_harmonized",
    ]
    task1 = []
    task2 = []
    target_fields = [
        "self_rated_health",
        "days_very_happy_past_week",
        "days_isolated_past_week",
        "days_lonely_past_week",
        "relatives_help_serious_problem",
        "friends_help_serious_problem",
        "wants_more_people_for_personal_concerns",
        "wants_more_people_for_good_time",
        "wants_more_people_for_practical_help",
        "personality",
        "ever_goes_online",
        "family_contact_methods_weekly",
        "friend_contact_methods_weekly",
        "voting_frequency",
        "party_identity",
        "religion_current",
        "religion_raised",
    ]
    for row in rows:
        if not complete_required(row, required):
            continue
        task1.append(
            task_record(
                row,
                "ucnets_w1",
                "s0_baseline_state",
                ucnets_prompt(row, include_health=False),
                ucnets_target(row, include_health=True),
                target_fields,
            )
        )
        if row.get("health_harmonized"):
            task2.append(
                task_record(
                    row,
                    "ucnets_w1",
                    "s1_health_conditioned",
                    ucnets_prompt(row, include_health=True),
                    ucnets_target(row, include_health=False),
                    [field for field in target_fields if field != "self_rated_health"],
                )
            )
    return task1, task2


def build_gss_jsonl(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = [
        "age_group_harmonized",
        "gender_harmonized",
        "race_ethnicity_harmonized",
        "education_harmonized",
        "marital_harmonized",
        "work_status_harmonized",
    ]
    task1 = []
    task2 = []
    target_fields = [
        "self_rated_health",
        "general_happiness",
        "people_can_be_trusted",
        "people_are_fair",
        "people_are_helpful",
        "party_identity",
        "political_ideology",
        "religion",
        "religious_service_attendance",
        "religious_affiliation_strength",
    ]
    for row in rows:
        if not complete_required(row, required):
            continue
        task1.append(
            task_record(
                row,
                "gss_2010",
                "s0_baseline_state",
                gss_prompt(row, include_health=False),
                gss_target(row, include_health=True),
                target_fields,
            )
        )
        if row.get("health_harmonized"):
            task2.append(
                task_record(
                    row,
                    "gss_2010",
                    "s1_health_conditioned",
                    gss_prompt(row, include_health=True),
                    gss_target(row, include_health=False),
                    [field for field in target_fields if field != "self_rated_health"],
                )
            )
    return task1, task2


def output_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def missingness(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    total = len(rows)
    summary = []
    for field in fields:
        missing = sum(1 for row in rows if row.get(field) in (None, ""))
        summary.append(
            {
                "field": field,
                "missing": missing,
                "total": total,
                "missing_pct": round(100 * missing / total, 2) if total else 0,
            }
        )
    return summary


def frequency_table(rows: list[dict[str, Any]], fields: list[str], limit: int = 8) -> dict[str, list[tuple[str, int]]]:
    tables = {}
    for field in fields:
        counter = Counter(str(row.get(field)) for row in rows if row.get(field) not in (None, ""))
        tables[field] = counter.most_common(limit)
    return tables


def create_crosswalk(
    ucnets_var_labels: dict[str, Any],
    gss_var_labels: dict[str, Any],
) -> list[dict[str, str]]:
    rows = []

    def add(dataset: str, fields: list[str], role: str, construct: str, task_usage: str) -> None:
        labels = ucnets_var_labels if dataset == "ucnets_w1" else gss_var_labels
        for field in fields:
            raw_label = labels.get(field, {})
            if isinstance(raw_label, dict):
                description = raw_label.get("label", "")
            else:
                description = str(raw_label)
            rows.append(
                {
                    "construct": construct,
                    "dataset": dataset,
                    "raw_field": field,
                    "cleaned_field": f"{field}_label",
                    "role": role,
                    "task_usage": task_usage,
                    "description": description,
                }
            )

    add("ucnets_w1", UCNETS_ANCHORS, "anchor", "background", "task1_and_task2")
    add("ucnets_w1", ["G1"], "target_or_condition", "health", "task1_target_task2_anchor")
    add("ucnets_w1", [f for f in UCNETS_TARGETS if f != "G1"], "target", "state", "task1_and_task2_target")
    return rows


def write_report(
    path: Path,
    ucnets_raw_n: int,
    gss_raw_n: int,
    ucnets_clean: list[dict[str, Any]],
    gss_clean: list[dict[str, Any]],
    ucnets_task1: list[dict[str, Any]],
    ucnets_task2: list[dict[str, Any]],
    gss_task1: list[dict[str, Any]],
    gss_task2: list[dict[str, Any]],
) -> None:
    ucnets_targets_for_report = ["G1_label", "H10", "H12", "H14", "E1A_label", "E2A_label", "E3_label", "E4_label", "E5_label"]
    gss_targets_for_report = ["health_label", "happy_label", "trust_label", "fair_label", "helpful_label", "partyid_label", "polviews_label", "relig_label", "attend_label", "reliten_label"]
    ucnets_freq_fields = ["age_group_harmonized", "gender_harmonized", "education_harmonized", "marital_harmonized", "health_harmonized"]
    gss_freq_fields = ["age_group_harmonized", "gender_harmonized", "race_ethnicity_harmonized", "education_harmonized", "marital_harmonized", "health_harmonized"]

    lines = [
        "# Phase 1 Data Cleaning Report",
        "",
        "## Sample Sizes",
        "",
        f"- UCNets Wave 1 raw rows: {ucnets_raw_n}",
        f"- UCNets clean rows: {len(ucnets_clean)}",
        f"- UCNets Task 1 JSONL rows: {len(ucnets_task1)}",
        f"- UCNets Task 2 JSONL rows: {len(ucnets_task2)}",
        f"- GSS 2010 raw rows: {gss_raw_n}",
        f"- GSS clean rows: {len(gss_clean)}",
        f"- GSS Task 1 JSONL rows: {len(gss_task1)}",
        f"- GSS Task 2 JSONL rows: {len(gss_task2)}",
        "",
        "## Missingness: UCNets Targets",
        "",
        "| Field | Missing | Total | Missing % |",
        "|---|---:|---:|---:|",
    ]
    for item in missingness(ucnets_clean, ucnets_targets_for_report):
        lines.append(f"| `{item['field']}` | {item['missing']} | {item['total']} | {item['missing_pct']} |")
    lines.extend(["", "## Missingness: GSS Targets", "", "| Field | Missing | Total | Missing % |", "|---|---:|---:|---:|"])
    for item in missingness(gss_clean, gss_targets_for_report):
        lines.append(f"| `{item['field']}` | {item['missing']} | {item['total']} | {item['missing_pct']} |")
    lines.extend(["", "## Frequency Checks: UCNets", ""])
    for field, values in frequency_table(ucnets_clean, ucnets_freq_fields).items():
        lines.append(f"### `{field}`")
        lines.extend([f"- {value}: {count}" for value, count in values])
        lines.append("")
    lines.extend(["## Frequency Checks: GSS", ""])
    for field, values in frequency_table(gss_clean, gss_freq_fields).items():
        lines.append(f"### `{field}`")
        lines.extend([f"- {value}: {count}" for value, count in values])
        lines.append("")
    lines.extend(
        [
            "## Caveats",
            "",
            "- UCNets loneliness/isolation items are direct state targets; GSS 2010 does not provide a direct equivalent in the selected target set.",
            "- UCNets support variables and GSS trust variables are construct-level counterparts, not item-equivalent measures.",
            "- Race/ethnicity fields are retained for subgroup calibration and included in the first JSONL prompt files; rerun variants can remove them if needed.",
            "- Task 2 conditions on observed health and excludes health from `target_json` to avoid target leakage.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(records: list[dict[str, Any]]) -> list[str]:
    errors = []
    for record in records:
        task = record["task"]
        prompt = record["prompt_json"]
        target = record["target_json"]
        overlap = sorted(set(prompt) & set(target))
        if overlap:
            errors.append(f"{record['dataset']} {task} {record['respondent_id']}: prompt/target overlap {overlap}")
        if task == "s0_baseline_state" and "self_rated_health" in prompt:
            errors.append(f"{record['dataset']} {record['respondent_id']}: health leaked into Task 1 prompt")
        if task == "s1_health_conditioned" and "self_rated_health" in target:
            errors.append(f"{record['dataset']} {record['respondent_id']}: health leaked into Task 2 target")
    return errors


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ucnets_labels = read_json(UCNETS_VALUE_LABELS)
    ucnets_var_labels = read_json(UCNETS_VARIABLE_LABELS)

    ucnets_raw = read_csv(UCNETS_SOURCE)

    ucnets_headers = set(ucnets_raw[0]) if ucnets_raw else set()
    missing_ucnets = [field for field in UCNETS_ANCHORS + UCNETS_TARGETS if field not in ucnets_headers]
    if missing_ucnets:
        raise SystemExit(f"Missing UCNets fields: {missing_ucnets}")

    ucnets_clean = clean_ucnets(ucnets_raw, ucnets_labels)
    ucnets_task1, ucnets_task2 = build_ucnets_jsonl(ucnets_clean)

    validation_errors = validate_outputs(ucnets_task1 + ucnets_task2)
    if validation_errors:
        error_path = OUT_DIR / "validation_errors.txt"
        error_path.write_text("\n".join(validation_errors) + "\n", encoding="utf-8")
        raise SystemExit(f"Validation failed; see {error_path}")

    write_csv(OUT_DIR / "ucnets_w1_state_clean.csv", ucnets_clean, output_fieldnames(ucnets_clean))

    # Crosswalk for UCNets; GSS network years use clean_state_gss.py.
    crosswalk = create_crosswalk(ucnets_var_labels, {})
    write_csv(
        OUT_DIR / "phase1_variable_crosswalk.csv",
        crosswalk,
        ["construct", "dataset", "raw_field", "cleaned_field", "role", "task_usage", "description"],
    )

    task_outputs = {
        "ucnets_w1_s0.jsonl": ucnets_task1,
        "ucnets_w1_s1.jsonl": ucnets_task2,
    }
    for filename, records in task_outputs.items():
        write_jsonl(OUT_DIR / filename, records, TASK_FIELD_ORDER)

    print("Wrote UCNets Phase 1 cleaned data to", OUT_DIR)
    print("UCNets Task 1 rows:", len(ucnets_task1))
    print("UCNets Task 2 rows:", len(ucnets_task2))
    print("GSS network years: run Testing/scripts/clean_state_gss.py")


if __name__ == "__main__":
    main()
