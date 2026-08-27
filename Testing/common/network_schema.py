#!/usr/bin/env python3
"""Shared Phase 2 schema helpers (allowed values + validation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "Testing" / "configs" / "network"
DEFAULT_DATASET = "ucnets_w1"
DEFAULT_CONFIG = CONFIG_DIR / f"{DEFAULT_DATASET}.yaml"

# Vocabulary lists live under `{field}_allowed` in the config; these fields deviate.
ALLOWED_LIST_ALIASES = {"same_race_as_ego": "same_race_allowed"}

# Fallback used only when a config predates the declarative `output_schema` block.
LEGACY_COMPOSITION_SCHEMA: dict[str, Any] = {
    "num_alters": {"type": "numeric", "min": 1, "max": 30},
    "size_bucket": None,
    "kin_share": {"type": "numeric", "min": 0, "max": 1},
    "friend_share": {"type": "numeric", "min": 0, "max": 1},
    "coworker_share": {"type": "numeric", "min": 0, "max": 1},
    "confidant_count": {"type": "numeric", "min": 0, "max": 30},
    "companion_count": {"type": "numeric", "min": 0, "max": 30},
    "pct_weekly_plus_contact": {"type": "numeric", "min": 0, "max": 1},
}
LEGACY_CORE_ALTER_SCHEMA: dict[str, Any] = {
    "rank": {"type": "numeric", "min": 1, "max": 5},
    "relationship": None,
    "support_roles": None,
    "contact_frequency": None,
    "gender": None,
    "same_race_as_ego": None,
}
LEGACY_ROSTER = {"key": "core_alters", "mode": "fixed", "size": 5, "rank_field": "rank"}


def config_path_for(dataset: str) -> Path:
    return CONFIG_DIR / f"{dataset}.yaml"


def load_config(path: Path | None = None, dataset: str | None = None) -> dict[str, Any]:
    """Load a Phase 2 content config by explicit path or by dataset id."""
    if path is None and dataset is not None:
        path = config_path_for(dataset)
    cfg_path = path or DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("dataset", cfg_path.stem)
    return cfg


def dataset_id(cfg: dict[str, Any]) -> str:
    return cfg.get("dataset") or DEFAULT_DATASET


def allowed_list(cfg: dict[str, Any], field: str) -> list[Any] | None:
    key = ALLOWED_LIST_ALIASES.get(field, f"{field}_allowed")
    values = cfg.get(key)
    return list(values) if values else None


def roster_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    """Roster shape for this survey.

    `fixed` means the model must return exactly `size` alters ranked 1..size (UCNets
    Top-5). `variable` means the roster is the person's complete nomination list, so
    its length is only bounded and must agree with num_alters (GSS name generators).
    `none` means there is no name generator: the estimand is a resource summary, not
    a network (ISSP).
    """
    raw = cfg.get("roster") or {}
    if raw.get("mode") == "none":
        return {"mode": "none", "key": None}
    spec = dict(LEGACY_ROSTER)
    spec.update(raw)
    return spec


def _resolve_field_schema(cfg: dict[str, Any], field: str, declared: Any) -> Any:
    """A null in the config means "use this field's vocabulary list"."""
    if declared is None:
        return allowed_list(cfg, field)
    return declared


def allowed_bundle(cfg: dict[str, Any]) -> dict[str, Any]:
    """Allowed-value spec for one survey, driven entirely by its config."""
    schema = cfg.get("output_schema") or {}
    roster = roster_spec(cfg)
    if roster.get("mode") == "none":
        resource_declared = schema.get("resource_summary") or {}
        resource = {
            field: _resolve_field_schema(cfg, field, declared)
            for field, declared in resource_declared.items()
        }
        bundle: dict[str, Any] = {
            "composition_summary": {},
            "core_alter": {},
            "roster_key": None,
            "resource_summary": {k: v for k, v in resource.items() if v is not None},
        }
        reverse_alias = {v: k for k, v in ALLOWED_LIST_ALIASES.items()}
        for key, values in cfg.items():
            if not key.endswith("_allowed") or not values:
                continue
            field = reverse_alias.get(key, key[: -len("_allowed")])
            bundle.setdefault(field, list(values))
        return bundle

    comp_declared = schema.get("composition_summary") or LEGACY_COMPOSITION_SCHEMA
    alter_declared = schema.get("core_alter_fields") or LEGACY_CORE_ALTER_SCHEMA

    composition = {
        field: _resolve_field_schema(cfg, field, declared)
        for field, declared in comp_declared.items()
    }
    core_alter = {
        field: _resolve_field_schema(cfg, field, declared)
        for field, declared in alter_declared.items()
    }
    composition = {k: v for k, v in composition.items() if v is not None}
    core_alter = {k: v for k, v in core_alter.items() if v is not None}

    bundle: dict[str, Any] = {
        "composition_summary": composition,
        "core_alter": core_alter,
        "roster_key": roster_spec(cfg)["key"],
    }
    # Flat vocabulary entries for callers that render prompt text. Every `*_allowed`
    # list in the config is exposed, so a survey can declare vocabularies UCNets lacks.
    reverse_alias = {v: k for k, v in ALLOWED_LIST_ALIASES.items()}
    for key, values in cfg.items():
        if not key.endswith("_allowed") or not values:
            continue
        field = reverse_alias.get(key, key[: -len("_allowed")])
        bundle.setdefault(field, list(values))
    return bundle


def extra_list_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Additional array-valued top-level keys beyond the alter roster.

    Surveys carrying an alter-alter matrix ask the model for the tie list as well, and
    that output has to be validated or the triadic-closure estimand rests on nothing.
    """
    schema = cfg.get("output_schema") or {}
    specs = {}
    for key, spec in (schema.get("extra_lists") or {}).items():
        fields = {
            field: _resolve_field_schema(cfg, field, declared)
            for field, declared in (spec.get("fields") or {}).items()
        }
        specs[key] = {
            "fields": {k: v for k, v in fields.items() if v is not None},
            "max_len": spec.get("max_len"),
        }
    return specs


def alter_tie_pair_diagnostics(pred: dict[str, Any] | None, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Check whether alter_ties represents every unordered roster pair once."""
    if not isinstance(pred, dict) or "alter_ties" not in extra_list_specs(cfg):
        return None
    roster = roster_spec(cfg)
    roster_key = roster.get("key")
    alters = pred.get(roster_key) if roster_key else None
    ties = pred.get("alter_ties")
    if not isinstance(alters, list) or not isinstance(ties, list):
        return None

    rank_field = roster.get("rank_field", "alter_id")
    alter_ids = {
        str(a.get(rank_field))
        for a in alters
        if isinstance(a, dict) and a.get(rank_field) is not None
    }
    expected = len(alter_ids) * (len(alter_ids) - 1) // 2
    seen: set[tuple[str, str]] = set()
    duplicate_rows = self_pairs = unknown_endpoint_rows = malformed_rows = 0
    for tie in ties:
        if not isinstance(tie, dict) or tie.get("src_id") is None or tie.get("tgt_id") is None:
            malformed_rows += 1
            continue
        src, tgt = str(tie["src_id"]), str(tie["tgt_id"])
        if src == tgt:
            self_pairs += 1
            continue
        if src not in alter_ids or tgt not in alter_ids:
            unknown_endpoint_rows += 1
            continue
        pair = tuple(sorted((src, tgt)))
        if pair in seen:
            duplicate_rows += 1
        else:
            seen.add(pair)
    missing_pairs = max(expected - len(seen), 0)
    complete = (
        len(seen) == expected
        and duplicate_rows == 0
        and self_pairs == 0
        and unknown_endpoint_rows == 0
        and malformed_rows == 0
    )
    return {
        "n_alters": len(alter_ids),
        "expected_pairs": expected,
        "tie_rows": len(ties),
        "unique_valid_pairs": len(seen),
        "missing_pairs": missing_pairs,
        "duplicate_rows": duplicate_rows,
        "self_pairs": self_pairs,
        "unknown_endpoint_rows": unknown_endpoint_rows,
        "malformed_rows": malformed_rows,
        "complete_c_n_2": complete,
    }


def top_level_keys(cfg: dict[str, Any]) -> list[str]:
    schema = cfg.get("output_schema") or {}
    declared = schema.get("top_level_keys")
    if declared:
        return list(declared)
    return ["composition_summary", roster_spec(cfg)["key"]]


def _validate_value(value: Any, allowed: Any) -> Any | None:
    if allowed is None:
        return None
    if isinstance(allowed, dict) and allowed.get("type") == "numeric":
        try:
            num = float(value)
        except (TypeError, ValueError):
            return {"got": value, "expected": "numeric", "spec": allowed}
        lo, hi = allowed.get("min"), allowed.get("max")
        if lo is not None and num < float(lo):
            return {"got": value, "expected_min": lo}
        if hi is not None and num > float(hi):
            return {"got": value, "expected_max": hi}
        return None
    if isinstance(allowed, dict) and "type" not in allowed:
        if not isinstance(value, dict):
            return {"got": value, "expected": "object", "keys": list(allowed.keys())}
        nested: dict[str, Any] = {}
        for k, sub in allowed.items():
            if k not in value:
                nested[k] = {"missing": True}
                continue
            err = _validate_value(value[k], sub)
            if err is not None:
                nested[k] = err
        extra = [k for k in value if k not in allowed]
        if extra:
            nested["_extra_keys"] = extra
        return nested or None
    if isinstance(allowed, list):
        if isinstance(value, list):
            bad = [v for v in value if v not in allowed]
            return {"got": value, "bad_items": bad, "allowed": allowed} if bad else None
        if value not in allowed:
            return {"got": value, "allowed": allowed}
        return None
    return None


def _roster_length_error(
    core: Any, roster: dict[str, Any], composition: Any
) -> dict[str, Any] | None:
    if not isinstance(core, list):
        return {"got_len": None, "expected_len": roster.get("size")}
    if roster.get("mode", "fixed") == "fixed":
        size = int(roster.get("size", 5))
        return None if len(core) == size else {"got_len": len(core), "expected_len": size}

    lo = int(roster.get("min_size", 1))
    hi = int(roster.get("max_size", 5))
    if not lo <= len(core) <= hi:
        return {"got_len": len(core), "expected_range": [lo, hi]}
    # A complete-roster survey must not disagree with its own reported size.
    if isinstance(composition, dict):
        declared = composition.get("num_alters")
        try:
            declared_int = int(declared)
        except (TypeError, ValueError):
            declared_int = None
        if declared_int is not None and declared_int != len(core):
            return {"got_len": len(core), "num_alters": declared_int, "expected": "equal"}
    return None


def validate_network_prediction(pred: dict[str, Any] | None, cfg: dict[str, Any]) -> dict[str, Any]:
    allowed = allowed_bundle(cfg)
    result: dict[str, Any] = {
        "parsed": pred is not None,
        "exact_keys": False,
        "missing_keys": [],
        "extra_keys": [],
        "invalid_values": {},
        "schema_ok": False,
    }
    expected = top_level_keys(cfg)
    roster = roster_spec(cfg)
    roster_key = roster.get("key")
    if pred is None:
        result["missing_keys"] = expected
        return result

    missing = [k for k in expected if k not in pred]
    extra = [k for k in pred if k not in expected]
    result["missing_keys"] = missing
    result["extra_keys"] = extra
    result["exact_keys"] = not missing and not extra

    invalid: dict[str, Any] = {}
    if "resource_summary" in pred and allowed.get("resource_summary"):
        err = _validate_value(pred["resource_summary"], allowed["resource_summary"])
        if err is not None:
            invalid["resource_summary"] = err
    if "composition_summary" in pred and allowed.get("composition_summary"):
        err = _validate_value(pred["composition_summary"], allowed["composition_summary"])
        if err is not None:
            invalid["composition_summary"] = err

    if roster.get("mode") != "none" and roster_key and roster_key in pred:
        core = pred[roster_key]
        length_err = _roster_length_error(core, roster, pred.get("composition_summary"))
        if length_err is not None:
            invalid[roster_key] = length_err
        else:
            core_errs: dict[str, Any] = {}
            rank_field = roster.get("rank_field", "rank")
            ranks = []
            for i, alter in enumerate(core):
                if not isinstance(alter, dict):
                    core_errs[str(i)] = {"expected": "object"}
                    continue
                err = _validate_value(alter, allowed["core_alter"])
                if err is not None:
                    core_errs[str(i)] = err
                ranks.append(alter.get(rank_field))
            expected_ranks = list(range(1, len(core) + 1))
            if sorted(r for r in ranks if isinstance(r, int)) != expected_ranks:
                core_errs["_ranks"] = {
                    "field": rank_field,
                    "got": ranks,
                    "expected": expected_ranks,
                }
            if core_errs:
                invalid[roster_key] = core_errs

    for key, spec in extra_list_specs(cfg).items():
        if key not in pred:
            continue
        items = pred[key]
        if not isinstance(items, list):
            invalid[key] = {"expected": "array", "got": items}
            continue
        max_len = spec.get("max_len")
        if max_len is not None and len(items) > int(max_len):
            invalid[key] = {"got_len": len(items), "max_len": max_len}
            continue
        item_errs: dict[str, Any] = {}
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                item_errs[str(i)] = {"expected": "object"}
                continue
            err = _validate_value(item, spec["fields"])
            if err is not None:
                item_errs[str(i)] = err
        if item_errs:
            invalid[key] = item_errs

    pair_diag = alter_tie_pair_diagnostics(pred, cfg)
    if pair_diag is not None and not pair_diag["complete_c_n_2"]:
        existing = invalid.get("alter_ties")
        if not isinstance(existing, dict):
            existing = {}
        existing["_pair_completeness"] = pair_diag
        invalid["alter_ties"] = existing

    result["invalid_values"] = invalid
    result["schema_ok"] = result["exact_keys"] and not invalid
    return result


def extract_json_object(text: str) -> dict[str, Any] | None:
    import json
    import re

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


def extract_json_any(text: str) -> Any | None:
    import json
    import re

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None
