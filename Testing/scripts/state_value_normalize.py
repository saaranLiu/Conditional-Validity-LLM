#!/usr/bin/env python3
"""Normalize near-miss LLM labels onto Phase 1 allowed values.

Handles common schema drift without inventing substantive answers:
  - Likert morphology: agreed/agrees/agre -> agree
  - Contact aliases: phone/phone_call -> cell_call; social media -> social_media
  - Minor typos / spelling variants on known categories
  - Day-count ranges like "2-3" -> midpoint integer

Cross-field swaps (e.g. trust field filled with fairness labels) are NOT remapped.
"""

from __future__ import annotations

import re
from typing import Any


LIKERT_ALLOWED = [
    "strongly disagree",
    "disagree",
    "neither agree nor disagree",
    "agree",
    "strongly agree",
]

# Exact aliases first (lowercase keys).
LIKERT_ALIASES: dict[str, str] = {
    "agree": "agree",
    "agrees": "agree",
    "agreed": "agree",
    "agre": "agree",
    "agreer": "agree",
    "agreem": "agree",
    "agreet": "agree",
    "agreeto": "agree",
    "agreement": "agree",
    "agrees slightly": "agree",
    "agrees somewhat": "agree",
    "agree slightly": "agree",
    "agree somewhat": "agree",
    "strongly agree": "strongly agree",
    "strongly agrees": "strongly agree",
    "strongly agreed": "strongly agree",
    "disagree": "disagree",
    "disagrees": "disagree",
    "disagreed": "disagree",
    "strongly disagree": "strongly disagree",
    "strongly disagrees": "strongly disagree",
    "strongly disagreed": "strongly disagree",
    "neither agree nor disagree": "neither agree nor disagree",
    "neither agrees nor disagrees": "neither agree nor disagree",
}

CONTACT_ALIASES: dict[str, str] = {
    "phone": "cell_call",
    "phone_call": "cell_call",
    "phone call": "cell_call",
    "cellphone": "cell_call",
    "cell phone": "cell_call",
    "cell": "cell_call",
    "mobile": "cell_call",
    "call": "cell_call",
    "social media": "social_media",
    "social-media": "social_media",
    "instant message": "instant_message",
    "instant messaging": "instant_message",
    "im": "instant_message",
    "video call": "video",
    "videocall": "video",
    "e-mail": "email",
    "emails": "email",
    "texts": "text",
    "sms": "text",
    "letter": "letters",
    "land line": "landline",
    "land-line": "landline",
}

# Field-specific scalar aliases (lowercase -> canonical).
SCALAR_ALIASES: dict[str, dict[str, str]] = {
    "relatives_help_serious_problem": {
        "probably do not have": "probably dont have",
        "probably don't have": "probably dont have",
        "probably doesnt have": "probably dont have",
    },
    "friends_help_serious_problem": {
        "probably dont have": "probably do not have",
        "probably don't have": "probably do not have",
    },
    "people_are_helpful": {
        "trying out for themselves": "looking out for themselves",
        "look out for themselves": "looking out for themselves",
        "looking out for themself": "looking out for themselves",
    },
}


def _norm_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_likert(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    key = _norm_key(value)
    if key in LIKERT_ALIASES:
        return LIKERT_ALIASES[key]
    if key in LIKERT_ALLOWED:
        return key

    # Morphology / soft matches.
    if "neither" in key and "agree" in key:
        return "neither agree nor disagree"
    if "strongly" in key and "disagree" in key:
        return "strongly disagree"
    if "strongly" in key and "agree" in key and "disagree" not in key:
        return "strongly agree"
    if re.fullmatch(r"agree[ds]?", key) or key.startswith("agree ") or key.startswith("agrees"):
        return "agree"
    if key.startswith("agre") and "disagree" not in key and "neither" not in key:
        # agreedisagree / ambiguous junk -> leave unchanged
        if "disagree" in key or "neither" in key or "hard to determine" in key:
            return value
        if key in {"agreedisagree"}:
            return value
        return "agree"
    if re.fullmatch(r"disagree[ds]?", key) or key.startswith("disagree"):
        return "disagree"
    return value


def normalize_contact_item(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    key = _norm_key(value)
    if key in CONTACT_ALIASES:
        return CONTACT_ALIASES[key]
    # already canonical snake_case
    return key if key else value


def normalize_day_count(value: Any) -> Any:
    """Map '2-3' / '0–1' style ranges to an int midpoint; keep plain ints/strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else value
    if not isinstance(value, str):
        return value
    text = value.strip()
    # Plain digits are already valid numerics for the validator — leave as-is.
    if re.fullmatch(r"-?\d+", text):
        return value
    m = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return int(round((lo + hi) / 2))
    return value


def normalize_scalar_field(field: str, value: Any) -> Any:
    if field in {
        "days_very_happy_past_week",
        "days_lonely_past_week",
        "days_isolated_past_week",
    }:
        return normalize_day_count(value)

    if not isinstance(value, str):
        return value
    key = _norm_key(value)
    aliases = SCALAR_ALIASES.get(field, {})
    if key in aliases:
        return aliases[key]
    return value


def normalize_prediction(pred: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return (normalized_pred, list of change records)."""
    if pred is None or not isinstance(pred, dict):
        return pred, []

    out = dict(pred)
    changes: list[dict[str, Any]] = []

    # Nested personality
    if isinstance(out.get("personality"), dict):
        pers = dict(out["personality"])
        for sub, val in list(pers.items()):
            new_val = normalize_likert(val)
            if new_val != val:
                changes.append(
                    {
                        "path": f"personality.{sub}",
                        "from": val,
                        "to": new_val,
                        "rule": "likert_alias",
                    }
                )
                pers[sub] = new_val
        out["personality"] = pers

    # Contact arrays
    for field in ("family_contact_methods_weekly", "friend_contact_methods_weekly"):
        if field not in out:
            continue
        val = out[field]
        if not isinstance(val, list):
            continue
        new_list = []
        changed = False
        for item in val:
            new_item = normalize_contact_item(item)
            if new_item != item:
                changed = True
                changes.append(
                    {
                        "path": field,
                        "from": item,
                        "to": new_item,
                        "rule": "contact_alias",
                    }
                )
            # dedupe while preserving order
            if new_item not in new_list:
                new_list.append(new_item)
        if changed or len(new_list) != len(val):
            out[field] = new_list

    # Other scalars present in prediction
    for field, val in list(out.items()):
        if field in {"personality", "family_contact_methods_weekly", "friend_contact_methods_weekly"}:
            continue
        new_val = normalize_scalar_field(field, val)
        if new_val != val:
            changes.append(
                {
                    "path": field,
                    "from": val,
                    "to": new_val,
                    "rule": "scalar_alias",
                }
            )
            out[field] = new_val

    return out, changes
