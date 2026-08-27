#!/usr/bin/env python3
"""Clean the Signos ego-network study into Phase 2 network-stage gold data.

Dataset id: `signos`. Emits the standard trio under Survey/ucnets/cleaned/:
`signos_network_gold.jsonl`, `signos_network_summary.csv`, `signos_population_priors.json`.

SUPPLEMENTARY dataset with one job: triadic closure at UCNets' network scale.

Every respondent named exactly 30 alters and judged all 435 alter pairs, so per-ego
density is measured with far more precision than the GSS modules allow (10 pairs at
most). GSS tests closure at k=5; this tests it at k=30, which is the size regime UCNets
actually operates in but cannot evaluate because UCNets never collected alter-alter ties.

Hard limit to keep in view: 70 respondents. That is enough to establish whether the
generator tracks a closure instruction at all, and not enough for a powered
distributional claim. Treat it as a structural stress test, not a replication of the
GSS result.

Coding hazards handled here:
  * alter `sex` is 0/1, not the 1/2 used elsewhere in the project.
  * ego `country` and alter `alter_country` use DIFFERENT code sets (3 = India for the
    ego, 3 = China for the alter), so origin matching goes through normalised labels
    rather than raw codes.
  * `alter_residence_country` is free text with a dozen misspellings of "España", so
    residence is reduced to a normalised in-Spain / abroad flag.
  * `meeting_place` has 83 largely uncoded values ('88, ce', '999. India') and is
    dropped rather than guessed at.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from network_cleaning_common import (  # noqa: E402
    assign_buckets,
    bucket_cutoff_block,
    bucket_share_block,
    code,
    mean_of,
    median_of,
    read_json,
    round_or_none,
    share,
    write_flat_csv,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "Survey" / "signos" / "processed"
OUT_DIR = ROOT / "Survey" / "signos" / "cleaned"
DATASET = "signos"

EGO_SEX = {0: "male", 1: "female"}
ALTER_SEX = {0: "male", 1: "female"}

EGO_COUNTRY = {0: "Spain", 1: "Philippines", 2: "Pakistan", 3: "India", 4: "China", 5: "other"}
EGO_RELIGION = {
    0: "muslim",
    1: "catholic",
    2: "evangelical",
    3: "none",
    4: "sikh",
    5: "confucian",
    6: "other",
}
EGO_MARITAL = {
    0: "single",
    1: "married",
    2: "divorced",
    3: "widowed",
    4: "separated",
    5: "domestic_partnership",
}
EGO_EDUCATION = {
    0: "no_formal_schooling",
    1: "primary",
    2: "secondary",
    3: "university",
    4: "postgraduate",
    5: "quranic_school",
    6: "vocational",
    7: "other",
}
EGO_EMPLOYMENT = {
    0: "full_time_contract",
    1: "part_time",
    2: "unemployed",
    3: "self_employed_for_others",
    4: "business_owner",
    5: "seasonal_work",
    6: "other",
}
EGO_LEGAL_STATUS = {0: "yes", 1: "no", 2: "declined"}
EGO_MIGRATION_REASON = {
    0: "knew_someone_here",
    1: "expected_to_find_work",
    2: "business",
    3: "knew_spain_from_television",
    4: "to_join_family",
    5: "born_here",
    6: "other",
}
EGO_REMITTANCES = {0: "never", 1: "sometimes", 2: "often"}
COMMUNITY = {"chinese": "chinese", "filipinos": "filipino", "sikhs": "sikh"}

CLOSENESS = {
    0: "not_at_all_close",
    1: "not_very_close",
    2: "fairly_close",
    3: "close",
    4: "very_close",
}
CONFIDANT = {0: "never", 1: "sometimes", 2: "often"}
RELATION_TYPE = {
    0: "spouse_or_partner",
    1: "blood_relative",
    2: "in_law",
    3: "coworker",
    4: "neighbor",
    5: "association_or_club",
    6: "place_of_worship",
    7: "through_third_person",
    8: "school",
    9: "since_youth",
    10: "other",
}
KIN_RELATIONS = {"spouse_or_partner", "blood_relative", "in_law"}

# 0 = not likely, 1 = might be, 2 = very likely. These are the respondent's judgements
# about whether two alters would interact independently of them, not observed ties.
TIE_PROBABILITY = {0: "not_likely", 1: "might_be", 2: "very_likely"}

SPAIN_FORMS = {"espana", "espanya", "catalunya", "cataluna", "barcelona", "espanol", "espanola"}
SPAIN_MATCH_THRESHOLD = 0.78


def normalize_text(value: Any) -> str:
    """Lowercase, strip accents, keep letters only."""
    text = str(value if value is not None else "").strip()
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(ch for ch in stripped.lower() if ch.isalpha())


def is_spain(value: Any) -> bool | None:
    """Free-text residence country reduced to in-Spain / abroad.

    The field carries a dozen misspellings, so exact matching is not enough; anything
    close to a known Spanish form counts, and genuinely mixed answers stay unknown.
    """
    norm = normalize_text(value)
    if not norm:
        return None
    if norm in SPAIN_FORMS:
        return True
    if "/" in str(value):  # e.g. "Espanya/Xina" names two countries
        return None
    return any(
        SequenceMatcher(None, norm, form).ratio() >= SPAIN_MATCH_THRESHOLD for form in SPAIN_FORMS
    )


def same_origin(ego_country_label: str | None, alter_country_label: Any) -> bool | None:
    """Origin match via normalised labels, since the two code sets disagree."""
    ego_norm = normalize_text(ego_country_label)
    alter_norm = normalize_text(alter_country_label)
    if not ego_norm or not alter_norm:
        return None
    if ego_norm in {"other"} or alter_norm in {"otropais", "otropaisue"}:
        return None
    # Catalonia is recorded as an origin distinct from Spain; treat both as Spain.
    if is_spain(ego_country_label) and is_spain(alter_country_label):
        return True
    return SequenceMatcher(None, ego_norm, alter_norm).ratio() >= 0.85


def build_state(ego: dict[str, Any]) -> dict[str, Any]:
    state = {
        "age": code(ego.get("age")),
        "gender": EGO_SEX.get(code(ego.get("sex"))),
        "origin_community": COMMUNITY.get(ego.get("community"), ego.get("community")),
        "country_of_origin": EGO_COUNTRY.get(code(ego.get("country"))),
        "religion": EGO_RELIGION.get(code(ego.get("religion"))),
        "marital_status": EGO_MARITAL.get(code(ego.get("marital_status"))),
        "children_count": code(ego.get("num_children")),
        "education": EGO_EDUCATION.get(code(ego.get("education"))),
        "employment_status": EGO_EMPLOYMENT.get(code(ego.get("employment_status"))),
        "has_residence_permit": EGO_LEGAL_STATUS.get(code(ego.get("legal_status"))),
        "migration_reason": EGO_MIGRATION_REASON.get(code(ego.get("migration_reason"))),
        "sends_remittances": EGO_REMITTANCES.get(code(ego.get("remittances"))),
    }
    return {k: v for k, v in state.items() if v is not None}


def build_alter(row: dict[str, Any], ego_sex: int | None, ego_country_label: str | None) -> dict[str, Any]:
    alter_sex = code(row.get("sex"))
    origin_match = same_origin(ego_country_label, row.get("alter_country_label"))
    in_spain = is_spain(row.get("alter_residence_country_label"))
    out = {
        "alter_id": code(row.get("alter_id")),
        "relationship": RELATION_TYPE.get(code(row.get("relation_type"))),
        "gender": ALTER_SEX.get(alter_sex),
        "closeness_to_ego": CLOSENESS.get(code(row.get("closeness"))),
        "confides_personal_problems": CONFIDANT.get(code(row.get("confidant"))),
        "same_gender_as_ego": (
            None if alter_sex is None or ego_sex is None else ("yes" if alter_sex == ego_sex else "no")
        ),
        "same_origin_as_ego": None if origin_match is None else ("yes" if origin_match else "no"),
        "resides_in_spain": None if in_spain is None else ("yes" if in_spain else "no"),
    }
    return {k: v for k, v in out.items() if v is not None}


def composition_of(
    ego: dict[str, Any], alters: list[dict[str, Any]], ties: list[dict[str, Any]]
) -> dict[str, Any]:
    n = len(alters)
    ego_sex = code(ego.get("sex"))
    ego_country_label = ego.get("country_label")

    rels = [RELATION_TYPE.get(code(a.get("relation_type"))) for a in alters]
    rels_present = [r for r in rels if r is not None]

    def rel_share(*names: str) -> float | None:
        return share(sum(1 for r in rels_present if r in names), len(rels_present))

    sexes = [code(a.get("sex")) for a in alters]
    sexes = [s for s in sexes if s in (0, 1)]
    same_sex = (
        share(sum(1 for s in sexes if s == ego_sex), len(sexes)) if ego_sex in (0, 1) and sexes else None
    )

    close_vals = [code(a.get("closeness")) for a in alters]
    close_vals = [v for v in close_vals if v is not None]
    conf_vals = [code(a.get("confidant")) for a in alters]
    conf_vals = [v for v in conf_vals if v is not None]

    origin_flags = [same_origin(ego_country_label, a.get("alter_country_label")) for a in alters]
    origin_present = [f for f in origin_flags if f is not None]
    spain_flags = [is_spain(a.get("alter_residence_country_label")) for a in alters]
    spain_present = [f for f in spain_flags if f is not None]

    comp: dict[str, Any] = {
        "num_alters": n,
        "kin_share": round_or_none(rel_share(*KIN_RELATIONS)),
        "coworker_share": round_or_none(rel_share("coworker")),
        "neighbor_share": round_or_none(rel_share("neighbor")),
        "place_of_worship_share": round_or_none(rel_share("place_of_worship")),
        "through_third_person_share": round_or_none(rel_share("through_third_person")),
        "relationship_coverage": round_or_none(share(len(rels_present), n)),
        # closeness 3-4 = close / very close; confidant 1-2 = ever confided in
        "close_share": round_or_none(share(sum(1 for v in close_vals if v >= 3), len(close_vals))),
        "confidant_share": round_or_none(share(sum(1 for v in conf_vals if v >= 1), len(conf_vals))),
        "same_sex_share": round_or_none(same_sex),
        "same_origin_share": round_or_none(
            share(sum(1 for f in origin_present if f), len(origin_present))
        ),
        "same_origin_coverage": round_or_none(share(len(origin_present), n)),
        "resides_in_spain_share": round_or_none(
            share(sum(1 for f in spain_present if f), len(spain_present))
        ),
        "resides_in_spain_coverage": round_or_none(share(len(spain_present), n)),
    }

    tie_vals = [code(t.get("value")) for t in ties]
    tie_vals = [v for v in tie_vals if v in (0, 1, 2)]
    possible = n * (n - 1) // 2
    if tie_vals:
        denom = len(tie_vals)
        comp["density"] = round_or_none(share(sum(1 for v in tie_vals if v == 2), denom))
        comp["density_incl_maybe"] = round_or_none(share(sum(1 for v in tie_vals if v >= 1), denom))
        comp["dyad_coverage"] = round_or_none(share(denom, possible))
        comp["n_dyads_observed"] = denom
    else:
        comp["density"] = None
        comp["density_incl_maybe"] = None
        comp["dyad_coverage"] = None
        comp["n_dyads_observed"] = 0
    return comp


BUCKETED: list[tuple[str, Any]] = [
    ("density", None),
    ("density_incl_maybe", None),
    ("same_sex_share", None),
    ("same_origin_share", None),
    ("resides_in_spain_share", None),
    ("close_share", None),
    ("confidant_share", None),
]


def build_priors(
    records: list[dict[str, Any]],
    cuts: dict[str, tuple[float, float] | None],
    counts: dict[str, Counter],
) -> dict[str, Any]:
    comps = [r["target_json"]["composition_summary"] for r in records]
    n = len(comps)
    priors: dict[str, Any] = {
        "dataset": DATASET,
        "n_egos": n,
        "roster_design": "fixed 30 alters per respondent, complete alter-alter census",
        "mean_num_alters": mean_of([c["num_alters"] for c in comps]),
        "median_num_alters": median_of([c["num_alters"] for c in comps]),
        "bucket_cutoffs": {},
        "bucket_shares": {},
        "underpowered_warning": (
            "70 respondents: enough to detect whether closure instructions are followed, "
            "not enough for a powered distributional claim."
        ),
    }
    for field in (
        "kin_share",
        "coworker_share",
        "neighbor_share",
        "place_of_worship_share",
        "through_third_person_share",
        "close_share",
        "confidant_share",
        "same_sex_share",
        "same_origin_share",
        "resides_in_spain_share",
        "density",
        "density_incl_maybe",
    ):
        vals = [c.get(field) for c in comps if c.get(field) is not None]
        if vals:
            priors[f"mean_{field}"] = mean_of(vals)
            priors[f"median_{field}"] = median_of(vals)
            priors[f"n_{field}"] = len(vals)

    for field, _ in BUCKETED:
        block = bucket_cutoff_block(cuts.get(field))
        if block:
            priors["bucket_cutoffs"][field] = block
        share_block = bucket_share_block(counts[field])
        if share_block:
            priors["bucket_shares"][field] = share_block
    return priors


def main() -> None:
    egos = read_json(SRC / "ego.json")
    alters = read_json(SRC / "alter.json")
    ties = read_json(SRC / "alter_alter.json")

    alters_by_ego: dict[str, list[dict[str, Any]]] = {}
    for row in alters:
        alters_by_ego.setdefault(row["ego_id"], []).append(row)
    for rows in alters_by_ego.values():
        rows.sort(key=lambda r: code(r.get("alter_id")) or 0)

    ties_by_ego: dict[str, list[dict[str, Any]]] = {}
    for row in ties:
        ties_by_ego.setdefault(row["ego_id"], []).append(row)

    records: list[dict[str, Any]] = []
    dropped = Counter()
    for ego in egos:
        ego_id = ego["ego_id"]
        roster = alters_by_ego.get(ego_id, [])
        if not roster:
            dropped["no_alters"] += 1
            continue
        state = build_state(ego)
        if state.get("age") is None or state.get("gender") is None:
            dropped["incomplete_state"] += 1
            continue

        ego_sex = code(ego.get("sex"))
        ego_country_label = ego.get("country_label")
        comp = composition_of(ego, roster, ties_by_ego.get(ego_id, []))
        records.append(
            {
                "respondent_id": ego_id,
                "dataset": DATASET,
                "ego_id": ego_id,
                "state_json": state,
                "target_json": {
                    "composition_summary": comp,
                    "alters": [build_alter(a, ego_sex, ego_country_label) for a in roster],
                    "alter_ties": [
                        {
                            "src_id": code(t.get("source_id")),
                            "tgt_id": code(t.get("target_id")),
                            "tie_probability": TIE_PROBABILITY.get(code(t.get("value"))),
                        }
                        for t in ties_by_ego.get(ego_id, [])
                        if TIE_PROBABILITY.get(code(t.get("value"))) is not None
                    ],
                },
                "metadata": {
                    "community": ego.get("community"),
                    "questionnaire": ego.get("questionnaire"),
                    "complete": ego.get("complete"),
                    "source_path": str(SRC.relative_to(ROOT)),
                },
            }
        )

    cuts, counts = assign_buckets(records, BUCKETED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / f"{DATASET}_network_gold.jsonl", records)

    summary_rows = []
    for rec in records:
        comp = rec["target_json"]["composition_summary"]
        flat = {"respondent_id": rec["respondent_id"], "community": rec["metadata"]["community"]}
        flat.update({k: v for k, v in comp.items() if not isinstance(v, dict)})
        summary_rows.append(flat)
    write_flat_csv(OUT_DIR / f"{DATASET}_network_summary.csv", summary_rows)

    priors = build_priors(records, cuts, counts)
    (OUT_DIR / f"{DATASET}_population_priors.json").write_text(
        json.dumps(priors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"{DATASET}: gold={len(records)}/{len(egos)} dropped={dict(dropped)}")
    for field, _ in BUCKETED:
        block = priors["bucket_shares"].get(field)
        cut = cuts.get(field)
        if block and cut:
            print(
                f"    {field:24s} n={block['n_labelled']:3d} "
                f"low={block.get('low', 0):.0%} med={block.get('medium', 0):.0%} "
                f"high={block.get('high', 0):.0%}  切点={cut[0]:.3f}/{cut[1]:.3f}"
            )
        else:
            print(f"    {field:24s} 分位退化，未分箱")


if __name__ == "__main__":
    main()
