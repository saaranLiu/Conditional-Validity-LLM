#!/usr/bin/env python3
"""Follow-up pilot: how to elicit alter-alter ties among 30 alters without biasing length.

The first pilot's sparse edge-list arm produced 3-8 edges out of 435 (density 0.007-0.018),
far below the human minimum of 0.025. But that prompt told the model "most pairs of 30
acquaintances do not [interact], so this list is normally far shorter than 435", which
pushes toward brevity. That makes the result an artifact of my wording, not a measurement
of the model, so it cannot be used to choose a design.

This pilot separates the two by holding the roster fixed and varying only the elicitation
format:

  neutral_edges   sparse edge list, no hint about expected length
  adjacency       for each alter, list the others they would likely interact with
                  (30 short answers instead of 435 objects, then symmetrise)
  count_first     ask for the number of connected pairs out of 435, then the list

Human reference (70 egos): density mean 0.319, range 0.025-0.703, i.e. roughly 11-306
connected pairs out of 435.

Run: python3 Testing/scripts/pilot_signos_tie_elicitation.py [--n 4]
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_COMMON = _HERE.parents[1] / "common"
for _p in (_COMMON, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


import argparse
import json
from typing import Any

from pilot_signos_output_design import (  # noqa: E402
    COMMON_HEADER,
    POSSIBLE_DYADS,
    ROSTER_SIZE,
    call,
    extract_json,
    prompt_c1,
)

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "Survey" / "ucnets" / "cleaned" / "signos_network_gold.jsonl"
PRIORS = ROOT / "Survey" / "ucnets" / "cleaned" / "signos_population_priors.json"
OUT = ROOT / "_archive" / "signos_tie_elicitation_pilot.json"


def compact_roster(roster: list[dict[str, Any]]) -> str:
    return json.dumps(
        [
            {
                "alter_id": a.get("alter_id"),
                "relationship": a.get("relationship"),
                "gender": a.get("gender"),
            }
            for a in roster
        ],
        ensure_ascii=False,
    )


def p_neutral(roster: list[dict[str, Any]]) -> str:
    return (
        COMMON_HEADER
        + "This person's network:\n"
        + compact_roster(roster)
        + """

Some pairs of these people would interact with each other independently of this person,
and some would not. List the pairs who would.

Required schema:
{"alter_ties": [ {"src_id": <int>, "tgt_id": <int>} ]}

Rules:
- src_id < tgt_id, and both must be alter_id values listed above.
"""
    )


def p_adjacency(roster: list[dict[str, Any]]) -> str:
    return (
        COMMON_HEADER
        + "This person's network:\n"
        + compact_roster(roster)
        + f"""

For EACH of the {ROSTER_SIZE} people above, list which of the others they would interact
with independently of this person.

Required schema:
{{"adjacency": [ {{"alter_id": <int>, "interacts_with": [<alter_id>, ...]}} ]}}

Rules:
- Include one object for every alter_id from 1 to {ROSTER_SIZE}.
- interacts_with may be empty.
"""
    )


def p_count_first(roster: list[dict[str, Any]]) -> str:
    return (
        COMMON_HEADER
        + "This person's network:\n"
        + compact_roster(roster)
        + f"""

These {ROSTER_SIZE} people form {POSSIBLE_DYADS} possible pairs. Some pairs would interact
independently of this person.

First state how many of the {POSSIBLE_DYADS} pairs would interact, then list them.

Required schema:
{{"n_connected_pairs": <int>, "alter_ties": [ {{"src_id": <int>, "tgt_id": <int>}} ]}}

Rules:
- src_id < tgt_id, and both must be alter_id values listed above.
- The length of alter_ties must equal n_connected_pairs.
"""
    )


def pairs_from_edges(obj: dict[str, Any]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for tie in obj.get("alter_ties") or []:
        if not isinstance(tie, dict):
            continue
        a, b = tie.get("src_id"), tie.get("tgt_id")
        if isinstance(a, int) and isinstance(b, int) and a != b:
            if 1 <= a <= ROSTER_SIZE and 1 <= b <= ROSTER_SIZE:
                out.add((min(a, b), max(a, b)))
    return out


def pairs_from_adjacency(obj: dict[str, Any]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for row in obj.get("adjacency") or []:
        if not isinstance(row, dict):
            continue
        src = row.get("alter_id")
        if not isinstance(src, int) or not 1 <= src <= ROSTER_SIZE:
            continue
        for tgt in row.get("interacts_with") or []:
            if isinstance(tgt, int) and 1 <= tgt <= ROSTER_SIZE and tgt != src:
                out.add((min(src, tgt), max(src, tgt)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--model", default="llama3.1:latest")
    args = ap.parse_args()

    priors = json.loads(PRIORS.read_text(encoding="utf-8"))
    human_mean = priors.get("mean_density")
    human_lo, human_hi = 0.025, 0.703

    with GOLD.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f][: args.n]

    arms = {
        "neutral_edges": (p_neutral, pairs_from_edges, 8192),
        "adjacency": (p_adjacency, pairs_from_adjacency, 8192),
        "count_first": (p_count_first, pairs_from_edges, 8192),
    }
    results: dict[str, list[dict[str, Any]]] = {k: [] for k in arms}

    for rec in records:
        raw, _ = call(args.model, prompt_c1(rec["state_json"]), num_predict=4096)
        obj1 = extract_json(raw)
        roster = (obj1 or {}).get("alters")
        if not isinstance(roster, list) or not roster:
            print(f"  {rec['respondent_id']}: 名单生成失败，跳过")
            continue

        for name, (builder, parser, npred) in arms.items():
            raw2, secs = call(args.model, builder(roster), num_predict=npred)
            obj2 = extract_json(raw2)
            entry: dict[str, Any] = {
                "respondent_id": rec["respondent_id"],
                "parsed": obj2 is not None,
                "seconds": round(secs, 1),
            }
            if obj2:
                pairs = parser(obj2)
                entry["n_pairs"] = len(pairs)
                entry["density"] = round(len(pairs) / POSSIBLE_DYADS, 4)
                entry["in_human_range"] = human_lo <= entry["density"] <= human_hi
                if name == "count_first":
                    stated = obj2.get("n_connected_pairs")
                    entry["stated_count"] = stated
                    entry["count_matches_list"] = stated == len(pairs)
            results[name].append(entry)
        print(f"  {rec['respondent_id']} 完成")

    print("\n" + "=" * 78)
    print(f"连边征询格式对比（{args.model}）")
    print(f"人类参照：密度均值 {human_mean}，范围 {human_lo}-{human_hi}"
          f"（即 435 对中约 {round(human_lo*POSSIBLE_DYADS)}-{round(human_hi*POSSIBLE_DYADS)} 对相连）")
    print("=" * 78)
    for name, rows in results.items():
        if not rows:
            continue
        n = len(rows)
        parsed = sum(1 for r in rows if r.get("parsed"))
        inrange = sum(1 for r in rows if r.get("in_human_range"))
        dens = [r.get("density") for r in rows if r.get("density") is not None]
        print(f"\n[{name}]  n={n}")
        print(f"  可解析          {parsed}/{n}")
        print(f"  密度落在人类范围 {inrange}/{n}")
        print(f"  密度取值        {dens}")
        print(f"  边数            {[r.get('n_pairs') for r in rows]}")
        print(f"  平均耗时        {sum(r.get('seconds') or 0 for r in rows)/n:.0f}s")
        if name == "count_first":
            match = sum(1 for r in rows if r.get("count_matches_list"))
            print(f"  自报数与列表一致 {match}/{n}  自报数={[r.get('stated_count') for r in rows]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"model": args.model, "human_mean_density": human_mean, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n明细写入 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
