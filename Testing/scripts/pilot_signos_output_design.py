#!/usr/bin/env python3
"""Pilot: decide how to ask for a 30-alter network with alter-alter ties.

Signos is the only dataset that can test triadic closure at UCNets' network scale
(30 alters), but a complete dyad census is C(30,2) = 435 tie objects. On the 5-alter
GSS/UCNets arms llama3.1 passes schema validation 90-98% of the time; nobody should
assume that survives a 30x longer output with an exact combinatorial count attached.

So measure it before writing the config. Three candidate designs:

  A_full_matrix  one call, 30 alters + all 435 tie objects
  B_density_only one call, 30 alters + a density number, no tie list
  C_multistep    call 1 for the roster, call 2 for a SPARSE edge list (connected pairs
                 only), density derived as len(ties) / 435

Reports, per design: JSON parse rate, roster-length correctness, tie-count behaviour and
whether a density estimate is recoverable. The winner is the design that yields a usable
closure measurement, not the one that merely returns something.

Run: python3 Testing/scripts/pilot_signos_output_design.py [--n 8] [--model llama3.1:latest]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "Survey" / "ucnets" / "cleaned" / "signos_network_gold.jsonl"
OUT = ROOT / "_archive" / "signos_output_design_pilot.json"

HOST = "http://127.0.0.1:11436"
ROSTER_SIZE = 30
POSSIBLE_DYADS = ROSTER_SIZE * (ROSTER_SIZE - 1) // 2  # 435

RELATIONSHIPS = [
    "spouse_or_partner",
    "blood_relative",
    "in_law",
    "coworker",
    "neighbor",
    "association_or_club",
    "place_of_worship",
    "through_third_person",
    "school",
    "since_youth",
    "other",
]
CLOSENESS = ["not_at_all_close", "not_very_close", "fairly_close", "close", "very_close"]

COMMON_HEADER = (
    "You are simulating one *randomly selected* synthetic individual sampled from "
    "first-generation immigrants living in Barcelona, Spain.\n"
    "Return ONLY a single JSON object. No prose, no markdown fences.\n"
)


def state_block(state: dict[str, Any]) -> str:
    return "Observed state attributes:\n" + json.dumps(state, ensure_ascii=False, indent=None) + "\n"


def prompt_a(state: dict[str, Any]) -> str:
    return (
        COMMON_HEADER
        + state_block(state)
        + f"""
Generate this person's personal network of exactly {ROSTER_SIZE} people they know.

Required schema:
{{
  "composition_summary": {{"num_alters": {ROSTER_SIZE}, "kin_share": <0-1>, "close_share": <0-1>}},
  "alters": [ {{"alter_id": 1..{ROSTER_SIZE}, "relationship": one of {RELATIONSHIPS}, "gender": "male"|"female", "closeness_to_ego": one of {CLOSENESS}}} ],
  "alter_ties": [ {{"src_id": <int>, "tgt_id": <int>, "tie_probability": "not_likely"|"might_be"|"very_likely"}} ]
}}

Rules:
- alters must contain exactly {ROSTER_SIZE} objects.
- alter_ties must contain one entry for EVERY unordered pair of alters, so exactly
  {POSSIBLE_DYADS} entries, with src_id < tgt_id.
"""
    )


def prompt_b(state: dict[str, Any]) -> str:
    return (
        COMMON_HEADER
        + state_block(state)
        + f"""
Generate this person's personal network of exactly {ROSTER_SIZE} people they know.

Required schema:
{{
  "composition_summary": {{"num_alters": {ROSTER_SIZE}, "kin_share": <0-1>, "close_share": <0-1>, "density": <0-1>}},
  "alters": [ {{"alter_id": 1..{ROSTER_SIZE}, "relationship": one of {RELATIONSHIPS}, "gender": "male"|"female", "closeness_to_ego": one of {CLOSENESS}}} ]
}}

Rules:
- alters must contain exactly {ROSTER_SIZE} objects.
- density is the share of all {POSSIBLE_DYADS} possible pairs of these {ROSTER_SIZE} people
  who would very likely interact with each other independently of this person.
"""
    )


def prompt_c1(state: dict[str, Any]) -> str:
    return (
        COMMON_HEADER
        + state_block(state)
        + f"""
Generate this person's personal network of exactly {ROSTER_SIZE} people they know.

Required schema:
{{
  "composition_summary": {{"num_alters": {ROSTER_SIZE}, "kin_share": <0-1>, "close_share": <0-1>}},
  "alters": [ {{"alter_id": 1..{ROSTER_SIZE}, "relationship": one of {RELATIONSHIPS}, "gender": "male"|"female", "closeness_to_ego": one of {CLOSENESS}}} ]
}}

Rules:
- alters must contain exactly {ROSTER_SIZE} objects.
"""
    )


def prompt_c2(roster: list[dict[str, Any]]) -> str:
    compact = [
        {
            "alter_id": a.get("alter_id"),
            "relationship": a.get("relationship"),
            "gender": a.get("gender"),
        }
        for a in roster
    ]
    return (
        COMMON_HEADER
        + "This person's network:\n"
        + json.dumps(compact, ensure_ascii=False)
        + f"""

List ONLY the pairs of these people who would VERY LIKELY interact with each other
independently of this person. Omit every pair that would not.

Required schema:
{{
  "alter_ties": [ {{"src_id": <int>, "tgt_id": <int>}} ]
}}

Rules:
- src_id < tgt_id, and both must be alter_id values listed above.
- Include a pair only if the two would very likely interact on their own. Most pairs of
  30 acquaintances do not, so this list is normally far shorter than {POSSIBLE_DYADS}.
"""
    )


def call(model: str, prompt: str, num_predict: int, timeout_s: float = 900.0) -> tuple[str, float]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.8, "num_predict": num_predict, "num_ctx": 16384},
    }
    req = urllib.request.Request(
        f"{HOST}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode())
        return body.get("response", ""), time.time() - start
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"__CALL_FAILED__ {exc}", time.time() - start


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def score_ties(ties: Any, expect_full: bool) -> dict[str, Any]:
    if not isinstance(ties, list):
        return {"tie_list": False}
    valid = [
        t
        for t in ties
        if isinstance(t, dict)
        and isinstance(t.get("src_id"), int)
        and isinstance(t.get("tgt_id"), int)
        and 1 <= t["src_id"] <= ROSTER_SIZE
        and 1 <= t["tgt_id"] <= ROSTER_SIZE
        and t["src_id"] != t["tgt_id"]
    ]
    pairs = {(min(t["src_id"], t["tgt_id"]), max(t["src_id"], t["tgt_id"])) for t in valid}
    out: dict[str, Any] = {
        "tie_list": True,
        "n_returned": len(ties),
        "n_valid_unique": len(pairs),
    }
    if expect_full:
        out["complete_census"] = len(pairs) == POSSIBLE_DYADS
        linked = [
            t for t in valid if str(t.get("tie_probability", "")).lower() == "very_likely"
        ]
        out["density"] = round(len(linked) / len(pairs), 4) if pairs else None
    else:
        out["density"] = round(len(pairs) / POSSIBLE_DYADS, 4) if pairs else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--model", default="llama3.1:latest")
    args = ap.parse_args()

    with GOLD.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f][: args.n]

    results: dict[str, list[dict[str, Any]]] = {
        "A_full_matrix": [],
        "B_density_only": [],
        "C_multistep": [],
    }

    for rec in records:
        state = rec["state_json"]
        rid = rec["respondent_id"]

        raw, secs = call(args.model, prompt_a(state), num_predict=12288)
        obj = extract_json(raw)
        entry: dict[str, Any] = {"respondent_id": rid, "seconds": round(secs, 1), "parsed": obj is not None}
        if obj:
            alters = obj.get("alters")
            entry["roster_len"] = len(alters) if isinstance(alters, list) else None
            entry["roster_ok"] = entry["roster_len"] == ROSTER_SIZE
            entry.update(score_ties(obj.get("alter_ties"), expect_full=True))
        results["A_full_matrix"].append(entry)

        raw, secs = call(args.model, prompt_b(state), num_predict=4096)
        obj = extract_json(raw)
        entry = {"respondent_id": rid, "seconds": round(secs, 1), "parsed": obj is not None}
        if obj:
            alters = obj.get("alters")
            entry["roster_len"] = len(alters) if isinstance(alters, list) else None
            entry["roster_ok"] = entry["roster_len"] == ROSTER_SIZE
            comp = obj.get("composition_summary") or {}
            d = comp.get("density")
            entry["density"] = d if isinstance(d, (int, float)) else None
            entry["density_ok"] = isinstance(d, (int, float)) and 0 <= d <= 1
        results["B_density_only"].append(entry)

        raw, secs1 = call(args.model, prompt_c1(state), num_predict=4096)
        obj1 = extract_json(raw)
        entry = {"respondent_id": rid, "parsed": obj1 is not None}
        roster = (obj1 or {}).get("alters")
        if isinstance(roster, list) and roster:
            entry["roster_len"] = len(roster)
            entry["roster_ok"] = len(roster) == ROSTER_SIZE
            raw2, secs2 = call(args.model, prompt_c2(roster), num_predict=8192)
            obj2 = extract_json(raw2)
            entry["step2_parsed"] = obj2 is not None
            entry["seconds"] = round(secs1 + secs2, 1)
            if obj2:
                entry.update(score_ties(obj2.get("alter_ties"), expect_full=False))
        else:
            entry["seconds"] = round(secs1, 1)
            entry["roster_ok"] = False
        results["C_multistep"].append(entry)

        print(f"  {rid} 完成")

    print("\n" + "=" * 78)
    print(f"Signos 输出设计试跑（模型 {args.model}，{len(records)} 个 ego）")
    print("=" * 78)
    for design, rows in results.items():
        n = len(rows)
        parsed = sum(1 for r in rows if r.get("parsed"))
        roster_ok = sum(1 for r in rows if r.get("roster_ok"))
        secs = [r.get("seconds") or 0 for r in rows]
        print(f"\n[{design}]  n={n}")
        print(f"  JSON 可解析      {parsed}/{n}")
        print(f"  名单恰好 30 人   {roster_ok}/{n}")
        print(f"  平均耗时         {sum(secs)/n:.0f}s")
        if design == "A_full_matrix":
            full = sum(1 for r in rows if r.get("complete_census"))
            counts = [r.get("n_valid_unique") for r in rows if r.get("n_valid_unique") is not None]
            print(f"  连边普查完整(435){full}/{n}")
            print(f"  实际返回对数     {counts}")
        elif design == "B_density_only":
            dok = sum(1 for r in rows if r.get("density_ok"))
            print(f"  density 可用     {dok}/{n}")
            print(f"  density 取值     {[r.get('density') for r in rows]}")
        else:
            s2 = sum(1 for r in rows if r.get("step2_parsed"))
            print(f"  第二步可解析     {s2}/{n}")
            print(f"  边数 / 推得密度  {[(r.get('n_valid_unique'), r.get('density')) for r in rows]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"model": args.model, "n": len(records), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n明细写入 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
