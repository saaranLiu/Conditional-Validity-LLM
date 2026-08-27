"""
Does the LLM's core_alters same_race_as_ego rate actually vary across the
provided same_race_share_bucket condition (low / medium / high), and does it
track the TRUE bucket-conditional rate in the human data? This is the direct
test of whether B5's explicit homophily conditioning is heeded at all, as
opposed to just checking whether the pooled marginal average happens to match.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/bld/data/data2/jiaxin/simulation")
GOLD = ROOT / "Survey" / "ucnets" / "cleaned"
N4_IN = ROOT / "Testing" / "T3_interval_following" / "ucnets_w1" / "input" / "n4_input.jsonl"
N4_OUT = ROOT / "Testing" / "T3_interval_following" / "ucnets_w1" / "output" / "llama3.1_latest" / "n4_homophily" / "ucnets_w1_output.jsonl"


def last_by_id(records):
    out = {}
    for r in records:
        out[str(r.get("respondent_id"))] = r
    return out


def extract_pred(rec):
    return rec.get("prediction_json")


def main() -> int:
    gold = {}
    for line in (GOLD / "ucnets_w1_network_gold.jsonl").read_text().splitlines():
        g = json.loads(line)
        gold[str(g["respondent_id"])] = g

    b5_input = {}
    for line in N4_IN.read_text().splitlines():
        r = json.loads(line)
        b5_input[str(r["respondent_id"])] = r["prompt_json"]["same_race_share_bucket"]

    recs = [json.loads(l) for l in N4_OUT.read_text().splitlines()]
    recs_by_id = last_by_id(recs)

    buckets = ["low", "medium", "high"]
    human_by_bucket = {b: [0, 0] for b in buckets}  # [yes, total]
    llm_by_bucket = {b: [0, 0] for b in buckets}
    n_by_bucket = {b: 0 for b in buckets}

    for rid, g in gold.items():
        bucket = b5_input.get(rid)
        if bucket is None:
            continue
        rec = recs_by_id.get(rid)
        if rec is None:
            continue
        val = rec.get("validation") or {}
        if not val.get("schema_ok"):
            continue
        pred = extract_pred(rec)
        if not pred:
            continue
        n_by_bucket[bucket] += 1
        for a in g["target_json"].get("core_alters") or []:
            if a.get("same_race_as_ego") is None:
                continue
            human_by_bucket[bucket][1] += 1
            if a["same_race_as_ego"] == "yes":
                human_by_bucket[bucket][0] += 1
        for a in pred.get("core_alters") or []:
            if not isinstance(a, dict) or a.get("same_race_as_ego") is None:
                continue
            llm_by_bucket[bucket][1] += 1
            if str(a["same_race_as_ego"]) == "yes":
                llm_by_bucket[bucket][0] += 1

    print(f"{'bucket':8s} {'n_egos':>7s} {'human_rate':>11s} {'llm_rate':>9s} {'gap':>7s}")
    rows = []
    for b in buckets:
        n = n_by_bucket[b]
        hy, ht = human_by_bucket[b]
        ly, lt = llm_by_bucket[b]
        hr = hy / ht if ht else float("nan")
        lr = ly / lt if lt else float("nan")
        print(f"{b:8s} {n:7d} {hr:11.3f} {lr:9.3f} {lr - hr:7.3f}")
        rows.append({"bucket": b, "n_egos": n, "human_rate": hr, "llm_rate": lr, "gap": lr - hr})

    out_json = ROOT / "Reports" / "cross_phase_analysis" / "b5_bucket_conditional_results.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
