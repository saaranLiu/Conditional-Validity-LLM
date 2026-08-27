"""
Analysis C: does combining an information lever (B2 size_bucket, B3 population
priors) with a format lever (B4 roster multistep) produce additive gains
(B1->B6 approx (B1->B2 gain) + (B1->B4 gain)), or does the combination
interact (sub-additive, super-additive, or backfiring)?

Each four-arm contrast uses the common schema_ok egos available in its required
arms (within the fixed 400-person candidate subsample),
using Testing/T1_association/ucnets_w1/eval/phase2_eval_summary.json
(subset_400 block, produced by evaluate_network.py).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/bld/data/data2/jiaxin/simulation")
EVAL_JSON = ROOT / "Testing" / "T1_association" / "ucnets_w1" / "eval" / "phase2_eval_summary.json"
OUT = ROOT / "Reports" / "cross_phase_analysis"


def decompose(table: dict, info_short: str, base: str = "B1", fmt: str = "B4", combo: str = None) -> dict:
    out = {}
    for metric, vals in table.items():
        b1, info, f4, cb = vals.get(base), vals.get(info_short), vals.get(fmt), vals.get(combo)
        if None in (b1, info, f4, cb):
            continue
        info_gain = b1 - info
        fmt_gain = b1 - f4
        additive_pred = b1 - info_gain - fmt_gain
        actual = cb
        out[metric] = {
            "B1": b1, info_short: info, fmt: f4, combo: cb,
            "info_gain": round(info_gain, 4),
            "format_gain": round(fmt_gain, 4),
            "additive_prediction": round(additive_pred, 4),
            "actual": round(actual, 4),
            "interaction": round(actual - additive_pred, 4),
        }
    return out


def main() -> int:
    d = json.loads(EVAL_JSON.read_text())
    sub = d["subset_400"]
    size_arm = decompose(sub["combo_table_size_arm"], "B2", combo="B6")
    prior_arm = decompose(sub["combo_table_prior_arm"], "B3", combo="B7")
    # Compatibility for existing plotting code.  The primary T5 estimand is the
    # explicitly documented fixed composition-field subset, never core-alter TV.
    if "mean_tv_composition" in size_arm:
        size_arm["mean_tv"] = size_arm["mean_tv_composition"]
    if "mean_tv_composition" in prior_arm:
        prior_arm["mean_tv"] = prior_arm["mean_tv_composition"]

    results = {
        "n_subsample_candidates": sub["n_ids"],
        "primary_metric": "mean_tv_composition",
        "primary_metric_definition": "fixed composition fields declared by evaluate_network.COMPOSITION_TV_FIELDS",
        "size_arm_n_common_schema_ok": sub["size_arm_n_common_schema_ok"],
        "prior_arm_n_common_schema_ok": sub["prior_arm_n_common_schema_ok"],
        "size_arm_B1_B2_B4_B6": size_arm,
        "prior_arm_B1_B3_B4_B7": prior_arm,
    }
    (OUT / "analysis_c_results.json").write_text(json.dumps(results, indent=2))

    def fmt_row(m, v):
        sign = "+" if v["interaction"] >= 0 else ""
        return (f"  {m:28s} additive_pred={v['additive_prediction']:.4f}  actual={v['actual']:.4f}  "
                f"interaction={sign}{v['interaction']:.4f}")

    print(
        f"candidate_n={sub['n_ids']} "
        f"size_common_n={sub['size_arm_n_common_schema_ok']} "
        f"prior_common_n={sub['prior_arm_n_common_schema_ok']}\n"
    )
    print("=== Size arm: B1 -> B2(info) / B4(format) / B6(info+format) ===")
    print("(lower is better for mean_tv; interaction<0 means combo did BETTER than naive additive sum, interaction>0 means WORSE / backfired)")
    for m, v in size_arm.items():
        print(fmt_row(m, v))
    print("\n=== Prior arm: B1 -> B3(info) / B4(format) / B7(info+format) ===")
    for m, v in prior_arm.items():
        print(fmt_row(m, v))

    (OUT / "analysis_c_summary.txt").write_text("placeholder")
    print(f"\nwrote {OUT / 'analysis_c_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
