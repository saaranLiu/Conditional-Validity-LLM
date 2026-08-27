"""
Analysis A: does the LLM's fidelity gain from an intervention track the
TRUE (human-data) explanatory power of the information that intervention adds,
or is the gain/inflation roughly independent of that true value?

This directly tests the "predictor vs sampler" reframing: a calibrated sampler
should show fidelity gains, and association strengths, that scale with the
true signal. A mode-seeking predictor is expected to show gains/inflation that
are large-and-roughly-uniform (or field/topic specific) rather than tracking
the true explanatory power of the specific piece of information supplied.

Produces:
  Reports/cross_phase_analysis/analysis_a_results.json
  Reports/cross_phase_analysis/analysis_a_summary.md
  Reports/cross_phase_analysis/fig_eval2_human_vs_llm_association.png
  Reports/cross_phase_analysis/fig_phase2_info_sufficiency.png
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/bld/data/data2/jiaxin/simulation")
P1 = ROOT / "Testing" / "T1_association" / "eval"
GOLD = ROOT / "Survey" / "ucnets" / "cleaned"
NET_EVAL = ROOT / "Testing" / "T1_association" / "ucnets_w1" / "eval"
OUT = ROOT / "Reports" / "cross_phase_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def part_a_eval2_calibration() -> dict[str, Any]:
    """Across ALL Phase1 eval2 pairs (3 datasets x 3 tasks), does the LLM's
    reproduced association strength (stat_llm) track the true one (stat_human)?
    A calibrated sampler => stat_llm ~= stat_human (points near the diagonal,
    high correlation, slope near 1). A predictor that manufactures a stereotyped
    level of association regardless of the true value => weak correlation
    and/or systematic inflation (slope >> 1, poor rank agreement)."""
    df = pd.read_csv(P1 / "eval2_evaluation" / "eval2_pair_metrics.csv")
    df = df.dropna(subset=["stat_human", "stat_llm"])
    # cramers_v is bounded [0,1]; pearson_r/eta-like stats can be signed. Use abs
    # value for a "strength" comparison across mixed stat types.
    df["abs_human"] = df["stat_human"].abs()
    df["abs_llm"] = df["stat_llm"].abs()

    out: dict[str, Any] = {"by_task": {}}
    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task]
        if len(sub) < 4:
            continue
        pear = stats.pearsonr(sub["abs_human"], sub["abs_llm"])
        spear = stats.spearmanr(sub["abs_human"], sub["abs_llm"])
        slope, intercept = np.polyfit(sub["abs_human"], sub["abs_llm"], 1)
        out["by_task"][task] = {
            "n_pairs": int(len(sub)),
            "mean_human": round(float(sub["abs_human"].mean()), 4),
            "mean_llm": round(float(sub["abs_llm"].mean()), 4),
            "mean_inflation_llm_minus_human": round(float((sub["abs_llm"] - sub["abs_human"]).mean()), 4),
            "pearson_r_human_vs_llm": round(float(pear.statistic), 4),
            "pearson_p": float(pear.pvalue),
            "spearman_rho_human_vs_llm": round(float(spear.statistic), 4),
            "spearman_p": float(spear.pvalue),
            "ols_slope_llm_on_human": round(float(slope), 4),
            "share_llm_gt_human": round(float((sub["abs_llm"] > sub["abs_human"]).mean()), 4),
        }

    # Pooled across all tasks/datasets
    pear = stats.pearsonr(df["abs_human"], df["abs_llm"])
    spear = stats.spearmanr(df["abs_human"], df["abs_llm"])
    out["pooled"] = {
        "n_pairs": int(len(df)),
        "pearson_r_human_vs_llm": round(float(pear.statistic), 4),
        "spearman_rho_human_vs_llm": round(float(spear.statistic), 4),
        "mean_human": round(float(df["abs_human"].mean()), 4),
        "mean_llm": round(float(df["abs_llm"].mean()), 4),
        "share_llm_gt_human": round(float((df["abs_llm"] > df["abs_human"]).mean()), 4),
    }
    out["rows"] = df[["dataset", "task", "var_a", "var_b", "abs_human", "abs_llm"]].to_dict(orient="records")
    return out


def part_b_eval3_calibration() -> dict[str, Any]:
    """Same calibration test but for Eval3 predictive-strength (R2 / pseudo-R2
    of outcome ~ 5 demographic anchors), task1 baseline only. Human R2 is
    uniformly weak (demographics barely explain attitudes/health); LLM R2 varies
    hugely by outcome. If driven by true signal we would NOT expect this
    heterogeneity, since true signal is uniformly weak across all outcomes."""
    df = pd.read_csv(P1 / "eval3_evaluation" / "eval3_outcome_metrics.csv")
    t1 = df[(df["task"].isin(["task1", "s0"])) & df["metric_llm"].notna()].copy()
    t1["inflation"] = t1["metric_llm"] - t1["metric_human"]
    pear = stats.pearsonr(t1["metric_human"], t1["metric_llm"])
    return {
        "n_outcomes": int(len(t1)),
        "human_r2_range": [round(float(t1["metric_human"].min()), 4), round(float(t1["metric_human"].max()), 4)],
        "llm_r2_range": [round(float(t1["metric_llm"].min()), 4), round(float(t1["metric_llm"].max()), 4)],
        "pearson_r_human_vs_llm": round(float(pear.statistic), 4),
        "pearson_p": float(pear.pvalue),
        "mean_inflation": round(float(t1["inflation"].mean()), 4),
        "rows": t1[["dataset", "outcome", "metric_human", "metric_llm", "inflation"]]
        .sort_values("inflation", ascending=False)
        .to_dict(orient="records"),
    }


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    d = pd.DataFrame({"y": values, "g": groups}).dropna()
    if d["g"].nunique() < 2 or len(d) < 10:
        return float("nan")
    grand_mean = d["y"].mean()
    ss_between = d.groupby("g")["y"].apply(lambda s: len(s) * (s.mean() - grand_mean) ** 2).sum()
    ss_total = ((d["y"] - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


def part_c_phase2_info_sufficiency() -> dict[str, Any]:
    """For Phase2 UCNets B1 -> B2 (size_bucket) and B1 -> B3 (population
    priors): does the true informativeness of the added condition for a given
    field predict the LLM's fidelity gain on that field?"""
    gold = [json.loads(l) for l in (GOLD / "ucnets_w1_network_gold.jsonl").read_text().splitlines()]
    comp = pd.DataFrame([g["target_json"]["composition_summary"] for g in gold])

    fields = ["num_alters", "kin_share", "friend_share", "coworker_share", "confidant_count", "companion_count"]
    eta2_size = {f: round(eta_squared(comp[f], comp["size_bucket"]), 4) for f in fields}

    eval_summary = json.loads((NET_EVAL / "phase2_eval_summary.json").read_text())
    # B1/B2/B3 are compared on the same N0-N3 schema_ok ego intersection.
    common = eval_summary.get("n0_n3_common", {}).get("per_task", eval_summary)
    b1_fields = common["B1"]["layer1"]["fields"]
    b2_fields = common["B2"]["layer1"]["fields"]
    b3_fields = common["B3"]["layer1"]["fields"] if "B3" in common else {}

    def gain_row(f: str, base: dict, cond: dict) -> dict[str, Any]:
        if f not in base or f not in cond:
            return {}
        bt = base[f].get("tv_binned", base[f].get("tv"))
        ct = cond[f].get("tv_binned", cond[f].get("tv"))
        be = base[f].get("entropy_ratio_binned", base[f].get("entropy_ratio"))
        ce = cond[f].get("entropy_ratio_binned", cond[f].get("entropy_ratio"))
        bk, ck = base[f].get("kappa"), cond[f].get("kappa")
        return {
            "tv_B1": bt, "tv_cond": ct, "tv_gain": None if bt is None or ct is None else round(bt - ct, 4),
            "entropy_ratio_B1": be, "entropy_ratio_cond": ce,
            "entropy_ratio_gain_abs": None if be is None or ce is None else round(abs(be - 1) - abs(ce - 1), 4),
            "kappa_B1": bk, "kappa_cond": ck,
            "kappa_gain": None if bk is None or ck is None else round(ck - bk, 4),
        }

    b2_table = {}
    for f in fields:
        row = gain_row(f, b1_fields, b2_fields)
        row["eta2_vs_size_bucket"] = eta2_size.get(f)
        b2_table[f] = row

    # B3: informativeness proxied by how far B1's own mean already was from the
    # true population mean (the "correction distance" the prior needed to close).
    priors = json.loads((GOLD / "ucnets_w1_population_priors.json").read_text())
    b3_table = {}
    for f in fields:
        row = gain_row(f, b1_fields, b3_fields)
        mean_human = b1_fields.get(f, {}).get("mean_human")
        mean_llm_b1 = b1_fields.get(f, {}).get("mean_llm")
        row["correction_needed_abs"] = (
            None if mean_human is None or mean_llm_b1 is None else round(abs(mean_llm_b1 - mean_human), 4)
        )
        b3_table[f] = row

    # Correlate eta2 (true informativeness) with tv_gain across fields, for B2
    valid = [(v["eta2_vs_size_bucket"], v["tv_gain"]) for v in b2_table.values() if v.get("eta2_vs_size_bucket") == v.get("eta2_vs_size_bucket") and v.get("tv_gain") is not None]
    corr_b2 = None
    if len(valid) >= 3:
        xs, ys = zip(*valid)
        corr_b2 = round(float(stats.spearmanr(xs, ys).statistic), 4)

    valid3 = [(v["correction_needed_abs"], v["tv_gain"]) for v in b3_table.values() if v.get("correction_needed_abs") is not None and v.get("tv_gain") is not None]
    corr_b3 = None
    if len(valid3) >= 3:
        xs, ys = zip(*valid3)
        corr_b3 = round(float(stats.spearmanr(xs, ys).statistic), 4)

    return {
        "eta2_field_vs_size_bucket": eta2_size,
        "B1_to_B2_table": b2_table,
        "B1_to_B3_table": b3_table,
        "spearman_eta2_vs_tv_gain_B2": corr_b2,
        "spearman_correction_needed_vs_tv_gain_B3": corr_b3,
        "population_priors": priors,
    }


def make_figures(a: dict, b: dict, c: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    rows = a["rows"]
    xs = [r["abs_human"] for r in rows]
    ys = [r["abs_llm"] for r in rows]
    axes[0].scatter(xs, ys, alpha=0.6, s=25)
    lim = max(max(xs), max(ys)) * 1.05
    axes[0].plot([0, lim], [0, lim], "k--", lw=1, label="calibrated (y=x)")
    axes[0].set_xlabel("True association strength (human data)")
    axes[0].set_ylabel("LLM-reproduced association strength")
    axes[0].set_title(f"Eval2 pairs, all tasks/datasets (n={len(rows)})\nSpearman rho={a['pooled']['spearman_rho_human_vs_llm']}")
    axes[0].legend()

    b_rows = b["rows"]
    xs2 = [r["metric_human"] for r in b_rows]
    ys2 = [r["metric_llm"] for r in b_rows]
    axes[1].scatter(xs2, ys2, alpha=0.7, s=30, color="darkorange")
    lim2 = max(max(xs2), max(ys2)) * 1.05
    axes[1].plot([0, lim2], [0, lim2], "k--", lw=1, label="calibrated (y=x)")
    axes[1].set_xlabel("True R2 / pseudo-R2 (demographics -> outcome)")
    axes[1].set_ylabel("LLM-reproduced R2 / pseudo-R2")
    axes[1].set_title(f"Eval3 task1 outcomes (n={len(b_rows)})\nPearson r={b['pearson_r_human_vs_llm']}")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_eval2_human_vs_llm_association.png", dpi=150)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(7, 5.5))
    b2 = c["B1_to_B2_table"]
    fx = [v["eta2_vs_size_bucket"] for v in b2.values()]
    fy = [v["tv_gain"] for v in b2.values()]
    labels = list(b2.keys())
    ax.scatter(fx, fy, s=60, color="steelblue")
    for lbl, x, y in zip(labels, fx, fy):
        if x is not None and y is not None:
            ax.annotate(lbl, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("True eta2(field ~ size_bucket) in human data")
    ax.set_ylabel("B1->B2 TV-distance fidelity gain (positive = improved)")
    ax.set_title(f"Phase2 UCNets: does true informativeness of size_bucket\npredict the B2 fidelity gain? Spearman rho={c['spearman_eta2_vs_tv_gain_B2']}")
    fig2.tight_layout()
    fig2.savefig(OUT / "fig_phase2_info_sufficiency.png", dpi=150)
    plt.close(fig2)


def main() -> int:
    a = part_a_eval2_calibration()
    b = part_b_eval3_calibration()
    c = part_c_phase2_info_sufficiency()
    results = {"eval2_calibration": a, "eval3_calibration": b, "phase2_info_sufficiency": c}
    (OUT / "analysis_a_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    make_figures(a, b, c)

    lines = ["# Analysis A: Information Sufficiency vs LLM Fidelity Gain\n"]
    lines.append("## Part 1: Eval2 pairwise association calibration (Phase1, all datasets/tasks)\n")
    lines.append(f"Pooled n={a['pooled']['n_pairs']} pairs. Mean true |association|={a['pooled']['mean_human']}, ")
    lines.append(f"mean LLM-reproduced |association|={a['pooled']['mean_llm']}. ")
    lines.append(f"Share of pairs where LLM overshoots human={a['pooled']['share_llm_gt_human']}. ")
    lines.append(f"Pearson r(human,llm)={a['pooled']['pearson_r_human_vs_llm']}, Spearman rho={a['pooled']['spearman_rho_human_vs_llm']}.\n")
    lines.append("By task:\n")
    for task, v in a["by_task"].items():
        lines.append(
            f"- {task}: n={v['n_pairs']}, mean_human={v['mean_human']}, mean_llm={v['mean_llm']}, "
            f"inflation={v['mean_inflation_llm_minus_human']}, spearman={v['spearman_rho_human_vs_llm']}, "
            f"slope={v['ols_slope_llm_on_human']}, share_overshoot={v['share_llm_gt_human']}"
        )
    lines.append("\n## Part 2: Eval3 predictive-strength calibration (task1 baseline)\n")
    lines.append(
        f"n={b['n_outcomes']} outcomes. Human R2 range={b['human_r2_range']} (uniformly weak). "
        f"LLM R2 range={b['llm_r2_range']} (highly heterogeneous). "
        f"Pearson r(human,llm)={b['pearson_r_human_vs_llm']}. Mean inflation={b['mean_inflation']}.\n"
    )
    lines.append("Top over-associated outcomes (LLM manufactures far more demographic determinism than exists):\n")
    for r in b["rows"][:5]:
        lines.append(f"- {r['dataset']}/{r['outcome']}: human={r['metric_human']:.3f} llm={r['metric_llm']:.3f} inflation=+{r['inflation']:.3f}")
    lines.append("\n## Part 3: Phase2 UCNets info-sufficiency (does true relevance predict the fidelity gain)\n")
    lines.append(f"Spearman rho(eta2_size_bucket, B1->B2 TV gain) across fields = {c['spearman_eta2_vs_tv_gain_B2']}")
    lines.append(f"Spearman rho(correction_needed, B1->B3 TV gain) across fields = {c['spearman_correction_needed_vs_tv_gain_B3']}\n")
    lines.append("Field-level detail (B1->B2):\n")
    for f, v in c["B1_to_B2_table"].items():
        lines.append(f"- {f}: eta2={v.get('eta2_vs_size_bucket')}, tv_gain={v.get('tv_gain')}, kappa_gain={v.get('kappa_gain')}")
    lines.append("\nField-level detail (B1->B3):\n")
    for f, v in c["B1_to_B3_table"].items():
        lines.append(f"- {f}: correction_needed={v.get('correction_needed_abs')}, tv_gain={v.get('tv_gain')}, kappa_gain={v.get('kappa_gain')}")

    (OUT / "analysis_a_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {OUT / 'analysis_a_results.json'}")
    print(f"wrote {OUT / 'analysis_a_summary.md'}")
    print(f"wrote {OUT / 'fig_eval2_human_vs_llm_association.png'}")
    print(f"wrote {OUT / 'fig_phase2_info_sufficiency.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
