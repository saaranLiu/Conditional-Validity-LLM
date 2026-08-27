"""Regenerate fig_info_format_interaction.png from analysis_c_results.json.

Bar chart of mean_tv across B1 / info-only / format-only(B4) / combo / additive
prediction, for the size arm (B2/B6) and the prior arm (B3/B7), on the same
400-person paired subsample.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/bld/data/data2/jiaxin/simulation")
OUT = ROOT / "Reports" / "cross_phase_analysis"

d = json.loads((OUT / "analysis_c_results.json").read_text())
size = d["size_arm_B1_B2_B4_B6"]["mean_tv"]
prior = d["prior_arm_B1_B3_B4_B7"]["mean_tv"]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["B1\n(baseline)", "info\nalone", "B4\nformat alone", "info+format\ncombo (actual)", "additive\nprediction"]
size_vals = [size["B1"], size["B2"], size["B4"], size["B6"], size["additive_prediction"]]
prior_vals = [prior["B1"], prior["B3"], prior["B4"], prior["B7"], prior["additive_prediction"]]

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
colors = ["#888888", "#4c72b0", "#dd8452", "#c44e52", "#55a868"]

for ax, vals, title in (
    (axes[0], size_vals, f"size arm (B2/B6), interaction={size['interaction']:+.4f}"),
    (axes[1], prior_vals, f"prior arm (B3/B7), interaction={prior['interaction']:+.4f}"),
):
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("mean TV (lower is better)")
    ax.set_ylim(0, max(size_vals + prior_vals) * 1.25)

fig.suptitle("Info x Format interaction decomposition: actual combo vs additive prediction", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "fig_info_format_interaction.png", dpi=150)
plt.close(fig)
print(f"wrote {OUT / 'fig_info_format_interaction.png'}")
