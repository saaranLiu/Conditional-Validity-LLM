#!/usr/bin/env python3
"""人类侧参照量：每个候选字段有多少「可检测空间」。

这是 logit 实验的设计输入，不调用任何模型。要回答的问题是：给定人口学变量，人类的条件
分布到底比边缘分布窄多少。差距越小，就越能说「人口学本来决定不了这个结果」，模型若在该
字段上给出很尖的分布，就越是它自己的问题而不是数据本身可预测。

对每个字段报四个量：

  H(边缘)          人类边缘分布的熵，单位 bit
  H(条件)          多项 logit 以模型看得到的同一批人口学变量预测该字段，逐 ego 条件熵取均值
  可检测空间       H(边缘) - H(条件)，人口学真正削掉的不确定性
  基线准确率差     多项 logit 准确率减去「总是猜众数」的准确率

「可检测空间」小意味着：人类条件分布几乎和边缘分布一样宽，这个字段适合用来检验模型是否
过度确定。若某字段这个量很大，说明人口学确实能定下结果，模型分布尖就不算错，该字段不适合
承担这个判据。

条件熵用 5 折交叉验证的样本外预测算，避免用训练集内的过拟合分布把参照熵压低。

用法: python3 Reports/Predictor_vs_Sampler_Analysis/scripts/human_reference_headroom.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
ROOT = ANALYSIS_DIR.parent.parent
GOLD = ROOT / "Survey" / "ucnets" / "cleaned" / "ucnets_w1_s0.jsonl"
INPUT = ROOT / "Testing" / "T1_association" / "ucnets_w1" / "input" / "s0_input.jsonl"
OUT_JSON = ANALYSIS_DIR / "output" / "human_reference_headroom.json"

# 模型在 task1 里看得到的条件变量，逐字对齐 prompt_json 的键。
# 条件熵必须用「模型看得到的同一批信息」来算，否则参照点不公平。
DEMOGRAPHICS = [
    "age_group",
    "gender",
    "race_ethnicity",
    "education",
    "work_status",
    "marital_status",
    "children_status",
    "romantic_relationship",
    "household_other_people_count",
]

# 候选字段。personality 是嵌套的，用 "personality.<子项>" 定位。
CANDIDATE_FIELDS = [
    "self_rated_health",
    "religion_current",
    "days_lonely_past_week",
    "days_isolated_past_week",
    "days_very_happy_past_week",
    "wants_more_people_for_personal_concerns",
    "wants_more_people_for_good_time",
    "relatives_help_serious_problem",
    "voting_frequency",
    "personality.handles_stress_well",
    "personality.nervous_easily",
    "personality.sociable",
    "personality.trusting",
    "personality.lazy",
]


def get_field(target: dict[str, Any], field: str) -> Any:
    if "." in field:
        head, tail = field.split(".", 1)
        sub = target.get(head)
        return sub.get(tail) if isinstance(sub, dict) else None
    return target.get(field)


def entropy_bits(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log2(p)).sum())


def main() -> None:
    gold = {json.loads(l)["respondent_id"]: json.loads(l) for l in GOLD.open(encoding="utf-8")}
    inputs = {json.loads(l)["respondent_id"]: json.loads(l) for l in INPUT.open(encoding="utf-8")}
    shared = [rid for rid in inputs if rid in gold]

    allowed = next(iter(inputs.values()))["allowed_values"]

    results: list[dict[str, Any]] = []
    for field in CANDIDATE_FIELDS:
        ys: list[str] = []
        xs: list[list[str]] = []
        for rid in shared:
            y = get_field(gold[rid]["target_json"], field)
            if y is None or y == "":
                continue
            pj = inputs[rid]["prompt_json"]
            xs.append([str(pj.get(d, "missing")) for d in DEMOGRAPHICS])
            ys.append(str(y))

        n = len(ys)
        counts = Counter(ys)
        # 极稀有类别会让分层交叉验证无法切分，且本身也不是可检验的对比。
        rare = {k for k, v in counts.items() if v < 10}
        if rare:
            keep = [i for i, y in enumerate(ys) if y not in rare]
            ys = [ys[i] for i in keep]
            xs = [xs[i] for i in keep]
            counts = Counter(ys)

        labels = sorted(counts)
        k = len(labels)
        if k < 2 or len(ys) < 100:
            results.append({"field": field, "skipped": f"类别数 {k}，样本 {len(ys)}"})
            continue

        marg = np.array([counts[c] / len(ys) for c in labels])
        h_marg = entropy_bits(marg)

        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X = enc.fit_transform(np.array(xs))
        idx = {c: i for i, c in enumerate(labels)}
        y_arr = np.array([idx[c] for c in ys])

        # 样本外条件概率：用训练集内的预测会高估人口学的解释力，把参照熵压得过低。
        oof = np.zeros((len(y_arr), k))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260814)
        for tr, te in skf.split(X, y_arr):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X[tr], y_arr[tr])
            proba = clf.predict_proba(X[te])
            for j, cls in enumerate(clf.classes_):
                oof[te, cls] = proba[:, j]

        oof = np.clip(oof, 1e-12, 1.0)
        oof /= oof.sum(axis=1, keepdims=True)
        h_cond = float(np.mean([entropy_bits(row) for row in oof]))

        acc_model = float((oof.argmax(axis=1) == y_arr).mean())
        acc_mode = float(marg.max())

        opts = allowed.get(field.split(".")[0])
        if isinstance(opts, dict) and "type" in opts:
            n_opts = int(opts["max"]) - int(opts["min"]) + 1
        elif isinstance(opts, dict):
            sub = opts.get(field.split(".", 1)[1]) if "." in field else None
            n_opts = len(sub) if isinstance(sub, list) else None
        elif isinstance(opts, list):
            n_opts = len(opts)
        else:
            n_opts = None

        results.append(
            {
                "field": field,
                "n": len(ys),
                "n_classes_observed": k,
                "n_options_in_prompt": n_opts,
                "dropped_rare_classes": sorted(rare) or None,
                "human_marginal": {c: round(float(p), 4) for c, p in zip(labels, marg)},
                "H_marginal_bits": round(h_marg, 4),
                "H_conditional_bits": round(h_cond, 4),
                "headroom_bits": round(h_marg - h_cond, 4),
                "H_uniform_bits": round(math.log2(k), 4),
                "acc_logit": round(acc_model, 4),
                "acc_mode_baseline": round(acc_mode, 4),
                "acc_gain": round(acc_model - acc_mode, 4),
            }
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {"dataset": "ucnets_w1", "demographics_used": DEMOGRAPHICS, "fields": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    usable = [r for r in results if "skipped" not in r]
    usable.sort(key=lambda r: r["headroom_bits"])
    print(f"{'字段':40s} {'n':>5s} {'选项':>4s} {'H(边缘)':>8s} {'H(条件)':>8s} {'可检测空间':>10s} {'准确率增益':>10s}")
    print("-" * 96)
    for r in usable:
        print(
            f"{r['field']:40s} {r['n']:5d} {r['n_classes_observed']:4d} "
            f"{r['H_marginal_bits']:8.3f} {r['H_conditional_bits']:8.3f} "
            f"{r['headroom_bits']:10.3f} {r['acc_gain']:+10.3f}"
        )
    for r in results:
        if "skipped" in r:
            print(f"{r['field']:40s} 跳过：{r['skipped']}")
    print(f"\n明细写入 {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
