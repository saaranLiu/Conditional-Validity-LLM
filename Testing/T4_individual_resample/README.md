# T4 个体重复抽样（R0 / R1）

UCNets Wave 1。比较真实 State 与 LLM State 喂给同一网络生成器后，总体分布是否稳定、个体输出是否仍响应具体输入。

## 条件

| 条件 | 定义 | 路径 |
|---|---|---|
| **N_H** | 人类金标 | `Survey/ucnets/cleaned/ucnets_w1_network_gold.jsonl` |
| **R0** | \(N_L(S_H)\)，等于 N0 | `Testing/T1_association/ucnets_w1/output/.../n0_baseline_network/`（本目录 `output/r0` 为符号链接） |
| **R1** | \(N_L(S_L)\)：S0 的 \(S_L\) → 同 N0 协议 | 本目录 `input/` `output/` |
| **S0** | \(S_L\) vs \(S_H\)（辅助） | `Testing/T1_association/ucnets_w1/` |

归因（网络量 \(d\)）：

```text
A_state→net = d(N_L(S_L), N_H) − d(N_L(S_H), N_H)
```

## 目录

```text
Testing/T4_individual_resample/
  README.md
  Design.md
  ucnets_w1/
    input/          # R1 prompts
    output/         # R1 generations
    eval/           # 评估报告
```

## 快速命令

```bash
python Testing/scripts/build_r1_inputs.py
bash Testing/scripts/launch_r1_ucnets_gpu3.sh
python Testing/scripts/evaluate_t4.py
```

当前范围：**仅 UCNets，仅 N0 协议，R1 生成**。臂名对照见 [`../NAMING.md`](../NAMING.md)。
