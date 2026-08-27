# T4 个体重复抽样：设计说明

**状态：** R1 已生成；评估见 `ucnets_w1/eval/T4_Evaluation_Report.md`。  
**数据：** UCNets Wave 1 only  
**Network 协议：** 与 **N0**（旧 B1）相同（oneshot，`format_only`）

---

## 1. 研究问题

在尚无 Response/Dynamics 时，问：

> 网络集体失真有多少来自 **坏 State（\(S_L\)）传入**，有多少是 **Network 生成器本身**（即便给定 \(S_H\)）？

形式化：

```text
R0:  S_H → N_L(S_H)     # 等于 N0，不另生成
R1:  S_L → N_L(S_L)     # Full LLM 两阶段串联

A_state→net(k) = d_k(N_L(S_L), N_H) − d_k(N_L(S_H), N_H)
```

- \(A \approx 0\)：Network 主导  
- \(A \gg 0\)：State 失真向 Network 传播  

---

## 2. R1 输入构造

对每个 `PRIM_KEY` ∈（N0 gold ∩ S0 input ∩ S0 schema_ok）：

```text
prompt_json = X  ∪  S_L
```

- \(X\)：S0 `prompt_json`（域 A–C）  
- \(S_L\)：S0 `prediction_json`（归一化后、`schema_ok`）  
- **不用** \(S_H\)；**不泄漏**任何网络真值  
- `gold_target` / `metadata` 仍挂人类 \(N_H\)，仅供评估  
- `task` 名：`r1_llm_state_network`  
- Prompt 正文复用 N0 instruction / rules / schema（`prompt_text` 不改）

数值字段（幸福/孤独天数等）写入 prompt 时统一为**字符串**，与 N0 一致。

---

## 3. 生成与评估

| 项 | 设定 |
|---|---|
| 模型 | `llama3.1:latest` |
| GPU | 3 → ollama `127.0.0.1:11436` |
| Workers | 16（可调） |
| Resume | 是 |
| 主指标 | Mean TV、`num_alters`、shares、\(D_h\)、Layer4 机制项；按字段报 \(A\) |

分析样本 = R0 与 R1 均 `schema_ok` 且同 `respondent_id` 的交集。

脚本：`Testing/scripts/build_r1_inputs.py`、`launch_r1_ucnets_gpu3.sh`、`evaluate_t4.py`。
