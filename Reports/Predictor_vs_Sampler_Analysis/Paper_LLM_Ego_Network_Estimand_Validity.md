# LLM 生成个体网络的集体效度：Predictor 对 Sampler 的诊断

**版本：** 2026-08-25（五模型完整结果 + native-path decoding 更新版）

**模型（5）：** `llama3.1:8b`、 `Qwen3-235b`、`DeepSeek-V4-Flash`、Ollama `qwen3:8b`、`qwen3:14b`。每条生成臂 41/41 齐全，评估已跑完。

**主数据：** UCNets Wave 1（唯一跑完 State S0–S2、Network N0–N6、R0/R1 的 survey）。补充 survey 只补 UCNets 测不干净的识别力（见 §3）。臂名对照：`Testing/NAMING.md`。

**独立机制实验：** `Testing/decoding/` 用 HF 本地 Llama-3.1-8B，在同一原始 JSON prompt 与推理栈下比较 native JSON slot sequence scores 和 matched counterfactual full-JSON generation（§4.7）

---

## 摘要

RQ：单个 LLM agent 能否生成在**总体层面**（边际、关联、条件依赖）上有效的个体状态 \(S\) 与自我中心网络 \(N\)？管线拆成 **State → Network** 两阶段，先用 T1–T3 确认两阶段均存在系统性失真，再用 T4–T6 与 decoding 探针检验更根本的机制：**LLM 是条件采样器（sampler）还是检索式预测器（predictor）**？

**机制对照**（Sampler vs Predictor）：

```text
Sampler:   X,B ──► 从 P_H(·|X,B) 抽样 ──► Y_L          （P_L ≈ P_H；换人则分布变）
Predictor: X,B ──► 检索典型模板 g(X,B) ──► Y_L ≈ g+ε  （g 平坦；偏社会平均）
```

**主要结论**（五模型 × 六判据，异质，并非全为 predictor）：

- **T3 条件跟随：** 在 prompt 中写入闭合度 high/low 分箱后，用 **span**（high 组与 low 组靶量均值的差）衡量输出是否随条件移动。GSS 情感闭合度上，Llama span\(_L=0.03\)，high/low 几乎无差别；DeepSeek 为 0.53，人类 span\(_H=0.76\)——模型间调制能力差异大，但多数仍远未跟上人类幅度。
- **基线与归因：** **composition TV** 比较 LLM 与人类在各 network composition 字段上的边际分布差距（0=一致，1=完全不重叠）；UCNets N0 上所有模型仍偏高（0.45–0.65），说明无条件生成的网络集体分布已与人类偏离较大。**T4 归因 \(A=\mathrm{TV}(R1,N_H)-\mathrm{TV}(R0,N_H)\)** 比较 R1（LLM State）与 R0（真 State）生成网络后距人类的 TV 差；\(A\) 普遍接近 0，表明修 State 对 Network 集体误差的作用很小。
- **Native-path：** 对同一 400 ego 反事实替换 bucket，**high−low \(\Delta\)** 为换到 high 分箱后靶量组均值相对 low 的抬升。Llama 完整 JSON 确有响应（同性别 \(\Delta=0.0348\)，闭合度 \(\Delta=0.0498\)），但仅为 human span 的 ~6%/7%，即条件信号只被微弱放大。进一步，每个 ego 上 native slot 的 \(\Delta\) 与 full JSON 的 \(\Delta\) 几乎不相关（Pearson \(r=-0.039/0.007\)），说明 teacher-forced 下的局部条件偏好不能预测完整 JSON 实现。
- **综合机制：** 符合**条件语义可访问，但默认模板主导、联合网络响应高度压缩**。

---

## 1. 故事线：从分布拟合到生成机制

```text
背景 X  ──►  State S  ──►  Network N  ──►  集体估计量 Y
              │                │
         Eval 1–3          evaluate_network
         (T1 State)        (T1–T3 Network, T5)
                              │
                    §5.2 收束：模板 → 可访问 → 调制 → 联合实现
```

1. **现象层（§4.0）：** State 与 Network 无条件基线均已失真（TV 高、熵压缩、κ≈0）。
2. **机制层（T1–T3）：** 关联是否校准？条件信息是否按真实信息量被使用？显式分箱写进 prompt 后是否跟随？
3. **归因层（T4）：** 换 LLM State 能否改善 Network 误差？
4. **工程层（T5）：** 信息 × 格式是否可加？（非 sampler 必要条件）
5. **外部效度（T6）：** 换国家/年份/人群，输出是否随人类总体差异移动？
6. **Decoding 层（§4.7）：** 同一 ego 反事实条件下，native slot 是否移动、能否预测 full JSON 实现？

§5.2 沿上述六步收束为四层生成机制（模板 → 可访问 → 调制 → 联合实现），并给出 RQ 答案与评估/改进含义。
---

## 2. 理论框架与记号

### 2.1 生成对象

| 符号 | 含义 |
|---|---|
| \(i\) | 受访者（ego） |
| \(X_i\) | 背景协变量（年龄、性别、教育、婚姻等，**两阶段均用真值**） |
| \(S_i^H, S_i^L\) | 人类 / LLM 状态向量 |
| \(N_i^H, N_i^L\) | 人类 / LLM 网络（composition + roster） |
| \(B_i\) | 显式条件（健康、size_bucket、闭合度分箱等） |
| \(Y\) | 集体估计量（TV、η²、分箱均值等） |

**臂的输入输出：** \(S_L = f_S(X)\)（S0）；\(N_L = f_N(X, S^*)\)（N0 用 \(S^H\)；R1 用 \(S^L\)）。T3 在 N0 协议上加 \(B=\) 靶量分箱。

### 2.2 Sampler vs Predictor（可检验含义）

| 机制 | 形式化 | 在本文中的可观察含义 |
|---|---|---|
| **Sampler** | \(P_L(Y\mid X,B) \approx P_H(Y\mid X,B)\) | 校准散点近 \(y=x\)；T3 组间跨度 span\(_L \approx\) span\(_H\)；T6 跨国/跨年同号 |
| **Predictor** | \(Y_L \approx g(X,B)\)，\(g\) 平坦 | T3 span\(_L \ll\) span\(_H\)；T2 改善与真实 η² 错配；T6 输出跨总体几乎不变 |

**识别边界：** T3 与 native-path full-JSON counterfactual 直接检验输出靶量是否随显式条件移动；native slot score 只识别固定 teacher-forced 上下文中的**局部条件可访问性**，不能解释为完整网络的潜在字段概率。其余多数判据只能**拒绝完全校准**，不能唯一识别 predictor；T5 明确不是 sampler 必要条件。

### 2.3 核心统计量

| 符号 | 定义 | 主要用途 |
|---|---|---|
| TV | \(\mathrm{TV}(P,Q)=\frac12\sum_k|P_k-Q_k|\) | 边际分布差距（Eval1；Network composition） |
| ER | \(H(Y_L)/H(Y_H)\) | 熵比，\(\ll1\) 为压缩 |
| η² | 组间平方和 / 总平方和 | T1 关联；T3 组间效应 |
| κ | Cohen's kappa | **个体** exact match（低 κ 不单独否定 sampler） |
| \(R^2\), pseudo-\(R^2\) | OLS / McFadden | Eval3 人口学解释力 |
| span | \(\max_b \bar Y_b - \min_b \bar Y_b\) | T3 区间跟随 |
| \(A\) | \(\mathrm{TV}(R1,N_H)-\mathrm{TV}(R0,N_H)\) | T4 State→Network 归因 |

**评估口径（2026-08-20 起）：** 先过滤 `schema_ok`；跨臂比较用 ego 交集；Network 主指标为 composition 字段 mean TV。

---

## 3. 实验设计：Survey、臂与五模型覆盖

### 3.1 H & testing

| 判据 | 主 Survey | 补充 Survey | 各模型生成 | 各模型评估 |
|---|---|---|---|:---:|
| **T1** 关联 | UCNets State+Net | GSS 1987/2010 State；GSS85/04、NSHAP N0 | ✅ | ✅ |
| **T2** 信息使用 | UCNets S1/S2, N1/N2 | GSS 1987/2010 S1/S2 | ✅ | ✅ |
| **T3** 区间跟随 | UCNets N4（退化） | GSS85/04/24, NSHAP, Signos | ✅ | ✅ |
| **T4** State 替换 | UCNets R0/R1 | GSS 1985, NSHAP R0/R1 | ✅ | ✅ |
| **T5** 可加性 | UCNets N5/N6 | — | ✅ | ✅ |
| **T6** 跨总体 | — | ISSP T6；GSS85↔04；NSHAP↔UCNets；GSS24 | ✅ | ✅ |
| **Decoding** | — | GSS85 同性别/闭合；UCNets size | —（仅 llama HF） | ✅ |

**脚本：** State `T1_association/scripts/evaluate_state_eval{1,2,3}.py`；Network `T1_association/scripts/evaluate_network.py`；T4 `T4_individual_resample/scripts/evaluate_t4.py`；T6 `T6_cross_population/scripts/evaluate_t6.py`。批量入口：`Testing/common/run_model_evaluation.sh`。

### 3.2 T3 靶量

| Survey | 臂 | 靶量 | 分箱 |
|---|---|---|---|
| GSS 1985/2004 | N4c / N4s | 情感闭合 / 同性别 | 见 `NAMING.md` §3.4 |
| NSHAP r1 | N4c | 周联系密度 | 0.20 / 0.50 |
| Signos | N4o / N4r | 同源国 / 住西班牙 | 0.70/0.90 等 |
| GSS 2024 | N4 | 每周联系占比 | 0.75 / 1.0 |
| UCNets | N4 | 同种族（**90.6% 顶格，探索性**） | — |

---

## 4. 结果

### 4.0 基线：两阶段均已失真

**State（S0，UCNets）mean TV：**

| 模型 | mean TV | pass_rate（Eval1 χ²/K-S） |
|---|---:|---:|
| llama3.1_latest | 0.294 | 0/26 |
| Qwen | 0.375 | 1/26 |
| DeepSeek-V4-Flash | 0.420 | 0/26 |
| qwen3_8b | 0.574 | 0/26 |
| qwen3_14b | 0.507 | 0/26 |

**Network（N0，UCNets composition mean TV / entropy ratio）：**

| 模型 | mean TV | ER | \(n\) |
|---|---:|---:|---:|
| llama3.1_latest | 0.509 | 1.54 | 1094 |
| Qwen | 0.481 | 1.15 | 1089 |
| DeepSeek-V4-Flash | 0.446 | 1.24 | 1140 |
| qwen3_8b | 0.543 | 0.72 | 1030 |
| qwen3_14b | 0.648 | 0.78 | 1154 |

个体 κ 全程 ≈0：模型不是点预测器，但总体仍可能看起来合理。

**结果解释：**

1. **State 阶段已经存在系统性总体误差。** Mean TV 可理解为：对一个典型字段，需要重新分配约 29%–57% 的概率质量，LLM 边际分布才能与人类样本对齐。即使表现最好的 Llama，26 个字段也没有一个通过当前 Eval1 检验；因此问题不是少数困难变量拖累平均值，而是跨字段的普遍偏移。
2. **Network 误差不能简单归因于错误 State。** N0 已经向模型提供真实 \(S_H\)，composition mean TV 仍为 0.45–0.65，且整体高于 State 阶段。这说明即使上游状态信息正确，Network generator 仍会把人口背景映射到偏离真实分布的网络模板；该观察为后续 T4 的 State 替换实验提供了基线。
3. **熵失真方向并不统一。** qwen3_8b/14b 的 ER<1，表现为输出集中和模式压缩；Llama、Qwen、DeepSeek 的 ER>1，则说明某些构成字段被过度摊平或产生额外离散性。共同问题并非所有模型都呈现低熵，而是**生成分布的集中程度也未校准**。
4. **κ≈0 不能单独否定 sampler。** 一个正确 sampler 本来就不应逐人复现同一个真实网络；这里真正具有否证力的是 marginal TV、entropy 和后续 association/conditional response 同时失真。换言之，论文批评的不是逐人猜中，而是合成人群未能恢复正确的统计规律。

#### I/O 示例：State S0 → Network N0

**Input（S0，`s0_input.jsonl` 片段）：** 只给背景 \(X\)，要求生成状态。

```json
{
  "respondent_id": "3000000000036",
  "prompt_json": {
    "age": "53", "gender": "female", "marital_status": "divorced",
    "education": "some_college_or_associate", "work_status": "taking care of home or family"
  }
}
```

**Output（S0，`ucnets_w1_output.jsonl`）：** LLM 补全的状态 \(S_L\)。

```json
{
  "prediction_json": {
    "self_rated_health": "good",
    "personality": "somewhat outgoing",
    "party_identity": "independent"
  },
  "validation": { "schema_ok": true }
}
```

**Input（N0）：** 同一 \(X\) **加上真值** \(S_H\)（非 LLM 状态），生成网络。

**Output（N0，`composition_summary`）：**

```json
{
  "num_alters": 20, "kin_share": 0.05, "friend_share": 0.70,
  "pct_weekly_plus_contact": 0.80
}
```

（人类该 ego 的 friend_share 远低于 0.70——构成模板化网络的入口。）

---

### 4.1 T1：关联强度是否与真实强度成正比

**检验：** Sampler ⇒ 配对 \((\eta^2_H, \eta^2_L)\)、\((R^2_H,R^2_L)\) 近 \(y=x\)，Pearson/Spearman → 1。

**方法：** State Eval2/3（Cramér's V、η²、OLS/McFadden \(R^2\)）；Network UCNets Layer2 η² 及补充 survey N0。

**Survey：** UCNets S0–S2；GSS 1987/2010 S0–S2（信任三题）；GSS85/04、NSHAP N0（Network 关联点）。

**llama 参考结果（State，153 pairs）：** Pearson \(r=0.45\)，Spearman \(\rho=0.64\)，55.6% 夸大；Eval3 22 outcomes 的 \(R^2\) 配对 \(r=-0.13\)（\(p=0.56\)）——虚假决定论与真实可解释力不同步。

**结果解释：**

1. **模型保留了部分排序信息，但没有恢复关联幅度。** Spearman \(\rho=0.64\) 高于 Pearson \(r=0.45\)，意味着模型有时知道哪些关系相对更强，但强弱差距经过非线性放大或压缩，散点不会贴近 sampler 所要求的 \(y=x\)。
2. **超过一半关联被夸大，符合刻板化决定论。** 模型容易把年龄、婚姻、政治身份等背景线索转译成过强的状态差异，让合成人群显得比真实人群更可预测。这种看似有社会学规律的输出反而可能导致虚假的效应量和过窄的下游不确定性。
3. **Eval3 的 \(R^2\) 配对相关为负，说明问题不只是统一尺度偏差。** 如果模型只是把所有真实关联按同一比例放大，真实与生成 \(R^2\) 的排序仍应保持为正；\(r=-0.13\) 表明模型对哪些 outcome 最能被人口学解释也缺乏稳定识别。
4. **但条件化能够在特定话题上显著改善排序。** GSS S2 的 \(\rho=0.88\)–0.95 说明政治宗教条件明确进入 prompt 时，模型可以恢复部分关联结构。因此 T1 不支持模型一律关联盲的论断，而支持**话题依赖、条件依赖的非均匀 predictor**：局部结构可能较好，整体校准仍失败。

**识别边界：** 高相关只表示跨关联点的相对排序接近，不等于每个关联的数值校准；低相关则可以拒绝完整 sampler，但不能单独区分记忆不足、刻板先验或生成噪声。后续 T2/T3 通过显式增加信息和直接干预靶量进一步缩小解释空间。

---

### 4.2 T2：条件信息是否按信息量被使用？

**检验：** 真实 η² 越高的字段，条件化后 TV 改善应越大。严格 sampler 应比较 \(P_L(Y|B)\) vs \(P_H(Y|B)\)；当前用 Spearman\((\eta^2, \Delta\mathrm{TV})\) 作代理。

**Survey / 臂：**

| 阶段 | 臂 | Survey |
|---|---|---|
| State | S1 健康、S2 政治宗教 | UCNets；GSS 1987/2010 |
| Network | N1 size_bucket、N2 population prior | UCNets  |

**llama 关键数（UCNets N0→N1，6 字段）：** Spearman\((\eta^2, \Delta\mathrm{TV})=-0.89\)——**改善与真实相关性负相关**；旧评估中 size consistency 从 0.82→0.99，指的是模型生成的 `num_alters` 与模型自己生成的 `size_bucket` 相互一致，**不是**对输入 bucket 的遵循率；N1 对 friend/kin 几乎无效，N3（仅改格式）构成反而最好。§4.7 用 input bucket 重新计算后显示 size 跟随具有明显非对称性。

**结果解释：**

1. **条件是否有效，与它在人类数据中是否真正重要并不一致。** 若模型按 \(P_H(Y\mid B)\) 更新，真实 \(\eta^2\) 较大的 outcome 应获得更大 TV 改善；六字段代理相关为 −0.89，方向恰好相反，说明当前改善更可能由字段名称、任务格式或模型熟悉度驱动。
2. **N1 的高 size self-consistency 不能证明使用了输入信息。** 模型可以让自己生成的 `num_alters` 与自己生成的 `size_bucket` 相互自洽，却仍不遵循 prompt 中的 bucket。§4.7 直接以输入 bucket 为参照后，small 条件的遵循率只有 23.5%，证明输出内部一致与条件校准是两个不同问题。
3. **Population priors 更像均值锚点。** N2 能推动输出中心，但没有同步恢复真实离散程度，表现为模型把总体摘要当作一个典型值，而不是从该总体分布中重新采样个体异质性。
4. **格式变化有时比真实信息更有效。** N3 只改变生成结构却可能得到更好的 composition，说明 prompt 协议会改变模型进入的默认生成轨迹。信息增量与行为改善之间不存在简单单调关系。

**识别边界：** T2 只有少数字段，Spearman 对字段集合敏感，因此 −0.89 应作为探索性机制诊断，而非稳定总体参数。它不能单独证明模型故意忽略信息，但与 T1 的关联错配、T3 的响应压缩共同说明模型没有按人类信息结构进行一致更新。

---

### 4.3 T3：显式分箱下是否跟随？

**检验：** 把靶量分箱 \(B\) 写入 prompt 后，组间均值应随 \(B\) 移动，span\(_L \approx\) span\(_H\)，η²\(_L > 0\)。

**Survey：** 主证据 GSS 1985 N4c（情感闭合）；复核 GSS 2004、NSHAP、Signos；UCNets 。

#### I/O 示例（GSS 1985 N4c）

**Input（`n4c_input.jsonl`）：**

```json
{
  "prompt_json": {
    "age": "33", "marital_status": "married",
    "alter_closure_bucket": "medium"
  },
  "prompt_text": "…Given … the observed degree of closeness among their alters…"
}
```

**Output（`gss_1985_output.jsonl`，LLM 名单 + composition）：**

```json
{
  "prediction_json": {
    "composition_summary": {
      "num_alters": 5, "friend_share": 0.30, "kin_share": 0.20
    },
    "alters": [ "…" ]
  }
}
```

评估时从 `alter_ties` **重算** `esp_close_density`，再按 prompt 中的 bucket 分组（非直接读 composition 字段）。

#### 五模型：GSS 1985 闭合度（N4c）

| 模型 | span\(_H\) | span\(_L\) | η²\(_L\) | 是否跟随 |
|---|---:|---:|---:|:---:|
| llama3.1_latest | 0.756 | **0.032** | 0.025 | 幅度极弱 |
| Qwen | 0.757 | 0.314 | 0.427 | 部分 |
| DeepSeek-V4-Flash | 0.750 | **0.531** | 0.665 | **最强** |
| qwen3_8b | 0.747 | 0.089 | 0.153 | 弱 |
| qwen3_14b | 0.755 | 0.214 | 0.364 | 中等 |

人类三分位均值约 **0.02 / 0.28 / 0.78**；llama 的 LLM 均值钉在 ~0.28 附近（经典 predictor 模板）；DeepSeek 能把高桶抬到 ~0.5+，但仍低于人类高桶。

**UCNets N4（同种族，探索性）：** 三档 LLM 同种族率 0.55/0.56/0.57（span\(_L=0.01\)），人类 0.49→1.00——条件对象（全网）与评价对象（Top-5 core）不同构 + 90.6% 顶格。

**结果解释：**

1. **T3 是全文最直接的 sampler 检验之一。** 条件 \(B\) 本身就是待评估靶量的分箱；若模型是校准 conditional sampler，low/medium/high 的生成均值应接近人类三档均值，而不只是方向正确。
2. **Llama 展现清晰的默认模板。** 人类 closure 从约 0.02 增至 0.78，Llama 却集中在约 0.28，span 只恢复人类的约 4%。这不是完全随机，因为输出落在一个稳定、社会上合理的中档中心；但它也不是条件采样，因为极端条件几乎没有改变联合网络。
3. **五模型差异说明能力不是二元属性。** Qwen、qwen3_14b 和尤其 DeepSeek 能明显抬高 high bucket；DeepSeek 恢复约 71% 的 human span，而 Llama、qwen3_8b 分别只有约 4% 和 12%。因此全模型均为平坦 predictor 的假设被拒绝，更合适的是比较模型在 predictor–sampler 连续体上的调制强度。
4. **较大的 \(\eta^2_L\) 仍不等于完全校准。** DeepSeek 的分组解释力很强，但 high bucket 仍低于人类约 0.78；模型可以很好地区分三档，却继续压缩档内位置与端点幅度。
5. **UCNets N4 不承担严格识别。** 其 condition 基于全网络、评价却基于 Top-5 core alters，且 human bucket 高度顶格；近零 span 可能同时来自模型不跟随、estimand 错配和缺少 headroom。因此正文主结论依赖 GSS/NSHAP 等同构靶量，UCNets 只作探索性复核。

**与 §4.7 的关系：** 原 T3 是按事实 bucket 分组的跨 ego 结果；native-path 实验进一步对同一 ego 反事实替换三个 bucket，并重复生成，确认 Llama full JSON 确有小幅因果响应，但幅度仍只恢复 human factual span 的约 6%–7%。两套结果共同支持可调制但高度压缩，而非完全无响应。

---

### 4.4 T4：替换 State 后 Network 误差是否变化？

**检验：** \(A = \mathrm{TV}(N_L(S_L), N_H) - \mathrm{TV}(N_L(S_H), N_H)\)。Sampler 下换 \(S\) 应移动 \(N\)；若 \(A \approx 0\) 且个体相关低，则 Network 生成器主导误差。

**Survey：** UCNets、GSS 1985、NSHAP r1（S0→R1；R0 = 各 survey 的 N0）。

#### I/O 示例（R1）

**Input（`r1_input.jsonl`）：** \(X\) 真值 + **LLM 状态** \(S_L\)（来自 S0）。

```json
{
  "prompt_json": {
    "age": "53", "marital_status": "divorced",
    "self_rated_health": "good",
    "personality": "somewhat outgoing"
  },
  "state_source": "phase1_task1_S_L"
}
```

**Output：** 与 N0 同 schema 的 `composition_summary` + roster（协议相同，仅 State 来源不同）。

#### 五模型：\(A\) = mean TV(R1) − mean TV(R0)

| 模型 | UCNets \(A\) | GSS 1985 \(A\) | NSHAP \(A\) |
|---|---:|---:|---:|
| llama3.1_latest | 0.005 | 0.020 | −0.000 |
| Qwen | 0.016 | −0.017 | −0.026 |
| DeepSeek-V4-Flash | 0.036 | 0.010 | −0.002 |
| qwen3_8b | 0.014 | 0.018 | 0.002 |
| qwen3_14b | **0.062** | 0.000 | 0.013 |

**结果解释：**

1. **大多数 \(A\) 接近 0。** \(A>0\) 表示把真实 State 换成 LLM State 后网络更差，\(A<0\) 表示反而略好；除 qwen3_14b 在 UCNets 的 0.062 外，大多数绝对变化不超过约 0.04，且正负方向跨 survey 不稳定。
2. **这把主要误差定位到 Network generator。** R0 已经使用真实 State，如果 State 是关键瓶颈，换成有误差的 \(S_L\) 应普遍显著恶化 Network；实际边际变化很小，而 R0 自身 TV 已经很高。因此只提高 State 预测准确率，不足以解决网络分布失真。
3. **小 \(A\) 不等于输入完全没作用。** TV 是多个 composition 字段的平均，某些字段可能向好、另一些向坏后相互抵消；同时 R0/R1 各只有一次生成，sampling noise 会降低个体对应关系。故 T4 支持的是 State 并非主要误差杠杆，而非 Network 对输入严格独立。
4. **跨 survey 的近零结果增强了定位可信度。** UCNets、GSS 1985 和 NSHAP 的网络定义不同，但多数模型都没有因 State 来源改变而产生大幅系统变化，说明默认网络生成机制具有跨数据环境的稳定影响。

**实践含义：** 后续方法不应只投入更多资源生成更精细的 synthetic State；更直接的改进对象是 Network 层的联合约束、分布校准和重复采样稳定性。

---

### 4.5 T5：信息 × 格式是否可加？（工程检查）

**检验：** N5 = N1+N3、N6 = N2+N3 的 TV 是否接近可加性预测；interaction 大 ⇒ 格式默认轨迹压制信息。

**Survey：** 仅 UCNets（400 人子样本，四臂共同 N=307/289）。

**llama 结果：** size 臂 interaction = **0.009**（近可加）；prior 臂 interaction = **0.122**（组合差于 N3 单独）。


**结果解释：**

1. **Size 信息与多步格式近似可加。** Interaction=0.009 表明在该 400 人子样本上，把 size condition 与 roster-style format 合并，并没有产生明显额外损失；两种改动的效果大体可以独立叠加。
2. **Population prior 与格式存在明显冲突。** Interaction=0.122 表示组合臂比由单独效果推算的结果更差。多步 roster 可能让模型优先维持逐人叙事一致性，从而弱化 population prior 对 composition 的锚定；也可能是更长上下文把先验信息稀释。
3. **更多信息叠加更复杂推理并不保证更好。** N3 仅改变格式却可能优于含真实先验的组合臂，说明工程组件会改变模型默认轨迹，而不是像统计模型中的独立特征那样稳定贡献信息。

**识别边界：** T5 检查的是提示组件的工程可组合性，interaction 大并不是拒绝 sampler 的必要或充分证据；共同样本只有 307/289，也不宜把一个 Llama interaction 数值外推到所有 survey。它在全文中的作用是解释为何追加信息可能无法沿复杂生成流程稳定传递。

---

### 4.6 T6：换总体后是否跟随人类差异？

**检验：** ISSP 跨国 Pearson；GSS 1985↔2004、NSHAP↔UCNets 的 Δ 是否同号；GSS 2024 污染检查。

**Survey / 臂：** ISSP `t6_country_resources`；其余用已有 N0。

#### I/O 示例（ISSP T6）

**Input：** 国家分层子样本 + 该国人类资源先验（priors）。

```json
{
  "metadata": { "country": "Austria", "country_code": "AT" },
  "prompt_json": { "age": "42", "gender": "male", "…" }
}
```

**Output：**

```json
{
  "prediction_json": {
    "resource_summary": {
      "occupational_access_count": 5,
      "kin_channel_share": 0.4,
      "friend_channel_share": 0.3
    }
  }
}
```

#### 五模型摘要

| 模型 | ISSP mean Pearson | GSS 85↔04 同号字段 |
|---|---:|---|
| llama3.1_latest | 0.067 | 3/5 |
| Qwen | 0.261 | 1/5 |
| DeepSeek-V4-Flash | **0.311** | **4/5** |
| qwen3_8b | 0.201 | 1/5 |
| qwen3_14b | 0.175 | 3/5 |

GSS 跨年：人类朋友占比 0.68→0.53；五模型 LLM 侧均在 ~0.47–0.48 窄幅波动——**跨总体模板化**在多数模型上仍成立，DeepSeek 在 ISSP 上相对最好。

**结果解释：**

1. **外部效度不仅要求单个总体内拟合，还要求总体差异方向正确。** 如果模型从国家/年份条件分布中采样，人类资源或网络构成在总体间升降时，LLM 输出也应同号并具有相近幅度。
2. **绝对相关普遍偏低。** 即使表现最好的 DeepSeek，ISSP mean Pearson 也只有 0.311；Llama 为 0.067，接近没有跨国排序信息。模型可能产生合理的国家级均值，却没有恢复国家之间真实的相对位置。
3. **GSS 跨年结果显示强烈中心收缩。** 人类 friend share 下降约 0.15，五模型却都停留在 0.47–0.48 附近。这与 T3 的中档默认模板一致：改变总体标签能产生有限调制，但输出被拉回一个跨人口共享的典型中心。
4. **模型异质性再次出现。** DeepSeek 在 ISSP 和 GSS 同号数上相对最好，与其 T3 较强调制相互印证；但相关仍远低于 1，不能据此称为跨总体校准 sampler。
5. **T6 把模板化结论从单一 survey 推向外部环境。** T1–T3 的失真可能被解释为某个数据集或任务格式特有；国家、年份和调查制度变化下仍出现响应压缩，说明默认中心具有更广泛的稳定性。

**识别边界：** 跨总体相关会受到国家样本量、测量口径和可用 covariates 影响，低相关不能唯一归因于模型忽略国家。但当其与同号数不足、幅度压缩及 T3/T4 结果一致时，证据整体指向有限外部调制，而非完整分布迁移。

---

### 4.7 Native-path Decoding：局部条件可访问性是否转化为联合网络响应？

**研究问题与共同设计。** 对同一 ego 反事实替换条件 bucket，同时测量固定 JSON prefix 下的 **native slot score** 和自由生成后的 **full-JSON target**。三个实验使用同一 HF Llama-3.1-8B-Instruct BF16、chat template、原始 T3/N1 prompt 和 temperature \(=0.7\)。

| 实验 | 数据与条件 | Native slot 测量 | Full-JSON 靶量 | 
|---|---|---|---|
| **1. Sex homophily** | GSS 1985；low/medium/high | 5 个 `gender` slots 的同 ego 性别偏好 | roster 重算 same-sex share | 
| **2. Closure** | GSS 1985；low/medium/high | 10 个 dyads 的 `especially_close` 偏好 | ties 重算 closure share | 
| **3. Size** | UCNets N1；small/medium/large | `num_alters` 对 1–30 的候选分布 | 生成的 `num_alters` | 

共得到 19,200 个 native slot decisions 和 8,400 个 full-JSON samples。Native constrained score 只表示固定上下文中的相对候选偏好，**不等同于校准生成概率**。

#### 实验 1–2：关系结构条件

| 靶量 | Human factual means（low/medium/high） | Native means | Full JSON means | Full high−low（95% CI） | Human span 恢复率 |
|---|---|---|---|---:|---:|
| sex homophily | 0.287 / 0.600 / 0.865 | 0.761 / 0.766 / 0.779 | 0.589 / 0.596 / 0.624 | **+0.0348** (0.0235, 0.0471) | **6.0%** |
| closure | 0.022 / 0.286 / 0.770 | 0.632 / 0.649 / 0.671 | 0.229 / 0.234 / 0.279 | **+0.0498** (0.0422, 0.0580) | **6.7%** |

Human factual means 是按观测 bucket 分组的群体校准参照，不是同一 ego 的反事实真值；full-JSON high−low 则来自同 ego 配对干预。

#### 实验 3：网络规模条件

| 输入条件 | Native expected bucket index（0–2） | Full `num_alters` 均值 | 区间遵循率 |
|---|---:|---:|---:|
| small | 0.472 | 8.16 | 23.5% |
| medium | 0.998 | 11.04 | 100.0% |
| large | 1.945 | 21.44 | 99.2% |

Size 在 native 和 full 两层都能区分三个条件，但 small 输出明显向 medium 收缩。因此该实验既是条件可访问性的正控，也表明即使条件直接对应单个字段，默认规模先验仍会造成非对称遵循。

#### 跨实验机制诊断

| 机制诊断 | Sex | Closure |
|---|---:|---:|
| Full medium−low（95% CI） | 0.0068 (−0.0042, 0.0180) | 0.0048 (−0.0038, 0.0132) |
| Full high−medium（95% CI） | 0.0281 (0.0163, 0.0399) | 0.0451 (0.0372, 0.0536) |
| 同 ego×condition 采样 SD | 0.1156 | 0.0818 |
| Native/full 个体 high−low \(\Delta\) 相关 | −0.039 | 0.007 |

完整 JSON parse rate 为 99.8%–100%，结构完整率为 95.9%–100%；仅保留结构完整样本后结论不变，因而主要结果不是格式失败造成的。

#### 机制解释

1. **条件可访问，但只形成弱调制。** Native 与 full JSON 均随 bucket 同方向变化，full high−low CI 不跨 0，因此模型确实读取了条件；但完整网络仅恢复约 6%–7% 的 human span，说明条件不是生成分布的主要决定因素。
2. **默认模板主导，并呈现阈值响应。** Sex 输出集中在 same-sex share≈0.60，closure 集中在 0.23–0.28；medium−low 不显著，而 high−medium 显著。模型更像在默认中心上对 “high” 做一次有限修正，而非连续复现三个条件分布。
3. **局部偏好不是完整网络的稳定中介。** Native 与 full 的个体 high−low 响应几乎零相关。固定 prefix 下的局部候选偏好不足以预测自由生成的联合结果；roster 一致性、多项 ties 决策和自回归采样路径共同决定最终靶量，不能将两层解释为同一概率信号的简单衰减。
4. **采样噪声限制个体级控制。** 同条件重复生成的 SD 大于平均条件效应，说明 bucket 提供的是可检测的群体级偏置，但单个 ego 的具体网络主要受生成随机性影响。
5. **直接字段约束比关系结构约束更易执行。** Size 可以直接落在 `num_alters` 字段上，整体遵循明显强于需要跨多个 alters/dyads 联合实现的 sex homophily 和 closure；但 small 条件仍受默认规模先验牵引，说明直接映射也不是充分条件。

综上，最符合证据的行为机制是：

> **条件语义在局部决策中可访问，但完整网络主要由强默认模板驱动；条件只能产生有限、阈值化且受采样噪声影响的联合调制。**

该结论不等于识别了模型内部神经算法：native score 依赖 teacher-forced prefix，两层不在同一概率尺度，且实验目前仅覆盖 Llama。

---

## 5. 讨论与结论

### 5.1 模型横向对比

| 模型 | 相对优势 | 相对劣势 |
|---|---|---|
| llama3.1_latest | N0 TV 中等；T4 \(A\) 最小 | T3 跟随极弱；ISSP 跨国最弱 |
| Qwen | N0 TV 较低 | T4 GSS \(A\) 为负；GSS 跨年不同号多 |
| DeepSeek-V4-Flash | **T3 闭合度、ISSP 跨国最好** | State/N0 TV 仍偏高 |
| qwen3_8b | ER 较低（分布较平） | N0 TV 偏高 |
| qwen3_14b | — | **N0 TV 最高**；UCNets T4 \(A\) 最大 |

**没有**全面 sampler 模型；**也没有**五判据全为 predictor 的模型。调制强度呈连续谱：DeepSeek 在 T3/T6 上相对最强，Llama 最弱但仍非零响应——下文 §5.2 据此收束全篇机制叙事。

### 5.2 综合结论：从集体失真到生成机制

1. **RQ 的回答。** LLM 能否在总体层面生成有效的个体状态与自我中心网络——不是逐人猜对，而是合成人群的边际、关联与条件依赖是否与人类一致。沿 State → Network 两阶段管线，五模型 × T1–T6 与 native-path decoding 给出的答案是否定的：这些模型不能被视为校准的条件采样器。它们更像落在 Predictor–Sampler 连续体上：能读懂 prompt 里的条件，输出也会随条件同向移动，但长期被一套社会常识式的默认网络模板锚住，条件只能做有限、往往还是阈值化的修正，而无法把整张网络的联合分布拉向 \(P_H\)。

2. **集体效度在两阶段均已失守。** 无条件基线表明，State 与 Network 的边际分布都系统性地偏离人类；低 κ 只说明模型不是逐人复现器，不能替总体层面的失真开脱。更关键的是，即使 Network 阶段已注入真实 State，输出仍明显不对——这意味着不能把问题主要归咎于 State 预测不准，Network 生成器本身就在引入独立的集体误差。

3. **失真不是完全不懂社会学，而是懂一些但用错方式。** T1 说明模型常能分辨哪些背景变量与哪些状态/outcome 更相关，却倾向于把关联夸大成决定论式的刻板差异；T2 说明追加条件信息后，改善幅度与这些条件在人类数据中的真实重要性并不对齐，模型更像按字段名称和任务格式做启发式更新，而非按 \(P_H(Y\mid B)\) 重采样。T3 则把问题推到最直接的可检验层面：把靶量分箱写进 prompt 后，组间均值确实会动，但移动幅度远小于人类——有的模型几乎钉在中档模板，有的模型能明显抬高 high 桶，却没有任何模型把三档分布完整复现出来。五模型之间的差异因此很重要：它拒绝所有 LLM 都是完全平坦 predictor 的二元说法，但也同样拒绝已有可用 sampler 的乐观说法。

4. **主要瓶颈在 Network 层的联合生成，而非 State 输入。** T4 的 State 替换实验给出清晰的归因：把上游 State 从真值换成 LLM 预测，Network 的集体误差几乎不变。换言之，在现有协议下，修 State 不是修复合成网络效度的主杠杆。T5–T6 进一步说明，这种模板化并不只是某个 prompt 写法的副作用：信息组件与生成格式并不能稳定叠加，换国家、换年份、换调查人群后，输出仍倾向于回到跨环境共享的典型中心，而不是随人类总体差异同幅度迁移。

5. **Decoding 实验把机制形态说得更具体。** 在同一 ego 上做反事实 bucket 替换，并同时测量 teacher-forced 下的 native slot 分数与自由生成的 full JSON，可以得到比 T3 更细的分辨率。两层都检测到条件响应，因此模型并非对显式条件完全盲视；但 full JSON 的条件效应只有人类幅度的一小部分，而且同一 ego 上 native 与 full 的条件响应几乎互不相关。这排除了两种常见但过于简单的叙事：一种认为模型根本不懂条件，另一种认为模型内部已有正确网络概率、只是 JSON 输出把它衰减了。更符合证据的是：条件语义在局部可读，但无法稳定传导为联合网络的实现——名单一致性、多个 dyad 决策、自回归采样路径共同决定最终靶量，局部偏好不是完整网络的中介变量。

6. **生成机制图景。** 上述结果叠成四层结构，而不是四个彼此独立的发现：

| 层级 | 核心含义 |
|---|---|
| 模板层 | 模型先回到一套强默认网络（典型构成、典型闭合度、典型规模），锚定无条件边际与跨总体输出 |
| 可访问层 | 显式条件能改变局部候选偏好，说明 bucket 被读取；但这不等于已持有校准的条件分布 |
| 调制层 | 完整 JSON 会随 bucket、国家、年份同向移动，但幅度被大幅压缩；high 条件的修正通常强于 low，呈阈值响应 |
| 联合实现层 | 最终网络由多字段、多关系的联合约束共同决定；协议与生成路径改变默认轨迹，native 与 full 因此弱耦合 |

T4 把改进杠杆明确指向调制层与联合实现层所在的 Network 生成器。同时，T3 与 decoding 也表明：只有 prompt 条件与 estimand 定义对齐（例如从 ties 重算 closure，而非读自报 summary），才能识别出真实的条件响应；对齐本身是必要条件，却不足以把 predictor 推成 sampler。

7. **对评估与应用的含义。** 判断 LLM 合成网络是否有集体效度，不能停留在看起来像合理网络或无条件边际大致相近；必须主动检验条件跟随、跨总体方向一致性，并从生成结果重算与研究问题一致的 estimand。改进路径同样不应只押在更好的 State prompt 或更多背景变量上——在默认模板主导的前提下，需要直接约束和校准 Network 的联合输出，并用重复采样区分群体级条件偏置与个体级生成噪声。只有把协议对齐、分布校准与联合结构约束放在一起，才有可能把连续体上已有的有限调制，推进到真正可用的 collective validity。

---

## 附录 A：产物索引

| 路径 | 内容 |
|---|---|
| `Testing/T1_association/eval/{<slug>/}eval{1,2,3}_evaluation/` | State T1/T2 |
| `Testing/T1_association/ucnets_w1/eval/{<slug>/}phase2_eval_summary.json` | UCNets T1–T5 Network |
| `Testing/T3_interval_following/<ds>/eval/{<slug>/}network_eval_summary.json` | T3 补充 |
| `Testing/T4_individual_resample/<ds>/eval/{<slug>/}sr1_summary.json` | T4 |
| `Testing/T6_cross_population/eval/{<slug>/}t6_summary.json` | T6 |
| `Testing/decoding/output/main/shard_{0..3}/` | Native slot 与 full-JSON 分片原始结果 |
| `Testing/decoding/output/main/combined_analysis.json` | 四分片合并、paired bootstrap、native/full 耦合与稳健性分析 |
| `Testing/decoding/temp/` | 已替换的旧 A/B/C forced-choice 脚本、输入与结果 |
| `Reports/EXPERIMENT_REGISTRY.md` | 登记与分箱细节 |
| `Testing/common/run_model_evaluation.sh` | 五模型批量评估 |

## 附录 B：臂名速查

| 短名 | 配置键 | 主 Testing |
|---|---|---|
| S0 / S1 / S2 | `s0_baseline_state` 等 | T1 / T2 |
| N0–N6 | `n0_baseline_network` 等 | T1–T5 |
| R1 | `r1_llm_state_network` | T4 |
| T6 | `t6_country_resources` | T6 |

（完整表见 `Testing/NAMING.md`。）
