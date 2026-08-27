# Network 臂设计（N0–N6；主 Testing：T1 / T2 / T3 / T5）

臂名对照见 [`NAMING.md`](NAMING.md)。下文若仍写 B1–B7，即 N0–N6。

**状态：** 生成完成；评估与报告见各 Testing 的 `eval/`（N0b 暂缓）  
**对应 RP：** Sub-Experiment B — Network Generation（[`LLM_Estimand_Validity_RP.md`](../Reports/LLM_Estimand_Validity_RP.md) §4.4）  
**主数据：** UCNets Wave 1  
**范围：** 仅 UCNets；GSS 作为后续独立稳健性文档

---

## 0. 研究问题定位

Phase 1 已证明：**在生成任何网络之前，State 失真已经很大且系统性**。UCNets Task 1 基线（`llama3.1:latest`）：

| 指标 | 数值 | 读法 |
|---|---:|---|
| Mean TV | 0.294 | 边际分布偏离人类 |
| Mean 熵比 | 0.700 | 输出比人类更集中 |
| Eval3 平均 \|ΔR²\| | 0.060 | 人口学过度可预测程度（UCNets 相对 GSS 较轻） |
| Mean κ | ≈ 0 | 个体层面预测不可用 |

失败模式：**方差/熵压缩 + 刻板关联**。κ≈0 说明不能把 Phase 1/2 解读为个体标签预测。

Phase 2 做**隔离设计**：

> 假设 State 阶段完美（输入 = 真实人类 State \(S_{i,H}^1\)），单独问：LLM 能否为这个人「长出」结构上合理的自我中心网络 \(N_{i,L}^1\)？

形式化（RP Sub-Experiment B）：

```text
S_{i,H}^1  →  N_{i,L}^1
与 N_{i,H}^1 在总体层面对比
```

与 State 臂拼起来，是为后续 T4（RQ3/RQ4）铺垫：同一批 `PRIM_KEY` 上可做「State 真、Network 假」的反事实，分解下游 \(Y\) 失真有多少来自 Network 阶段。

核心张力不变：**局部/组件合理 ≠ 对集体量（规模、构成、同质性、角色分化）有效**。本文额外追问：LLM 隐含的**网络形成机制**是否符合已知社会网络规律，以及哪些任务设计能修复失真。

---

## 1. 数据与样本

| 项 | 路径 / 说明 |
|---|---|
| Alter 层 | `UCNets .../processed/by_wave/Wave1/alters.csv`（每行一个 alter） |
| State 层 | `Survey/ucnets/cleaned/ucnets_w1_state_clean.csv` |
| 关联键 | `PRIM_KEY` |
| 预期 N | alters 中有 alter 的 ego（约 1,156）∩ Phase1 State（约 1,159）；清洗脚本须做交集校验并报告差集 |
| 规模 | 人均约 **17.57** 个 alter，槽位上限 30 → 不能像 GSS 那样全量逐人生成/评估 |

---

## 2. 变量与双层输出

### 2.1 为何双层

全网约 20 人时，逐 alter 生成不现实。拆成：

| 层 | 内容 | 目的 |
|---|---|---|
| `core_alters` | Top-5 重要 alters，**逐人**生成细字段 | 微观 tie 结构可评估 |
| `composition_summary` | 对**全部** alters 的汇总统计 | 集体量（规模/构成）可评估 |

**Top-5 排序（须固定、可复现）：**  
主序：`contact_frequency`（越频繁越靠前）→ `C2B_NCLOSE`（尤其亲近优先）→ tie-break：`B4A`（confidant 提名顺序更靠前优先）。

### 2.2 `core_alters` 字段（每个 Top-5 alter）

| 字段 | 含义 | 允许取值（草案） | 人类真值来源 |
|---|---|---|---|
| `rank` | 排序名次 1–5 | 1…5 | 排序规则计算 |
| `relationship` | 与 ego 的主关系类型（由多选 flag 归约；若多选取优先序：配偶 > 子女 > 父母 > 手足 > 其他亲属 > 朋友 > 同事 > 邻居 > 其他） | 见下表 | `C1A_*` |
| `support_roles` | 支持功能（**多选数组**） | `companion`, `confidant`, `advisor`, `practical_help`, `injury_help`, `r_helps_them`, `difficult`（实现时可收敛为前 4–5 个主角色） | `B2A`/`B4A`/`B5A`/`B6B`/`B7C`/`B8A`/`B9A` 是否被提名 |
| `contact_frequency` | 综合联系频率 | `daily`, `weekly`, `monthly`, `several_times_year`, `yearly_or_less`, `never` | `D1F`/`D1G`/`D1H` 取最频繁渠道（子样本；缺失则用缺失码或降级规则） |
| `gender` | alter 性别 | `male`, `female` | `N_GENDER` |
| `same_race_as_ego` | 是否与 ego 同种族/族裔 | `yes`, `no` | `C2G_NSAMERACE` |

**`relationship` ← `C1A_*` 对照：**

| 码本字段 | 含义 | 归约标签（示例） |
|---|---|---|
| `C1A_1` | 配偶/伴侣 | `spouse` |
| `C1A_2` | 恋爱对象 | `romantic_partner` |
| `C1A_3` / `C1A_23` | 父母 / 继父母 | `parent` |
| `C1A_4` / `C1A_24` | 子女 / 继子女 | `child` |
| `C1A_5` / `C1A_25`/`26` | 兄弟姐妹 / 继/同父异母等 | `sibling` |
| `C1A_6` | 其他亲属 | `other_relative` |
| `C1A_10` | 室友 | `housemate` |
| `C1A_11` | 邻居 | `neighbor` |
| `C1A_12` | 工作认识 | `coworker` |
| `C1A_13` | 学校认识 | `school` |
| `C1A_14` | 教会认识 | `church` |
| `C1A_20` | 朋友 | `friend` |
| `C1A_21` | 熟人 | `acquaintance` |
| `C1A_22` | 其他方式认识 | `other` |

**`support_roles` ← roster 提名：**

| 原始 | 题意 | 映射角色 |
|---|---|---|
| `B2A` | 社交陪伴 | `companion` |
| `B4A` | confidant（倾诉心事） | `confidant` |
| `B5A` | 重大决定建议 | `advisor` |
| `B6B` | 家务/实际帮助 | `practical_help` |
| `B7C` | 受伤时帮助 | `injury_help` |
| `B8A` | ego 帮助对方 | `r_helps_them` |
| `B9A` | 关系困难的人 | `difficult` |

**`contact_frequency` ← `D1F`（见面）/`D1G`（电话）/`D1H`（电邮短信）原始码：**  
1=至少每天，2=至少每周，3=至少每月，4=一年几次，5=一年一次或更少，6=从不 → 清洗时取三渠道中**最高频**并映射到英文枚举。

### 2.3 `composition_summary` 字段（全网汇总）

| 字段 | 含义 | 计算 |
|---|---|---|
| `num_alters` | 网络规模 | 该 `PRIM_KEY` 下 alter 行数（可另报 size bucket） |
| `size_bucket` | 规模分档（B2 条件化用） | `small` 1–5 / `medium` 6–15 / `large` 16–30 |
| `kin_share` | 亲属占比 | 关系为配偶/父母/子女/手足/其他亲属等的比例 |
| `friend_share` | 朋友占比 | `friend` 比例 |
| `coworker_share` | 同事占比 | `coworker` 比例 |
| `confidant_count` | confidant 人数 | `B4A` 非空人数 |
| `companion_count` | 陪伴角色人数 | `B2A` 非空人数 |
| `pct_weekly_plus_contact` | 每周及以上联系占比 | `contact_frequency` ∈ {daily, weekly} 的比例（有频率数据的子样本上算；全样本需注明覆盖） |

### 2.4 输入：真实 State（Phase 1 目标字段反转为条件）

Phase 2 **不生成**这些字段，全部视为观测 \(S_H\)：

| 大类 | 字段（中文） |
|---|---|
| A–C 背景 | `age`/`age_group`（年龄）, `gender`（性别）, `race_ethnicity`, `education`, `marital_status`, `romantic_relationship`, `household_other_people_count`, `children_status`, `work_status` |
| D 健康福祉 | `self_rated_health`, `days_very_happy_past_week`, `days_isolated_past_week`, `days_lonely_past_week` |
| F–G | `party_identity`, `voting_frequency`, `religion_current`, `religion_raised` |
| H 支持 | `relatives_help_serious_problem`, `friends_help_serious_problem`, `wants_more_people_for_*` |
| I 人格 | `personality.*`（10 项 Likert） |
| J 接触 | `ever_goes_online`, `family_contact_methods_weekly`, `friend_contact_methods_weekly` |

**防泄漏：** prompt / `prompt_json` 不得包含任何网络真值（`num_alters`、`kin_share`、alter 列表等），除非该 Task 明确条件化（B2 的 size bucket、B3 的人群先验）。

### 2.5 统一输出 schema（B1–B4 最终对齐）

```json
{
  "composition_summary": {
    "num_alters": 12,
    "size_bucket": "medium",
    "kin_share": 0.42,
    "friend_share": 0.33,
    "coworker_share": 0.08,
    "confidant_count": 3,
    "companion_count": 4,
    "pct_weekly_plus_contact": 0.55
  },
  "core_alters": [
    {
      "rank": 1,
      "relationship": "spouse",
      "support_roles": ["confidant", "companion", "practical_help"],
      "contact_frequency": "daily",
      "gender": "male",
      "same_race_as_ego": "yes"
    }
  ]
}
```

B4 四步中间结果由代码拼接成上述结构，评估脚本不分支。

---

## 3. 任务矩阵

每一档对应一个**可证伪的修复假设**（并挂上 Phase 1 证据）。

| Task | 条件 | Phase 1 依据 → Phase 2 要验证什么 |
|---|---|---|
| **B1** 基线 | 仅真实 State；`format_only`；**一次**生成完整 `core_alters` + `composition_summary` | Task1 Mean TV 0.294、熵比 0.700：默认输出会塌缩 → 网络上是否表现为 `num_alters` 众数化、`kin_share`/`friend_share` 趋同 |
| **B1b** | B1 + 反类型学（保持异质性，勿默认「配偶+2 朋友」） | Phase1：反类型学跨任务/跨库不稳定 → 网络阶段是否仍无效 |
| **B2** | B1 + **观测规模分档** `size_bucket`（不给精确人数） | 类比 Task2 条件化健康：Eval3 大改善、边际未必修好 → 「难猜结构性锚点」能否修构成/同质性，即使边际仍塌 |
| **B3** | B1 + **人群网络先验**（如平均规模、亲属占比、confidant 中位数） | Phase1 **从未测过** → 分布锚定能否缓解方差压缩（方法论新增贡献） |
| **B4** | 信息量同 B1，但改为 **roster 同构的多步生成**（见 §4） | 唯一「改任务分解、不改信息量」的修复 → 部分失真是否来自「一次性倒出整网」这种不自然格式 |

**Task3 教训（评估读法）：** UCNets Task3 收益高度局部（H 改善、D 变差、J 无效）。因此 B2/B3 **不能**只报全字段均值，必须按 `composition_summary` 字段与 `core_alters` 子结构拆开报告。

---

## 4. B1 vs B4：流程与走查例子

### 4.1 对比一览

| | B1 | B4 |
|---|---|---|
| LLM 调用 | 1 次 / 人 | 4 次 / 人 |
| 决策粒度 | 规模+构成+角色+频率同时决定 | 每步一类判断；后步可见前步人物 |
| 角色多重性 | 一次「编」出角色组合，易套模板 | 多轮追问中涌现（同一人被反复提名） |
| 新增 alter | 一步给齐 | Step2 可为具体功能「想起」新人 |
| 对应假设 | 基线塌缩 | 若同质性/角色分化/Blau 改善 → **格式因素**可分离归因 |
| 成本 | 低 | ≈4×（成本本身是该修复手段的一部分） |

一句话：**B1 问「你的社交圈长什么样」；B4 问「最近跟谁聊过心事？谁帮你跑腿？谁信得过？」**——把开放整体想象拆成与真实问卷同构的检索线索。

### 4.2 例子人物（B1 / B4 输入相同）

```json
{
  "age_group": "50-70",
  "gender": "female",
  "marital_status": "married",
  "education": "bachelors",
  "self_rated_health": "fair",
  "days_lonely_past_week": 2,
  "relatives_help_serious_problem": "definitely have",
  "wants_more_people_for_personal_concerns": "wish i knew more"
}
```

### 4.3 B1：一次性整体生成

**一次调用**，prompt 给出完整 schema：

```text
[背景 State 如上]
Generate this person's ego network as JSON with composition_summary and
core_alters (exactly 5 ranks). Use only allowed values.
```

**典型塌缩式输出（示意）：**

```json
{
  "composition_summary": {
    "num_alters": 8,
    "kin_share": 0.5,
    "friend_share": 0.375,
    "coworker_share": 0.125,
    "confidant_count": 2,
    "companion_count": 3,
    "pct_weekly_plus_contact": 0.6
  },
  "core_alters": [
    {"rank": 1, "relationship": "spouse", "support_roles": ["confidant", "companion", "practical_help"], "contact_frequency": "daily", "gender": "male", "same_race_as_ego": "yes"},
    {"rank": 2, "relationship": "friend", "support_roles": ["confidant", "companion"], "contact_frequency": "weekly", "gender": "female", "same_race_as_ego": "yes"},
    {"rank": 3, "relationship": "friend", "support_roles": ["companion"], "contact_frequency": "weekly", "gender": "female", "same_race_as_ego": "yes"},
    {"rank": 4, "relationship": "sibling", "support_roles": ["confidant"], "contact_frequency": "monthly", "gender": "female", "same_race_as_ego": "yes"},
    {"rank": 5, "relationship": "coworker", "support_roles": ["practical_help"], "contact_frequency": "weekly", "gender": "male", "same_race_as_ego": "yes"}
  ]
}
```

风险：换成「30 岁未婚、健康很好、不孤独」的人，仍可能得到高度相似的「配偶+朋友模板」——这是 Phase 1 众数塌缩在网络层的翻版（State 与 \(N\) 弱条件依赖）。

### 4.4 B4：四步有依赖生成（对齐 UCNets roster）

**Step 1 — Roster A（重要事务讨论对象）**

```text
[State]
List up to 5 people this person would discuss important personal matters
with. For each: anonymized label, relationship, gender only.
Do NOT give contact frequency or support roles yet.
```

示例输出：

```json
[
  {"label": "P1", "relationship": "spouse", "gender": "male"},
  {"label": "P2", "relationship": "friend", "gender": "female"},
  {"label": "P3", "relationship": "sibling", "gender": "female"}
]
```

**Step 2 — Roster B（分角色提名；可复用可新增）**

```text
[State] + P1–P3
For companionship / practical_help / confidant / advice:
which of P1–P3 provide it, OR add a new person if needed.
```

示例输出：

```json
{
  "companionship": ["P1", "P2"],
  "practical_help": ["P1", "P4"],
  "confidant": ["P1"],
  "advice": ["P1", "P3"],
  "new_people": [{"label": "P4", "relationship": "neighbor", "gender": "female"}]
}
```

P1 的多重角色是**多轮追问涌现**，不是一次编造；且允许 Step1 未出现的 P4。

**Step 3 — 人口学补全**

```text
Full unique list P1–P4. For each: same_race_as_ego yes/no.
```

**Step 4 — 联系频率**

```text
For each of P1–P4: contact_frequency in {daily, weekly, monthly, ...}.
```

**代码拼接** → 与 B1 相同的 `composition_summary` + `core_alters`（取排序后 Top-5；全网 summary 若 B4 只生成了核心名单，则 `num_alters`/`shares` 需约定：仅基于生成名单估算，或 B4 另要一步 summary——**实现前拍板**；推荐：B4 最终另调一次轻量 summary 头，或只用 core 可算的指标做 B1–B4 公平对比）。

---

## 5. 评估设计（四层）

读表原则与 Phase 1 一致：全样本 Pass 率常近 0；主读 **TV、熵比、\|Δ关联\|、\|ΔR²\|、\(D_h\)、机制指标**；κ 作个体一致性参考（预期仍≈0）。

### 5.0 Layer 0：Schema 与内部逻辑（不比人类）

| 检查 | 说明 |
|---|---|
| Schema | 键齐全；取值 ∈ allowed |
| `spouse` ≤ 1 | core 中配偶至多一人 |
| 角色–关系相容 | 如 `coworker` 极少同时是唯一 `confidant`（软警告或硬规则，实现时定） |
| 规模一致 | 若同时输出 `num_alters` 与名单，逻辑不矛盾 |
| 频率梯度软检 | 配偶/子女的 `daily`/`weekly` 占比应高于邻居（Layer 0 可做粗检；精细比较在 Layer 4） |

### 5.1 Layer 1 = Eval 1：边际

| 对象 | 方法 | 含义 |
|---|---|---|
| `kin_share`, `friend_share`, …（连续/比例） | TV、JS、熵比（可分箱后当离散） | 构成是否塌缩到众数 |
| `relationship`, `contact_frequency`（分类） | χ²/TV；Cohen's κ（个体） | 类型边际与个体命中 |
| `num_alters` | TV + **过散布**：负二项 \(\mathrm{Var}/\mathrm{Mean}\)（人类 vs LLM） | 直接量化规模上的方差压缩 |

**熵比** \(H(N_L)/H(N_H)\)：≪1 表示 LLM 更集中（mode collapse）。

### 5.2 Layer 2 = Eval 2：关联与同质性

| 检验 | 方法 | 类比 Phase 1 |
|---|---|---|
| `kin_share` ~ `age_group` / `marital_status` | η² 或相关；\|Δ\| ≤ 阈值通过 | 人口学–态度过度关联 |
| `num_alters` ~ `education` / `self_rated_health` | 同上 | Task2 式「锚点是否校准结构」 |
| **同质性 \(D_h\)** | ego–alter 同性别 / 同年龄段 / 同种族比例，人类 vs LLM 的差 | Chang；Homophily |
| **角色共现** | 如 P(confidant \| kin) vs P(confidant \| friend)；\|ΔV\| | party→ideology 过度绑定 |

### 5.3 Layer 3 = Eval 3：预测强度

| 检验 | 方法 |
|---|---|
| State \(X\) → `num_alters`, `kin_share`, … | OLS / 伪 \(R^2\)；看 \|ΔR²\|（LLM 是否过度由人口学决定网络） |
| **Blau 指数 / Shannon 熵**（关系类型构成） | 单一标量总结异质性；人类 vs LLM |

Blau（示例）：\(1-\sum p_k^2\)，\(p_k\) 为关系类型份额；越低越单一（塌缩）。

### 5.4 Layer 4：生成机制一致性（回应「形成能力」）

不问「分布像不像」，问「生成机制像不像社会网络」。

| 检验 | 理论 | 做法 |
|---|---|---|
| **纽带强度梯度** | Granovetter (1973) | Kendall's τ：关系类型 → 联系频率 / 角色数 的排序，人类 vs LLM（期望大致 spouse > child > friend > coworker > neighbor） |
| **角色多重性/分化** | Wellman & Wortley (1990) | 每 alter 平均 `support_roles` 数；过高 = 「全能型」mode collapse |
| **亲属可得性 vs 朋友选择性** | Fischer (1982) | `kin_share` 对 `marital_status`（如丧偶）的敏感模式是否与人类同向 |
| **B1 vs B4 分解** | Name generator（McCallister & Fischer 1978） | 上三项 + Blau：B4−B1 的差 = **仅任务格式**贡献（信息量相同） |

### 5.5 非 LLM 统计基线

脚本设想：`statistical_baseline_network.py`

- `num_alters`：负二项拟合后采样  
- composition / roles / contact：给定 ego 协变量的**经验条件频次表**采样  

回答：LLM 是否显著好于「不懂社会关系的采样器」？若 Layer 4 上 LLM ≤ baseline，则问题可能在「State→一次想象整网」范式本身，而非 prompt 措辞。

---

## 6. 理论基础（压缩）

| 来源 | 落点 |
|---|---|
| OpinionQA / SSDataBench；\(D_{prior}\), \(D_{var}\) | Eval1；B3 分布锚定 |
| Chang \(D_h\)；McPherson et al. Homophily (2001) | Eval2 同质性 |
| Granovetter；Wellman & Wortley；Fischer | Layer 4 |
| Argyle et al. algorithmic fidelity | B2/B3 |
| PIMMUR Unawareness | B1b 勿让模型轻易猜到实验假设 |
| RP Stage-restoration | \(S_H\)+\(N_L\) 接口预留 |

Dynamics 用的 Friedkin–Johnsen 等**不**纳入本阶段。

---

## 7. 与 Phase 1 / 未来阶段的接口

```text
Phase 1   X → S_L     vs S_H
Phase 2   S_H → N_L   vs N_H     ← 本文件（隔离 Network）
未来      S_H + N_L → Y_L       （Network 误差向下游传播）
```

同 `PRIM_KEY` 可拼 State 与 Network，支撑 T4。

---

## 8. One sentence

Phase 2（UCNets）在 **真实 State** 条件下测 Network 生成：用 **Top-5 + 全网汇总** 解决大规模 ego-net 的工程约束；用 **B1–B4** 分别检验基线塌缩、反类型学、规模锚、分布锚、**任务分解**；评估在 Phase1 的 Eval1–3 之上增加 **\(D_h\)、过散布、Blau、Layer4 机制一致性** 与 **非 LLM 基线**，直接回答「LLM 能否充当网络形成模拟器、哪种修复对哪类失真有效」。
