# GSS Ego-Network 数据

本目录是 **检索便利副本**，不是另一套裁剪数据。

- 内容与 [`Survey/gss/processed/by_year/{year}/network/`](../processed/by_year/) **逐文件同步**（同 MD5）
- `ego` = 当年网络模块受访者 × **该年完整 survey 全部列** + 少量派生网络标记（不删问卷变量）
- 原始来源：累积文件 `Survey/gss/GSS_stata/gss7224_r3.dta`（1985/1987/2004/2024）；2010 为 `Survey/gss/processed/by_year/2010/GSS2010merged_R8.dta`
- `alter` / `alter_alter` 仅为 long 形检索表；对应 wide 列仍保留在 `ego` 上

完整处理流程见 [`Survey/gss/processed/README.md`](../processed/README.md)。

---

## 目录结构

```
Survey/gss/network/
├── README.md
├── 1985/
├── 1987/
├── 2004/
├── 2010/          # 另含 network_source.json
└── 2024/
```

每个年份文件夹只保留：

| 文件 | 说明 |
|------|------|
| `ego.csv` / `.parquet` | **网络模块受访者** × 该年**完整 survey 列** + 网络标记。不裁剪变量。 |
| `alter.csv` / `.parquet` | alter long 表：由 `sex1`… 等 wide 列展开；原始 wide 列仍留在 `ego` |
| `alter_alter.csv` / `.parquet` | alter 间关系边（仅 1985、2004）；含 `close_var`（如 `close12`）与 `weight` |
| `value_labels.json` | 取值码（按 `ego` / `alter` / `alter_alter` 分组） |
| `network_source.json` | **仅 2010**：数据来源与样本说明 |

已删除的冗余表：`ego_network`、`survey_network`、`network_wide`（信息已由 `ego` 覆盖）。当年全样本问卷仍在 `Survey/gss/processed/by_year/{year}/survey.csv`。

---

## 五个网络年份一览

GSS 在 1972–2024 共 35 个截面年份中，仅 **5 年**施测了可导出为 ego/alter 的讨论网模块：

| 年份 | 样本类型 | ego（网络模块） | alter | alter_alter | max_alters |
|------|----------|----------------:|------:|------------:|----------:|
| **1985** | 截面（important matters） | 1,531 | 4,483 | 6,200 | 5 |
| **1987** | 截面（政治讨论网） | 1,800 | 4,182 | — | 3 |
| **2004** | 截面（important matters） | 1,426 | 2,845 | 3,384 | 5 |
| **2010** | **Panel only**（2006 cohort W3） | 1,272 | 2,984 | — | 5 |
| **2024** | 截面（部分模块） | 1,062 | 3,895 | — | 5 |

- `ego`：`numgiven` 非空（被问到网络题，含报 0 人者）
- `alter`：仅 `numgiven > 0` 的 ego
- `alter_alter`：仅 `numgiven > 1` 的 ego 的 alter 对；来自 `close{i}{j}`

---

## 表关系

```
ego                                  alter
┌─────────────────────┐             ┌──────────────────────────┐
│ ego_id  (PK)        │─── 1:N ───►│ ego_id + alter_id  (PK)  │
│ 完整 survey + 标记   │             │ sex, race, spouse, kin…  │
└─────────────────────┘             └──────────────────────────┘
         │
         │              alter_alter
         └──────────── 1:N ────────► ego_id + src_id + tgt_id
                                     close_var, weight
```

---

## 字段说明

以下分别介绍核心表字段。各字段英文原文及完整码表见当年 `value_labels.json`（按 `ego` / `alter` / `alter_alter` 分组）。

### 数据类型约定

| 类型 | 说明 |
|------|------|
| `String` | 文本，如 `ego_id` |
| `Int64` | 整数；问卷分类变量以**数值码**存储（非文字标签），**可为空**（`.parquet` 为 `null`，`.csv` 为空单元格） |
| `Float64` | 浮点数，如权重 `wtssall` |
| `Boolean` | 布尔值（`true`/`false`），用于派生标记列 |

空值通常表示 **IAP（未问到）**、**拒答**或**不知道**，具体含义见 GSS Codebook。读取 `.csv` 时注意将空字符串解析为缺失值；`.parquet` 已保留 nullable 类型。

### ego

`ego` 记录**进入网络模块的受访者**的完整问卷：与当年 `survey` 列一致（对该子集），并附加网络标记。全样本问卷见 `by_year/{year}/survey.csv`。

下列为基础标识与网络标记；其余列为该年全部 survey 变量（如 `trust`、`polviews`、`sex1`…），见当年 `value_labels.json`。

#### 标识

| 字段 | 类型 | 含义 |
|------|------|------|
| `ego_id` | String | 主键，格式 `{year}_{id}`（如 `1985_42`），避免跨年 ID 冲突 |
| `id` | Int64 | GSS 原始受访者编号 |
| `year` | Int64 | 调查年份 |

#### 人口统计

| 字段 | 类型 | 含义 | 取值说明 |
|------|------|------|----------|
| `age` | Int64 | 受访者年龄 | 年龄（岁）；部分年份 89 表示 89+ |
| `sex` | Int64 | 受访者性别 | 1=male, 2=female |
| `race` | Int64 | 受访者种族 | 编码随年代变化（与 alter 的 `race` 码表不同，详见码表） |
| `educ` | Int64 | 最高受教育年数 | 0–20（0=未上学, 12=高三, 16=本科四年, …） |
| `degree` | Int64 | 最高学位类别 | 0=低于高中, 1=高中, 2=副学士/专科, 3=学士, 4=研究生 |
| `marital` | Int64 | 婚姻状况 | 1=married, 2=widowed, 3=divorced, 4=separated, 5=never married |
| `relig` | Int64 | 宗教归属 | 1=protestant, 2=catholic, 3=jewish, 4=none, 5=other |
| `partyid` | Int64 | 政党认同 | 0=strong dem … 6=strong rep, 7=other |
| `region` | Int64 | 访谈区域 | 美国四大区域 |
| `happy` | Int64 | 总体幸福感 | 1=very happy, 2=pretty happy, 3=not too happy |
| `health` | Int64 | 自评健康状况 | 1=excellent, 2=good, 3=fair, 4=poor |

#### 网络模块

| 字段 | 类型 | 含义 | 取值说明 |
|------|------|------|----------|
| `numgiven` | Int64 | 受访者提名的人数 | 见下表；未进入模块时为 null |
| `in_network_module` | Boolean | （派生）是否进入网络模块 | `numgiven` 非空 |
| `has_alters` | Boolean | （派生）是否有被提名者 | `numgiven > 0` |
| `numgiven_capped` | Int64 | （派生）截断后提名数 | `min(numgiven, max_alters)` |
| `numgiven_is_topcoded` | Boolean | （派生）是否顶码 | `numgiven >= 6` |

`numgiven` 取值：

| 值 | 含义 |
|----|------|
| 空 | 未进入网络模块（IAP） |
| `0` | 进入模块，但无人可讨论 |
| `1`–`5` | 具体人数 |
| `6` | "6 or more"；`alter` 按 5 人展开，`numgiven_is_topcoded=True` |

#### 抽样设计

**抽样设计**（sampling design）字段描述的是 GSS **如何抽取样本、如何代表总体**，不是受访者填写的问卷内容。GSS 采用分层 + 群集（cluster）的复杂抽样；做全国推断或假设检验时，通常需要权重和设计变量，否则标准误可能偏小。

| 字段 | 类型 | 含义 | 用途 |
|------|------|------|------|
| `wtssall` | Float64 | 综合权重 | 加权计算均值、比例等点估计 (该样本可以代表几个人, 1985 全样本 wtssall 范围约 0.52 ~ 3.63，均值约 1.0) |
| `vpsu` | Int64* | Primary Sampling Unit，初级抽样单元 | 与 `vstrat` 配合，做聚类稳健方差估计 |
| `vstrat` | Int64* | Stratum，抽样层 | 与 `vpsu` 配合，做设计校正的标准误 |
| `ballot` | Int64 | 问卷 ballot 版本 | 标识不同问卷版本（部分题目有差异） |
| `form` | Int64 | 分卷形式 | 标识分卷/表单版本（当年有则导出） |
| `samptype` | Int64 | **仅 2010**：样本类型 | 本数据均为 2006 panel cohort（值 2006） |
| `version` | Int64 | **仅 2010**：问卷版本 | Panel merged 文件中的版本号 |

\* 2010 年 `vpsu` / `vstrat` 在 `.parquet` 中为 `String`（原始 merged 文件缺失较多）；其余年份为 `Int64`。

**简要用法**：描述性推断用 `wtssall`；回归或检验时在统计软件中指定 `weights=wtssall`、`cluster=vpsu`、`strata=vstrat`（具体语法因软件而异）。2010 panel 网络分析优先用 Merged 中的 panel 权重（如 `WTPAN123`，现已保留在 `ego` 全列中，若该列存在）。

#### 各年 `ego` 抽样设计字段

| 字段 | 1985 | 1987 | 2004 | 2010 | 2024 |
|------|:----:|:----:|:----:|:----:|:----:|
| `wtssall` | ✓ | ✓ | ✓ | | |
| `ballot` | | | ✓ | ✓ | ✓ |
| `samptype` | | | | ✓ | |

---

### alter

`alter` 为 **long 表**：每行对应一位被提名者（alter），由问卷中的 `sex1`…`friend5` 等 wide 列按 `alter_id` 展开。仅包含 `numgiven > 0` 的 ego。

#### 主键

| 字段 | 类型 | 含义 |
|------|------|------|
| `ego_id` | String | 所属受访者，与 `ego` 表关联 |
| `alter_id` | Int64 | 该 alter 在提名列表中的序号（1, 2, …, `max_alters`），仅在同一 ego 内有效 |

#### 人口统计

受访者被问到：**第 n 位被提名者的……**（`#n` 即 `alter_id`）。以下字段按年施测，并非每年都有。

| 字段 | 类型 | 含义 | 取值说明 |
|------|------|------|----------|
| `sex` | Int64 | 该 alter 的性别 | 1=male, 2=female |
| `age` | Int64 | 该 alter 的年龄 | 年龄（岁） |
| `race` | Int64 | 该 alter 的种族 | **alter 专用编码**（与 ego 的 `race` 不同）：1=asian, 2=black, 3=hispanic, 4=white, 5=other |
| `educ` | Int64 | 该 alter 的教育程度 | 0=1–6 年, 1=7–9 年, 2=10–12 年, 3=高中毕业, 4=大专在读, 5=副学士, 6=学士, 7=研究生 |
| `relig` | Int64 | 该 alter 的宗教归属 | 1=protestant, 2=catholic, 3=jewish, 4=none, 5=other |

#### 关系类型

受访者从一组**并列**关系类别中勾选该 alter 所属类型。各字段为二元标记：`1`=mentioned（属于该类），`2`=not mentioned（不属于）。**同一 alter 可同时属于多种关系**（如既是 `friend` 又是 `cowork`），不是互斥单选。以下字段类型均为 `Int64`。

| 字段 | 含义 |
|------|------|
| `spouse` | 该 alter 是受访者的配偶 |
| `parent` | 该 alter 是受访者的父母 |
| `sibling` | 该 alter 是受访者的兄弟姐妹 |
| `child` | 该 alter 是受访者的子女 |
| `othfam` | 该 alter 是受访者的其他家庭成员 |
| `cowork` | 该 alter 是受访者的工作同事 |
| `memgrp` | 该 alter 是受访者所属团体/组织的成员 |
| `neighbr` | 该 alter 是受访者的邻居 |
| `friend` | 该 alter 是受访者的朋友 |
| `advisor` | 该 alter 是受访者的顾问/指导者 |
| `other` | 该 alter 是其他类型的熟人 |

#### 互动

| 字段 | 类型 | 含义 | 取值说明 |
|------|------|------|----------|
| `talkto` | Int64 | 受访者与该 alter 交谈的频率 | 1=几乎每天, 2=至少每周一次, 3=至少每月一次, 4=少于每月一次 |
| `known` | Int64 | 受访者认识该 alter 多久 | 1=少于 3 年, 2=3–6 年, 3=6 年以上 |
| `rclose` | Int64 | 受访者是否感到与该 alter 特别亲密 | 1=especially close, 2=not esp. close |

#### 派生

| 字段 | 类型 | 含义 | 规则 |
|------|------|------|------|
| `kin` | Int64 | 该 alter 是否为亲属 | `parent` / `sibling` / `child` / `othfam` 任一为 1 → `kin=1`；否则 `kin=2` |

#### 各年 `alter` 可用列

| 字段 | 1985 | 1987 | 2004 | 2010 | 2024 |
|------|:----:|:----:|:----:|:----:|:----:|
| sex, age, educ, relig | ✓ | | ✓ | sex | |
| race | ✓ | | ✓ | ✓ | ✓ |
| spouse … other（关系类型） | ✓ | ✓ | ✓ | ✓ | ✓ |
| sibling | ✓ | ✓ | ✓ | ✓ | |
| talkto | ✓ | | ✓ | | ✓ |
| known, rclose | ✓ | rclose | | | |
| kin（派生） | ✓ | ✓ | ✓ | ✓ | ✓ |

---

### alter_alter

`alter_alter` 记录**同一 ego 网络内、两位 alter 之间的熟悉程度**，由问卷变量 `close12`…`close45` 展开为边列表。**仅 1985、2004** 有此文件（原始数据含完整 `close**` 变量；1987 / 2010 / 2024 无）。

#### 字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `ego_id` | String | 所属受访者，与 `alter` 表中的 `ego_id` 一致 |
| `src_id` | Int64 | 边的起点 alter 序号，对应 `alter.alter_id` |
| `tgt_id` | Int64 | 边的终点 alter 序号，对应 `alter.alter_id` |
| `close_var` | String | 原始 survey 变量名（如 `close12`） |
| `weight` | Int64 | 该 `close_var` 的原始取值 |

`src_id` 恒小于 `tgt_id`（对应 `close12`、`close13`…`close45` 共 10 对组合）。仅当 ego 的 `numgiven > 1` 且双方均在提名范围内时才会产生边。原始 `close**` 宽表列仍保留在 `ego` 上。

#### `weight` 取值

受访者被问到：**第 i 位与第 j 位被提名者之间是什么关系？**

| 值 | 含义 |
|----|------|
| `1` | especially close（特别亲密） |
| `2` | know each other（互相认识） |
| `3` | total strangers（完全陌生） |

导出时 **`weight` 为 0 或缺失的行不写入**（表示未作答或跳题）。

---

**注意**

- 跨年比较 alter 属性前，先查「各年 `alter` 可用列」或当年 `alter.csv` 列名
- 关系类型列（`spouse`, `friend` 等）是并列二元标记，非互斥分类
- 1985/2004 做加权推断时使用 `wtssall`；2010 优先用 panel 权重（如 `WTPAN123`，见 `ego` 全列）

---

*正式发表请对照 [GSS Codebook](https://gss.norc.org/) 核验变量定义与权重。*
