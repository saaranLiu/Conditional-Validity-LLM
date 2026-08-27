# UCNets 数据说明

UC Berkeley Social Networks Study（UCNets）本地副本，面向 Python（Polars）社会网络分析。

数据来源：[ICPSR 36975](https://www.icpsr.umich.edu/web/NACDA/studies/36975)（San Francisco Bay Area, 2015–2018，public-use 版本）。

---

## 快速开始

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("Survey/ucnets/scripts")))
from load_ucnets import list_waves, load_wave, load_value_labels, load_variable_labels

print(list_waves())  # ['Wave1', 'Wave2', 'Wave3', 'combined']

tables = load_wave("Wave1")
ego = tables["ego"]        # Ego 层（1,159 × 555）
alters = tables["alters"]  # Alter 长表（20,313 × 71）

# 合并 Ego + Alters（链接键 PRIM_KEY）
df = ego.join(alters, on="PRIM_KEY", how="inner")

# 解码分类变量
labels = load_value_labels("Wave1", "alters")
# labels["A2B"] -> {'0': 'no', '1': 'yes', ...}

# 查变量英文题项
var_labels = load_variable_labels("Wave1", "ego")
# var_labels["GENDER"] -> "gender"
# var_labels["A2A"] -> "marital status"
```

```bash
# 原始 .rda 更新后，重新导出
Rscript Survey/ucnets/scripts/export_rda.R
python3 Survey/ucnets/scripts/csv_to_parquet.py

# 仅补导出/更新标签 JSON（不重写 CSV）
Rscript Survey/ucnets/scripts/export_rda.R --labels-only
```

---

## 目录结构

```
Survey/ucnets/
├── README.md
├── ICPSR_36975/               # ICPSR 原始下载（codebook、.rda 等）
│   ├── DS0001/ … DS0008/
│   └── 36975-Codebook-PI.pdf
├── processed/                 # Python 友好格式（由 scripts 生成）
│   ├── manifest.csv           # 导出索引（行数、列数、源文件路径）
│   └── by_wave/
│       ├── Wave1/
│       ├── Wave2/
│       ├── Wave3/
│       └── combined/
└── scripts/
    ├── export_rda.R           # .rda → CSV + 标签 JSON
    ├── csv_to_parquet.py      # CSV → Parquet
    └── load_ucnets.py         # Python 读取工具
```

---

## 原始数据（ICPSR 36975）

| DS | 文件 | 行 × 列 | 说明 |
|----|------|---------|------|
| DS0001 | `alters` Wave 1 | 20,313 × 71 | Alter 长表 |
| DS0002 | `alters` Wave 2 | 20,313 × 73 | Alter 长表 |
| DS0003 | `alters` Wave 3 | 20,313 × 82 | Alter 长表 |
| DS0004 | `alters` combined | 20,313 × 222 | 三波合并 Alter |
| DS0005 | `ego` Wave 1 | 1,159 × 555 | Ego 受访者 |
| DS0006 | `ego` Wave 2 | 1,159 × 425 | Ego 受访者 |
| DS0007 | `ego` Wave 3 | 1,159 × 439 | Ego 受访者 |
| DS0008 | `ego` combined | 1,159 × 1,415 | 三波合并 Ego |

各 DS 目录下有 `36975-000N-Codebook-ICPSR.pdf` 及 `36975-000N-Data.rda`。  
研究级文档见 `36975-Codebook-PI.pdf`、`36975-User_guide.pdf`。

---

## 处理后数据（`processed/by_wave/`）

每个 wave 导出为 `{ego,alters}.csv`、`.parquet`，以及：

- `variable_labels_{name}.json` — 变量名 → 英文题项/字段说明
- `value_labels_{name}.json` — 分类变量取值码表

| Wave | ego | alters |
|------|-----|--------|
| **Wave1** | 1,159 × 555 | 20,313 × 71 |
| **Wave2** | 1,159 × 425 | 20,313 × 73 |
| **Wave3** | 1,159 × 439 | 20,313 × 82 |
| **combined** | 1,159 × 1,415 | 20,313 × 222 |

完整索引见 [`processed/manifest.csv`](processed/manifest.csv)。

---

## 访谈与纵向追踪

### 研究设计

UCNets 是 **固定样本面板**：Wave 1 招募 1,159 名受访者，之后尽量对同一批人追踪。三波 **不是** 独立横截面，而是同一 `PRIM_KEY` 的重复测量。

| 波次 | 时间 | 实际完成访谈 | 占 Wave 1 比例 |
|------|------|-------------|----------------|
| Wave 1 | 2015-04 — 2016-01 | 1,159 | 100% |
| Wave 2 | 2017-02 — 2017-06 | 1,022 | 88.2% |
| Wave 3 | 2018-02 — 2018-05 | 962 | 83.0% |
| 三波均参与 | — | 962 | 83.0% |

- Wave 1 → Wave 2 流失 137 人（11.8%）；Wave 2 → Wave 3 流失 60 人（5.9%）
- 无人跳过 Wave 2 却在 Wave 3 回来
- 每个 ego 文件始终 1,159 行（面板骨架）；是否参与某波次看 `WAVE2`/`WAVE3` 标志

### 访谈模式

抽样时随机分配访谈模式，Wave 1 以面对面（FTF）为主，后续波次 Web 比例上升：

| 变量 | 说明 | Wave 1 分布 |
|------|------|-------------|
| `INTENT_MODE` | 随机分配的**预期**模式 | 1=FTF, 3=Web |
| `SELECTED_MODE` | 受访者**实际选择**的模式 | 1=FTF, 3=Web |
| `MODE` | **最终完成**的访谈模式 | 1=FTF (647), 3=Web (512) |

Wave 2/3 对应变量为 `INTENT_MODE_W2`、`MODE_W2` 等（带波次后缀）。  
Alter 表中也有 `MODE`（该 alter 记录来自 FTF 还是 Web 访谈）。

### 访谈管理变量（Ego 表）

以下变量记录访谈过程元数据，Wave 2/3 在 `combined` 中带 `_W2`/`_W3` 后缀：

| 变量 | 类型 | 含义 |
|------|------|------|
| `PRIM_KEY` | ID | 受访者唯一标识（13 位整数） |
| `WAVE1` / `WAVE2` / `WAVE3` | 标志 | 是否参与该波次（1=是，null=未参与） |
| `MODE` | 分类 | 访谈模式：1=FTF, 3=Web |
| `INTENT_MODE` | 分类 | 随机分配的预期模式 |
| `SELECTED_MODE` | 分类 | 受访者选择的模式 |
| `BEGINTIME` / `ENDTIME` | 时间戳 | 访谈起止时间 |
| `TOT_TIME_MIN` | 数值 | 访谈总时长（分钟） |
| `I_DURATION` | 数值 | 访谈时长（秒） |
| `LANGUAGE` | 分类 | 访谈语言 |
| `INTERVIEWER_W1` | ID | 访员编号（96=无录音） |
| `INT1_W1` — `INT4_W1` | 分类 | 访员性别、年龄组、种族、经验 |
| `S1` / `S1S1`…`S1S7` | 分类 | 访谈地点（客厅、厨房、工作场所等） |
| `WEIGHT_W1_DEMO` | 数值 | Wave 1 人口加权 |
| `WEIGHT_W2_DEMO` / `WEIGHT_W3_DEMO` | 数值 | Wave 2/3 人口加权（combined 中） |
| `WT_W1_DEM_95_INF` | 数值 | Wave 1 权重 95% 置信区间 |
| `SUBSAMP_ALGO_W1` | 分类 | 子样本算法（影响 alter 详细描述模块） |
| `FB_FLAG` | 标志 | 是否通过 Facebook 招募 |
| `REF_FLAG` | 标志 | 是否由其他受访者推荐 |
| `AGEGROUP` | 分类 | 0=21–30 岁, 1=50–70 岁 |
| `AGE` / `AGE_EX` | 数值 | 年龄（整数 / 含小数） |
| `YEAR_PRELOAD` / `MONTH_PRELOAD` | 数值 | 预载出生年/月 |

### 样本与链接键

- **地区**：旧金山湾区 6 县
- **队列**：21–30 岁 & 50–70 岁两个年龄组
- **Ego 链接键**：`PRIM_KEY`（跨波匹配同一受访者）
- **Alter 链接键**：`ID_W1_W2_W3`（= `PRIM_KEY` + 2 位序号，跨波匹配同一网络成员）
- **Ego–Alter 链接**：`PRIM_KEY`

---

## 三表逻辑结构

UCNets 的社会网络数据对应 ego-centric **三表**，但 ICPSR 只物理拆成 ego + alters 两个文件；Alter–Alter 关系嵌在 alter 表的宽变量中。

```
Ego (1,159 行)                    Alter 长表 (20,313 行)
┌─────────────────────┐           ┌──────────────────────────────┐
│ PRIM_KEY            │──1:N─────│ PRIM_KEY                     │
│ 人口统计/健康/人格   │           │ ID_W1_W2_W3, NAME_NMBR2      │
│ 生命历程事件         │           │ Roster A/B/C/D 变量          │
│ 访谈管理变量         │           │ Alter 属性 (C2*, D1*)        │
└─────────────────────┘           │ Alter–Alter 矩阵 (N*, C1A_*) │
                                  └──────────────────────────────┘
```

| 层级 | 物理文件 | 行含义 | Wave 1 规模 |
|------|----------|--------|-------------|
| **Ego** | `ego` | 1 行 = 1 名受访者 | 1,159 × 555 |
| **Alter** | `alters` | 1 行 = 1 个 alter 槽位 | 20,313 × 71（约 17.6 alter/ego） |
| **Alter–Alter** | `alters` 内宽变量 | 嵌在 alter 行中的关系矩阵 | `N*_N*` + `C1A_*` |

---

## 表 1：Ego 字段说明

Wave 1 共 **555** 个变量，按问卷模块字母前缀分类。完整题项见 `variable_labels_ego.json`，取值码见 `value_labels_ego.json`。

### 模块总览

| 模块 | 前缀 | 变量数 | 内容 |
|------|------|--------|------|
| 筛选/同意 | SC | 7 | 互联网接入、设备类型、计算机使用意愿 |
| 抽样/访谈环境 | S | 42 | 访谈地点、抽样信息 |
| 生命历程 | A | 117 | 婚姻、生育、家庭变迁、父母/子女事件 |
| 家庭/住户 | B | 52 | 家庭共餐频率、与 difficult alter 的应对方式 |
| 就业/社会支持 | E | 10 | 亲属/朋友帮助意愿、社交网络满意度 |
| 健康行为/生活事件 | F | 61 | 工作/经济/关系/死亡等重大生活事件 |
| 健康状况 | G | 83 | 自评健康、慢性病、ADL/IADL、抑郁/焦虑 |
| 人格/在线交友 | H | 36 | 在线约会经历、幸福感、孤独感、PHQ 量表 |
| 互联网/联系方式 | I | 53 | 上网频率、与家人/朋友联系方式 |
| Big Five | J | 13 | 外向性、宜人性、尽责性、神经质、开放性 |
| Ego 背景 | K | 77 | 教育、宗教、政治、族裔、配偶信息 |
| 管理/权重 | — | ~20 | 见上文「访谈管理变量」 |

### 各模块代表变量

#### A — 生命历程 / 人口统计

| 变量 | 类型 | 含义 |
|------|------|------|
| `A2A` | 分类 | 婚姻状况（1=已婚, 2=丧偶, 3=离婚, 4=分居, 5=未婚） |
| `A2B` / `A2B2` | 数值/分类 | 结婚年份、婚前是否同居 |
| `A3A` — `A3C` | 分类/数值 | 同居/恋爱关系 |
| `A6` — `A8A` | 分类 | 子女相关（有无、数量） |
| `A10` — `A18B` | 分类 | 父母状况（健在、婚姻、离婚） |
| `A12A` / `A12B` | 分类 | 近一年是否有人搬入/搬出 |
| `A15A` — `A21S` | 分类 | 家庭里程碑事件（毕业、换工作、搬家等） |

#### G — 健康状况

| 变量 | 类型 | 含义 |
|------|------|------|
| `GENDER` | 分类 | 性别（1=男, 2=女） |
| `G1` | 分类 | 自评总体健康（excellent → poor） |
| `G2S1` — `G2S8` | 分类 | 是否被诊断各类疾病 |
| `G8A` — `G8C` | 分类 | 吸烟/饮酒/运动 |
| `G9A` — `G9BS` | 分类 | 睡眠、体重 |
| `G10A` — `G10AS6` | 分类 | 日常活动困难（ADL） |

#### F — 健康行为 / 重大生活事件

| 变量 | 类型 | 含义 |
|------|------|------|
| `F1A` / `F1B` | 分类 | 近一年工作/经济重大事件及影响 |
| `F10A` / `F10B` | 分类 | 近一年关系变化及影响 |
| `F11A` — `F11E` | 分类/数值 | 近一年亲近者去世及影响 |

#### J — Big Five 人格

| 变量 | 类型 | 含义 |
|------|------|------|
| `J1A` | 分类 | 内向/ reserved |
| `J1B` | 分类 | 外向/ sociable |
| `J1C` | 分类 | 信任/ trusting |
| `J1D` — `J1J` | 分类 | 挑错、抗压、想象力、艺术兴趣、懒惰、情绪稳定、条理、创新 |

#### K — Ego 背景（含配偶 pipe-in 题）

| 变量 | 类型 | 含义 |
|------|------|------|
| `K1` / `K2` | 分类/数值 | 最高学历及完成年份 |
| `K10A` / `K10B` | 分类 | 投票习惯、政党倾向 |
| `K11` — `K13` | 分类 | 配偶/父母出生国 |
| `K15` / `K16` | 分类 | 现宗教 / 成长宗教 |
| `K3` — `K9D` | 分类/数值 | 配偶教育、就业、族裔等（pipe-in 配偶姓名） |

> 每个模块还有 `_OTHER` 开放文本补充变量。555 个变量的完整英文题项见 `processed/by_wave/Wave1/variable_labels_ego.json`。

---

## 表 2：Alter 字段说明

Wave 1 共 **71** 个变量。每行 = 某 ego 网络中的一个 alter 槽位（`NAME_NMBR2` 1–30）。同一 alter 可出现在多个 Roster 中，通过对应列非空判断。

### 2.1 链接与编号

| 变量 | 类型 | 含义 |
|------|------|------|
| `PRIM_KEY` | ID | 所属 ego 的 `PRIM_KEY` |
| `ID_W1_W2_W3` | ID | Alter 跨波唯一标识（`PRIM_KEY` + 2 位序号） |
| `NAME_NMBR2` | 数值 | Wave 1 内 alter 编号（1–30） |
| `NAME_SUBNMBR` | 数值 | 子样本编号（Section D 问题用） |
| `MERGED_NAME` | 标志 | 1=该 alter 由多个重复提名合并 |
| `N_GENDER` | 分类 | Alter 性别（1=男, 2=女） |
| `MODE` | 分类 | 该记录来源访谈模式（1=FTF, 3=Web） |

### 2.2 Roster A — 讨论对象（Discussion）

| 变量 | 类型 | 含义 |
|------|------|------|
| `A2B` | 分类 | 是否被提名为配偶（0=no, 1=yes） |
| `A3B` | 分类 | 是否被提名为恋爱伴侣 |
| `A12` | 数值 | 在 Roster A 中的提名顺序 |
| `A12X` | 分类 | 是否被提名为 household 成员 |

非空即表示该 alter 属于 Roster A。Wave 1 约 12,211 行有 A roster 数据。

### 2.3 Roster B — 情感/支持（Support）

| 变量 | 类型 | 含义 |
|------|------|------|
| `B2A` | 数值 | 社交伙伴（social companion）提名顺序 |
| `B4A` | 数值 | 知己（confidant）提名顺序 |
| `B5A` | 数值 | Advisor 提名顺序 |
| `B6B` | 数值 | 家务帮手提名顺序 |
| `B7C` | 数值 | 伤病帮助者提名顺序 |
| `B8A` | 数值 | ego 帮助的对象提名顺序 |
| `B9A` | 数值 | "difficult" 关系人提名顺序 |
| `G10D` | 分类 | 是否被提名为 disability helper |

### 2.4 Roster C — 活动伙伴（Activity）

Roster C 是 alter 表的**骨架**，所有 20,313 行均有 C roster 数据。

**关系类型（`C1A_*`，0=no, 1=yes）：**

| 变量 | 含义 |
|------|------|
| `C1A_1` | 配偶/伴侣 |
| `C1A_2` | 恋爱关系 |
| `C1A_3` | 父母 |
| `C1A_4` | 子女 |
| `C1A_5` | 兄弟姐妹 |
| `C1A_6` | 其他亲属 |
| `C1A_10` | 室友 |
| `C1A_11` | 邻居 |
| `C1A_12` | 工作场所认识 |
| `C1A_13` | 学校认识 |
| `C1A_14` | 教会认识 |
| `C1A_20` | 朋友 |
| `C1A_21` | 熟人 |
| `C1A_22` | 其他方式认识 |
| `C1A_23` — `C1A_26` | 继父母/继子女/继兄弟姐妹/同父异母 |

**Alter 属性（`C1B`/`C1C`/`C2*`）：**

| 变量 | 类型 | 含义 |
|------|------|------|
| `C1A` | 数值 | 该 alter 在 C roster 中的编号 |
| `C1B_NSAMESEX` | 分类 | 与 ego 同性别 |
| `C1C_NSAMEAGE` | 分类 | 与 ego 年龄相近（±6 岁） |
| `C1C_NOLDER` | 分类 | 比 ego 大 6 岁以上 |
| `C2A_NMET` | 分类 | 近一年新认识 |
| `C2B_NCLOSE` | 分类 | ego 感到特别亲近 |
| `C2C_N5MINUTES` | 分类 | 5 分钟车程内 |
| `C2D_N1HOUR` | 分类 | 1 小时以上车程 |
| `C2E1_NSAMEWORK` | 分类 | 从事同类工作 |
| `C2E2_NSAMESCHOOL` | 分类 | 也在上学 |
| `C2E3_NALSOHOME` | 分类 | 也照顾家庭 |
| `C2E4_NALSOUNEMP` | 分类 | 也失业 |
| `C2F_NSAMEREL` | 分类 | 同宗教 |
| `C2G_NSAMERACE` | 分类 | 同种族/族裔 |
| `C2H_NDIFPOLITICS` | 分类 | 政治观点不同 |

### 2.5 Roster D — 其他重要联系人

| 变量 | 类型 | 含义 |
|------|------|------|
| `D1A` | 分类 | 认识方式（1=同家族, 7=工作, 10=朋友介绍, 13=网上认识, 14=其他…） |
| `D1B` | 数值 | 认识时长（年） |
| `D1C` | 分类 | 该 alter 是否有配偶 |
| `D1F` | 分类 | 见面频率（1=每天, 2=每周, …, 6=从不） |
| `D1G` | 分类 | 电话频率 |
| `D1H` | 分类 | 邮件/短信频率 |
| `D1I` | 分类 | 是否经历重大生活变化 |
| `D1K` | 分类 | ego 是否感到有义务帮助（1=非常, 4=没有） |
| `D1L` | 数值 | 该 alter 是否可能向 ego 求助 |

### 2.6 纵向追踪变量（Wave 2/3 / combined）

Wave 2/3 的 alter 文件变量名带 `_W2`/`_W3` 后缀；`combined` 中三波并存。

| 变量 | 波次 | 含义 |
|------|------|------|
| `W2_NAME_NMBR` / `W3_NAME_NMBR` | W2/W3 | 该波次内 alter 编号 |
| `W1_MISSING_W2` | W2 | Wave 1 的 alter 在 Wave 2 未出现 |
| `W1_WHY_W2` | W2 | 未出现原因 |
| `W1_WHY_REL_W2` | W2 | 关系如何变化 |
| `W2_NEWNAME` | W2 | Wave 2 新提名 |
| `W2_MISSING_W3` | W3 | Wave 2 的 alter 在 Wave 3 未出现 |
| `W2_WHY_W3` / `W2_WHY_REL_W3` | W3 | 未出现原因 / 关系变化 |
| `W3_AGAIN` | W3 | Wave 1 提名、Wave 2 缺席、Wave 3 回归 |
| `W3_AGAIN_WHY` | W3 | 回归原因 |
| `W3_NEW` / `W3_NEW_WHY` | W3 | Wave 3 全新提名 / 原因 |
| `NAME_IN_W1` / `NAME_IN_W2` / `NAME_IN_W3` | W3 | 跨波 alter 对应编号 |
| `CORR_NOTE` | W3 | 数据修正说明 |

---

## 表 3：Alter–Alter 字段说明

Alter–Alter 关系 **不单独成表**，嵌在 alter 长表的宽变量中。需按 ego 分组后从矩阵变量提取边。

### 3.1 讨论网前 5 人：`N*_N*`

针对 Roster A 中前 5 名 alter，两两询问熟悉程度：

| 变量 | 含义 | 取值 |
|------|------|------|
| `N1_N2` | alter 1 与 alter 2 | 1=very well, 2=know a little, 3=do not know, 4=something else |
| `N1_N3` … `N1_N5` | alter 1 与 alter 3/4/5 | 同上 |
| `N2_N3` … `N4_N5` | alter 2–5 之间所有组合 | 同上 |

共 10 对（C(5,2)），非空行 752–2,112 条。  
**存储方式**：值存在 alter 1 对应的行上（`NAME_NMBR2=1` 的行含 `N1_N2`…`N1_N5`）。

### 3.2 活动伙伴网：`C1A_*` 矩阵

针对 Roster C 中的 alter，询问与其他 C roster alter 是否有联系：

| 变量 | 含义 | 取值 |
|------|------|------|
| `C1A` | 该 alter 在 C roster 中的编号 | 数值（如 4, 11, 20, 22） |
| `C1A_j` (j=1–6, 10–14, 20–26) | 该 alter 是否与 C roster 编号 j 的 alter 有联系 | 0=no, 1=yes |

**示例**（ego `3000000000036`）：

| NAME_NMBR2 | C1A (C roster #) | C1A_20=1 含义 |
|------------|----------------|---------------|
| 6 | 20 | alter 6 与 C roster 中 #20 的 alter 有联系 |

**提取边列表示例**：

```python
# 从 C1A_* 提取 alter-alter 边（概念示意）
c1a_cols = [c for c in alters.columns if c.startswith("C1A_")]
# 对每行：若 C1A_j == 1，则 (C1A, j) 为一条边
```

### 3.3 三表对应关系小结

| 分析需求 | 使用变量 | 所在表 |
|----------|----------|--------|
| Ego 属性 | `GENDER`, `AGE`, `A2A`, `G1`, `J1A`… | ego |
| Alter 是否在某 roster | `A2B`, `B4A`, `C1A_20`, `D1A` 等非空 | alters |
| Alter 属性 | `N_GENDER`, `C2B_NCLOSE`, `D1F`, `D1A` | alters |
| Alter–Alter 熟悉度 | `N1_N2` … `N4_N5` | alters（宽矩阵） |
| Alter–Alter 活动联系 | `C1A_*` | alters（宽矩阵） |
| 跨波 alter 追踪 | `ID_W1_W2_W3`, `NAME_IN_W1/W2/W3` | alters (combined) |
| 面板 attrition | `WAVE2`, `WAVE3`, `W1_MISSING_W2` | ego / alters |

---

## 波次文件命名规则

| 文件 | 变量命名 | Alter 槽位列 |
|------|----------|-------------|
| `Wave1/` | 无前缀（`A2B`, `C1A`, `D1F`） | `NAME_NMBR2` |
| `Wave2/` | 全部 `_W2` 后缀 | `W2_NAME_NMBR` |
| `Wave3/` | 全部 `_W3` 后缀 | `W3_NAME_NMBR` |
| `combined/` | W1 无前缀 + W2/W3 带后缀并存 | 三波编号列都在 |

- **单波截面分析** → 用 `Wave1`/`Wave2`/`Wave3`
- **面板变化分析** → 用 `combined`，或自行 join 三波

---

## 标签文件说明

| 文件 | 内容 | 示例 |
|------|------|------|
| `variable_labels_*.json` | 变量名 → 英文题项 | `"D1F"` → `"How often R sees sub-name"` |
| `value_labels_*.json` | 变量名 → 取值码表 | `"D1F"` → `{"1": "At least once a day", ...}` |

```python
var_labels = load_variable_labels("Wave1", "alters")  # 字段含义
val_labels = load_value_labels("Wave1", "alters")   # 取值含义
```

更完整的 skip logic 和问卷流程见 PDF codebook（`36975-Codebook-PI.pdf`）。

---

## 数据转换说明

ICPSR 的 R 格式将分类变量存为 factor，level 形如 `(1) male`。

`export_rda.R` 转换规则：

1. 带 `(数字)` 前缀的 factor → **数值编码**（写入 CSV/Parquet）
2. 取值标签 → `value_labels_{dataset}.json`
3. 变量题项 → `variable_labels_{dataset}.json`（来自 ICPSR `.rda` 的 `variable.labels` 属性）
4. 纯字符串 factor（如 `PRIM_KEY`、`ID_W1_W2_W3`）→ 保留字符串
5. 同步生成 Parquet（Polars 读取更快）

---

## 脚本

| 脚本 | 作用 |
|------|------|
| [`scripts/export_rda.R`](scripts/export_rda.R) | 扫描 `ICPSR_36975/DS*/*-Data.rda`，导出 CSV + 标签 JSON |
| [`scripts/csv_to_parquet.py`](scripts/csv_to_parquet.py) | 批量 CSV → Parquet |
| [`scripts/load_ucnets.py`](scripts/load_ucnets.py) | `load_wave()` / `load_value_labels()` / `load_variable_labels()` / `list_waves()` |

---

## 引用

> Fischer, Claude S. UC Berkeley Social Networks Study (UCNets), San Francisco Bay Area, 2015-2018. Inter-university Consortium for Political and Social Research.  
> https://doi.org/10.3886/ICPSR36975.v2
