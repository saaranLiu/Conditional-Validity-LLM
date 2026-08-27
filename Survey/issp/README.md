# ISSP 2017 Social Networks 数据说明

International Social Survey Programme（ISSP）2017 年「社会网络与社会资源」模块本地副本，由 GESIS ZA6980 v2.0.0 导出，面向 Python（Polars）分析。

**官方来源**：[GESIS ZA6980](https://www.gesis.org/en/issp/data-and-documentation/social-networks/2017

**引用**：
> ISSP Research Group (2019). *International Social Survey Programme: Social Networks and Social Resources - ISSP 2017*. GESIS Data Archive, Cologne. ZA6980 Data file Version 2.0.0, https://doi.org/10.4232/1.13322

---

## 快速开始

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("Survey/issp/scripts")))
from load_issp import (
    load_survey,
    load_variable_labels,
    load_value_labels,
    load_manifest,
)

df = load_survey()              # 44,492 × 356，优先读 parquet
manifest = load_manifest()      # 版本、国家列表等元数据
var_labels = load_variable_labels()  # 变量名 → 英文题项
val_labels = load_value_labels()     # 变量名 → {label, values}

# 查某一列含义
print(var_labels["v1"])
print(val_labels["v1"]["values"])   # {"1": "Family or relative", ...}

# 按国家筛选
import polars as pl
cn = df.filter(pl.col("c_alphan") == "CN")
```

```bash
# 原始 .dta 更新后重新导出
Rscript Survey/issp/scripts/export_dta.R
python3 Survey/issp/scripts/csv_to_parquet.py
```

---

## 目录结构

```
Survey/issp/
├── README.md                          # 本文档
├── ZA6980_v2-0-0.dta/                 # GESIS 原始下载
│   ├── ZA6980_v2-0-0.dta              # Stata 整合数据（~29 MB）
│   ├── ZA6980_cdb.pdf                 # Variable Report（1003 页）
│   └── ZA6980_v2-0-0_missing.txt      # Stata 缺失值编码（非下载缺失清单）
├── processed/                         # Python 友好格式
│   ├── survey.parquet                 # 主表（推荐）
│   ├── survey.csv
│   ├── variable_labels.json           # 356 变量英文标签
│   ├── value_labels.json              # 356 变量取值码
│   └── manifest.json
└── scripts/
    ├── export_dta.R                   # Stata → CSV + 标签 JSON
    ├── csv_to_parquet.py              # CSV → Parquet
    └── load_issp.py                   # Python 读取工具
```

| 文件 | 粒度 | 说明 |
|------|------|------|
| `survey` | 受访者 × 356 变量（wide） | 一行 = 一位 ego |
| `variable_labels.json` | 变量名 → 英文题项/字段说明 | 356 条 |
| `value_labels.json` | 变量名 → `{label, values}` | 356 条 |
| `manifest.json` | 研究级元数据 | 版本、DOI、国家列表 |

---

## 数据概览

| 项目 | 内容 |
|------|------|
| 档案号 | ZA6980 |
| 版本 | 2.0.0（2019-08-19） |
| 模块主题 | Social Networks and Social Resources |
| 总样本 | **44,492** 受访者 |
| 变量数 | **356** |
| 国家/地区 | **30**（整合文件） |
| 纵向 | 否（横截面） |
| 权重 | `WEIGHT`（多数国家有；部分国家 = 1） |

### 参与国家与样本量

| 代码 | N | 代码 | N | 代码 | N |
|------|---|------|---|------|---|
| CN | 4219 | DE | 1701 | AT | 1200 |
| ZA | 3098 | JP | 1609 | PH | 1200 |
| TW | 1955 | GB-GBN | 1595 | US | 1173 |
| ES | 1733 | RU | 1559 | SE | 1125 |
| IN | 1510 | TH | 1500 | DK | 1079 |
| FR | 1489 | IS | 1450 | FI | 1074 |
| CZ | 1405 | SK | 1404 | CH | 1066 |
| NZ | 1357 | AU | 1317 | LT | 1052 |
| SR | 1273 | IL | 1267 | SI | 1047 |
| | | | | HR | 1026 |
| | | | | HU | 1007 |
| | | | | MX | 1002 |

---

## 核心设计：网络信息如何采集

ISSP 2017 **不使用 name generator**（不像 GSS 那样提名 5 个讨论对象并追问性别/年龄）。模块采用 **定位法 + 角色/情境题**，更适合跨国比较和自填问卷，但 **无法还原 alter-level 个体表**。

| 维度 | ISSP 2017 | GSS name generator |
|------|-----------|-------------------|
| 提名具体联系人 | 否 | 是（最多 5 人） |
| Alter 性别/年龄/教育 | **否** | 是 |
| Alter–Alter 关系 | 否 | 部分年份有 |
| 关系类型 | 是（角色/职业槽位） | 是 |
| 接触频率 | 是（按角色槽位） | 部分有 |

### 四类「社会关系」相关测量

#### 1. 定位法 Position Generator（`v1`–`v10`，Q1）

对 10 种职业，问亲友圈中是否认识该职业的人（需知道姓名且能联系）；若多人符合，选 **最亲近** 者，只记录 **关系类型**：

| 编码 | 含义 |
|------|------|
| 1 | 家人/亲属 |
| 2 | 密友 |
| 3 | 其他熟人 |
| 4 | 无人 |

职业：公交/货车司机、大公司高管、清洁工、理发师、HR、律师、机修工、护士、警察、教师。

#### 2. 情感/日常求助（`v21`–`v25`，Q7）

5 种情境下 **最先向谁求助**（近亲/远亲/密友/邻居/同事/其他人/无人）：家务、生病、抑郁、家庭建议、社交陪伴。

#### 3. Instrumental 求助（`v26`–`v30`，Q8）

5 种情境下向 **哪类来源** 求助：借大钱、找工作、行政事务、找住处、重病照护。选项为家人朋友、其他人、私营/公共/非营利机构等。

#### 4. 角色锚定接触频率（`v48`–`v52`, `v65`–`v66`，Q21–Q25 / Q35–Q36）

问与 **联系最频繁** 的一位父母/兄弟姐妹/成年子女/其他亲属/密友/同乡/教友的接触频率（非具名 alter）。另有 `v46`–`v47` 工作日接触人数、`v53` 互联网联系占比等聚合指标。

### 配偶/伴侣信息（唯一接近「关系对象属性」的部分）

对 **配偶/稳定伴侣** 有部分背景变量（非网络提名）：

| 变量 | 含义 |
|------|------|
| `v67` | 配偶最高学历（国际分类） |
| `SPWORK`, `SPWRKHRS`, `SPISCO08`, `SPMAINST` 等 | 配偶就业、工时、职业 |
| `PARTLIV`, `MARITAL` | 是否有稳定伴侣、婚姻状态 |

**没有**配偶年龄、性别等单独字段。

---

## 变量结构（356 列）

### 1. 档案 / 协议变量（14 列）

| 变量 | 含义 |
|------|------|
| `studyno` | GESIS 档案号（6980） |
| `version` | 数据版本 |
| `doi` | DOI |
| `country` | 国家 ISO 3166 数字码 |
| `c_sample` | 国家/子样本 ISO 码 |
| `c_alphan` | 国家字母码（分析常用） |
| `CASEID` | 个案唯一 ID |
| `SUBSCASE` | 子样本内 ID |
| `WEIGHT` | 权重 |
| `MODE` | 采集方式（面访 PAPI/CAPI、自填等） |
| `PARTIALS` | 是否部分完成 |
| `DATEYR`, `DATEMO`, `DATEDY` | 访问日期 |

### 2. 问卷题 `v1`–`v67`（67 列）

| 题组 | 变量 | 主题 |
|------|------|------|
| Q1 | `v1`–`v10` | 定位法（10 职业 × 关系类型） |
| Q2 | `v11`–`v14` | 收入不平等态度 |
| Q3–Q4 | `v15`–`v16` | 医疗/养老责任归属 |
| Q5 | `v17`–`v19` | 组织参与频率 |
| Q6 | `v20` | 政治 efficacy |
| Q7 | `v21`–`v25` | 情感/日常求助对象 |
| Q8 | `v26`–`v30` | Instrumental 求助来源 |
| Q9–Q16 | `v31`–`v43` | 孤独、信任、家庭规范与压力 |
| Q17–Q20 | `v44`–`v47` | 社交外出、接触人数 |
| Q21–Q26 | `v48`–`v53` | 角色锚定接触频率 |
| Q27–Q36 | `v54`–`v66` | 健康、满意度、友谊规范、同乡/教友联系 |
| 配偶 | `v67` | 配偶学历 |

完整英文题项见 `processed/variable_labels.json`。

### 3. 跨国可比背景变量（约 31 列）

ISSP 标准背景模块，各国统一编码：

| 类别 | 代表变量 |
|------|----------|
| 人口学 | `SEX`, `BIRTH`, `AGE`, `MARITAL`, `PARTLIV`, `HOMPOP`, `HHCHILDR`, `HHTODD` |
| 教育 | `EDUCYRS`, `DEGREE` |
| 宗教/政治 | `RELIGGRP`, `ATTEND`, `VOTE_LE`, `PARTY_LR`, `TOPBOT` |
| 就业（本人） | `MAINSTAT`, `WORK`, `WRKHRS`, `WRKSUP`, `EMPREL`, `ISCO08`, `NEMPLOY`, `NSUP`, `TYPORG1`, `TYPORG2`, `UNION` |
| 就业（配偶） | `SPWORK`, `SPWRKHRS`, `SPWRKSUP`, `SPEMPREL`, `SPISCO08`, `SPMAINST` |
| 地理/移民 | `URBRURAL`, `F_BORN`, `M_BORN` |

### 4. 各国特有变量（240 列 = 30 国 × 8 类）

命名规则 `{国家码}_{类型}`，如 `CN_DEGR`、`DE_REG`。**仅该国受访者有有效值**。

| 后缀 | 含义 |
|------|------|
| `*_DEGR` | 该国学历分类 |
| `*_RELIG` | 该国宗教/教派 |
| `*_PRTY` | 该国投票政党 |
| `*_ETHN1`, `*_ETHN2` | 该国族群 |
| `*_RINC` | 该国个人收入 |
| `*_INC` | 该国家庭收入 |
| `*_REG` | 该国地区 |

国家码：`AT AU CH CN CZ DE DK ES FI FR GB HR HU IL IN IS JP LT MX NZ PH RU SE SI SK SR TH TW US ZA`（英国为 `GB_*`）。

---

## 缺失值编码

常见编码（因题而异，以 `value_labels.json` 为准）：

| 编码 | 常见含义 |
|------|----------|
| 0 | 不适用 / 未问 |
| 8 | 无法选择 |
| 9 | 无回答 / 拒答 |
| 95–96 | 特殊跳过（如无兄弟姐妹、对象同住） |
| 98 | 不知道 |
| 99 | 无回答 |

`ZA6980_v2-0-0_missing.txt` 为 Stata 格式的缺失值定义文件，**不是**下载缺失清单。

---

## 数据流

```
ZA6980_v2-0-0.dta
    │
    └─ export_dta.R ──► survey.csv
                        variable_labels.json
                        value_labels.json
                        manifest.json
            │
            └─ csv_to_parquet.py ──► survey.parquet
```

| 脚本 | 作用 |
|------|------|
| `export_dta.R` | Stata → CSV + 标签 JSON |
| `csv_to_parquet.py` | CSV → Parquet |
| `load_issp.py` | Python 读取封装 |

---

## 适用场景与局限

### 适合

- 跨国比较社会资本（定位法职业可达性）
- 求助/支持网络的角色结构（家庭 vs 朋友 vs 同事）
- 不同社会角色的接触频率分布
- 结合 ego 背景变量（`SEX`, `AGE`, `DEGREE`, `ISCO08` 等）的宏观分析

### 不适合

- 需要 **alter-level 性别、年龄、教育** 的微观网络仿真（无 name generator）
- 构建 **alter–alter 边列表** 或完整 ego-network 图
- 作为 GSS / UCNets / LASA 式提名网络的 ground truth

### 在本项目中的定位

见 [`Datasets.md`](../../Reports/Datasets.md) 第 5 节。ISSP 2017 评级 ★☆☆：含网络相关子模块，但 **不满足 alter-level 完整人口统计** 需求。

---

## 参考资料

- [GESIS ISSP 2017 Social Networks 模块页](https://www.gesis.org/en/issp/data-and-documentation/social-networks/2017)
- Variable Report：`ZA6980_v2-0-0.dta/ZA6980_cdb.pdf`
- Joye, Sapin & Wolf (2019). *Measuring Social Networks and Social Resources: An Exploratory ISSP Survey around the World*. GESIS-Schriftenreihe 22. https://doi.org/10.21241/ssoar.62256
- 2017 模块设计说明：2017 版 **放弃 name generator**，改用 position generator（见上述 GESIS 设计文档）
