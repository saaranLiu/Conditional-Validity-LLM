# NSHAP 数据说明

National Social Life, Health, and Aging Project（NSHAP）本地副本，面向 Python（Polars）社会网络分析。

数据来源：[ICPSR NACDA Series 706](https://www.icpsr.umich.edu/web/NACDA/series/706/studies)（public-use 版本，手动下载 R 格式）。

---

## 快速开始

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("Survey/nshap/scripts")))
from load_nshap import list_rounds, load_network_round, load_value_labels

print(list_rounds())  # ['Round1', 'Round2', 'Round3']

tables = load_network_round("Round3")
core = tables["core"]                    # Ego 层
networks = tables["social_networks"]     # Alter 长表

# 合并 Core + Social Networks（链接键 ID）
df = core.join(networks, on="ID", how="inner")

# 解码分类变量
labels = load_value_labels("Round3", "core")
# labels["GENDER"] -> {'1': 'male', '2': 'female'}
```

```bash
# 原始 .rda 更新后，重新导出
Rscript Survey/nshap/scripts/export_rda.R Survey/nshap Survey/nshap/processed
python3 Survey/nshap/scripts/csv_to_parquet.py
```

---

## 目录结构

```
Survey/nshap/
├── README.md
├── manifest.json              # ICPSR 各轮 public/restricted 文件清单（参考）
├── Round1/ … Round4/          # ICPSR 原始下载（codebook、.rda 等）
├── processed/                 # Python 友好格式（由 scripts 生成）
│   ├── manifest.csv           # 导出索引（行数、列数、源文件路径）
│   └── by_round/
│       ├── Round1/
│       ├── Round2/
│       └── Round3/
└── scripts/
    ├── export_rda.R           # .rda → CSV + value_labels JSON
    ├── csv_to_parquet.py      # CSV → Parquet
    └── load_nshap.py          # Python 读取工具
```

---

## 原始数据（ICPSR 下载）

| 目录 | ICPSR | 年份 | 说明 |
|------|-------|------|------|
| `Round1/` | [20541](https://www.icpsr.umich.edu/web/NACDA/studies/20541) | 2005–06 | Cohort 1 基线 |
| `Round2/` | [34921](https://www.icpsr.umich.edu/web/NACDA/studies/34921) | 2010–11 | C1 追踪 + 伴侣 |
| `Round3/` | [36873](https://www.icpsr.umich.edu/web/NACDA/studies/36873) | 2015–16 (+ COVID 2020–21) | C1 追踪 + C2 新队列 |
| `Round4/` | [39511](https://www.icpsr.umich.edu/web/NACDA/studies/39511) | 2021–23 | 目前仅 documentation，无 .rda |

各轮 `ICPSR_*/DS*/` 下为 ICPSR 分 dataset 打包的 `.rda` 文件及 codebook PDF。

---

## 处理后数据（`processed/by_round/`）

每轮每个 dataset 导出为 `{name}.csv`、`.parquet`，以及 `value_labels_{name}.json`（分类变量码表）。

| Round | 文件 | 行 × 列 | 用途 |
|-------|------|---------|------|
| **Round1** | `core` | 3,005 × 820 | Ego 人口统计、健康、权重等 |
| | `social_networks` | 13,116 × 20 | Alter 长表（每名册一条） |
| **Round2** | `core` | 3,196 × 861 | Ego |
| | `social_networks` | 14,980 × 23 | Alter |
| | `social_networks_update` | 9,537 × 5 | 网络成员跨轮变化 |
| | `partner_disposition_r1` | 3,005 × 3 | R1 伴侣追踪 |
| | `proxy` | 433 × 19 | 失访/死亡 proxy |
| | `sleep_statistics` | 2,128 × 11 | 加速度计睡眠 |
| **Round3** | `core` | 4,377 × 726 | Ego（C1 + C2） |
| | `social_networks` | 19,442 × 20 | Alter |
| | `partner_disposition` | 3,317 × 4 | 伴侣追踪 |
| | `proxy` | 587 × 19 | Proxy |
| | `covid` | 2,429 × 99 | COVID-19 子研究 |

完整索引见 [`processed/manifest.csv`](processed/manifest.csv)。各表字段说明见下文「各表字段说明」。

---

## Cohort 与纵向链接

| Cohort | 出生年 | 首次入样 |
|--------|--------|----------|
| C1 | 1920–1947 | Round 1 |
| C2 | 1948–1965 | Round 3 |

- **纵向链接键**：`ID`（跨轮匹配同一受访者；public-use 数据中无 `SU_ID`）
- **截面权重**：各轮 Core 中的 `WEIGHT_ADJ`
- **区分 C1/C2**（Round 3）：Core 变量 `SURVEYTYPE3`（1 = returning / C1，2 = new cohort / C2）
- **样本类型**（Round 2）：`SURVEYTYPE`（1 = W1 追踪，2 = W1 未访重联，3 = partner）

分类变量取值标签见各轮 `value_labels_{dataset}.json`；完整题项说明见 ICPSR codebook PDF。

---

## 各表字段说明

以下描述 `processed/by_round/` 中各导出表。`core` 变量极多（700+ 列），此处按模块归纳；其余表逐字段列出。

### `social_networks`（Alter 长表，一名册一行）

| 字段 | 类型 | 说明 |
|------|------|------|
| `ID` | int | Ego 标识，链接 `core` |
| `SECTION` | int | 所属名册：1=a 倾诉，2=b 配偶/伴侣，3=c 其他重要联系人，4=d 家庭成员 |
| `LINENO` | int | 该 Ego 网络内的 Alter 序号（用于 `TALKFREQ*` 配对） |
| `ANYMORE` | int | 该名册是否还有更多人：0=否，1=是 |
| `RELAT` / `RELAT2` | int | 与 Ego 关系类型（R1 为 `RELAT`，R2/R3 为 `RELAT2`，含配偶/子女/朋友等 18–19 类） |
| `NODE_GENDER` | int | Alter 性别：1=男，2=女 |
| `LIVEWITH` | int | 是否同住：0=否，1=是，2=部分年份同住 |
| `NODE_AGE` | int | Alter 年龄（同住者常可报告；否则缺失） |
| `NODE_AGEDK` | int | 相对年龄：1=更年轻，2=差不多，3=更年长 |
| `NODE_LOCAL` | int | 是否本地（仅 R3）：0=否，1=是 |
| `TALKFREQ0` | int | 该 Alter 与 **Ego** 的交谈频率（0–8 有序量表，见下） |
| `TALKFREQ1`–`TALKFREQ7` | int | 该 Alter 与名册中 **LINENO 1–7** 的其他 Alter 的交谈频率 |
| `HOWCLOSE` | int | 情感亲密度：1–4（仅 R1/R2；R3 无此变量） |
| `HEALTHTALK` | int | 讨论健康问题的可能性：1–3（仅 R1/R2） |
| `SPCURRENT` | int | 是否当前配偶（仅 R1/R2） |
| `ROSTER_W1` | int | 是否在 Round 1 名册中：0=否，1=是（R2/R3） |
| `LINENO_W1` | int | Round 1 对应行号（仅 R2） |
| `LONGKNOW` | int | 认识时长：1=<1 年 … 4=>6 年（R2/R3） |

`TALKFREQ*` 编码（0–8）：0=从未交谈，1=少于一年一次，2=一年一次，3=一年几次，4=一月一次，5=两周一次，6=一周一次，7=一周几次，8=每天。

R3 注意：`social_networks` 仅含 SECTION 1/2/4，无 `HOWCLOSE`、`HEALTHTALK`、`TALKFREQ7`。

---

### `social_networks_update`（仅 Round 2，R1→R2 网络成员追踪）

| 字段 | 类型 | 说明 |
|------|------|------|
| `ID` | int | Ego 标识 |
| `LINENO_W1` | int | Round 1 名册行号 |
| `LINENO_W2` | int | Round 2 对应行号（缺失表示该成员未进入 R2 名册） |
| `ROSTER_STLLIV` | int | 该成员是否仍存活：0=否，1=是 |
| `ROSTER_REASON` | int | 若不在网中：1=受访者搬家，2=对方搬家，3/4=健康原因，5=其他 |

---

### `partner_disposition_r1`（Round 2）/ `partner_disposition`（Round 3）

追踪回归受访者伴侣的样本处置状态。

| 字段 | 类型 | 说明 |
|------|------|------|
| `ID` | int | 关联受访者 ID |
| `SAMPLETYPE` | int | 仅 R3：1=returning respondent，2=new cohort (household)，3=new cohort (respondent) |
| `DISP_GRP` | int | 处置分组：1=已访谈，2=死亡，3=健康不佳，4=养老院，5=其他（推定在世），6=其他（生死不明） |
| `DECEASED` | int | 0=在世，1=死亡 |

---

### `proxy`（Round 2/3，失访/死亡受访者的代理人访谈）

| 字段 | 类型 | 说明 |
|------|------|------|
| `ID` | int | 原受访者 ID |
| `FI_ID` | str | 访员 ID |
| `PROXY_DATE` | date | 代理访谈日期 |
| `RSTATUS` | int | 受访者状态 |
| `PROXYREL` | int | 代理人与受访者关系 |
| `KNOW_DUR` / `KNOW_UNITS` | num/int | 认识受访者时长及单位 |
| `HAVSPOUS` | int | 受访者是否有配偶 |
| `SPOUSLIV` | int | 配偶是否在世 |
| `DEATHMO` / `DEATHYR` | int | 死亡月/年 |
| `CONDITION` / `CONDITION2` | int | 死亡/疾病原因（R2/R3 变量名略异） |
| `ILL_DUR` / `ILL_UNITS` | num/int | 患病时长 |
| `LIVE_DUR` / `LIVE_UNITS` | num/int | 独居/居住安排时长 |
| `NEEDHELP` | int | 是否需要帮助 |
| `WHOHELP` | int | 帮助来源 |
| `WHERELIV` | int | 去世前居住安排 |

---

### `sleep_statistics`（仅 Round 2，加速度计睡眠）

| 字段 | 类型 | 说明 |
|------|------|------|
| `ID` | int | 受访者 ID |
| `NIGHT` | int | 第几晚记录 |
| `REST_START` / `REST_STOP` | time | 休息起止时间 |
| `START` / `STOP` | time | 实际睡眠起止 |
| `ASSUMED_SLEEP` | num | 推断睡眠时长 |
| `ACTIGRAPH_SLEEP` | num | 加速度计测得睡眠时长 |
| `WASO` | num | _wake after sleep onset_（入睡后清醒时间） |
| `FRAGMENTATION` | num | 睡眠碎片化指数 |
| `PERCENT_SLEEP` | num | 睡眠占比 |

---

### `covid`（仅 Round 3，2020–21 COVID 子研究，99 列）

与 `core` 通过 `ID` 链接；含独立权重 `COVID_WT`。

| 模块 | 代表字段 | 内容 |
|------|----------|------|
| 访问与感染 | `MODE`, `CONCERN`, `DIAGNOS`, `COVIDDIE` | 访谈模式、担忧程度、是否确诊、是否因 COVID 失亲友 |
| 健康与就医 | `PHYSHLTH`, `PHYSHLTH_C19`, `DELAYCARE_*`, `COMPLTCARE`, `HOWCOMPLT_*` | 自评健康及疫情前后变化、延迟/完成就医 |
| 经济与工作 | `WORKAFFECT`, `INCEXP`, `INCAFFECT` | 工作受影响、收入变化 |
| 家庭联络 | `FAMPHN_*`, `FAMEML_*`, `FAMVID_*`, `FAMINPERS_*`, `FAMSOCWISH` | 与家人电话/邮件/视频/见面频率及社交意愿 |
| 朋友联络 | `FRDPHN_*`, `FRDEML_*`, `FRDVID_*`, `FRDINPERS_*`, `FRDSOCWISH` | 与朋友联络方式及意愿 |
| 居住变化 | `LIVECHANGE`, `WHYLIVECHANGE`, `WHEREMOVE_*`, `MOVEIN`, `WHOMOVE_*`, `HHSIZE` | 疫情期间搬家、同住人数 |
| 心理与社会隔离 | `HAPPY`, `MNTLHLTH4`, `MNTLHLTH_C19`, `GAD_*`, `FLTDEP`, `COMPANION2`, `LEFTOUT2`, `ISOLATED2`, `LONELY` | 情绪、焦虑、抑郁、孤独感 |
| 社会参与 | `ENGAGE_1`–`ENGAGE_7`, `EMNAMES`, `EMSERIOUS` | 社区/宗教/志愿活动参与 |
| 行为变化 | `PHYSACT3`, `PHYSACT_C19`, `ALCOHOL_*`, `SMOKING_*`, `RESTED2`, `RESTED_C19`, `POSCHANGE` | 运动、饮酒、吸烟、睡眠及积极变化 |

---

### `core`（Ego 主文件，一人一行）

各轮列数：R1 820 / R2 861 / R3 726。按主题模块归纳如下；药理学分类变量（`ANTIBIOTICS`、`SSRIANTIDEPRESSA` 等）在三轮均有大量 0/1 指标，表示是否服用该类药物，此处不逐一列出。

#### 标识、权重与样本类型

| 字段 | 说明 |
|------|------|
| `ID` | 受访者唯一标识（纵向链接键） |
| `FI_ID` | 访员 ID |
| `WEIGHT_SEL` / `WEIGHT_ADJ` | 抽样权重 / 调整后权重（分析用 `WEIGHT_ADJ`） |
| `STRATUM` / `CLUSTER` | 分层 / 聚类变量 |
| `SURVEYTYPE` | 仅 R2：1=W1 追踪，2=W1 未访重联，3=伴侣样本 |
| `SURVEYTYPE3` / `VERSION3` | 仅 R3：1=returning (C1)，2=new cohort (C2) |
| `INT_START` | 访谈开始时间 |
| `CASECOMP` / `CASEDIF` | 案例完成状态 / 难度标记 |

#### 人口统计

| 字段 | 说明 |
|------|------|
| `GENDER` | 性别：1=男，2=女 |
| `AGE` | 年龄 |
| `AGEGRP` | 年龄组（仅 R1） |
| `EDUC` / `DEGREE_RECODE` | 教育程度 |
| `RACE_RECODE` / `HISPANIC` / `ETHGRP` | 种族/族裔 |
| `MILITARY` / `BORN_US` | 军旅经历 / 是否美国出生 |
| `MARITLST` / `SPARTNER` / `SPNAMED` / `SPLINENO` | 婚姻状态 / 是否有伴侣 / 伴侣是否在名册 / 伴侣行号 |

#### 社会网络相关（core 内汇总项，非 Alter 长表）

| 字段 | 说明 |
|------|------|
| `ROSTERINTRO` | 是否进入名册模块 |
| `ALTERS` | 网络成员总数（仅 R1） |
| `SPTIME`–`SPCRITZE` | 与配偶/伴侣的关系质量（开放、依赖、批评等） |
| `FAMOPEN`–`FAMCRITZ` | 与家人的关系质量 |
| `FROPEN`–`FRCRITZ` / `FRAMT` | 与朋友的关系质量 / 朋友数量 |
| `CLSREL` | 与最亲近家人的关系 |
| `VOLUNTEER` / `ATTEND` / `SOCIAL` | 志愿、宗教参与、社交活动 |
| `CONFIDNT` / `COMPANION` / `LEFTOUT` / `ISOLATED` | 孤独与社交隔离量表（R2/R3 部分变量名带 `2` 后缀） |
| `DOFAVORS` / `PERSADVICE` / `CLOSEKNIT` / `HELPNEIGH` | 邻里与社区社会资本（R2/R3） |
| `CAREREC_ROSTER` / `CAREREC_LINE` / `CAREREC_REL` | 日常帮助提供者是否在网络名册中 |

#### 健康、功能与认知

| 字段 | 说明 |
|------|------|
| `PHYSHLTH` / `MNTLHLTH`（R3 为 `MNTLHLTH3`） | 自评身体健康 / 心理健康 |
| `HEALTH` | 总体健康（仅 R1） |
| `CONDITNS_*` / `CDIAG_*` | 慢性病与癌症诊断史 |
| `WALKBLK`–`TOILET` / `DRIVED` | ADL/IADL 功能限制 |
| `*_HELP` / `*_LIKEHELP` | 各项活动是否需要/希望得到帮助（R3） |
| `MOCA_*` | Montreal 认知评估各子项得分（R2/R3） |
| `MEM*` / `SPMSQ_*` | 认知与记忆测试（仅 R1） |
| `HAPPY` / `FLTDEP` / `WORRY` / `PANIC` 等 | 情绪与心理量表条目 |

#### 生物测量与实验室

| 字段 | 说明 |
|------|------|
| `HEIGHT` / `WEIGHT` / `WAIST` / `BMI` | 人体测量 |
| `SYSTOLIC_*` / `DIASTOLIC_*` / `PULSE*` | 血压与脉搏（多次测量） |
| `TESTOSTERONE_*` / `ESTRADIOL_*` / `DHEA_*` 等 | 唾液激素 Assay |
| `CRP` / `EBV` / `HB` / `HBA1C` | 血液斑点 Assay（R1 血浆/全血；R2 为 DBS 格式） |
| `SML_*` / `BLUEPEN_*` | 嗅觉测试（R1/R2/R3） |
| `WALK_*` / `CHAIR_*` / `BALANCE_*` | 体能测试（R2/R3） |

#### 性行为与亲密关系（NSHAP 特色模块）

| 字段 | 说明 |
|------|------|
| `SEXIMPRT` / `THINKSEX` / `SEX3MOS` | 性重要性与近期性行为 |
| `OFTSEX*` / `PLEASURE` / `EMTSATFY` | 性频率与满意度 |
| `LACKSEX` / `SEXPAIN` / `AVOIDSEX` 等 | 性功能障碍相关 |
| `TOUCHED` / `HUGGING` / `CUDDLE` / `TOUCH_PRTNR` | 身体接触（R2） |
| `SAMEBED` / `SAMEBED2` / `SAMEBED3` | 是否与伴侣同床（各轮） |

#### 用药

| 字段 | 说明 |
|------|------|
| `DRUGS_COUNT` / `NP_COUNT` | 用药总数 / 处方药数量 |
| `ANTI*` / `SSRI*` / `INSULIN` 等 200+ 列 | 按药理分类的是否用药指示变量（0/1） |
| `ALTMEDS_*` / `ALTERNATIVEMEDS` | 替代医学/保健品（R1 为主） |

#### 社会经济与访谈管理

| 字段 | 说明 |
|------|------|
| `INCOME_*` / `IML*` / `HAML*` | 收入与资产 |
| `JOBSTAT_*` / `WORKPAY` / `HRSCJOB` | 就业状态 |
| `RELIGION` / `ATNDSERV` / `BELIEFS` | 宗教 |
| `CAREGIVER` / `CARE_*` | 照护经历 |
| `IWLOC*` / `STRUCTQ` / `BUILD` | 访谈环境与住房类型 |
| `RFHLTHR` / `RDESCR*` | 受访者自评健康对访谈的影响 |

> 完整变量列表与题项原文请查阅各轮 `Round*/ICPSR_*/DS0001/*-Codebook-ICPSR.pdf`，或运行 `list(df.columns)` / 读取 `value_labels_core.json` 获取全部编码。

---

## 社会网络模块

NSHAP 采用 **多名册 Roster** 设计：

| Roster | 内容 |
|--------|------|
| A | 倾诉对象 |
| B | 配偶/伴侣 |
| C | 其他重要联系人 |
| D | 家庭成员 |

`social_networks` 为 Alter 长表；R2/R3 另有 `social_networks_update` 追踪关系变化。  
与 GSS 1985 不同，NSHAP **不提供完整 Alter–Alter 边列表**，但含 Alter 间接触频率等变量，可构建加权矩阵。

### 对应 ego-centric 三表

| 层级 | 来源 |
|------|------|
| Ego | `core` |
| Alter | `social_networks` |
| Alter–Alter | 由网络文件中接触频率变量构建 |

---

## 数据转换说明

ICPSR 的 R 格式将分类变量存为 factor，level 形如 `(1) male`。

`export_rda.R` 转换规则：

1. 带 `(数字)` 前缀的 factor → **数值编码**（写入 CSV/Parquet）
2. 取值标签 → `value_labels_{dataset}.json`
3. 纯字符串 factor（如 `SU_ID`）→ 保留字符串
4. 同步生成 Parquet（Polars 读取更快）

---

## 脚本

| 脚本 | 作用 |
|------|------|
| [`scripts/export_rda.R`](scripts/export_rda.R) | 扫描 `Round*/ICPSR_*/*-Data.rda`，导出 CSV + 标签 JSON |
| [`scripts/csv_to_parquet.py`](scripts/csv_to_parquet.py) | 批量 CSV → Parquet |
| [`scripts/load_nshap.py`](scripts/load_nshap.py) | `load_round()` / `load_network_round()` / `load_value_labels()` |

---

## 引用

> Waite, Linda J., et al. National Social Life, Health, and Aging Project (NSHAP). Inter-university Consortium for Political and Social Research.  
> Round 1: https://doi.org/10.3886/ICPSR20541.v10  
> Round 2: https://doi.org/10.3886/ICPSR34921.v5  
> Round 3: https://doi.org/10.3886/ICPSR36873.v9

网络测量设计参考：Cornwell, B., & Waite, L. J. (2009). Social networks in the NSHAP study. *Journals of Gerontology*.
