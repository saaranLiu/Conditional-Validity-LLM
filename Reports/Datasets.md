# Ego-centric Network Data with Alter Attributes 数据集汇总

> **目标数据结构**（社会网络分析标准三层格式）：
>
> | 层级 | 内容 | 典型 ID 字段 |
> |------|------|-------------|
> | **Ego 表** | 受访者人口统计特征、态度/健康等结局变量 | `.egoID` |
> | **Alter 表** | 网络成员人口统计特征 + Ego–Alter 关系属性 | `.egoID` + `.altID` |
> | **Alter–Alter 表** | 网络成员之间的连边（结构信息） | `.egoID` + `.srcID` + `.tgtID` |
>
> 在 R 的 `egor` 包中，三层数据通过 `.egoID` 链接，可用 `egor(egos=..., alters=..., aaties=...)` 构建分析对象。

---

## 匹配度分级说明

| 等级 | 含义 |
|------|------|
| ★★★ | **完全匹配**：Ego 人口统计 + Alter 人口统计 + Alter–Alter 网络结构 |
| ★★☆ | **基本匹配**：Ego + Alter 人口统计 + 部分结构信息（如密度、接触频率矩阵） |
| ★☆☆ | **部分匹配**：Ego 人口统计 + 定位法/聚合指标，Alter 仅有职业/关系类型等有限属性 |
| ☆☆☆ | **不完全匹配**：仅有网络结构或仅有 Ego 层，不满足 alter attributes 需求 |

---

## 一、国际公开调查数据

### 1. GSS 1985 Ego Network Module ★★★

| 项目 | 内容 |
|------|------|
| **全称** | General Social Survey, 1985 — Social Networks Topical Module |
| **地区/人群** | 美国，18+ 成年人，全国代表性截面样本 |
| **样本量** | Ego：约 1,531 人（回答网络模块者）；Alter：约 4,483 条记录 |
| **网络界定** | "重要事宜讨论网"（important matters name generator）：过去 6 个月内与受访者讨论重要问题的人 |
| **Ego 人口统计** | 年龄、性别、种族、教育、宗教、政党、幸福感、健康、收入权重等 |
| **Alter 人口统计** | 年龄、性别、种族、教育、宗教、是否亲属/配偶、接触频率等 |
| **网络结构** | **完整 Alter–Alter 边列表**（是否认识、是否特别亲密），可计算密度、结构洞等 |
| **纵向** | 否（截面） |
| **获取方式** | ① 预处理三表 CSV（可直接用于 `egor`）：[ego](https://github.com/JeffreyAlanSmith/Integrated_Network_Science/raw/master/data/gss1985_ego_dat.csv) / [alter](https://github.com/JeffreyAlanSmith/Integrated_Network_Science/raw/master/data/gss1985_alter_dat.csv) / [alter-alter](https://github.com/JeffreyAlanSmith/Integrated_Network_Science/raw/master/data/gss1985_alteralter_dat.csv)；② 原始数据：[ICPSR 35314](https://www.icpsr.umich.edu/web/ICPSR/studies/35314)；③ [GSS SDA](https://sda.berkeley.edu/) |
| **备注** | 社会网络分析教学与实证研究的**经典基准数据集**；每名 Ego 最多报告 5 个 Alter 的详细属性 |

---

### 2. GSS 2004 / 2010 Ego Network Module ★★☆

| 项目 | 内容 |
|------|------|
| **全称** | General Social Survey — Social Networks Module (2004, 2010) |
| **地区/人群** | 美国成年人，全国代表性截面样本 |
| **样本量** | 2004 约 1,467 人（网络模块）；2010 含实验子样本 |
| **网络界定** | 与 1985 相同的 "important matters" 提名法 |
| **Ego 人口统计** | GSS 核心问卷全套人口统计与社会经济变量 |
| **Alter 人口统计** | 性别、种族、教育、年龄、关系类型、接触频率、认识时长等 |
| **网络结构** | 有 Alter 间是否认识的**汇总/矩阵式**信息，但**不如 1985 提供完整边列表** |
| **纵向** | 否 |
| **获取方式** | [GSS 官网](https://gss.norc.org/) / [ICPSR](https://www.icpsr.umich.edu/) / [SDA Berkeley](https://sda.berkeley.edu/) |
| **备注** | McPherson et al. (2006) 用 2004 数据与 1985 对比，发现核心讨论网规模显著缩小；2010 含问卷位置实验 |

---

### 3. NSHAP (National Social Life, Health, and Aging Project) ★★☆

| 项目 | 内容 |
|------|------|
| **全称** | National Social Life, Health, and Aging Project |
| **地区/人群** | 美国，57–85 岁（R1）社区居住老年人；R3 扩展至 1948–1965 出生队列 |
| **样本量** | R1：约 3,005 人；三轮累计 Alter 记录约 38,890 条 |
| **网络界定** | 多名册（Roster A：倾诉对象；B：配偶/伴侣；C：其他重要联系人；D：家庭成员） |
| **Ego 人口统计** | 性别、年龄、教育、种族/族裔、婚姻、健康、认知、生物测量等 |
| **Alter 人口统计** | 性别、关系类型（亲属/朋友等）、是否同住、年龄（同住者）、接触频率、情感亲密度、是否讨论健康问题 |
| **网络结构** | Alter 之间的接触频率（可构建加权 Alter–Alter 矩阵）；网络变化追踪（纵向） |
| **纵向** | **是**（R1: 2005–06；R2: 2010–11；R3: 2015–16；含 COVID-19 子研究） |
| **获取方式** | [ICPSR NAHDAP Series 706](https://www.icpsr.umich.edu/web/NAHDAP/series/706/studies)（需注册申请） |
| **备注** | 老年社会网络与健康研究的旗舰数据；含配偶/伴侣追踪文件 |

---

### 4. UCNets (UC Berkeley Social Networks Study) ★★☆

| 项目 | 内容 |
|------|------|
| **全称** | UC Berkeley Social Networks Study, San Francisco Bay Area, 2015–2018 |
| **地区/人群** | 美国旧金山湾区 6 县；21–30 岁 & 50–70 岁两个队列 |
| **样本量** | Wave 1 起约数千人（地址抽样，累积响应率较低） |
| **网络界定** | 个人网络提名 + 解释器（讨论、支持、活动等多维度） |
| **Ego 人口统计** | 年龄、性别、教育、种族、就业、健康、人格、家庭里程碑等 |
| **Alter 人口统计** | 性别、是否亲属、是否同种族/同年龄段、关系类型、接触频率等 |
| **网络结构** | Alter–Alter 关系数据（具体变量见 Social Networks 数据文件） |
| **纵向** | **是**（3 波：2015–16 / 2017 / 2018） |
| **获取方式** | [ICPSR 36975](https://www.icpsr.umich.edu/web/NACDA/studies/36975)（Restricted/公开版需申请）；[UCNets 项目网站](https://ucnets.berkeley.edu/) |
| **备注** | `egor` 包教程常用数据（原始文件需申请）；关注生命历程转变如何改变网络 |

---

### 5. ISSP 2017 Social Networks ★☆☆

| 项目 | 内容 |
|------|------|
| **全称** | International Social Survey Programme: Social Networks and Social Resources 2017 |
| **地区/人群** | 30 国/地区整合文件（ZA6980 v2.0.0）；**含中国**（N=4,219）；爱沙尼亚、土耳其单独发布、不在整合文件中 |
| **样本量** | 整合文件 **44,492** 人 × **356** 变量（各国 1,002–4,219 不等） |
| **网络界定** | **定位法**（10 职业 × 关系类型）+ **情境求助**（5 情感 + 5 instrumental 场景）+ **角色锚定接触频率**（父母/兄弟姐妹/密友等）；**无 name generator** |
| **Ego 人口统计** | 标准 ISSP 背景变量（性别、年龄、教育、职业、收入等）；配偶有部分就业/学历变量 |
| **Alter 人口统计** | **无** alter-level 性别/年龄/教育；仅有关系**类型**（家人/密友/同事等）、求助**角色**、各**职业槽位**可达性、按**社会角色**的接触频率 |
| **网络结构** | 无 alter 提名列表、无完整 Alter–Alter 边列表 |
| **纵向** | 否 |
| **获取方式** | [GESIS ZA6980](https://www.gesis.org/en/issp/data-and-documentation/social-networks/2017)（注册下载）；本地已处理见 [`Survey/issp/README.md`](../Survey/issp/README.md) |
| **备注** | 2017 模块**有意放弃** 1986/2001 的 name generator，改用定位法以利跨国比较；适合社会资本/求助结构分析，**不满足** alter-level 完整人口统计或 `egor` 三表构建 |

---

### 6. LASA (Longitudinal Aging Study Amsterdam) ★★☆

| 项目 | 内容 |
|------|------|
| **全称** | Longitudinal Aging Study Amsterdam — Personal Network Module |
| **地区/人群** | 荷兰，55+ 老年人（1992 起持续追踪） |
| **样本量** | 基线约 3,107 人；个人网络成员无上限（实际 < 80） |
| **网络界定** | 7 个领域提名（家庭成员、子女、其他亲属、邻居、同事、组织成员、其他） |
| **Ego 人口统计** | 完整老年学队列背景变量 |
| **Alter 人口统计** | 所有成员：关系类型、性别、接触频率；**联系最密切的 12 人**：年龄、婚姻状况、支持交换、负面关系等 |
| **网络结构** | 领域式网络，非完整两两关系矩阵 |
| **纵向** | **是**（多波，1992–至今） |
| **获取方式** | [LASA 数据申请](https://lasa-vu.nl/en/request-data/)（签署协议） |
| **备注** | 欧洲老年个人网络研究经典数据；文件号 047/247 系列 |

---

### 7. Add Health (National Longitudinal Study of Adolescent to Adult Health) ★★☆

| 项目 | 内容 |
|------|------|
| **全称** | National Longitudinal Study of Adolescent to Adult Health |
| **地区/人群** | 美国青少年，84 个学校社区，7–12 年级 |
| **样本量** | 约 90,000+ 学生；友谊提名网络 |
| **网络界定** | 校内友谊提名（send / receive / reciprocated） |
| **Ego 人口统计** | 性别、种族、年级、GPA、课外活动、健康行为等 |
| **Alter 人口统计** | 被提名朋友的性别、种族、年级（来自同一调查） |
| **网络结构** | **完整校内友谊网络**（可提取 Ego-centric 子图） |
| **纵向** | **是**（Wave I–V，1994–2018） |
| **获取方式** | [Add Health 官网](https://addhealth.cpc.unc.edu/)（合同申请）；预处理网络：[networkdata/addhealth](https://github.com/tillahoffmann/networkdata/tree/main/addhealth) |
| **备注** | 严格说是**学校内整体网**而非调查式提名，但 Alter 属性完整；适合青少年同辈网络研究 |

---

### 8. Signos / OPeN (Observatory of Personal Networks) ★★★

| 项目 | 内容 |
|------|------|
| **全称** | Signos — Personal Networks of Immigrants in Barcelona |
| **地区/人群** | 西班牙巴塞罗那，中国（21）、菲律宾（25）、印度锡克教（24）移民 |
| **样本量** | 70 个 EgoNet 访谈 |
| **网络界定** | 每个 Ego 提名最多 30 个 Alter |
| **Ego 人口统计** | 年龄、性别、移民背景、居住状况等 |
| **Alter 人口统计** | 来源国、居住国、年龄、与 Ego 的关系类型 |
| **网络结构** | **完整 Alter–Alter 无向边** |
| **纵向** | 否 |
| **获取方式** | [Signos_public_data.zip](https://visone.ethz.ch/wiki/index.php/Signos_(data))；更多数据见 [OPeN](https://www.open-personal-networks.org/) |
| **备注** | 跨文化移民网络研究；EgoNet 软件格式，可转 GraphML |

---

## 二、中国/东亚调查数据

### 9. CGSS 中国综合社会调查 ★☆☆

| 项目 | 内容 |
|------|------|
| **全称** | Chinese General Social Survey |
| **地区/人群** | 中国大陆，18+ 成年人 |
| **样本量** | 每年约 8,000–12,000 人 |
| **网络界定** | 历年模块不同：**定位法（拜年网/讨论网）**为主；2023 含 **EASS 社会网络提名法**模块 |
| **Ego 人口统计** | 性别、年龄、教育、职业、收入、户口、婚姻等 |
| **Alter 人口统计** | 定位法：仅记录各职业位置是否有人及关系类型；提名法（部分年份/EASS）：性别、年龄、教育、关系类型 |
| **网络结构** | 部分提名法模块含 Alter 间是否认识；无系统性完整边列表 |
| **纵向** | 重复截面（2003–至今） |
| **获取方式** | [CNSDA 中国学术调查数据资料库](https://www.cnsda.org/)（实名注册免费下载） |
| **备注** | 中国社会科学研究最主要数据来源；**2023 年 EASS 模块**是较接近目标结构的近期数据 |

---

### 10. CFPS 中国家庭追踪调查 ☆☆☆

| 项目 | 内容 |
|------|------|
| **全称** | China Family Panel Studies |
| **地区/人群** | 中国大陆 25 省，家庭层面追踪 |
| **样本量** | 基线 14,960 户 / 42,590 人 |
| **网络界定** | **非典型 ego-centric 提名法**；以家庭关系库（父母、子女、配偶）为主 |
| **Ego 人口统计** | 完整个人层面追踪变量 |
| **Alter 人口统计** | 家庭成员有完整属性；**非家庭成员无系统 alter-level 数据** |
| **网络结构** | 家庭内部关系可构建，但非个人社交网 |
| **纵向** | **是**（2010 起每两年） |
| **获取方式** | [北京大学 CFPS 数据平台](https://www.isss.pku.edu.cn/cfps/) |
| **备注** | **不推荐**作为 ego-centric network with alter attributes 的主数据源 |

---

### 11. 边燕杰等科技工作者/农民工调查（局部样本）★★☆

| 项目 | 内容 |
|------|------|
| **来源** | 边燕杰等（2017）"使用定位法测量个体社会网结构" |
| **地区/人群** | 科技工作者（全国多城市）；农民工（局部） |
| **样本量** | 科技工作者：合著网 8,888 人，讨论网 28,082 人；农民工：支持网 503 人 |
| **网络界定** | 合著网/支持网（提名法，最多 5 人）+ 讨论网/拜年网（定位法） |
| **Ego 人口统计** | 年龄、性别、教育、职业、收入等 |
| **Alter 人口统计** | 提名法：年龄、性别、职业等 |
| **网络结构** | 提名法含 Alter 间关系 |
| **纵向** | 否 |
| **获取方式** | 原始数据需联系作者；方法论文：[CASTED](http://en.casted.org.cn/upload/qikan/2017/vxXSNeWUe.pdf) |
| **备注** | 中国语境下提名法 vs 定位法效度比较的重要研究 |

---

## 三、教学示例 / 计算网络数据（非调查）

### 12. ideanet 内置示例数据 ★★★

| 项目 | 内容 |
|------|------|
| **来源** | R 包 `ideanet` 附带 vignette 数据 |
| **样本量** | 20 个 Ego |
| **Ego 人口统计** | 年龄、性别、种族/族裔、教育、政治倾向 |
| **Alter 人口统计** | 性别、种族/族裔；Ego–Alter 关系（亲属/朋友/其他） |
| **网络结构** | 完整 Alter–Alter 边 + 面对面/电话/短信接触频率 |
| **获取方式** | R: `data(egodata, package="ideanet")` |
| **备注** | 适合算法开发与方法测试，**非真实调查数据** |

---

### 13. SNAP Ego Networks (Facebook / Twitter / Google+) ★☆☆

| 项目 | 内容 |
|------|------|
| **来源** | Stanford SNAP |
| **样本量** | Facebook: 10 个 Ego 网（合计 4,039 节点）；Twitter: 973 个 Ego 网 |
| **Ego/Alter 属性** | 节点特征（profile metadata），**非标准人口统计调查变量** |
| **网络结构** | 完整 Ego-centric 子图（含 circles） |
| **获取方式** | [SNAP ego-Facebook](https://snap.stanford.edu/data/ego-Facebook.html) 等 |
| **备注** | 适合网络结构算法；**不适合**需要调查式人口统计的研究 |

---

## 四、快速选型建议

| 研究需求 | 推荐数据集 |
|----------|-----------|
| 方法开发 / 仿真验证 / 教学 | GSS 1985 预处理三表、ideanet 示例数据 |
| 美国成年人代表性样本 | GSS 1985（结构完整）或 GSS 2004（较新） |
| 老年健康 + 网络纵向变化 | NSHAP、LASA |
| 生命历程与网络动态 | UCNets |
| 跨国比较（社会资本/求助结构，非 alter 个体属性） | ISSP 2017 |
| 青少年同辈网络 | Add Health |
| 移民/跨文化网络 | Signos / OPeN |
| 中国代表性样本 | CGSS 2023（EASS 提名法模块）> 历年 CGSS 定位法 |
| 完整三层 + 公开免申请 | **GSS 1985 预处理 CSV**（首选） |

---

## 五、数据结构与处理工具

### 标准三表结构示例（GSS 1985）

```
ego 表:     CASEID | AGE | EDUC | RACE | SEX | ...
alter 表:   CASEID | ALTERID | AGE | EDUC | RACE | SEX | KIN | ...
aatie 表:   CASEID | ALTER1 | ALTER2 | Weight
```

### 常用 R 工具

| 包 | 用途 |
|----|------|
| [`egor`](https://cran.r-project.org/package=egor) | 三层 ego 数据管理、同质性/多样性指标 |
| [`ideanet`](https://cran.r-project.org/package=ideanet) | 从三表生成网络指标（密度、约束、核心讨论网大小等） |
| [`igraph`](https://igraph.org/) | 网络可视化与结构分析 |

```r
library(egor)
egonet <- egor(
  egos = ego_dat, alters = alter_dat, aaties = alteralter_dat,
  ID.vars = list(ego = "CASEID", alter = "ALTERID",
                 source = "ALTER1", target = "ALTER2")
)
```

---

## 六、关键参考文献

1. Marsden, P. V., & Campbell, K. E. (1984/1987). Measuring tie strength. *Social Forces* — GSS 1985 网络模块方法论基础
2. McPherson, M., Smith-Lovin, L., & Brashears, M. E. (2006). Social isolation in America. *American Sociological Review* — GSS 1985 vs 2004 比较
3. Marsden, P. V., & Fekete, M. D. (2019). Contributions of the GSS to egocentric network research. GSS Topical Report No. 45
4. Cornwell, B., & Waite, L. J. (2009). Social networks in the NSHAP study. *Journals of Gerontology* — NSHAP 网络测量设计
5. 边燕杰、张文宏等. 定位法与提名法效度比较. *社会* — 中国语境网络测量

---

*最后更新：2026-07-04*
