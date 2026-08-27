# Ego-Network Dataset Comparison

## Purpose

This document compares the datasets collected for the LLM social simulation project, focusing on three layers:

1. **Ego fields**: respondent-level attributes, states, and outcomes.
2. **Alter fields**: named or role-based network members and ego-alter relationship attributes.
3. **Alter-Alter fields**: ties, familiarity, contact, or inferred structure among alters within the same ego network.

The goal is to clarify which datasets can support the main UCNets-centered experiment and which datasets are better suited for stage-specific robustness checks.

---

## Quick Summary

| Dataset | Ego fields: categories and examples | Alter fields: categories and examples | Alter-Alter fields: categories and examples | Longitudinal | Role in Current RP |
|---|---|---|---|---:|---|
| **UCNets** | 人口学、家庭、健康、人格、政治/宗教、生活事件、权重；例：`PRIM_KEY`, `AGEGROUP`, `GENDER`, `K1`, `G1`, `H*`, `J1*`, `WEIGHT_W1_DEMO` | 具名 alter 槽位、关系类型、支持角色、接触/距离、部分 alter 属性；例：`ID_W1_W2_W3`, `NAME_NMBR2`, `N_GENDER`, `C1A_*`, `B4A`, `B7C`, `C2B_NCLOSE` | 嵌在 alter 宽表中的关系矩阵/接触结构；例：`N*_N*`, `C1A_*` related matrix fields | Yes, 3 waves | Main experiment |
| **GSS network modules + panels** | 截面核心变量 + panel 纵向问卷；例：`ego_id`, `age`, `sex`, `race`, `educ`, `partyid`, `health`, `happy`, panel `cohort_*` wave variables | 网络模块年份有 named discussants；例：`alter_id`, `sex`, `age`, `race`, `educ`, `relig`, `spouse`, `friend`, `cowork`, `talkto`, `rclose`, `kin` | 部分网络年份有 alter pair 熟悉程度；例：`alter_alter.csv`, `src_id`, `tgt_id`, `weight`; panel waves 通常不含 alter-alter | Yes for panel cohorts; network modules mostly cross-sectional | Network-stage + panel robustness |
| **NSHAP** | 老年人核心问卷、健康、孤独/隔离、家庭/伴侣、权重；例：`ID`, demographics, health modules, loneliness/isolation items, `WEIGHT_ADJ` | 名册 alter、关系类型、性别、同住、年龄/相对年龄、交谈频率、亲密度、健康讨论；例：`SECTION`, `LINENO`, `RELAT2`, `NODE_GENDER`, `LIVEWITH`, `NODE_AGE`, `TALKFREQ0`, `HOWCLOSE`, `HEALTHTALK` | alter 间交谈频率矩阵；例：`TALKFREQ1`-`TALKFREQ7`; 另有 R1-R2 更新：`social_networks_update`, `LINENO_W1`, `LINENO_W2` | Yes, 3 rounds | Response/Dynamics robustness |
| **Signos** | 移民 ego 背景、家庭、教育、职业、居留、汇款；例：`ego_id`, `community`, `sex`, `age`, `country`, `religion`, `education`, `years_residence`, `remittances` | 固定 30 个 alter、性别、亲近度、来源国/居住地、confidant、关系类型、见面地点；例：`alter_id`, `sex`, `closeness`, `alter_country`, `alter_residence_city`, `confidant`, `relation_type` | 完整 30-alter pair 判断；例：`source_id`, `target_id`, `value`, `probably_knows` | No | Dense ego-network stress test |
| **Add Health / Freeman** | 学校 whole-network 节点属性；例：community node ID, `sex`, `race`, `grade`, `school` | 友谊提名边，而非传统 alter 表；例：`source`, `target`, `weight` 1-6 based on joint activities | Whole-network 可诱导 alter-alter ties；例：给定 ego 的朋友 `j,k`，若 `j -> k` 或 `k -> j` 有边，则存在 alter-alter tie | No | Whole-network topology robustness |
| **Add Health Public-Use** | 多波 ego 纵向问卷、健康、行为、家庭、学业、成年结果；例：AID, sex, race, grade, health, behavior, family, work/income/partner outcomes | 朋友槽位描述但无可链接 alter AID；例：`H1MF*`, `H1FF*` for friend gender slot, school, grade, activities | 无 alter-alter；不能还原 named network | Yes, multiple waves | Not sufficient for full network experiment |
| **ISSP 2017** | 跨国 ego 背景、教育、就业、宗教/政治、健康/幸福、权重；例：`CASEID`, `c_alphan`, `WEIGHT`, `SEX`, `AGE`, `DEGREE`, `WORK`, `ISCO08`, `v54`, `v58` | 无具名 alter；只有角色/职业/求助来源；例：position generator `v1`-`v10`, help sources `v21`-`v30`, role contact `v48`-`v52`, spouse work `SPWORK` | 无 alter-alter；只能做 role-based / macro social resources | No | Macro role/support robustness |

Recommended use:

```text
UCNets = main experiment
GSS = ego-network structural robustness
NSHAP = health, loneliness, support, and network-change robustness
ISSP = cross-national role/support robustness
Add Health/Freeman = school whole-network robustness
Signos = dense personal-network stress test
```

---

## 1. UCNets

### Position in the Project

UCNets is the best dataset for the main experiment because it contains:

- The same egos followed across three waves.
- Rich ego-level state and outcome variables.
- Alter-level long tables.
- Some alter continuity across waves.
- Alter-alter information embedded in the alter table.

### File Structure

| Layer | File | Granularity | Keys |
|---|---|---|---|
| Ego | `processed/by_wave/{Wave}/ego.csv` | One row per respondent | `PRIM_KEY` |
| Alter | `processed/by_wave/{Wave}/alters.csv` | One row per alter slot | `PRIM_KEY`, `ID_W1_W2_W3`, `NAME_NMBR2` |
| Alter-Alter | Embedded in `alters` wide variables | Within-ego alter relationship matrix | `PRIM_KEY` + alter number |

### Ego Fields

Wave 1 ego has about 555 variables.

| Module | Prefix | Content | RP Use |
|---|---|---|---|
| Screening/interview | `SC`, `S` | Device, interview mode, location, duration | Controls |
| Life course | `A` | Marriage, cohabitation, children, parents, household changes | State |
| Household | `B` | Family meals, difficult alter handling | State / outcome |
| Work/support | `E` | Help availability from kin/friends, network satisfaction | State / outcome |
| Life events | `F` | Work, finances, relationships, deaths | Dynamics covariates |
| Health | `G` | Self-rated health, chronic illness, ADL, depression/anxiety | State / outcome |
| Personality/well-being | `H`, `J` | Happiness, loneliness, PHQ, Big Five | State / outcome |
| Internet/contact | `I` | Internet use, family/friend contact | Response/Dynamics |
| Background | `K` | Education, religion, politics, ethnicity, spouse info | State |
| Weights/admin | `WEIGHT_*`, `MODE` | Survey weights, mode, tracking indicators | Inference |

Key ego categories:

- `PRIM_KEY`: ego ID across waves.
- `WAVE1`, `WAVE2`, `WAVE3`: wave participation.
- `AGEGROUP`, `AGE`, `GENDER`: demographics.
- `K1`, `K2`, `K10A`, `K10B`, `K15`, `K16`: education, politics, religion.
- `G1`, `G2*`, `G10*`: health and functional limitations.
- `H*`, `J1*`: loneliness, happiness, personality.
- `WEIGHT_W1_DEMO`, `WEIGHT_W2_DEMO`, `WEIGHT_W3_DEMO`: weights.

### Alter Fields

Wave 1 alter table has about 20,313 rows and 71 columns. Each row is an alter slot.

| Field Group | Representative Fields | Meaning | RP Use |
|---|---|---|---|
| IDs | `PRIM_KEY`, `ID_W1_W2_W3`, `NAME_NMBR2`, `NAME_SUBNMBR` | Ego-alter link, alter continuity, order | Network construction, retention |
| Interview/admin | `MODE`, `MERGED_NAME` | Interview mode, duplicate-name merge | Data quality |
| Alter attributes | `N_GENDER`, `C1B_NSAMESEX`, `C1C_NSAMEAGE`, `C1C_NOLDER` | Gender and relative age | Composition / homophily |
| Roster A | `A2B`, `A3B`, `A12`, `A12X` | Discussion partner, spouse/partner, household member | Discussion network |
| Roster B | `B2A`, `B4A`, `B5A`, `B6B`, `B7C`, `B8A`, `B9A`, `G10D` | Companion, confidant, advisor, helpers, difficult alter | Support network |
| Roster C | `C1A_*`, `C2*` | Activity partner, role, closeness, distance, same religion/race, political difference | Composition / diversity |
| Roster D | `D1A`, `D1B` | Other important contact, how met, duration known | Extended network |

Important relationship role fields:

| Field | Meaning |
|---|---|
| `C1A_1` | Spouse/partner |
| `C1A_2` | Romantic partner |
| `C1A_3` | Parent |
| `C1A_4` | Child |
| `C1A_5` | Sibling |
| `C1A_6` | Other relative |
| `C1A_10` | Roommate |
| `C1A_11` | Neighbor |
| `C1A_12` | Work contact |
| `C1A_13` | School contact |
| `C1A_14` | Church contact |
| `C1A_20` | Friend |
| `C1A_21` | Acquaintance |
| `C1A_22` | Other |

### Alter-Alter Fields

UCNets does not provide a separate `alter_alter.csv`, but alter-alter structure is embedded in the alter table.

| Type | Field Pattern | Meaning |
|---|---|---|
| Alter matrix fields | `N*_N*` | Alter-alter matrix-related fields |
| Relationship/contact fields | `C1A_*` and related wide fields | Within-network relation/contact structure, to be interpreted with the codebook |

### Suitable Estimands

- Network size.
- Network composition.
- Alter retention.
- Support availability.
- Contact patterns.
- Role diversity.
- Loneliness transition.
- Health transition.
- Social satisfaction transition.

### Limitations

- Regional public-use sample, not nationally representative.
- Alter-alter structure requires parsing embedded wide variables.
- Some alter attributes are relative to ego rather than absolute attributes.

---

## 2. GSS Ego-Network and Panel Data

### Position in the Project

GSS should not be limited to 1985 and 2004. In this project it has two complementary uses:

1. **Network-module years** support ego-network robustness. The strongest alter-alter years are 1985 and 2004, but other network years also provide ego/alter information with different field coverage.
2. **GSS panel cohorts** support longitudinal robustness for response patterns, attitude changes, health/happiness transitions, and other repeated survey outcomes, even when detailed ego-network tables are not available.

### File Structure

| Layer | File | Granularity | Keys |
|---|---|---|---|
| Ego | `GSS/network/{year}/ego.csv` | Network-module respondents (full survey) | `ego_id` |
| Alter | `alter.csv` | One row per named alter | `ego_id`, `alter_id` |
| Alter-Alter | `alter_alter.csv` | One row per alter pair | `ego_id`, `src_id`, `tgt_id` |
| Panel survey | `Survey/gss/processed/panel/cohort_{year}/wave_*/survey.csv` | One row per panel respondent-wave | panel respondent ID |

### Ego Fields

| Field Group | Representative Fields | Meaning | RP Use |
|---|---|---|---|
| IDs | `ego_id`, `id`, `year` | Respondent and year | Linking |
| Demographics | `age`, `sex`, `race`, `educ`, `degree`, `marital`, `relig`, `partyid`, `region` | Baseline state | State |
| Outcomes | `happy`, `health` | Happiness and self-rated health | Response robustness |
| Network module | `numgiven`, `in_network_module`, `has_alters`, `numgiven_capped`, `numgiven_is_topcoded` | Number of named discussants | Network size |
| Design | `wtssall`, `vpsu`, `vstrat`, `ballot`, `form` | Weights and survey design | Inference |

Panel cohorts add repeated respondent-level survey variables across waves. They are useful for testing whether LLM-generated responses reproduce longitudinal changes in attitudes, health, happiness, political/religious views, and other repeated GSS items. Their role is closer to **Response/Dynamics robustness** than ego-network topology unless a given panel wave also contains relevant network measures.

### Alter Fields

| Field Group | Representative Fields | Meaning |
|---|---|---|
| IDs | `ego_id`, `alter_id` | Ego-alter link |
| Demographics | `sex`, `age`, `race`, `educ`, `relig` | Alter demographics |
| Relationship type | `spouse`, `parent`, `sibling`, `child`, `othfam`, `cowork`, `memgrp`, `neighbr`, `friend`, `advisor`, `other` | Multiple relationship roles |
| Interaction | `talkto`, `known`, `rclose` | Talk frequency, duration known, closeness |
| Derived | `kin` | Whether alter is kin |

Availability by year:

| Field | 1985 | 1987 | 2004 | 2010 | 2024 |
|---|---:|---:|---:|---:|---:|
| `sex`, `age`, `educ`, `relig` | Yes | No | Yes | Partial | No |
| `race` | Yes | No | Yes | Yes | Yes |
| Relationship roles | Yes | Yes | Yes | Yes | Yes |
| `talkto` | Yes | No | Yes | No | Yes |
| `known`, `rclose` | Yes | Partial | No | No | No |
| `kin` | Yes | Yes | Yes | Yes | Yes |

### Alter-Alter Fields

Only 1985 and 2004 provide complete `alter_alter.csv`.

| Field | Meaning |
|---|---|
| `ego_id` | Ego ID |
| `src_id` | Source alter |
| `tgt_id` | Target alter |
| `weight` | Familiarity between alters |

`weight` coding:

| Value | Meaning |
|---:|---|
| `1` | Especially close |
| `2` | Know each other |
| `3` | Total strangers |

### Suitable Estimands

- Network size / `numgiven`.
- Kin share.
- Friend share.
- Coworker share.
- Ego-alter homophily.
- Alter-alter closure.
- Network diversity.
- Panel response changes in repeated GSS items.
- Longitudinal shifts in health, happiness, attitudes, political/religious variables, where repeated in panel cohorts.

### Limitations

- Ego-network modules are mostly cross-sectional and small, usually up to 5 alters.
- Alter fields differ across network years.
- Complete alter-alter closure is only available in some network years, especially 1985 and 2004.
- GSS panel cohorts support longitudinal response analysis, but they do not generally provide full ego/alter/alter-alter network tables.

---

## 3. NSHAP

### Position in the Project

NSHAP is best for **Response/Dynamics robustness** in an older population, especially health, loneliness, support, and network change.

### File Structure

| Layer | File | Granularity | Keys |
|---|---|---|---|
| Ego | `processed/by_round/{Round}/core.csv` | One row per respondent | `ID` |
| Alter | `social_networks.csv` | One row per roster alter | `ID`, `LINENO` |
| Alter-Alter | In `social_networks.csv` via `TALKFREQ1`-`TALKFREQ7` | Alter-to-alter talk frequency | `ID`, `LINENO` |
| Network update | `social_networks_update.csv` | R1 to R2 alter tracking | `ID`, `LINENO_W1`, `LINENO_W2` |

### Ego Fields

| Field Group | Content | RP Use |
|---|---|---|
| Demographics | Gender, age, education, marital status, household, race | State |
| Health | Self-rated health, chronic conditions, functional limitations, mental health | Health outcome |
| Loneliness/isolation | Loneliness, left out, isolated, companionship | Well-being outcome |
| Partner/family | Spouse, family structure, caregiving | State / network context |
| Weights | `WEIGHT_ADJ` and related fields | Inference |

### Alter Fields

| Field | Meaning |
|---|---|
| `ID` | Ego ID |
| `SECTION` | Roster: 1=discussion, 2=spouse/partner, 3=other important contact, 4=family |
| `LINENO` | Alter number within ego |
| `ANYMORE` | Whether more people exist in the roster |
| `RELAT` / `RELAT2` | Relationship type |
| `NODE_GENDER` | Alter gender |
| `LIVEWITH` | Whether alter lives with ego |
| `NODE_AGE` | Alter age, often more complete for co-residents |
| `NODE_AGEDK` | Relative age |
| `NODE_LOCAL` | Whether alter is local, Round 3 only |
| `TALKFREQ0` | Ego-alter talk frequency |
| `HOWCLOSE` | Emotional closeness, R1/R2 |
| `HEALTHTALK` | Likelihood of discussing health, R1/R2 |
| `SPCURRENT` | Current spouse, R1/R2 |
| `ROSTER_W1`, `LINENO_W1` | Link to Round 1 alter |
| `LONGKNOW` | Duration known, R2/R3 |

### Alter-Alter Fields

| Field | Meaning |
|---|---|
| `TALKFREQ1`-`TALKFREQ7` | Current alter's talk frequency with other alters in the same ego roster |

`TALKFREQ*` is coded from 0 to 8, from never to daily.

### Suitable Estimands

- Older adult network size and change.
- Kin/spouse/friend support structure.
- Ego-alter talk frequency.
- Alter-alter talk frequency.
- Health transition.
- Loneliness and social isolation.
- Support availability.

### Limitations

- Older population.
- Some alter attributes are missing, especially for non-co-residents.
- Round 3 has fewer alter variables than R1/R2.

---

## 4. Signos

### Position in the Project

Signos is a small but structurally rich dense ego-network dataset. It is useful as a dense personal-network stress test.

### File Structure

| Layer | File | Granularity | Keys |
|---|---|---|---|
| Ego | `processed/ego.json` | One row per ego | `ego_id` |
| Alter | `processed/alter.json` | One row per alter | `ego_id`, `alter_id` |
| Alter-Alter | `processed/alter_alter.json` | One row per alter pair | `ego_id`, `source_id`, `target_id` |

Scale:

- 70 egos.
- 30 alters per ego.
- 435 alter-alter pairs per ego.
- 2,100 alter rows.
- 30,450 alter-alter rows.

### Ego Fields

| Field | Meaning |
|---|---|
| `ego_id` | Ego ID, e.g. `chinese0` |
| `community` | Migrant group: Chinese, Filipino, Sikh |
| `sex` | Gender |
| `age` | Age |
| `country` | Country/national origin |
| `religion` | Religion |
| `migration_reason` | Reason for migration |
| `father_country`, `mother_country` | Parents' countries of birth |
| `marital_status` | Marital status |
| `num_children` | Number of children |
| `employment_status` | Employment |
| `legal_status` | Legal residence status |
| `occupation` | Occupation |
| `education` | Education |
| `years_residence` | Years in Spain |
| `remittances` | Whether ego sends remittances |
| `family_in_origin` | Family in country of origin |

### Alter Fields

| Field | Meaning |
|---|---|
| `alter_id` | Alter number, 0-29 |
| `sex` | Alter gender |
| `closeness` | Closeness to ego, 0-4 |
| `alter_country` | Alter origin country |
| `alter_residence_country` | Alter residence country |
| `alter_residence_city` | Alter residence city |
| `confidant` | Whether ego confides in alter |
| `relation_type` | Relationship type |
| `meeting_place` | Usual meeting place |

Signos does not include alter age.

### Alter-Alter Fields

| Field | Meaning |
|---|---|
| `ego_id` | Ego ID |
| `source_id`, `target_id` | Alter pair |
| `value` | 0=unlikely to know, 1=possibly know, 2=very likely know |
| `label` | Original response label |
| `probably_knows` | Derived true when `value == 2` |

### Suitable Estimands

- Dense network closure.
- Alter composition.
- Migrant community differences.
- Alter-alter density.
- Relationship-type diversity.
- Origin/residence-country diversity.

### Limitations

- Small sample.
- Specialized migrant population.
- Better as a stress test than as a main inference dataset.

---

## 5. Add Health / Freeman

### Position in the Project

Freeman Add Health is a school whole-network dataset. It is not an ego/alter/alter-alter survey table in the usual sense; each student can be treated as both ego and alter in a whole graph.

### File Structure

| Layer | File | Granularity | Keys |
|---|---|---|---|
| Node / Ego | `raw/comm*_att.dat` | One row per student node | Community-level node ID |
| Edge / Alter relation | `raw/comm*.dat` | One row per friendship nomination | Source node, target node |
| Alter-Alter | Implied by whole network | Any pair of nodes can be checked for ties | Node IDs |

### Ego / Node Fields

| Field | Meaning |
|---|---|
| Node ID | Community-level integer ID, not public-use AID |
| `sex` | 1=male, 2=female, 0=unreported |
| `race` | 1=white, 2=black, 3=hispanic, 4=asian, 5=mixed/other, 0=unreported |
| `grade` | 7-12, 0=unreported |
| `school` | School code in two-school communities |

### Alter / Edge Fields

| Field | Meaning |
|---|---|
| `source` | Nominating student |
| `target` | Nominated friend |
| `weight` | Friendship strength, 1-6 |

The edge weight is based on five joint activities, such as visiting, hanging out after school, weekend time, talking about problems, and phone calls.

### Alter-Alter Fields

There is no ego-centric alter-alter table, but the whole network allows induced alter-alter structure:

```text
For ego i with friends {j, k, l},
if j -> k or k -> j exists, then j and k have an alter-alter tie.
```

### Suitable Estimands

- Degree distribution.
- In-degree and out-degree.
- Reciprocity.
- Race/sex/grade homophily.
- Clustering and transitivity.
- Component size.
- School-level topology.

### Limitations

- Node IDs do not link to Add Health public-use in-home AID.
- Node attributes are limited to sex, race, grade, and school.
- Not suitable for a full state-to-health/attitude outcome experiment.

---

## 6. Add Health Public-Use

### Position in the Project

Add Health public-use data contain rich longitudinal ego survey records but cannot reconstruct a complete friendship network.

### Ego Fields

| Wave | File | Ego Content |
|---|---|---|
| Wave I | `w1inhome_dvn.sav` | Adolescent demographics, health, behavior, school, family |
| Wave II | `w2inhome_dvn.sav` | Follow-up adolescent health, behavior, relationships, school |
| Wave III | `w3inhome_dvn.sav` | Young adult education, work, health, relationships |
| Wave IV | `w4inhome_dvn.sav` | Early adult work, income, health, relationships |
| Wave V/VI | `pwave5`, `pwave6` | Adult questionnaires and health/biomarkers |

Available ego fields include:

- Sex, race, grade, age.
- Health, behavior, psychological symptoms.
- School performance.
- Family background.
- Adult work, income, partner, and health outcomes.

### Alter Fields

Public-use files include friend-slot descriptions but no linkable alter AID.

| Field Group | Meaning |
|---|---|
| `H1MF*` | Up to 5 male friends' school, grade, joint activities |
| `H1FF*` | Up to 5 female friends' school, grade, joint activities |
| Wave II friend slots | Similar to Wave I |

Available alter information:

- Alter gender is implied by slot type.
- Alter school.
- Alter grade.
- Joint activities with each friend.

Unavailable:

- Alter AID.
- Alter race.
- Full alter survey attributes.
- Alter-alter relations.

### Alter-Alter Fields

None.

### Suitable Estimands

- Ego-level longitudinal health/behavior/life-course outcomes.
- Wave I network summary proxies.
- Friend-slot gender/grade/activity patterns.

### Limitations

- Cannot reconstruct `ego_AID -> alter_AID` edges.
- Cannot build whole-network structure.
- Cannot attach full survey attributes to alters.
- Not suitable as the full ego-network main dataset.

---

## 7. ISSP 2017

### Position in the Project

ISSP 2017 is a cross-national role-based social resources dataset, not a name-generator ego-network dataset.

### Ego Fields

| Field Group | Representative Variables | Meaning |
|---|---|---|
| ID/design | `CASEID`, `country`, `c_alphan`, `WEIGHT`, `MODE` | Case, country, weight, mode |
| Demographics | `SEX`, `AGE`, `BIRTH`, `MARITAL`, `PARTLIV`, `HOMPOP` | Sex, age, marital/household status |
| Education | `EDUCYRS`, `DEGREE`, `*_DEGR` | International and country-specific education |
| Work | `WORK`, `WRKHRS`, `EMPREL`, `ISCO08`, `MAINSTAT` | Work status and occupation |
| Politics/religion | `RELIGGRP`, `ATTEND`, `VOTE_LE`, `PARTY_LR` | Religion and politics |
| Income/class | `TOPBOT`, `*_RINC`, `*_INC` | Subjective class, personal/household income |
| Health/well-being | `v54`, `v55`, `v58` | Self-rated health, depression, life satisfaction |

### Alter Fields

ISSP has no named alter table. It measures roles, occupational positions, and help sources.

| Question Block | Variables | Content | Approximate Interpretation |
|---|---|---|---|
| Position generator | `v1`-`v10` | Whether ego knows someone in 10 occupations; closest relation type | Role/occupation access |
| Daily/emotional help | `v21`-`v25` | First source of help for household tasks, illness, depression, family advice, social occasion | Support role |
| Instrumental help | `v26`-`v30` | Help source for money, job, administration, housing, serious illness care | Institutional vs informal support |
| Role contact | `v48`-`v52`, `v65`-`v66` | Contact frequency with parent, sibling, adult child, relatives, close friend, hometown person, religious community | Role-anchored contact |
| Partner/spouse | `v67`, `SPWORK`, `SPWRKHRS`, `SPISCO08`, `SPMAINST` | Spouse education and work | Spouse attributes |

### Alter-Alter Fields

None.

ISSP does not provide:

- Named alters.
- Alter-level gender, age, education.
- Alter-alter edges.
- Ego-network topology.

### Suitable Estimands

- Country-level social resource access.
- Family/friend/institution help shares.
- Occupational reachability.
- Contact frequency by role.
- Social trust, loneliness, health, life satisfaction.

### Limitations

- Not suitable for micro-level ego-network topology validation.
- Best used for role-based or macro response robustness.

---

## Cross-Dataset Capability Matrix

| Capability | UCNets | GSS | NSHAP | Signos | Add Health/Freeman | Add Health Public | ISSP |
|---|---|---|---|---|---|---|---|
| Ego ID | `PRIM_KEY` | `ego_id` | `ID` | `ego_id` | community node ID | AID | `CASEID` |
| Ego demographics | age group, gender, education, family, religion/politics | age, sex, race, education, marital, party/religion | age, gender, education, marital/family | age, sex, country, education, migration | sex, race, grade, school | sex, race, grade, age, family | sex, age, education, work, country |
| Ego health/well-being | self-rated health, loneliness, PHQ, social satisfaction | `health`, `happy` | health, loneliness, isolation, mental health | limited background only | not measured | health, behavior, mental health across waves | `v54`, `v55`, `v58` |
| Ego longitudinal tracking | Wave 1-3 via `PRIM_KEY` | panel cohorts 2006/2008/2010/2016 | Round 1-3 via `ID` | no | no | Wave I-VI via AID | no |
| Alter records | `alters.csv` long table | `alter.csv` named discussants | `social_networks.csv` roster alters | `alter.json`, 30 alters per ego | edge list can define alters | friend slots only, no linkable alter ID | role/position items only |
| Alter ID | `ID_W1_W2_W3`, `NAME_NMBR2` | `alter_id` within ego | `LINENO` within ego | `alter_id` 0-29 | node ID | no linkable ID | no named alter ID |
| Alter demographics | gender, same sex/age/race/religion indicators | sex, age, race, education, religion best covered in 1985/2004; partial in other network years | gender, age/relative age, local status | sex, origin/residence country | sex, race, grade, school | gender slot, school, grade | no alter demographics |
| Alter relationship type | `C1A_*`, roster A/B/C/D roles | spouse, parent, sibling, friend, coworker, neighbor, etc. | `RELAT` / `RELAT2`, roster section | `relation_type`, confidant | friendship nomination | friend slot only | role categories: family, friend, institution, occupation |
| Ego-alter tie strength | closeness, support role, contact/distance fields | `talkto`, `known`, `rclose` | `TALKFREQ0`, `HOWCLOSE`, `HEALTHTALK` | `closeness` | edge `weight` 1-6 | joint activity items | contact frequency by role |
| Alter-alter ties | embedded matrix fields in alter table | `alter_alter.csv`, `close12`-`close45` | `TALKFREQ1`-`TALKFREQ7` | `alter_alter.json`, `probably_knows` | induced from whole network | none | none |
| Whole-network | no | no | no | no | yes | no | no |
| Best use | main experiment | network robustness + panel response robustness | dynamics/support robustness | dense network stress test | whole-network topology | ego longitudinal outcomes only | macro role/support robustness |

---

## Recommended Use in the RP

### Main Experiment

**UCNets**

Used for the full pipeline:

```text
State_human^1 + Network_human^1 -> Y_human^2 / Y_human^3
State_LLM^1   + Network_LLM^1   -> Y_LLM^2   / Y_LLM^3
```

Core estimands:

- Network size change.
- Alter retention.
- Relationship-role composition.
- Support availability.
- Loneliness.
- Social satisfaction.
- Self-rated health.

### Stage-Specific Robustness

| Stage | Recommended Dataset | What It Tests |
|---|---|---|
| State | GSS, ISSP, NSHAP | Whether generated/predicted state distributions are biased |
| Network | GSS network modules, Add Health/Freeman, Signos | Network size, composition, homophily, closure, degree distribution |
| Response/Dynamics | GSS panel cohorts, NSHAP, ISSP | Panel response changes, health/loneliness/support transitions, or role-based responses |
| Dense network stress | Signos | 30-alter closure and diversity |
| Whole-network topology | Add Health/Freeman | School friendship degree, reciprocity, clustering, homophily |

### Not Recommended

- Do not use Add Health Public-Use as the full network main dataset; it lacks linkable alter AID.
- Do not use ISSP as ego-network topology data; it has no named alters or alter-alter ties.
- Do not use Signos as the main inference dataset; it is small and specialized.
- Do not use GSS as the main dynamics dataset; its key network years are cross-sectional.

---

## Minimal Data Strategy

For a minimal paper:

1. **Main data: UCNets**
   - Run the full State, Network, Response/Dynamics, and Collective Estimand pipeline.
2. **One robustness dataset: GSS or NSHAP**
   - Choose GSS network modules if the emphasis is network distortion.
   - Choose GSS panel cohorts if the emphasis is repeated survey response dynamics.
   - Choose NSHAP if the emphasis is health/loneliness/support transitions.
3. **Other datasets**
   - Move to appendix or future work.

 