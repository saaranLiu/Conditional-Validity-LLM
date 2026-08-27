# 子实验命名对照

本文件是臂名的唯一对照表。论文与报告正文用短名（`S0`、`N0`）；配置键、目录名、jsonl 的 `task` 字段用全名。

| 短名 | 配置键 / 目录名 | 旧名 |
|---|---|---|
| S0 | `s0_baseline_state` | Task1 |
| S0b | `s0b_anti_typology` | Task1b |
| S1 | `s1_health_conditioned` | Task2 |
| S1b | `s1b_anti_typology` | Task2b |
| S2 | `s2_politics_religion_conditioned` | Task3 |
| N0 | `n0_baseline_network` | B1 |
| N0b | `n0b_anti_typology` | B1b（未跑） |
| N1 | `n1_size_bucket` | B2 |
| N2 | `n2_population_prior` | B3 |
| N3 | `n3_roster_multistep` | B4 |
| N4 | `n4_homophily` / `n4_contact` | B5 |
| N4c | `n4c_closure` | B5c |
| N4s | `n4s_sex_homophily` | B5s |
| N4o | `n4o_same_origin` | B5o |
| N4r | `n4r_resides_spain` | B5r |
| N5 | `n5_size_multistep` | B6 |
| N6 | `n6_prior_multistep` | B7 |
| R0 | 等于 N0，不另生成 | C1 |
| R1 | `r1_llm_state_network` | C2 |
| T6 | `t6_country_resources` | ISSP 跨国资源画像 |

## 主 Testing（文件只存一份）

其它 Testing 用相对符号链接引用同一份 input/output。

| 臂 | UCNets | GSS 1987/2010 | GSS 1985/2004/2024、NSHAP r1、Signos |
|---|---|---|---|
| S0, S0b | T1 | T1 | 1985 / NSHAP r1 仅 S0（供 T4）；其余 gold 在 Survey |
| S1, S1b, S2 | T2 | T2 | （gold 在 Survey，未生成） |
| N0 | T1 | — | T3 |
| N1, N2 | T2 | — | — |
| N3, N5, N6 | T5 | — | — |
| N4 变体 | T3 | — | T3 |
| R1 | T4 | — | 1985、NSHAP r1 R1 在 T4 |

T6 链接 GSS 1985/2004 与 NSHAP r1 的 N0。ISSP 的生成臂是 `t6_country_resources`，主目录 `T6_cross_population/issp_2017/`。
