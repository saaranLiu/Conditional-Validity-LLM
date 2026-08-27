# 实验与数据集登记表

本文件是加入新 survey 时的**唯一入口**：先在这里登记 dataset id 与状态，再动脚本。目的是避免数据集变多后不知道哪个跑过、哪个用的哪套 schema。

---

## 1. 目录与命名约定

实验按 **Testing（判据）× dataset × 臂** 组织。原始调查与清洗 gold 在 `Survey/`。一条生成臂只落在它的主 Testing；其它 Testing 用相对符号链接引用，不要复制 jsonl。臂名对照见 [`Testing/NAMING.md`](../Testing/NAMING.md)。

```text
Survey/{gss,ucnets,nshap,signos,issp,add_health}/
  ...原始表...
  cleaned/{dataset}_*.{csv,jsonl,json}   # 人类真值（gold）

Testing/
  NAMING.md
  configs/state/{dataset}.yaml
  configs/network/{dataset}.yaml
  scripts/                               # 清洗、造 input、生成、评估（共用）
  T1_association/{dataset}/{input,output,eval}
  T2_information_use/...
  T3_interval_following/...
  T4_individual_resample/...
  T5_additivity/...
  T6_cross_population/...
  logs/
```

约定要点：

| 规则 | 说明 |
|---|---|
| dataset id 全小写下划线 | `ucnets_w1`, `gss_1985`, `gss_2004`, `gss_2024` |
| 人类真值只放 `Survey/*/cleaned/` | 绝不进 `Testing/*/input/`，防答案泄漏 |
| `{dataset}/input/` 用短文件名 | dataset 已在父目录；如 `s0_input.jsonl`、`n0_input.jsonl` |
| 每 dataset 一份 state/network 配置 | schema 不同的数据集**不要**共用配置 |
| 一次性/被取代的产物 | 移到根目录 `_archive/`，不留在 `Testing/logs/` 下 |

路径一律走 [`Testing/scripts/paths.py`](../Testing/scripts/paths.py)：dataset → survey 文件夹、臂 → 主 Testing、旧 slug → 新 slug。

历史遗留：`gss_2010_network` 带 `_network` 后缀，其余 GSS 年份不带。新增年份统一**不带**后缀。该 id 已被跑过的 State 实验引用，故未改名；代价是网络阶段文件名出现 `gss_2010_network_network_gold.jsonl` 这种重复词。这是刻意保留 `f"{dataset}_network_gold.jsonl"` 机械拼接的结果，不要为它加特例。

---

## 2. dataset 登记表

| dataset id | 数据源 | ego N | alter N | alter-alter | max alters | 备注 |
|---|---|---:|---:|:-:|---:|---|
| `ucnets_w1` | `Survey/ucnets/processed/by_wave/Wave1/` | 1159 | 20313 行槽位 | 嵌在 alter 宽表 | 30 | 主数据 |
| `gss_1987` | `Survey/gss/network/1987/` | 1800 | 4182 | 无 | 3 | 政治讨论网 |
| `gss_2010_network` | `Survey/gss/network/2010/` | 1272 | 2984 | 无 | 5 | panel only（2006 cohort W3） |
| `gss_1985` | `Survey/gss/network/1985/` | 1531 | 4483 | **6200** | 5 | important matters；alter 属性最全 |
| `gss_2004` | `Survey/gss/network/2004/` | 1426 | 2845 | **3384** | 5 | 与 1985 同一提名法，隔 19 年 |
| `gss_2024` | `Survey/gss/network/2024/` | 1062 | 3895 | 无 | 5 | 采集晚于 llama3.1 训练截止 |
| `signos` | `Survey/signos/processed/` | 70 | 2100 | **30450**（全对普查） | 30 | 每人恰好 30 人名单；巴塞罗那移民社群 |
| `nshap_r1` | `Survey/nshap/processed/by_round/Round1/` | 2931 | 10182 | **实测频率矩阵** | 5 | 57 岁以上美国成年人 |
| `nshap_r2` | `Survey/nshap/processed/by_round/Round2/` | 3163 | 11984 | **实测频率矩阵** | 5 | 含 `ROSTER_W1` 跨轮 alter 链接 |
| `nshap_r3` | `Survey/nshap/processed/by_round/Round3/` | 4328 | 16720 | **实测频率矩阵** | 5 | 该轮取消 `HOWCLOSE` |
| `issp_2017` | `Survey/issp/processed/survey.csv` | 44044 | — | — | — | 30 国；**无提名法，无网络阶段** |

`signos` 与 `nshap_*` 的产物在 `Survey/{signos,nshap}/cleaned/`；`issp_2017` 没有网络阶段，产物在 `Survey/issp/cleaned/`。

---

## 3. 主数据与补充数据的分工

**一个 survey 不需要跑完所有判据。** UCNets Wave 1 是唯一的主实验数据，五条判据的完整证据链都建立在它上面；其他 survey 只承担自己数据结构真正能支撑的那一两条，作为稳健性或对照证据。

判据编号对应 [`Predictor_vs_Sampler_Analysis/Paper_LLM_Ego_Network_Estimand_Validity.md`](Predictor_vs_Sampler_Analysis/Paper_LLM_Ego_Network_Estimand_Validity.md) §2.3：
T1 关联强度、T2 条件信息使用、T3 显式区间跟随、T4 个体重复抽样、T5 干预叠加、T6 跨总体响应（候选）。

### 3.1 主数据

| dataset | State 清洗 | State 生成 | Network 清洗 | Network 生成 | 判据 |
|---|:-:|:-:|:-:|:-:|---|
| `ucnets_w1` | ● | ● S0–S2 | ● | ● N0–N6（N0b 未跑） | **T1–T5 全部** |

UCNets 的不可替代性：30 人上限的名单、支持角色名册（confidant/companion）、以及 N0–N6 七个干预臂的叠加设计。这些都不要求补充 survey 复现。

### 3.2 补充数据（各自只做一件事）

| dataset | 承担判据 | 需要的实验臂 | 明确**不做** | 理由 |
|---|---|---|---|---|
| `gss_1985` | **T3 主力** | N0 + 闭合度/同性别条件臂 | N1/N3/N5/N6、T5 | 唯一同时有 alter 人口学字段与 alter-alter 矩阵，`esp_close_density` 分箱 27/36/37 |
| `gss_2004` | **T6 跨总体对照** | 与 1985 完全相同的臂 | 同上 | 与 1985 同一套提名法、相隔 19 年，是 T3 结论的跨时点重复 |
| `gss_2024` | 训练截止后污染检查 | N0 + **联系频率**条件臂 | 其余 | 采集晚于 llama3.1 训练截止。原计划的同种族条件臂已撤销，见 §4 |
| `gss_1987` | T1、T2 | 无（仅 State 阶段） | 全部网络生成 | alter 表只有关系标记，无人口学字段，无法构造网络靶量 |
| `gss_2010_network` | T1、T2 | 无（可选 T3 同性别臂） | 其余 | 仅 `same_sex_share` 一个可用靶量，优先级低 |
| `signos` | **T3 同质性**（全项目唯一可用） | N0 + 同源群条件臂 + 跨国触达条件臂 | 闭合度（见 §4.4）与其余全部 | `same_race_share` 在 UCNets 与 GSS 全年份都退化，同质性这条线本来是断的；Signos 的 `same_origin_share` 铺满全距。只有 70 个 ego，见下 |
| `nshap_r1` | T3 大样本复核 + T4/T6 人群对照 | N0 + 闭合度条件臂 + R1 | T5、r2/r3 | alter–alter 是实测矩阵，2931 个 ego；人群为 57 岁以上，与 UCNets 形成对照 |
| `issp_2017` | **T6 跨国** | 已完成 30 国分层生成与评估（2400 输出，2283 `schema_ok`） | **全部网络阶段** | 问卷没有提名法，没有名单也没有 alter 属性，网络指标无从套用。它的独有价值是 30 国同题 |

清洗状态：十个 dataset 的真值均已产出。补充生成/评估已完成：GSS 1985/2004/2024 与 NSHAP、Signos 的 T3，GSS 1985/NSHAP 的 T4，以及 ISSP 2017 的 T6；各产物按 `schema_ok` 与所需共同样本口径汇报。

三点需要在动手前想清楚的限制：

- **Signos 只有 70 个 ego。** 足够判断生成器是否跟随闭合度指令，不足以支撑有功效的分布性结论。它是结构性压力测试，不是 GSS 结果的重复验证；`signos_population_priors.json` 里写了 `underpowered_warning`。
- **NSHAP 的跨轮 alter 留存（`ROSTER_W1`）属于 response/dynamics 阶段**，当前论文范围不含该阶段。字段已清洗落盘（`carried_from_round1_share`），先存不用。
- **ISSP 永远不会有网络阶段。** 这不是暂时缺口，是问卷设计使然，`issp_2017_population_priors.json` 里以 `has_network_stage: false` 显式标记。

### 3.3 脚本如何适配各 survey 的编码

每个 survey 的实验臂集合、取值词表、名单形状、分箱切点全部来自它自己的 `Testing/configs/network/{dataset}.yaml`，脚本不再持有 UCNets 常量。已落实：

| 脚本 | 适配方式 |
|---|---|
| `Testing/scripts/network_schema.py` | `allowed_bundle` / `top_level_keys` / `roster_spec` / `extra_list_specs` 全部读配置；所有 `*_allowed` 列表自动成为词表 |
| `Testing/scripts/build_network_inputs.py` | `--dataset` 参数；路径、`input_id` 前缀、提示词取值说明、条件区间措辞、priors 注入字段均由配置推导 |
| `Testing/scripts/run_network_generation.py` | 按每个输入文件自带的 dataset 解析配置，`--config` 仅作强制覆盖 |

清洗脚本则相反：**每个 survey 一个脚本**，因为原始编码差异太大，配置化的收益低于把编码陷阱写在代码旁边的收益。共用的只有分箱与统计语义：

| 脚本 | 覆盖 dataset |
|---|---|
| `Testing/scripts/network_cleaning_common.py` | 共用模块：容错数值解析、覆盖率份额、三分位切点与最小桶占比判据 |
| `Testing/scripts/clean_network_gss.py` | gss_1985/1987/2004/2010/2024 |
| `Testing/scripts/clean_network_signos.py` | signos |
| `Testing/scripts/clean_network_nshap.py` | nshap_r1/r2/r3 |
| `Testing/scripts/clean_issp_2017_resources.py` | issp_2017（state 阶段） |
| `Testing/scripts/validate_supplementary_cleaning.py` | 验收：答案泄漏、名单/连边内部一致、桶标签与切点一致、行数对齐 |

分箱语义统一后 GSS 产物经校验和比对确认逐字节不变；引入最小桶占比判据后只有被降级的桶标签发生变化，所有 `prompt_text` 与 `prompt_json` 仍逐字节一致（已在 1985/2004 全部 6 个输入文件上验证），既有生成结果依然有效。

各 survey 的编码陷阱写在对应脚本的模块 docstring 里，动手前先读。三个最容易造成静默错误的：Signos 的 ego 与 alter 国家用**不同码表**（ego 里 3=India，alter 里 3=China）；NSHAP 的 `HEALTH` 是"比以前好/差"的比较题而非自评健康（自评健康是 `PHYSHLTH`）；ISSP 的 Q21–Q25 里 95="无此亲属"必须移出分母、96="与我同住"是最高联系档而非缺失，混淆会让联系频率朝相反方向偏。

三处关键的编码差异已由配置表达：

- **名单形状**：UCNets 是 `roster.mode: fixed`（固定 Top-5，从最多 30 人的网络里挑）；GSS 是 `variable`（1–5 人的完整提名名单，长度必须等于 `num_alters`）。校验器据此分别检查。
- **可用字段**：UCNets 有 `support_roles` 名册与 confidant/companion 计数，GSS 没有，配置里直接不声明；GSS 2024 的 alter 表连性别、年龄都没有，且关系项缺 `sibling`，其 schema 相应更窄。
- **区间切点**：同一个 `esp_close_density` 在 1985 是 0.17/0.50，在 2004 是 0.20/0.60，各自写在自己配置的条件措辞里。

UCNets 的提示词文本是冻结的（生成结果已采集），重构后 7 个实验臂的输入文件经逐字节比对完全一致。

---

## 4. Testing 3 靶量的可用性结论

清洗补充 survey 的主要动机是给 T3（显式区间跟随）换一个非退化的靶量。UCNets 上用的同种族占比有 90.6% 的 ego 顶格在 1.0，三档几乎无法区分。

### 4.1 判据：最小桶占比 15%

三分箱只检查切点是否相等是不够的——切点可以不相等，而某一档仍然近乎空置。所以 `network_cleaning_common.py` 里加了 `MIN_BUCKET_SHARE = 0.15`：任何一档占比低于 15% 的靶量一律拒绝分箱，产物里不留桶标签。理由是这种情况下模型完全错过那一档也几乎不受惩罚，区间跟随检验失去意义。

这条判据下调了若干原先"可用"的靶量：`gss_1985` 的 `educ_homophily_share`(9.4%)、`pct_weekly_plus_contact`(6.5%)；`gss_2004` 的 `density`(12.4%)、`educ_homophily_share`(12.7%)；`gss_2024` 的 `same_race_share`(9.9%)。

**对 `gss_2024` 有实际后果**：它原定的同种族条件臂已改为联系频率条件臂（`pct_weekly_plus_contact`，31/18/51）。原写法带 `fallback: high`，桶标签全为空时会静默退回 "high"，臂看上去在跑、条件信号却毫无变化——这是最坏的失败形式。同时要认识到 2024 年的 alter 表只有种族、关系、联系频率，其他年份用的 `same_sex_share`、`esp_close_density` 都无法复现，所以**污染检查的落点是 N0 基线臂，不是配对的 T3 臂**。

### 4.2 各靶量可用性

| 靶量 | 可用数据集 | low/med/high | 说明 |
|---|---|---|---|
| `esp_close_density` | gss_1985、gss_2004 | 27/36/37、28/38/34 | 5 人名单上最佳。1985↔2004 的 T6 对照用它，不要用 `density` |
| `density`（Signos） | signos | 33/31/36 | **30 人名单尺度**上的闭合度，见 §4.3 |
| `density_weekly` | nshap_r1/r2/r3 | 33/28/38、33/31/36、33/32/35 | 实测 alter 间联系频率，每周以上算连边。样本量最大 |
| `same_sex_share` | gss_1985/2004/2010、signos、nshap_r1/r2/r3 | 均接近 30/33/37 | 覆盖最广 |
| `age_homophily_share` | gss_1985、gss_2004 | 29/35/36、33/32/35 | ±5 岁窗口。NSHAP 上不可用：`NODE_AGE` 只有两成填充，分布双峰于 0 和 1 |
| `same_origin_share` | signos | 31/33/36 | 同源国占比，移民社群特有 |
| `occupational_access_count` | issp_2017 | 28/33/39 | 位置发生器得分（0–10） |
| `same_race_share` | **无** | 1985/2004/2010 顶格退化，2024 仅 9.9% 中档 | 见下 |

### 4.3 两个影响论文写法的结论

**一、同种族占比不可用，是估计量本身的性质。** 它在 UCNets、GSS 全部四个年份上一致退化，包括原先认为唯一可用的 2024。这不是某个数据集的偶然，原论文对 T3 的处理需要据此改写。

**二、闭合度只有在大名单尺度上才有方差可解释。** 5 人名单上的闭合度接近天花板：NSHAP 三轮"任意联系"密度均值 0.80–0.85，中档只占 4%–17%，判据直接拒绝分箱；必须收紧到"每周以上联系"才铺得开。而 Signos 的 30 人名单密度均值 0.319、range 0.025–0.703，70 个 ego 里有 62 个不同取值，天然铺满三档。

这一点直接关系到 UCNets：UCNets 的名单上限正是 30 人，但它没有采集 alter–alter 连边。所以 GSS/NSHAP 只能在 k=5 上测闭合度，而 **Signos 是唯一能在 UCNets 实际运行的网络尺度上测闭合度的数据**。代价是只有 70 个 ego。

`issp_2017` 的 `kin_channel_share`、`friend_channel_share`、`institutional_reliance_share` 是第三种情形：**零膨胀**，三成半到四成的受访者恰好取 0，下三分位点落在 0 上使 low 档必空。这类靶量可测的是"有/无"，产物里给了对应的 `*_any` 二分标记（yes 占 60%–65%，均衡可用），份额本身按均值做跨国比较。

### 4.4 Signos 的 30 人闭合度：能不能测，取决于怎么问

写配置前先做了输出设计试跑（`pilot_signos_output_design.py`、`pilot_signos_tie_elicitation.py`，明细在 `_archive/signos_*_pilot.json`）。人类参照：密度均值 0.319、范围 0.025–0.703，即 435 对中约 11–306 对相连。

**一、要求完整 435 对普查不可行。** llama3.1 在 3 个 ego 上返回 0 个完整普查（实际给出 318 和 62 对），还有一个连 JSON 都没解析出来，单次耗时 51 秒。作为对照，5 人名单且不要连边列表时 UCNets 各臂的 schema 通过率是 90%–98%。

**二、格式决定结果，幅度超过一个数量级。** 同一模型、同一份已生成的 30 人名单，只换征询格式：

| 征询格式 | llama3.1 密度 | qwen3:14b 密度 | 可解析 | 备注 |
|---|---|---|---|---|
| 稀疏边表（中性措辞） | 0.018–0.046 | 0.058–0.120 | 6/6 | 两个模型都远低于人类均值 |
| 逐人邻接表 | 0.172–**1.000** | 0.074–**0.871** | 5/6 | 两个模型都出现近完全图 |
| 先报数再列表 | 0.023–0.064 | 0.099–0.120 | 6/6 | 自报数与列表一致：llama 2/4，qwen 2/2 |

结论有两条，都与论文主张相关：一是**生成器的三元闭合度不是一个稳定量**，它由征询格式决定，格式敏感性在 8B 和 14B 上都存在，不是小模型的缺陷；二是在能稳定解析的格式下，两个模型给出的 30 人网络都**比人类稀疏得多**（0.02–0.12 对 0.319）。

**决定：Signos 不做闭合度靶量。** 结论会是格式选择的产物而非模型性质，而 T3 的闭合度已有 `gss_1985` 和 `nshap_r1` 两个数据集支撑，它们的 alter–alter 是实测的、名单也短到能穷举。格式敏感性本身作为方法学发现写进论文，试跑明细留在 `_archive/signos_*_pilot.json`，将来若要加闭合度臂可直接接续。

**Signos 真正承担的是同质性。** 这比闭合度更要紧，因为**同质性这条 T3 测试线本来是断的**：`same_race_share` 在 UCNets 有 90.6% 挤在 1.0，在 GSS 全部网络年份都不满足 `MIN_BUCKET_SHARE`。Signos 抽的是同一城市里的三个移民社群，`same_origin_share` 因此铺满全距、三分位分成 31/33/36，是全项目唯一可用的同质性区间跟随测试。第二个条件臂用 `resides_in_spain_share`（跨国触达，分 31/27/41），使结论不依赖某一个特定份额。三个臂各 70 条，条件桶无一为空。

---

## 4A. T6（跨总体响应）的两条证据线

| 证据线 | 数据集 | 做法 | 人类侧的落差 |
|---|---|---|---|
| 跨时点 | gss_1985 → gss_2004 | 同一套提名法、完全相同的实验臂，相隔 19 年 | 用 `esp_close_density` 比对 |
| 跨国 | issp_2017 | 提示模板固定，只改国家，30 个国家样本同题 | 「找家人求助」从丹麦 0.50 到菲律宾 0.83 |

跨国这条线的力量在于人类侧的落差已经很大且方向明确。如果生成器对 30 个国家给出的画像基本相同，那它是在输出一个全球刻板印象，而不是在为被问的人群抽样——这正是 predictor 行为的定义。`issp_2017_population_priors.json` 的 `by_country` 块存了逐国均值与桶占比；分箱切点特意取自 30 国**合并**样本而非各国分别取，否则跨国差异会被各自的分位数抹平。

NSHAP 提供第三条较弱的线：57 岁以上人群的网络明显更小更偏亲属（均值名单 3.5–3.9 人）。若生成器对该人群仍返回与 UCNets 相同的众数网络，同样是人群不敏感的证据。

另外 1985/1987 的 `frndknow`（"你的朋友有多少互相认识"，四档）是**被调查者自报的密度**，已存入 `composition_summary.ego_reported`。它可以直接当 T3 的自然语言条件档位，无需我们自造区间。

注意 `esp_close_share`（ego 与各 alter 的亲近度）只有 38% 的 ego 有值：GSS 仅在 `eqclose = no` 时才追问 `rclose`，这是跳答设计而非缺失。

---

## 5. 新增 survey 的步骤清单

1. 在 §2 与 §3.2 登记 dataset id、数据源路径，并且**先想清楚它承担哪一条判据**。承担不了的判据就明确写"不做"，不要为了对齐而硬凑实验臂。
2. 写/扩清洗脚本，产出 `Survey/{survey}/cleaned/{dataset}_*` 与一份 cleaning report 记录样本损失。
3. 检查靶量分布是否非退化。清洗脚本会自动执行 §4.1 的最小桶占比判据（任一档 <15% 即拒绝分箱），但**判据只负责拒绝，换靶量要人来决定**：先看 cleaning report 里哪些靶量被拒，再确认剩下的能不能支撑该 survey 承担的判据。若一个都不剩，就把该判据标为"不做"，不要靠 `fallback` 凑一个跑得起来的臂。
4. 新建 `Testing/configs/network/{dataset}.yaml`（或 state 配置），只声明该 survey 需要的实验臂，并写全它自己的词表、`roster` 形状、`output_schema` 和区间切点。**不要复用 UCNets 的配置。**
5. `python Testing/scripts/build_network_inputs.py --dataset {id}` 先核对提示词渲染与不含 target，再正式产出。
6. 更新 §3 状态矩阵。

### 5.1 跑全量之前必做的小规模冒烟测试

`--limit` 十来条先跑一遍，看 schema 通过率。这一步已经抓到两个会静默毁掉整轮生成的问题，两个都不会报错、只会让结果变差或变假：

**一、比例字段被输出成数组。** 提示词里写 `composition_summary.kin_share / friend_share: proportions in [0, 1]` 时，llama3.1 会返回 `[0.5]` 而不是 `0.5`，schema 校验不通过。实测：

| 配置措辞 | nshap_r1 N0 | nshap_r1 N4c | gss_1985 N0 |
|---|---|---|---|
| `proportions in [0, 1]`（旧） | 5/12 | 0/12 | 7/10 |
| 每个字段单列 + `NOT an array`（现） | 12/12 | 11/12 | 10/10 |

UCNets 用同样的紧凑措辞却有 90%–98% 通过率，所以这不是措辞单独造成的——`alter_ties` 这个第二个数组字段放大了歧义。凡是 schema 里同时有 `alters` 和 `alter_ties` 的数据集，比例字段都要逐个单列并注明不是数组。四个用了 `schema_description_lines` 的配置均已修正。

**二、`fallback` 会把无定义的条件桶变成虚构标签。** 闭合度需要至少 2 个 alter；只提名 1 个人的 ego 没有配对，`esp_close_density_bucket` / `density_weekly_bucket` 为空。旧行为是套用 `fallback: medium`，这些 ego 于是被要求满足一个不可能成立的约束，而 gold 是空的、结果无法评估。各条件臂受影响的比例：

| 臂 | 条件靶量 | 空桶占比 | 剔除后条数 |
|---|---|---|---|
| `gss_2004` N4c | `esp_close_density` | **50.9%** | 1069 → 525 |
| `gss_1985` N4c | `esp_close_density` | **33.5%** | 1393 → 927 |
| `nshap_r1` N4c | `density_weekly` | 11.0% | 1000 → 890 |
| `ucnets_w1` N4 | `same_race_share` | 2.5% | 未改，见下 |
| 其余 N4s / N4 | 同性别份额等 | <0.5% | 基本无损 |

这不是边角问题：GSS 1985 之所以被选作 T3 主力，理由就是它的闭合度三分位能分成 27/36/37；但那个分布是在闭合度**有定义**的 927 个 ego 上算出来的，而输入文件把另外 466 个塞进了 medium。gss_2004 更极端，medium 桶会有过半是虚构的。这一臂原本会得出一个由 fallback 决定的结论。

现在任务可以声明 `require_condition_bucket: true`，条件桶为空时剔除该 ego，剔除数记入 manifest 的 `excluded_undefined_bucket`。GSS 三年与 `nshap_r1` 的条件臂均已打开，剔除后桶分布分别为 252/336/339（1985）、148/199/178（2004）、294/251/345（NSHAP），无 fallback 行。**新写条件臂时默认打开**，除非确认 fallback 是真实默认值而不是虚构。

`ucnets_w1` 的 N4 是唯一例外：它已有 1156 条生成结果，改输入会全部作废，而受影响只有 29 行（2.5%）。这 29 行**在评估阶段按 gold 桶为空剔除**，不重跑生成。

**三、长名单会撞上输出长度上限，而失败长得像格式错误。** Signos 要 30 个 alter × 7 个字段，约 2000 个输出 token，超过运行脚本默认的 `num_predict=2048`，JSON 在数组中途被截断。日志把这类行记成"未解析"而不是"被截断"，8 条里丢了 5 条；能跑完的那几条名单长度全是恰好 30，说明名单长度本身没问题。

配置现在可以声明自己的输出下限，`generation.num_predict_min`，运行脚本取它与命令行参数的较大值并打日志。Signos 设 4096 后 4/4 通过。**新数据集只要名单长于 5 人就要估一下输出长度并设这个值**，别指望调用者记得加参数。

代价要如实记：`gss_2004` 的闭合度臂样本减半（2004 年的网络本来就更小，能有 2 个以上 alter 的人更少）。525 个 ego、每桶约 150–200，仍够用，但它作为复核而非主力。

### 尚未适配的部分

`Testing/scripts/evaluate_network.py` 已按 survey 配置化：`--dataset` 读 `evaluation.t3_targets`，从生成的名单 / `alter_ties` 重算闭合度、同性别、同源国等 T3 靶量（GSS gold 里的 `alter_alter` 在评估时与 `alter_ties` 对齐）。UCNets 默认路径（Layer 0–4、N0–N6）不变。冒烟：`python Testing/scripts/evaluate_network.py --from-gold --all-supplementary`。

`Testing/scripts/evaluate_t4.py` 同样只对 UCNets 有效，但按 §3 的分工它不需要扩展到补充 survey，直到 GSS 1985 的 S0/R1 开跑。

`Testing/scripts/build_network_inputs.py` 目前假定每个数据集都有名单：`roster_spec` 是必需的，schema 校验也围绕名单长度展开。**`issp_2017` 没有名单**，因此还不能用它产生成输入。接入需要让配置支持"无名单数据集"（`roster: none`），并把靶量从 `composition_summary` 换成 `resource_summary`。清洗产物已按统一命名落盘，接入前不影响其他数据集。

`nshap_r1` 的配置与输入已产出（N0 = 1000 条，N4c = 890 条，共用同一批 ego），冒烟测试通过率 12/12 与 11/12，可以跑全量。`nshap_r2`/`nshap_r3` 已清洗但**故意不接入**：按 §3.2 的分工，一轮就能同时满足大样本闭合度复核与人群对照，跑三轮属于同一 survey 内的重复且生成成本三倍。

`signos` 的配置与输入已产出（N0 / N4o / N4r 各 70 条），冒烟测试 4/4，可以跑全量；需注意它每次调用约 90 秒（输出比 5 人名单长得多），三个臂 210 次调用在 4 并发下约 1.5 小时。它**不做**闭合度臂，理由见 §4.4。

---

## 6. 备份与归档

- 本目录**不是 git 仓库**，改动前请先做全量副本。
- 已有备份：`/home/bld/data/data2/jiaxin/simulation_backup_20260813T220610`；本次重组另有 `/home/bld/data/data2/jiaxin/simulation_backup_20260818T171600`。目录约定见文首与 [`MIGRATION.md`](../MIGRATION.md)。
- 被取代的一次性产物统一放根目录 `_archive/`，见 [`_archive/README.md`](../_archive/README.md)。
