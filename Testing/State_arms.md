# State 臂设计（S0–S2；主 Testing：T1 / T2）

臂名对照见 [`NAMING.md`](NAMING.md)。下文若仍写 Task 1/2/3，即 S0/S1/S2。

## 1. Goal

Phase 1 evaluates the **State** stage of the LLM ego-network generation pipeline.

The core question is:

> Given real ego-level background attributes, can a single LLM agent generate state variables whose population-level distributions and associations resemble human survey data?

Formally:

```text
X_i^{anchor} -> S_{i,L}^{heldout}
```

where:

- `X_i^{anchor}` is the set of observed background variables shown to the model.
- `S_{i,L}^{heldout}` is the LLM-generated state vector.
- `S_{i,H}^{heldout}` is the observed human state vector used later for evaluation.

This phase isolates whether distortion enters **before** any network generation. It is not primarily an individual prediction task; the main object is collective validity: whether the generated synthetic population reproduces human state distributions, subgroup patterns, and association structure.

---

## 2. Data

The implementation uses UCNets Wave 1 as the main dataset and two GSS network years as external robustness samples.

| Dataset | Role | Source File | Cleaned Outputs |
|---|---|---|---|
| UCNets Wave 1 | Main state-stage dataset | `Survey/ucnets/ego.csv` | `Survey/ucnets/cleaned/ucnets_w1_state_clean.csv` |
| GSS 1987 network | External robustness | `Survey/gss/network/1987/ego.csv` | `Survey/ucnets/cleaned/gss_1987_state_clean.csv` |
| GSS 2010 | External robustness | `Survey/gss/network/2010/ego.csv` | `Survey/ucnets/cleaned/gss_2010_network_state_clean.csv` |

UCNets is the main dataset because it contains richer ego-level **state constructs**: health and affective well-being, perceived social support, personality, and contact orientation (construct domains D, H–J below). GSS network years supply comparable **background** constructs (A–C) and classic **state** batteries for health/happiness, interpersonal trust, politics, and religion (D–G), plus `alter` tables on the same egos for a later Network stage.

The datasets are aligned at the **construct level**, not forced into exact item equivalence. For example, UCNets loneliness days and GSS general happiness both sit in domain D (health / subjective well-being), but they are not identical measures; GSS trust/fair/helpful (domain E) has no item-equivalent in UCNets.

---

## 3. Tasks

### Task 1: Baseline State Imputation

Task 1 gives the LLM only **background constructs (domains A–C)** and asks it to generate all held-out **state constructs (domains D–G on GSS; D–J on UCNets)**.

```text
background anchors (A–C) -> full state targets (D–G / D–J)
```

Purpose:

> Test whether the model can generate a realistic baseline state distribution from sparse socio-demographic descriptors alone.

Task 1 uses the **SSDataBench-aligned prompt baseline**: population framing only (no survey instrument name) and format rules 1–2 only.

### Task 1b: Anti-Typology Prompt Supplement (Task 1)

Task 1b uses the **same respondents, anchors, and targets** as Task 1. The only difference is the prompt rules.

### Task 2: Health-Conditioned State Completion

Task 2 gives the LLM the same background attributes plus **observed self-rated health (domain D)**. The model then generates the remaining state variables.

```text
background (A–C) + observed health (D) -> remaining state targets
```

Purpose:

> Test whether partially specifying the health / well-being domain improves calibration of remaining well-being items, support, personality, trust, politics, religion, and contact variables.

Task 2 has fewer rows than Task 1 because the conditioning health variable must be observed. This especially affects **GSS 2010**, where ballot missingness on `health` shrinks Task 2 (853 vs Task 1 1251). GSS 1987 network keeps nearly full Task 2 N (1785).

Task 2 uses the same **baseline prompt rules 1–2** as Task 1.

### Task 2b: Anti-Typology Prompt Supplement (Task 2)

Task 2b uses the **same respondents, anchors, and targets** as Task 2. The only difference is the prompt rules.

### Task 3: Politics / Religion / Voting–Conditioned State Completion

Task 3 gives the LLM the same background attributes plus **observed political and religious identity / participation (domains F and G)**. The model then generates the remaining state variables.

```text
background (A–C) + observed party / religion / voting (F–G) -> remaining state targets
```

| Dataset | Conditioning fields in input (excluded from output) | Domains |
|---|---|---|
| UCNets W1 | `party_identity`, `religion_current`, `voting_frequency` | F, G |
| GSS 1987 network | `party_identity`, `religion`, `voting_1984` (GSS `vote84`) | F, G |
| GSS 2010 | `party_identity`, `religion`, `voting_2008` (GSS `VOTE08`) | F, G |

Purpose:

> Test whether conditioning on civic / religious identity improves calibration of remaining well-being, trust, health, contact, and related state variables (complement to Task 2's health conditioning in domain D).

Task 3 uses **baseline prompt rules 1–2 only**. There is **no Task 3b** (no anti-typology ablation for this task).

Rows require all conditioning fields non-missing (UCNets especially loses cases with missing party/religion).

### Prompt Rule Tiers (All Tasks)

Every substantive task has two prompt variants:

| Tier | Tasks | `prompt_condition` | Rules |
|---|---|---|---|
| **Baseline (primary)** | Task 1, Task 2, Task 3 | `format_only` | 1–2 only |
| **Supplementary** | Task 1b, Task 2b | `format_anti_typology` | 1–2 + 3–4 |

Purpose of the supplementary tier:

> Test whether explicit anti-typology instructions reduce variance compression and stereotyped state generation, holding task design and respondents fixed.

After LLM generation, compare baseline vs supplementary pairs:

- Task 1 vs Task 1b
- Task 2 vs Task 2b

Use Eval 1–3 statistics (marginals, associations, entropy/variance ratios).

Panel State Projection is not part of the first Phase 1 implementation. It should be treated as a future extension because it begins to overlap with Response/Dynamics.

---

## 4. Variable Taxonomy (Construct Domains)

Phase 1 does **not** treat all targets as a single undifferentiated “Attitude” block. Variables are grouped into sociological **construct domains**, following GSS core topic structure (NORC Quick Guides: demographic / life-course, SES, political, religious, subjective well-being, social-life feelings / misanthropy) and standard construct definitions in the survey literature. UCNets adds support, personality, and contact domains that have no item-level GSS counterpart.

Cross-survey claims are made at the **domain** level only.

### 4.1 Domain map

| Domain | Construct | Typical role | Literature anchors |
|---|---|---|---|
| **A. Demography / life course** | Age, sex/gender, race/ethnicity, region | Input (Task 1–3) | GSS demographic core; status / life-course location |
| **B. Family / household** | Marital / partnership, children, household composition | Input | GSS marital / childs; family structure |
| **C. Socioeconomic status (SES)** | Education, degree, work status | Input | Education–occupation SES tradition (e.g. Blau–Duncan lineage in survey practice) |
| **D. Health / subjective well-being** | Self-rated health; happiness / affect / loneliness days | Output; Task 2 moves health into input | Self-rated health (Idler & Benyamini, 1997); GSS `HAPPY` as SWB |
| **E. Interpersonal trust / misanthropy** | Trust, fair, helpful triad | Output (GSS) | Rosenberg (1956) faith-in-people; GSS TRUST/FAIR/HELPFUL (Smith, GSS Topical Report on misanthropy) |
| **F. Political identity / participation** | Party ID, ideology, voting | Output; Task 3 moves party/voting into input | *The American Voter* party identification; GSS PARTYID / POLVIEWS / VOTE* |
| **G. Religion** | Affiliation, attendance, strength (and raised religion on UCNets) | Output; Task 3 moves affiliation into input (UCNets: both current and raised) | Affiliation–practice–salience triad (Glock–Stark tradition; GSS RELIG / ATTEND / RELITEN) |
| **H. Social support** | Help availability; wanting more alters for concerns / leisure / practical help | Output (UCNets) | Perceived support / network resource tradition |
| **I. Personality** | Big-Five–style Likert battery | Output (UCNets) | Individual differences (not GSS replicating core) |
| **J. Contact / media** | Online use; family/friend weekly contact methods | Output (UCNets) | Interaction channels |

**Task conditioning (design logic):**

| Task | Extra input domain | Held-out targets |
|---|---|---|
| Task 1 / 1b | — | All state domains |
| Task 2 / 2b | **D** (observed health) | Remaining state (D without health, plus E–J as available) |
| Task 3 | **F + G** (party, religion, voting) | Remaining state (D, E, and leftover F/G items) |

### 4.2 GSS network fields by domain (`gss_1987`, `gss_2010_network`)

Cleaned prompt / target names (raw GSS codes in parentheses). Schema is shared across the two network years; only the Task 3 vote field differs by year.

**Background (input in Task 1):**

| Domain | Cleaned fields | Raw (illustrative) |
|---|---|---|
| A | `age`, `age_group`, `sex`, `race_ethnicity`, `region` | `age`, `sex`, `race` / Hispanic / race census, `region` |
| B | `marital_status`, `children_count`, `children_status` | `marital`, `childs` |
| C | `education_years`, `degree`, `work_status` | `educ`, `degree`, `wrkstat` |

**State (output in Task 1):**

| Domain | Cleaned fields | Raw |
|---|---|---|
| D | `self_rated_health`, `general_happiness` | `health`, `happy` |
| E | `people_can_be_trusted`, `people_are_fair`, `people_are_helpful` | `trust`, `fair`, `helpful` |
| F | `party_identity`, `political_ideology` (+ year-specific vote in Task 3 input) | `partyid`, `polviews`, `vote84` / `VOTE08` |
| G | `religion`, `religious_service_attendance`, `religious_affiliation_strength` | `relig`, `attend`, `reliten` |

Task 2: add `self_rated_health` to input; drop it from output.  
Task 3: add `party_identity`, `religion`, and year-specific vote (`voting_1984` or `voting_2008`) to input; drop party and religion from output (voting is conditioning-only).

### 4.3 UCNets Wave 1 fields by domain

**Background (input in Task 1):**

| Domain | Cleaned fields | Raw (illustrative) |
|---|---|---|
| A | `age`, `age_group`, `gender`, `race_ethnicity` | `AGE`, `AGEGROUP`, `GENDER`, race items |
| B | `marital_status`, `romantic_relationship`, `household_other_people_count`, `children_status` | `A2A`, `A3A`, `A12_COUNT`, child items |
| C | `education`, `work_status` | `A21`, education items |

**State (output in Task 1):**

| Domain | Cleaned fields | Notes |
|---|---|---|
| D | `self_rated_health`, `days_very_happy_past_week`, `days_isolated_past_week`, `days_lonely_past_week` | Health + affect / isolation days |
| F | `party_identity`, `voting_frequency` | No separate ideology item |
| G | `religion_current`, `religion_raised` | Affiliation focus; no attendance/strength twin of GSS |
| H | `relatives_help_serious_problem`, `friends_help_serious_problem`, `wants_more_people_for_*` | Support / network need |
| I | `personality` (nested Likert) | Multi-item |
| J | `ever_goes_online`, `family_contact_methods_weekly`, `friend_contact_methods_weekly` | Contact / media |

No domain **E** (Rosenberg trust triad) on UCNets. Domains **H–J** are UCNets-only richness for the main pipeline.

Task 3 (UCNets): add `party_identity`, `religion_current`, `religion_raised`, and `voting_frequency` to input; drop all four from output (G domain fully conditioned).

### 4.4 Construct alignment (what may be compared across datasets)

| Comparable (cautiously) | Not item-equivalent |
|---|---|
| A–C background structure | Exact race / education coding |
| D health; D happiness vs happy-days | Loneliness days vs GSS happiness |
| F party; G current religion | GSS ideology, attendance, RELITEN vs UCNets raised religion |
| — | E trust triad (GSS only); H–J (UCNets only) |

Evaluation summaries should prefer **within-domain** aggregates (e.g. mean TV on E; mean TV on D) over a single mean across all targets.

---

## 5. Model Input Files

The LLM-facing input files are stored in:

```text
Testing/T1_association/input/
```

Current files:

| File | Rows | Task |
|---|---:|---|
| `ucnets_w1_task1_input.jsonl` | 1121 | UCNets Task 1 (baseline) |
| `ucnets_w1_task1b_input.jsonl` | 1121 | UCNets Task 1b (supplementary) |
| `ucnets_w1_task2_health_conditioned_input.jsonl` | 1120 | UCNets Task 2 (baseline) |
| `ucnets_w1_task2b_health_conditioned_input.jsonl` | 1120 | UCNets Task 2b (supplementary) |
| `ucnets_w1_task3_politics_religion_conditioned_input.jsonl` | 631 | UCNets Task 3 |
| `gss_1987_task1_input.jsonl` | 1787 | GSS 1987 network Task 1 |
| `gss_1987_task1b_input.jsonl` | 1787 | GSS 1987 network Task 1b |
| `gss_1987_task2_health_conditioned_input.jsonl` | 1785 | GSS 1987 network Task 2 |
| `gss_1987_task2b_health_conditioned_input.jsonl` | 1785 | GSS 1987 network Task 2b |
| `gss_1987_task3_politics_religion_conditioned_input.jsonl` | 1738 | GSS 1987 network Task 3 |
| `gss_2010_network_task1_input.jsonl` | 1251 | GSS 2010 Task 1 |
| `gss_2010_network_task1b_input.jsonl` | 1251 | GSS 2010 Task 1b |
| `gss_2010_network_task2_health_conditioned_input.jsonl` | 853 | GSS 2010 Task 2 |
| `gss_2010_network_task2b_health_conditioned_input.jsonl` | 853 | GSS 2010 Task 2b |
| `gss_2010_network_task3_politics_religion_conditioned_input.jsonl` | 1225 | GSS 2010 Task 3 |

Each input row contains:

```text
input_id
respondent_id
dataset
task
prompt_condition
prompt_json
prompt_text
expected_output_keys
allowed_values
metadata
```

The input files do **not** contain `target_json`. Human targets remain in:

```text
Survey/ucnets/cleaned/
```

This prevents answer leakage during LLM generation.

---

## 6. Prompt Design

The prompt follows SSDataBench / PNAS statistical realism logic: population framing, real background variables, target descriptions, allowed values, and strict JSON output.

Prompt conditions:

| Tier | Task | `prompt_condition` | Rules |
|---|---|---|---|
| Baseline | Task 1, Task 2 | `format_only` | 1–2 only |
| Supplementary | Task 1b, Task 2b | `format_anti_typology` | 1–2 + 3–4 |

Opening line (SSDataBench-style):

```text
You are simulating one *randomly selected* synthetic individual sampled from
{population context}
The data represent a cross-sectional snapshot collected in the year {reference_year}.
```

The model sees:

- population context (not the survey instrument name);
- task instruction;
- respondent background attributes;
- exact output keys;
- variable descriptions;
- allowed values for each target field.

Task 1b and Task 2b additionally see anti-typology rules about heterogeneity and avoiding stereotyped profiles.

The model does not see:

- true human target values;
- target distributions;
- evaluation statistics;
- other respondents' answers.

This is close to a multi-variable constrained survey-response format. It is not a single MCQ; the model must fill an entire JSON state vector, with each field constrained to allowed survey values.

---

## 7. Cleaning and Reproducibility

The cleaning pipeline is implemented in:

```text
Testing/scripts/clean_phase1_state_data.py
```

The model-input generation pipeline is implemented in:

```text
Testing/scripts/build_phase1_model_inputs.py
Testing/scripts/phase1_content_config.py
Testing/configs/state/*.yaml
```

Current cleaned outputs:

```text
Survey/ucnets/cleaned/ucnets_w1_state_clean.csv
Survey/ucnets/cleaned/gss_1987_state_clean.csv
Survey/ucnets/cleaned/gss_2010_network_state_clean.csv
Survey/ucnets/cleaned/gss_network_clean_summary.json
Survey/ucnets/cleaned/phase1_variable_crosswalk.csv
Survey/ucnets/cleaned/ucnets_w1_task1.jsonl
Survey/ucnets/cleaned/ucnets_w1_task2_health_conditioned.jsonl
Survey/ucnets/cleaned/ucnets_w1_task3_politics_religion_conditioned.jsonl
Survey/ucnets/cleaned/gss_1987_task1.jsonl
Survey/ucnets/cleaned/gss_1987_task2_health_conditioned.jsonl
Survey/ucnets/cleaned/gss_1987_task3_politics_religion_conditioned.jsonl
Survey/ucnets/cleaned/gss_2010_network_task1.jsonl
Survey/ucnets/cleaned/gss_2010_network_task2_health_conditioned.jsonl
Survey/ucnets/cleaned/gss_2010_network_task3_politics_religion_conditioned.jsonl
Survey/ucnets/cleaned/cleaning_report.md
```

Cleaning scripts:

```text
Testing/scripts/clean_phase1_state_data.py      # UCNets
Testing/scripts/clean_phase1_gss_network.py    # GSS 1987 / 2010
```

Important sample sizes:

| Dataset / Task | Rows |
|---|---:|
| UCNets raw / clean | 1159 |
| UCNets Task 1 | 1121 |
| UCNets Task 2 | 1120 |
| UCNets Task 3 | 631 |
| GSS 1987 network Task 1 | 1787 |
| GSS 1987 network Task 2 | 1785 |
| GSS 1987 network Task 3 | 1738 |
| GSS 2010 Task 1 | 1251 |
| GSS 2010 Task 2 | 853 |
| GSS 2010 Task 3 | 1225 |

---

## 8. Evaluation Plan

Evaluation will be implemented after LLM outputs are generated.

The first evaluation should adapt SSDataBench-style Eval 1–3:

| Eval | Question | Main Statistics |
|---|---|---|
| **Eval 1** — Univariate Distribution | Does each generated target match the human marginal distribution? | chi-square test for categorical variables; K-S test for numeric/ordinal variables; pass rate |
| **Eval 2** — Bivariate Association | Does the generated data reproduce pairwise association structure? | Cramer's V, Fisher z-test, eta-squared comparison |
| **Eval 3** — Multivariate Prediction | Do background variables predict generated outcomes with human-like strength? | real vs synthetic `R^2` / pseudo-`R^2`; bootstrap pass rate |

Additional diagnostics:

```text
entropy ratio
variance ratio
total variation distance
Jensen-Shannon distance
subgroup calibration error
schema-valid response rate
```

The main expected failure modes are:

- optimism bias, such as overestimating health or support;
- underestimating loneliness or isolation;
- variance / entropy compression;
- overly strong or overly stereotyped associations;
- subgroup-specific miscalibration.

---

## 9. Baselines

The LLM should be compared against simple statistical baselines:

| Baseline | Meaning |
|---|---|
| Marginal sampler | Sample each target from its unconditional human distribution |
| Conditional sampler | Sample targets within broad anchor cells such as age, gender, education, marital status |
| Logistic / multinomial regression | Predict each target from anchors |
| LLM | Generate targets from prompt |

These baselines clarify whether the LLM adds useful conditional structure or merely produces plausible but statistically distorted responses.

---

## 10. Key Risks

### Target Leakage

Task 1 prompts must not contain target variables. Task 2 may contain health, but health must be excluded from `target_json`.

### Over-Alignment Across Datasets

Do not force identical items across UCNets and GSS. Align on **construct domains A–G** where both exist; keep UCNets H–J and GSS E as dataset-specific. Report Eval metrics by domain when comparing datasets.

```text
UCNets loneliness days ≠ GSS happiness          (both in D, not item-equivalent)
UCNets perceived support ≠ GSS trust triad      (H ≠ E)
```

### Missingness

GSS **2010** has ballot missingness for `health`, `fair`, and `helpful`, which shrinks Task 2. GSS **1987 network** has near-complete coverage on domains D–E for the network-module sample.

### Individual Accuracy vs Collective Validity

The main evaluation should not stop at individual prediction accuracy. The central question is whether the generated population reproduces:

```text
marginal distributions
conditional distributions
association structure
subgroup calibration
```

---

## 11. Minimal Phase 1 Deliverable

The minimal Phase 1 paper result should answer:

1. Whether LLM-generated state variables match human marginal distributions.
2. Whether health, loneliness, support, trust, politics, religion, and personality show systematic distortion.
3. Whether supplementary anti-typology prompts (Task 1b / Task 2b) improve statistical realism over baseline Task 1 / Task 2.
4. Whether Task 2 improves calibration over Task 1.
5. Whether UCNets findings conceptually replicate in GSS.
6. Whether failures appear before any ego-network generation step.

An example finding format:

> At the State stage, LLM-generated egos reproduce some broad background-conditioned patterns but show systematic distortion in health / well-being (D), interpersonal trust (E), social support (H), and related state domains. The generated state distribution is often more optimistic and less variable than the human benchmark. Similar domain-level failures on GSS suggest that state-level distortion can enter the pipeline before network generation begins.
