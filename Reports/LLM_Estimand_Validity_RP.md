> 臂名与目录见 [`Testing/NAMING.md`](../Testing/NAMING.md)。下文保留历史叙述，不按新树重写。

# Collective Validity of LLM-Generated Ego Networks

---

## 1. Motivation

LLMs are increasingly used to generate synthetic agents and simulate social behavior. Existing evaluations often examine selected micro-level responses, marginal distributions, or component-level statistics. These checks are useful, but they do not establish that an LLM-generated ego network is valid for the **collective estimand** a researcher wants to use.

This project focuses on a narrower and more testable question:

> **Can a single LLM agent validly generate ego-level social networks and their downstream collective estimands? If not, which stage of the generation pipeline fails?**

The concern is not only whether one generated response looks plausible. A generated ego network may fail in several ways:

- The baseline **State** may be miscalibrated: health too good, loneliness too low, personality too average.
- The generated **Network** may be distorted: too many alters, too much kinship, too much support, too little tie loss.
- The **Response/Dynamics** mapping may be biased: even with the right baseline network, the LLM may underpredict loneliness increase, support loss, or health decline.

The core tension is therefore:

> **Passing local or component-level checks does not establish validity for the target collective estimand.**

This proposal evaluates LLM-generated ego networks at the level where many social-scientific claims are actually made: network composition, alter retention, support availability, loneliness transitions, health transitions, and related collective outcomes.

---

## 2. Research Questions

### RQ1: Collective Validity

Can a single LLM agent generate ego-level social networks that reproduce observed collective estimands in human network survey data?

Evidence:

```text
Delta Y_k^t = Y_{k,L}^t - Y_{k,H}^t
```

where `Y_{k,H}^t` is the human observed estimand and `Y_{k,L}^t` is the LLM-generated counterpart.

### RQ2: Stage Deviation

How far does each generated stage deviate from observed human data?

Stages:

```text
State -> Network -> Response / Dynamics -> Collective Estimand Y
```

Stage-specific metrics:

```text
M_state
M_network
M_response/dynamics
```

### RQ3: Stage Attribution

When collective distortion occurs, which stage restoration reduces the error most?

The key comparison is whether error shrinks most when restoring:

- observed human State;
- observed human Network;
- both State and Network;
- or a human/statistical Response/Dynamics benchmark.

### RQ4: Estimand Dependence

Does the main failure source differ by estimand?

For example:

- network size and relationship-role composition may depend mostly on the Network stage;
- loneliness and health transitions may depend more on Response/Dynamics;
- baseline health, loneliness, and personality may depend on State generation.

### RQ5: Robustness

Do the stage-level failure patterns found in UCNets replicate in other data contexts?

Robustness datasets:

- GSS network modules for ego-network structure and GSS panel cohorts for repeated response patterns;
- NSHAP for health, loneliness, support, and aging-related network change;
- ISSP for cross-national role-based support;
- Add Health/Freeman for school whole-network topology;
- Signos for dense 30-alter personal networks.

---

## 3. Data

### 3.1 Main Dataset: UCNets

The main experiment uses **UCNets** because it provides a single longitudinal ego-network panel. The same respondents are followed across three waves, and each wave contains ego-level attributes and alter-level network records.

Main time structure:

```text
Wave 1 = baseline
Wave 2 / Wave 3 = follow-up outcomes
```

UCNets supports the full pipeline:

```text
S_{i,H}^1 + N_{i,H}^1 -> Y_{k,H}^2 / Y_{k,H}^3
```

where:

- `S_{i,H}^1` is respondent `i`'s observed Wave 1 state;
- `N_{i,H}^1` is respondent `i`'s observed Wave 1 ego network;
- `Y_{k,H}^2` and `Y_{k,H}^3` are observed collective estimands at Waves 2 and 3.

### 3.2 UCNets Data Layers

| Layer | UCNets Measurement | Use in Experiment |
|---|---|---|
| Ego ID | `PRIM_KEY` | Tracks respondents across waves |
| State | Age, gender, education, work, family, health, personality, internet use, politics/religion | Baseline state and persona input |
| Network | Alter records, relationship roles, support roles, contact patterns, selected alter attributes | Observed baseline ego network |
| Alter continuity | `ID_W1_W2_W3` where available | Alter retention and network change |
| Outcomes | Health, loneliness, social satisfaction, support availability, contact patterns, network change | Follow-up collective estimands |

### 3.3 Key UCNets Estimands

| Estimand Family | Example Estimands |
|---|---|
| Network structure | network size, network size change, role diversity |
| Network composition | kin share, friend share, partner share, coworker share |
| Network dynamics | alter retention rate, new alter rate, tie loss |
| Support/contact | support availability, confidant availability, contact frequency, difficult alter prevalence |
| Well-being | loneliness distribution, loneliness increase rate, social satisfaction |
| Health | self-rated health distribution, health decline rate |

### 3.4 Robustness Datasets

| Dataset | Role | What It Tests | Main Limitation |
|---|---|---|---|
| **GSS network modules + panels** | Network and panel robustness | network size/composition in network-module years; longitudinal response patterns in panel cohorts | alter-alter closure is only available in some network years; panel waves do not always include ego-network tables |
| **NSHAP** | Response/Dynamics robustness | health, loneliness, support, aging-related network change | older population |
| **ISSP 2017** | Macro response robustness | family/friend/institution support roles across countries | no named alters |
| **Add Health/Freeman** | Whole-network topology robustness | school friendship degree, reciprocity, homophily, clustering | cannot link nodes to public-use in-home outcomes |
| **Signos** | Dense ego-network stress test | 30-alter composition and closure | small specialized migrant sample |

---

## 4. Experimental Design

### 4.1 Notation

| Notation | Meaning |
|---|---|
| `i` | respondent / ego |
| `t` | survey wave, e.g. `t = 1, 2, 3` |
| `k` | collective estimand index |
| `S_i^t` | state of respondent `i` at wave `t` |
| `N_i^t` | ego network of respondent `i` at wave `t` |
| `R` | Response/Dynamics mapping from baseline state/network to follow-up outcomes |
| `Y_k^t` | collective estimand `k` computed at wave `t` |
| superscript `H` | human observed data |
| superscript `L` | LLM-generated data |

Examples:

| Notation | Meaning |
|---|---|
| `S_{i,H}^1` | observed Wave 1 state of respondent `i` |
| `N_{i,H}^1` | observed Wave 1 ego network of respondent `i` |
| `S_{i,L}^1` | LLM-generated Wave 1 state for respondent `i` |
| `N_{i,L}^1` | LLM-generated Wave 1 ego network for respondent `i` |
| `Y_{k,H}^2` | human observed Wave 2 collective estimand `k` |
| `Y_{k,L}^2` | LLM-generated Wave 2 collective estimand `k` |

Central gap:

```text
Delta Y_k^t = Y_{k,L}^t - Y_{k,H}^t
```

### 4.2 Pipeline

The LLM generation pipeline is:

```text
State -> Network -> Response / Dynamics -> Collective Estimand Y
```

| Stage | Meaning | LLM Task | Human Benchmark |
|---|---|---|---|
| State | Baseline individual attributes | generate or impute `S_{i,L}^1` | `S_{i,H}^1` |
| Network | Baseline ego network | generate `N_{i,L}^1` | `N_{i,H}^1` |
| Response/Dynamics | Follow-up outcomes or transitions | generate `Y_{k,L}^2`, `Y_{k,L}^3` | `Y_{k,H}^2`, `Y_{k,H}^3` |

### 4.3 Sub-Experiment A: State Generation

Goal: test whether the LLM can generate or impute realistic baseline state variables.

Input:

```text
Minimal respondent descriptors
```

Example input:

```text
Age group: 50-70
Gender: female
Education: college
Employment: retired
Partnership: widowed
```

LLM output:

```text
S_{i,L}^1
```

Example output:

```json
{
  "self_rated_health": "fair",
  "loneliness": "rarely",
  "social_satisfaction": "moderate",
  "internet_contact_frequency": "weekly",
  "personality": {
    "extraversion": "medium",
    "neuroticism": "low",
    "openness": "medium"
  }
}
```

Comparison:

```text
S_{i,L}^1 vs S_{i,H}^1
```

### 4.4 Sub-Experiment B: Network Generation

Goal: test whether the LLM can generate realistic Wave 1 ego networks given observed baseline state.

Input:

```text
S_{i,H}^1
```

LLM output:

```text
N_{i,L}^1
```

Example output:

```json
{
  "wave1_network": {
    "num_alters": 4,
    "alters": [
      {
        "alter_id": 1,
        "relationship": "daughter",
        "support_role": "emotional_support",
        "contact_frequency": "daily"
      },
      {
        "alter_id": 2,
        "relationship": "friend",
        "support_role": "social_companionship",
        "contact_frequency": "weekly"
      }
    ]
  }
}
```

Comparison:

```text
N_{i,L}^1 vs N_{i,H}^1
```

### 4.5 Sub-Experiment C: Observed-Network Response/Dynamics

Goal: test whether the LLM can predict follow-up outcomes when given the true Wave 1 state and true Wave 1 network.

Input:

```text
S_{i,H}^1 + N_{i,H}^1
```

LLM output:

```text
Y_{k,L}^2 / Y_{k,L}^3
```

Comparison:

```text
Y_{k,L}^2 / Y_{k,L}^3 vs Y_{k,H}^2 / Y_{k,H}^3
```

This sub-experiment isolates the Response/Dynamics stage. If the LLM still underpredicts loneliness increase, support loss, health decline, or tie loss even when the observed baseline network is supplied, then the response/dynamics mapping itself is biased.

### 4.6 Sub-Experiment D: Generated-Network Response/Dynamics

Goal: test how generated network errors propagate into follow-up outcomes.

Input:

```text
S_{i,H}^1 + N_{i,L}^1
```

LLM output:

```text
Y_{k,L}^2 / Y_{k,L}^3
```

Key comparison:

```text
Observed-network condition:
S_{i,H}^1 + N_{i,H}^1 -> Y_{k,L}^2

Generated-network condition:
S_{i,H}^1 + N_{i,L}^1 -> Y_{k,L}^2
```

If the generated-network condition produces larger error, then Network-stage distortion propagates into downstream collective outcomes.

### 4.7 Sub-Experiment E: Full Pipeline Simulation

Goal: simulate the practical use case of building a synthetic ego-network population from limited baseline descriptors.

Input:

```text
Baseline respondent descriptors only
```

LLM pipeline:

```text
S_{i,L}^1 -> N_{i,L}^1 -> Y_{k,L}^2 / Y_{k,L}^3
```

Comparison:

```text
Y_{k,L}^2 / Y_{k,L}^3 vs Y_{k,H}^2 / Y_{k,H}^3
```

### 4.8 各子实验之间的关系

可以把这些子实验理解成一个逐步展开的诊断流程：

```text
A: State Generation
   LLM 生成的人本身对不对？

B: Network Generation
   给定真实人，LLM 生成人际网络对不对？

C: Observed-Network Dynamics
   给定真实人 + 真实网络，LLM 预测未来对不对？

D: Generated-Network Dynamics
   给定真实人 + LLM 网络，网络错误如何影响未来预测？

E: Full Pipeline
   LLM 从头生成整个 ego-network society，最终集体结果对不对？

Stage-Restoration
   如果 E 错了，恢复哪个阶段最能修正？
```

The point of this sequence is diagnostic. If we only evaluate the full pipeline, we may know that `Y_{k,L}^t` differs from `Y_{k,H}^t`, but we do not know whether the error comes from State generation, Network generation, or the Response/Dynamics mapping. Sub-Experiments A-E separate these sources before combining them again in the full pipeline and stage-restoration analysis.

### 4.9 Stage-Restoration Sub-Experiments

To diagnose which stage is associated with collective distortion, the experiment restores one stage at a time to the human value.

```text
Full LLM:
Y_k^t(S_L^1, N_L^1, R_L)

State restored:
Y_k^t(S_H^1, N_L^1, R_L)

Network restored:
Y_k^t(S_L^1, N_H^1, R_L)

State + Network restored:
Y_k^t(S_H^1, N_H^1, R_L)

Human/statistical benchmark:
Y_k^t(S_H^1, N_H^1, R_H)
```

Attribution score:

```text
A_{s,k}^t = |Delta Y_{k,full}^t| - |Delta Y_{k,restored(s)}^t|
```

Interpretation:

- If restoring State reduces error most, State generation is most associated with distortion.
- If restoring Network reduces error most, Network generation is most associated with distortion.
- If substantial error remains even after restoring State and Network, Response/Dynamics is likely biased.

This is **stage-restoration attribution**, not definitive causal responsibility.

---

## 5. Evaluation Methods

### 5.1 State Metrics

| Metric | Definition | Interpretation |
|---|---|---|
| Mean shift | Difference in means or category shares | Systematic state bias |
| Variance ratio | `Var_L / Var_H` | Variance compression or exaggeration |
| Distribution distance | Jensen-Shannon, Wasserstein, total variation | Full distribution mismatch |
| Subgroup calibration | Gaps within age, gender, education, partnership groups | Conditional mismatch |
| Correlation structure | Difference in variable correlations | Whether joint structure is preserved |

### 5.2 Network Metrics

| Metric | Definition |
|---|---|
| Size error | Error in alter count distribution |
| Composition error | Error in relationship-role shares |
| Support-role error | Error in support-role composition |
| Contact error | Error in contact frequency distribution |
| Diversity error | Error in network diversity |
| Retention error | Error in alter retention across waves |

### 5.3 Response / Dynamics Metrics

| Metric | Definition |
|---|---|
| Transition error | Error in Wave 1 to Wave 2/3 transition probabilities |
| Outcome distribution error | Error in follow-up outcome distributions |
| Loneliness increase error | Error in loneliness increase rate |
| Health decline error | Error in health decline rate |
| Support loss error | Error in loss or absence of support |
| Network change error | Error in network size and composition change |

### 5.4 Baselines

| Baseline | Purpose |
|---|---|
| Marginal sampler | Samples from population marginals |
| Conditional sampler | Samples from distributions conditional on demographics |
| Regression / random forest | Transparent predictive benchmark |
| Human-data bootstrap | Upper-bound benchmark for sampling variability |
| LLM with observed network | Separates Response/Dynamics error from Network generation error |
| LLM with generated network | Tests propagation of Network generation error |

### 5.5 Statistical Inference

Recommended reporting:

```text
Delta Y_k^t
SE(Delta Y_k^t)
standardized error
A_{s,k}^t
```

Recommended procedures:

- Use UCNets weights where appropriate.
- Bootstrap over egos.
- Cluster alter-level analyses by ego.
- Separate individual prediction accuracy from collective distribution error.
- Pre-specify primary estimands and treat others as exploratory.
- Control false discovery rate when reporting many estimands.

---

## 6. Expected Contributions

### Conceptual Contribution

The project reframes LLM social simulation evaluation around **target collective estimands**, rather than assuming that local or component-level checks are sufficient.

### Empirical Contribution

It evaluates LLM-generated ego networks against a real longitudinal ego-network panel, using UCNets as a continuous human benchmark.

### Diagnostic Contribution

It separates failures across three stages:

```text
State
Network
Response / Dynamics
```

This allows the study to say not only whether the LLM-generated network is invalid, but also which stage is most associated with the observed error.

### Methodological Contribution

It provides a reusable **stage-restoration evaluation framework** for future generative social simulation studies.

### Substantive Expected Findings

Potential hypotheses include:

- LLMs may overestimate social stability.
- LLMs may underestimate alter loss and support loss.
- LLM-generated networks may be too family-centered or too supportive.
- LLMs may underestimate loneliness increase.
- LLMs may underestimate health shocks.
- Failure sources may differ by estimand.

These are hypotheses, not assumed results.

---
