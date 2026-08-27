# Testing content configs

SSDataBench-style YAML：`state/` 与 `network/` 各一份 `{dataset}.yaml`。臂名见 [`../NAMING.md`](../NAMING.md)。

## What lives here

- `state/ucnets_w1.yaml`、`state/gss_1987.yaml`、`state/gss_2010_network.yaml`
- `network/{dataset}.yaml` — 各 survey 自己的臂、词表、roster、区间切点

## What still lives in JSONL

- `Survey/{survey}/cleaned/*_{s0,s1,s2}.jsonl` — per-respondent `prompt_json` and human `target_json`
- `Testing/T*/{dataset}/input/*_input.jsonl` — LLM inputs（不含 target）

## Edit workflow

1. Update variable definitions / allowed values / descriptions in the YAML file.
2. Re-run cleaning:
   - UCNets state: `python Testing/scripts/clean_state_ucnets.py`
   - GSS state: `python Testing/scripts/clean_state_gss.py`
   - Network: `python Testing/scripts/clean_network_{ucnets,gss,nshap,signos}.py`
3. Rebuild model inputs:
   `python Testing/scripts/build_state_inputs.py` 或 `build_network_inputs.py`

The builder reads schema and prompt text from YAML. It does **not** re-derive allowed values from data.

## Prompt tiers (state)

| Tier | Arms | `prompt_condition` | Rules |
|---|---|---|---|
| **Baseline (primary)** | S0, S1, S2 | `format_only` | baseline rules 1–2 |
| **Supplementary** | S0b, S1b | `format_anti_typology` | baseline + anti-typology rules 3–4 |

S0b reuses S0 cleaned JSONL via `reuse_task_file`.
S1b reuses S1 cleaned JSONL via `reuse_task_file`.
