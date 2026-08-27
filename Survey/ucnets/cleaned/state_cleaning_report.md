# Phase 1 Data Cleaning Report

## Sample Sizes

### UCNets Wave 1 (main)

- UCNets Wave 1 raw / clean rows: 1159
- UCNets Task 1 JSONL rows: 1121
- UCNets Task 2 JSONL rows: 1120
- UCNets Task 3 JSONL rows: 631（条件化 `party_identity` / `religion_current` / `religion_raised` / `voting_frequency`；因 raised 非缺失要求相对旧 684 略少）

### GSS 1987 / 2010

| Dataset | ego N | Task 1 | Task 2 | Task 3 | Vote field |
|---|---:|---:|---:|---:|---|
| `gss_1987` | 1800 | 1787 | 1785 | 1738 | `vote84` → `voting_1984` |
| `gss_2010_network` | 1272 | 1251 | 853 | 1225 | `VOTE08` → `voting_2008` |

Source tables: `Survey/gss/network/{1987,2010}/ego.csv`. Cleaning script: `Testing/scripts/clean_state_gss.py`. Summary JSON: `gss_network_clean_summary.json`.

UCNets cleaning script: `scripts/clean_phase1_state_data.py` (UCNets only).

## Missingness: UCNets Targets

| Field | Missing | Total | Missing % |
|---|---:|---:|---:|
| `G1_label` | 2 | 1159 | 0.17 |
| `H10` | 3 | 1159 | 0.26 |
| `H12` | 3 | 1159 | 0.26 |
| `H14` | 3 | 1159 | 0.26 |
| `E1A_label` | 11 | 1159 | 0.95 |
| `E2A_label` | 0 | 1159 | 0.0 |
| `E3_label` | 20 | 1159 | 1.73 |
| `E4_label` | 32 | 1159 | 2.76 |
| `E5_label` | 18 | 1159 | 1.55 |

## GSS network notes

- **1987**: domains A–G nearly complete; Task 2 almost as large as Task 1 because `health` is observed for nearly all network respondents.
- **2010**: ballot missingness on `health` / `fair` / `helpful` shrinks Task 2 (853) relative to Task 1 (1251); Task 3 uses `voting_2008`.
- GSS trust triad (domain E) has no UCNets item equivalent; UCNets H–J have no GSS item equivalent.

## Caveats

- Align UCNets and GSS at the **construct-domain** level (A–G), not item equivalence.
