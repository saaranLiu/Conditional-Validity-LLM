#!/usr/bin/env python3
"""Build NSHAP r1 S0 gold from network-gold state_json.

X (kept human on R0 and R1): age, gender, race, education, marital_status.
S (LLM-generated for R1): self_rated_health + UCLA loneliness items.

Rows missing any X or S field are dropped. loneliness_scale is derived later
when R1 prompts are built, so it is not an S0 target.
"""

from __future__ import annotations

import json
from pathlib import Path

from paths import network_gold

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Survey" / "nshap" / "cleaned" / "nshap_r1_s0.jsonl"

X_KEYS = ("age", "gender", "race", "education", "marital_status")
S_KEYS = ("self_rated_health", "lacks_companionship", "feels_left_out", "feels_isolated")


def main() -> int:
    kept = 0
    dropped = 0
    with OUT.open("w", encoding="utf-8") as out_f:
        with network_gold("nshap_r1").open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                state = rec.get("state_json") or {}
                if any(state.get(k) is None for k in (*X_KEYS, *S_KEYS)):
                    dropped += 1
                    continue
                row = {
                    "respondent_id": rec["respondent_id"],
                    "dataset": "nshap_r1",
                    "task": "s0_baseline_state",
                    "prompt_json": {k: state[k] for k in X_KEYS},
                    "target_json": {k: state[k] for k in S_KEYS},
                    "metadata": {
                        "weight": (rec.get("metadata") or {}).get("weight"),
                        "sensitive_fields_included": True,
                        "missing_target_fields": [],
                        "source": "nshap_r1_network_gold.state_json",
                    },
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    print(json.dumps({"wrote": str(OUT), "n_kept": kept, "n_dropped_incomplete": dropped}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
