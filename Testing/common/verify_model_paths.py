#!/usr/bin/env python3
"""Preflight checks for generation + R1 path consistency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paths import TESTING, arm_output_jsonl, model_slug

R1_DATASETS = ("ucnets_w1", "gss_1985", "nshap_r1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Generation --model name")
    parser.add_argument(
        "--log-slug",
        default=None,
        help="Optional tmux/log slug (must differ from output slug when set)",
    )
    parser.add_argument(
        "--require-s0",
        action="store_true",
        help="Fail if S0 outputs are missing (use before R1 backfill)",
    )
    args = parser.parse_args()

    slug = model_slug(args.model)
    errors: list[str] = []
    warnings: list[str] = []

    if args.log_slug and args.log_slug == slug:
        errors.append(
            f"log slug '{args.log_slug}' equals output slug '{slug}'; "
            "use log slug only for tmux/logs (e.g. hkust_gpt53), not for output paths"
        )

    print(f"model={args.model}")
    print(f"output_slug={slug}")
    if args.log_slug:
        print(f"log_slug={args.log_slug}")

    for ds in R1_DATASETS:
        s0 = arm_output_jsonl(ds, "s0_baseline_state", slug)
        r1_in = TESTING / "T4_individual_resample" / ds / "input" / slug / "r1_input.jsonl"
        r1_out = arm_output_jsonl(ds, "r1_llm_state_network", slug)
        print(f"  [{ds}] S0 -> {s0}")
        print(f"  [{ds}] R1 in -> {r1_in}")
        print(f"  [{ds}] R1 out -> {r1_out}")
        if args.require_s0 and not s0.exists():
            errors.append(f"missing S0 output for {ds}: {s0}")

    report = {
        "model": args.model,
        "output_slug": slug,
        "log_slug": args.log_slug,
        "r1_datasets": list(R1_DATASETS),
        "errors": errors,
        "warnings": warnings,
    }
    if errors:
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
