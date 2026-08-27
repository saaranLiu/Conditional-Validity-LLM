#!/usr/bin/env python3
"""Smoke-test ChatAnywhere OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import sys

from llm_client import (
    DEFAULT_CHATANYWHERE_BASE,
    LLMConfig,
    llm_generate,
    load_dotenv,
    resolve_api_key,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3.5-plus")
    parser.add_argument("--api-base", default=DEFAULT_CHATANYWHERE_BASE)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    load_dotenv()
    try:
        resolve_api_key(args.api_key, backend="chatanywhere")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Copy .env.example to .env and set CHATANYWHERE_API_KEY.", file=sys.stderr)
        return 1

    llm = LLMConfig(backend="chatanywhere", api_base=args.api_base, api_key=args.api_key)
    prompt = (
        'Return ONLY a JSON object: {"status":"ok","provider":"chatanywhere"}'
    )
    try:
        resp = llm_generate(
            llm=llm,
            model=args.model,
            prompt=prompt,
            temperature=0.0,
            num_predict=128,
            timeout_s=60.0,
        )
    except Exception as exc:  # noqa: BLE001 — surface provider errors to CLI
        print(f"API call failed: {exc}", file=sys.stderr)
        return 1

    raw = resp.get("response", "")
    print("model:", args.model)
    print("elapsed_s:", resp.get("_client_elapsed_s"))
    print("tokens:", resp.get("prompt_eval_count"), "+", resp.get("eval_count"))
    print("response:", raw[:500])
    try:
        obj = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        if obj.get("status") == "ok":
            print("OK: JSON parse succeeded")
            return 0
    except json.JSONDecodeError:
        pass
    print("WARN: response received but JSON smoke check did not pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
