#!/usr/bin/env python3
"""Smoke-test an OpenAI-compatible cloud API (ChatAnywhere or HKUST AIGC)."""

from __future__ import annotations

import argparse
import sys

from llm_client import LLMConfig, add_llm_cli_args, default_api_base, llm_config_from_args, llm_generate, load_dotenv, resolve_api_key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_llm_cli_args(parser)
    parser.add_argument(
        "--model",
        default=None,
        help="Model id (default: Qwen for hkust, qwen3.5-plus for chatanywhere)",
    )
    args = parser.parse_args()
    if args.backend == "ollama":
        print("ERROR: use --backend hkust or chatanywhere for this smoke test", file=sys.stderr)
        return 1

    if args.model is None:
        args.model = "Qwen" if args.backend == "hkust" else "qwen3.5-plus"

    load_dotenv()
    try:
        resolve_api_key(args.api_key, backend=args.backend)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("See .env.example for required env vars.", file=sys.stderr)
        return 1

    llm = llm_config_from_args(args)
    prompt = "hello." if args.backend == "hkust" else (
        f'Return ONLY a JSON object: {{"status":"ok","backend":"{args.backend}"}}'
    )
    temperature = 1.0 if args.backend == "hkust" else 0.0
    try:
        resp = llm_generate(
            llm=llm,
            model=args.model,
            prompt=prompt,
            temperature=temperature,
            num_predict=256,
            timeout_s=60.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"API call failed: {exc}", file=sys.stderr)
        return 1

    raw = resp.get("response", "")
    print("backend:", llm.backend)
    print("api_base:", llm.api_base)
    print("model:", args.model)
    print("elapsed_s:", resp.get("_client_elapsed_s"))
    print("tokens:", resp.get("prompt_eval_count"), "+", resp.get("eval_count"))
    print("finish_reason:", resp.get("_finish_reason"))
    print("message_keys:", resp.get("_message_keys"))
    print("response:", raw[:500])
    if raw.strip():
        print("OK: got a non-empty response")
        return 0
    raw_body = resp.get("_raw") or {}
    if raw_body.get("error"):
        print("API error:", raw_body["error"], file=sys.stderr)
        return 1
    print("FAIL: empty response (likely thinking tokens with content=null; retry after update)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
