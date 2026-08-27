"""Unified LLM client for local Ollama and OpenAI-compatible cloud APIs."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_CHATANYWHERE_BASE = "https://api.chatanywhere.tech/v1"
DEFAULT_HKUST_AIGC_BASE = "https://aigc-api.hkust-gz.edu.cn/v1"

OPENAI_COMPAT_BACKENDS = frozenset({"chatanywhere", "hkust"})


def extract_chat_content(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pull assistant text from an OpenAI-style chat/completions body."""
    choices = body.get("choices") or []
    if not choices:
        return "", {}
    choice = choices[0]
    message = choice.get("message") or {}
    # Prefer final answer fields; some Qwen gateways leave content empty while
    # spending the token budget on reasoning/thinking channels.
    for key in ("content", "reasoning_content", "reasoning"):
        val = message.get(key)
        if isinstance(val, str) and val.strip():
            return val, choice
    text = choice.get("text")
    if isinstance(text, str) and text.strip():
        return text, choice
    return "", choice


def _apply_provider_payload_tweaks(
    payload: dict[str, Any],
    *,
    provider: str,
    model: str,
) -> None:
    name = model.casefold()
    if provider != "hkust":
        return
    # Qwen thinking burns max_tokens with empty content; DeepSeek rejects
    # chat_template_kwargs on this gateway — Qwen-only tweaks.
    if "qwen" in name:
        payload["enable_thinking"] = False
        kwargs = payload.setdefault("chat_template_kwargs", {})
        if isinstance(kwargs, dict):
            kwargs["enable_thinking"] = False
    # OpenAI o-series / GPT-5.x on HKUST reject legacy max_tokens.
    if name.startswith("gpt-5") or name.startswith("o"):
        if "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        # Gateway only accepts default temperature (=1); omit explicit value.
        payload.pop("temperature", None)


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "ollama"
    host: str = DEFAULT_OLLAMA_HOST
    api_base: str = DEFAULT_CHATANYWHERE_BASE
    api_key: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_api_base(backend: str) -> str:
    if backend == "hkust":
        return DEFAULT_HKUST_AIGC_BASE
    if backend == "chatanywhere":
        return DEFAULT_CHATANYWHERE_BASE
    return DEFAULT_CHATANYWHERE_BASE


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ (does not override existing)."""
    env_path = path or (_repo_root() / ".env")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve_api_key(api_key: str | None = None, *, backend: str = "chatanywhere") -> str:
    load_dotenv()
    if api_key:
        return api_key
    env_keys: list[str]
    if backend == "hkust":
        env_keys = ["HKUST_AIGC_API_KEY", "OPENAI_API_KEY"]
    elif backend == "chatanywhere":
        env_keys = ["CHATANYWHERE_API_KEY", "OPENAI_API_KEY"]
    else:
        env_keys = ["OPENAI_API_KEY"]
    for name in env_keys:
        val = os.environ.get(name)
        if val:
            return val
    if backend == "hkust":
        hint = "HKUST_AIGC_API_KEY"
    elif backend == "chatanywhere":
        hint = "CHATANYWHERE_API_KEY"
    else:
        hint = "OPENAI_API_KEY"
    raise ValueError(f"Missing API key: set {hint} or pass --api-key")


def add_llm_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("ollama", "chatanywhere", "hkust"),
        default="ollama",
        help="ollama=local; chatanywhere/hkust=OpenAI /v1/chat/completions",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_OLLAMA_HOST,
        help="Ollama base URL when --backend=ollama",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="OpenAI-compatible base URL (default depends on --backend)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (else backend-specific env var from .env)",
    )


def llm_config_from_args(args: argparse.Namespace) -> LLMConfig:
    api_base = args.api_base or default_api_base(args.backend)
    return LLMConfig(
        backend=args.backend,
        host=args.host,
        api_base=api_base,
        api_key=args.api_key,
    )


def ollama_generate(
    *,
    host: str,
    model: str,
    prompt: str,
    temperature: float,
    num_predict: int,
    timeout_s: float,
    num_ctx: int = 4096,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Top-level, not options: Qwen3 otherwise spends num_predict on thinking
        # and returns an empty JSON body.
        "think": False,
        "keep_alive": "24h",
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    body["_client_elapsed_s"] = time.perf_counter() - started
    return body


def chat_completions_generate(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    num_predict: int,
    timeout_s: float,
    provider: str = "openai",
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": num_predict,
        "stream": False,
    }
    _apply_provider_payload_tweaks(payload, provider=provider, model=model)
    base = api_base.rstrip("/")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    started = time.perf_counter()
    body: dict[str, Any] = {}
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            api_err = body.get("error")
            if api_err:
                msg = api_err.get("message", api_err) if isinstance(api_err, dict) else str(api_err)
                code = api_err.get("code") if isinstance(api_err, dict) else None
                retryable = str(code) == "429" or (
                    isinstance(msg, str) and "try again" in msg.lower()
                )
                if retryable and attempt < 3:
                    time.sleep(min(15.0, 3.0 * (attempt + 1)))
                    continue
                raise RuntimeError(f"API error ({code}): {msg}")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 3:
                time.sleep(min(15.0, 3.0 * (attempt + 1)))
                continue
            raise
    elapsed = time.perf_counter() - started

    content, choice = extract_chat_content(body)
    usage = body.get("usage") or {}
    return {
        "response": content,
        "eval_count": usage.get("completion_tokens"),
        "prompt_eval_count": usage.get("prompt_tokens"),
        "_client_elapsed_s": elapsed,
        "_provider": provider,
        "_finish_reason": choice.get("finish_reason"),
        "_message_keys": list((choice.get("message") or {}).keys()),
        "_raw": body,
    }


def llm_generate(
    *,
    llm: LLMConfig,
    model: str,
    prompt: str,
    temperature: float,
    num_predict: int,
    timeout_s: float,
    num_ctx: int = 4096,
) -> dict[str, Any]:
    if llm.backend == "ollama":
        return ollama_generate(
            host=llm.host,
            model=model,
            prompt=prompt,
            temperature=temperature,
            num_predict=num_predict,
            timeout_s=timeout_s,
            num_ctx=num_ctx,
        )
    if llm.backend in OPENAI_COMPAT_BACKENDS:
        return chat_completions_generate(
            api_base=llm.api_base,
            api_key=resolve_api_key(llm.api_key, backend=llm.backend),
            model=model,
            prompt=prompt,
            temperature=temperature,
            num_predict=num_predict,
            timeout_s=timeout_s,
            provider=llm.backend,
        )
    raise ValueError(f"Unknown backend: {llm.backend}")
