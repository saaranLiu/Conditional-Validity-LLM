#!/usr/bin/env python3
"""Download Llama-3.1-8B-Instruct via HF mirror (preferred on this host)."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path("/home/bld/data/data2/jiaxin/simulation")
HF_LOCAL = ROOT / "Env" / "models" / "Meta-Llama-3.1-8B-Instruct"

NEED = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
}


def needed_ok(path: Path) -> bool:
    names = {p.name for p in path.iterdir() if p.is_file()}
    missing = sorted(NEED - names)
    if missing:
        print("missing:", missing, flush=True)
        return False
    return True


def main() -> int:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download

    HF_LOCAL.mkdir(parents=True, exist_ok=True)
    print("=== HF mirror unsloth/Meta-Llama-3.1-8B-Instruct ===", flush=True)
    try:
        path = snapshot_download(
            "unsloth/Meta-Llama-3.1-8B-Instruct",
            local_dir=str(HF_LOCAL),
            token=False,
            max_workers=1,
            ignore_patterns=["*.pth", "original/*", "*.bin", "*.gguf"],
        )
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return 1
    path = Path(path)
    print("NEEDED_OK" if needed_ok(path) else "NEEDED_INCOMPLETE", flush=True)
    print(f"DONE path={path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
