#!/usr/bin/env bash
# Sequential resume download from hf-mirror (faster than modelscope / hub client here).
set -eu
DEST=/home/bld/data/data2/jiaxin/simulation/Env/models/Meta-Llama-3.1-8B-Instruct
LOGDIR=/home/bld/data/data2/jiaxin/simulation/Env/logs
BASE=https://hf-mirror.com/unsloth/Meta-Llama-3.1-8B-Instruct/resolve/main
mkdir -p "$DEST" "$LOGDIR"

# expected sizes
declare -A EXP=(
  [model-00001-of-00004.safetensors]=4976698672
  [model-00002-of-00004.safetensors]=4999802720
  [model-00003-of-00004.safetensors]=4915916176
  [model-00004-of-00004.safetensors]=1168138808
)

# remove truncated incomplete shards so curl does clean resume from 0 if corrupt
for f in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors; do
  if [[ -f "$DEST/$f" ]]; then
    got=$(stat -c%s "$DEST/$f")
    exp=${EXP[$f]}
    if [[ "$got" -ne "$exp" ]]; then
      echo "[$(date +%H:%M:%S)] remove truncated $f ($got != $exp)"
      rm -f "$DEST/$f"
    else
      echo "[$(date +%H:%M:%S)] keep complete $f"
    fi
  fi
done

download_one() {
  local f=$1
  local exp=${EXP[$f]}
  local out=$DEST/$f
  local log=$LOGDIR/curl_$f.log
  if [[ -f "$out" ]] && [[ "$(stat -c%s "$out")" -eq "$exp" ]]; then
    echo "[$(date +%H:%M:%S)] skip $f"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] start $f" | tee -a "$log"
  # single connection, resume, long retries; avoid parallel 429
  curl -L --retry 50 --retry-delay 5 --retry-all-errors \
    -C - --connect-timeout 30 --fail \
    -o "$out" "$BASE/$f" >>"$log" 2>&1
  local ec=$?
  local sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
  echo "[$(date +%H:%M:%S)] done $f exit=$ec size=$sz exp=$exp" | tee -a "$log"
  [[ "$ec" -eq 0 && "$sz" -eq "$exp" ]]
}

for f in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  download_one "$f" || { echo FAIL "$f"; exit 1; }
done

# small sidecar files
for f in config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json model.safetensors.index.json README.md; do
  if [[ ! -f "$DEST/$f" ]]; then
    curl -L --fail -o "$DEST/$f" "$BASE/$f" || true
  fi
done

echo ALL_OK
ls -lh "$DEST"
