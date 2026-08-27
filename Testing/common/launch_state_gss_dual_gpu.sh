#!/usr/bin/env bash
# Background Phase 1 re-run for GSS network years on dedicated GPUs.
#   GPU 2 + port 11435 -> gss_1987 (all tasks 1/1b/2/2b/3)
#   GPU 3 + port 11436 -> gss_2010_network (all tasks 1/1b/2/2b/3)
#
# Usage: bash Testing/scripts/launch_state_gss_dual_gpu.sh
set -euo pipefail

ROOT="/home/bld/data/data2/jiaxin/simulation"
COMMON="${ROOT}/Testing/common"
OUT="${ROOT}/Testing/logs/state"
MODEL="${MODEL:-llama3.1:latest}"
SLUG="${SLUG:-llama3.1_latest}"
WORKERS="${WORKERS:-8}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"

cd "${ROOT}"
mkdir -p "${OUT}/logs/${SLUG}" "${OUT}/logs/ollama_servers"

start_ollama() {
  local gpu="$1"
  local port="$2"
  local log="${OUT}/logs/ollama_servers/ollama_gpu${gpu}_port${port}.log"
  if ss -ltn 2>/dev/null | grep -q ":${port} "; then
    echo "[ollama] port ${port} already listening; reuse"
    return 0
  fi
  echo "[ollama] starting on GPU ${gpu} port ${port}"
  # System ollama service stores weights here (user ~/.ollama is empty).
  local models="${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"
  CUDA_VISIBLE_DEVICES="${gpu}" OLLAMA_HOST="127.0.0.1:${port}" OLLAMA_MODELS="${models}" \
    nohup ollama serve >"${log}" 2>&1 &
  echo $! >"${OUT}/logs/ollama_servers/ollama_gpu${gpu}_port${port}.pid"
  for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${port}/api/tags" >/dev/null 2>&1; then
      echo "[ollama] GPU ${gpu} ready on :${port}"
      return 0
    fi
    sleep 1
  done
  echo "[ollama] ERROR: GPU ${gpu} port ${port} did not become ready" >&2
  return 1
}

start_ollama 2 11435
start_ollama 3 11436

# Ensure model visible to each server (shared ~/.ollama)
curl -sf "http://127.0.0.1:11435/api/tags" | head -c 200 >/dev/null
curl -sf "http://127.0.0.1:11436/api/tags" | head -c 200 >/dev/null

launch_year() {
  local dataset="$1"
  local port="$2"
  local gpu="$3"
  local run_id="${RUN_TS}_${dataset}_gpu${gpu}_w${WORKERS}"
  local console="${OUT}/logs/${SLUG}/console_${run_id}.log"
  local session="phase1_${dataset}_gpu${gpu}"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[launch] kill existing tmux ${session}"
    tmux kill-session -t "${session}"
  fi

  local py_cmd=(
    python3 "${COMMON}/run_state_generation.py"
    --model "${MODEL}"
    --host "http://127.0.0.1:${port}"
    --dataset "${dataset}"
    --limit 0
    --workers "${WORKERS}"
    --resume
    --num-ctx "${NUM_CTX:-4096}"
    --temperature 0.7
    --num-predict 2048
    --timeout 300
    --run-id "${run_id}"
  )

  local joined="${py_cmd[*]}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && export PYTHONUNBUFFERED=1 && \
     nohup ${joined} > '${console}' 2>&1; \
     echo EXIT_CODE=\$? >> '${console}'; \
     echo FINISHED_AT=\$(date -u +%Y-%m-%dT%H:%M:%SZ) >> '${console}'; \
     exec bash"

  echo "[launch] dataset=${dataset} gpu=${gpu} port=${port} session=${session}"
  echo "[launch] console=${console}"
}

launch_year gss_1987 11435 2
launch_year gss_2010_network 11436 3

echo "[launch] both years started in background (tmux + nohup)."
echo "[launch] tmux ls; tail -f ${OUT}/logs/${SLUG}/console_${RUN_TS}_*_gpu*.log"
