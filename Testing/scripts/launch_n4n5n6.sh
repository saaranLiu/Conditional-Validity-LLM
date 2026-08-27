#!/usr/bin/env bash
# Parallel UCNets generation for N4/N5/N6.
#
# N4: homophily-conditioned oneshot (full N=1156, ~90 min)
# N5: N1+N3 combo, multistep_roster (subsample N=400, ~2-3h)
# N6: N2+N3 combo, multistep_roster (subsample N=400, ~2-3h)
#
# Default mapping:
#   GPU3 :11436 -> B5
#   GPU2 :11435 -> B6
#   GPU0 :11437 -> B7
#
# Usage:
#   bash Testing/scripts/launch_n4n5n6.sh
set -euo pipefail

ROOT="/home/bld/data/data2/jiaxin/simulation"
SCRIPTS="${ROOT}/Testing/scripts"
OUT="${ROOT}/Testing/logs/network"
PY="${PY:-/home/bld/anaconda3/bin/python}"
MODEL="${MODEL:-llama3.1:latest}"
SLUG="${SLUG:-llama3.1_latest}"
WORKERS="${WORKERS:-16}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
MODELS="${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"

cd "${ROOT}"
mkdir -p "${OUT}/logs/${SLUG}" "${OUT}/logs/ollama_servers"
export PYTHONUNBUFFERED=1

start_ollama() {
  local gpu="$1"
  local port="$2"
  local log="${OUT}/logs/ollama_servers/ollama_gpu${gpu}_port${port}.log"
  if ss -ltn 2>/dev/null | grep -q ":${port} "; then
    echo "[ollama] port ${port} already listening; reuse"
    return 0
  fi
  echo "[ollama] starting on GPU ${gpu} port ${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" OLLAMA_HOST="127.0.0.1:${port}" OLLAMA_MODELS="${MODELS}" \
    nohup ollama serve >"${log}" 2>&1 &
  echo $! >"${OUT}/logs/ollama_servers/ollama_gpu${gpu}_port${port}.pid"
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${port}/api/tags" >/dev/null 2>&1; then
      echo "[ollama] GPU ${gpu} ready on :${port}"
      return 0
    fi
    sleep 1
  done
  echo "[ollama] ERROR: GPU ${gpu} port ${port} did not become ready" >&2
  return 1
}

launch_task() {
  local task="$1"
  local port="$2"
  local gpu="$3"
  local short="$4"
  local session="phase2_${short}_gpu${gpu}"
  local run_id="${RUN_TS}_${short}_gpu${gpu}_w${WORKERS}"
  local console="${OUT}/logs/${SLUG}/console_${run_id}.log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[launch] kill existing tmux ${session}"
    tmux kill-session -t "${session}"
  fi

  local py_cmd=(
    "${PY}" "${SCRIPTS}/run_network_generation.py"
    --model "${MODEL}"
    --host "http://127.0.0.1:${port}"
    --limit 0
    --workers "${WORKERS}"
    --resume
    --num-ctx "${NUM_CTX:-4096}"
    --temperature 0.7
    --num-predict 2048
    --timeout 300
    --run-id "${run_id}"
    --dataset ucnets_w1
    --task "${task}"
  )
  local joined="${py_cmd[*]}"

  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && export PYTHONUNBUFFERED=1 && \
     nohup ${joined} > '${console}' 2>&1; \
     echo EXIT_CODE=\$? >> '${console}'; \
     echo FINISHED_AT=\$(date -u +%Y-%m-%dT%H:%M:%SZ) >> '${console}'; \
     exec bash"

  echo "[launch] ${short} -> GPU${gpu} :${port} session=${session}"
  echo "         console: ${console}"
}

# Ensure ollama servers (reuse if already running from a prior phase2 launch)
start_ollama 3 11436
start_ollama 2 11435
start_ollama 0 11437

# Warm tags (confirm models visible)
for port in 11436 11435 11437; do
  curl -sf "http://127.0.0.1:${port}/api/tags" >/dev/null
done

launch_task n4_homophily 11436 3 b5
launch_task n5_size_multistep 11435 2 b6
launch_task n6_prior_multistep 11437 0 b7

echo "[launch] all three tasks started (workers=${WORKERS} each)"
echo "[launch] tmux ls | grep phase2_"
tmux ls 2>/dev/null | grep phase2_ || true
