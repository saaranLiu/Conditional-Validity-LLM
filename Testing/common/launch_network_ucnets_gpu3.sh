#!/usr/bin/env bash
# Launch UCNets network generation N0–N3 (no N0b) on GPU3 ollama.
#
# Usage:
#   bash Testing/scripts/launch_network_ucnets_gpu3.sh
#
# Env overrides:
#   MODEL WORKERS SESSION HOST NUM_CTX
set -euo pipefail

ROOT="/home/bld/data/data2/jiaxin/simulation"
COMMON="${ROOT}/Testing/common"
OUT="${ROOT}/Testing/logs/network"
PY="${PY:-/home/bld/anaconda3/bin/python}"
MODEL="${MODEL:-llama3.1:latest}"
SLUG="${SLUG:-llama3.1_latest}"
SESSION="${SESSION:-phase2_ucnets_gpu3}"
WORKERS="${WORKERS:-16}"
HOST="${HOST:-http://127.0.0.1:11436}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_ucnets_b1234_gpu3_w${WORKERS}"

cd "${ROOT}"
export PYTHONUNBUFFERED=1

LOG_DIR="${OUT}/logs/${SLUG}"
mkdir -p "${LOG_DIR}" "${OUT}/logs/ollama_servers"
CONSOLE_LOG="${LOG_DIR}/console_${RUN_ID}.log"

if ! curl -sf "${HOST}/api/tags" >/dev/null; then
  echo "[launch] ollama not reachable at ${HOST}; starting on GPU3..."
  MODELS="${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"
  CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=127.0.0.1:11436 OLLAMA_MODELS="${MODELS}" \
    nohup ollama serve >"${OUT}/logs/ollama_servers/ollama_gpu3_port11436.log" 2>&1 &
  echo $! >"${OUT}/logs/ollama_servers/ollama_gpu3_port11436.pid"
  for i in $(seq 1 60); do
    if curl -sf "${HOST}/api/tags" >/dev/null; then
      echo "[launch] ollama ready"
      break
    fi
    sleep 1
  done
  if ! curl -sf "${HOST}/api/tags" >/dev/null; then
    echo "[launch] ERROR: failed to start ollama on ${HOST}"
    exit 1
  fi
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[launch] ERROR: tmux session '${SESSION}' already exists."
  echo "  attach: tmux attach -t ${SESSION}"
  echo "  kill:   tmux kill-session -t ${SESSION}"
  exit 1
fi

PY_CMD=(
  "${PY}" "${COMMON}/run_network_generation.py"
  --model "${MODEL}"
  --host "${HOST}"
  --limit 0
  --workers "${WORKERS}"
  --resume
  --num-ctx "${NUM_CTX:-4096}"
  --temperature 0.7
  --num-predict 2048
  --timeout 300
  --run-id "${RUN_ID}"
  --dataset ucnets_w1
  --task n0_baseline_network
  --task n1_size_bucket
  --task n2_population_prior
  --task n3_roster_multistep
)

PY_JOINED="${PY_CMD[*]}"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && export PYTHONUNBUFFERED=1 && \
   nohup ${PY_JOINED} > '${CONSOLE_LOG}' 2>&1; \
   echo EXIT_CODE=\$? >> '${CONSOLE_LOG}'; \
   echo FINISHED_AT=\$(date -u +%Y-%m-%dT%H:%M:%SZ) >> '${CONSOLE_LOG}'; \
   exec bash"

echo "[launch] started tmux session: ${SESSION}"
echo "[launch] run_id: ${RUN_ID}"
echo "[launch] host: ${HOST} (GPU3)"
echo "[launch] workers: ${WORKERS}"
echo "[launch] tasks: B1 B2 B3 B4 (no B1b)"
echo "[launch] console log: ${CONSOLE_LOG}"
echo "[launch] script log:  ${LOG_DIR}/${RUN_ID}.log"
echo "[launch] attach: tmux attach -t ${SESSION}"
echo "[launch] tail:   tail -f ${CONSOLE_LOG}"
