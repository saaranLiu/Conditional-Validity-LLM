#!/usr/bin/env bash
# Launch T4 R1 (N_L | S_L) on GPU3 ollama via tmux.
#
# Usage:
#   bash Testing/scripts/launch_r1_ucnets_gpu3.sh
#
# Env overrides: MODEL WORKERS SESSION HOST NUM_CTX PY
set -euo pipefail

ROOT="/home/bld/data/data2/jiaxin/simulation"
T4="${ROOT}/Testing/T4_individual_resample/ucnets_w1"
SCRIPTS="${ROOT}/Testing/scripts"
OUT="${T4}/output"
PY="${PY:-/home/bld/anaconda3/bin/python}"
MODEL="${MODEL:-llama3.1:latest}"
SLUG="${SLUG:-llama3.1_latest}"
SESSION="${SESSION:-t4_r1_ucnets_gpu3}"
WORKERS="${WORKERS:-16}"
HOST="${HOST:-http://127.0.0.1:11436}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_t4_r1_gpu3_w${WORKERS}"

cd "${ROOT}"
export PYTHONUNBUFFERED=1

LOG_DIR="${ROOT}/Testing/logs/network/logs/${SLUG}"
mkdir -p "${LOG_DIR}" "${ROOT}/Testing/logs/network/logs/ollama_servers" "${T4}/input"

echo "[launch] building R1 inputs if needed..."
"${PY}" "${SCRIPTS}/build_r1_inputs.py"

if ! curl -sf "${HOST}/api/tags" >/dev/null; then
  echo "[launch] ollama not reachable at ${HOST}; starting on GPU3..."
  MODELS="${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"
  CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=127.0.0.1:11436 OLLAMA_MODELS="${MODELS}" \
    nohup ollama serve >"${ROOT}/Testing/logs/network/logs/ollama_servers/ollama_gpu3_port11436.log" 2>&1 &
  echo $! >"${ROOT}/Testing/logs/network/logs/ollama_servers/ollama_gpu3_port11436.pid"
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
else
  echo "[launch] ollama already up at ${HOST}"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[launch] ERROR: tmux session '${SESSION}' already exists."
  echo "  attach: tmux attach -t ${SESSION}"
  echo "  kill:   tmux kill-session -t ${SESSION}"
  exit 1
fi

CONSOLE_LOG="${LOG_DIR}/console_${RUN_ID}.log"

PY_CMD=(
  "${PY}" "${SCRIPTS}/run_network_generation.py"
  --model "${MODEL}"
  --host "${HOST}"
  --input-dir "${T4}/input"
  --output-dir "${OUT}"
  --config "${ROOT}/Testing/configs/network/ucnets_w1.yaml"
  --limit 0
  --workers "${WORKERS}"
  --resume
  --num-ctx "${NUM_CTX:-4096}"
  --temperature 0.7
  --num-predict 2048
  --timeout 300
  --run-id "${RUN_ID}"
  --dataset ucnets_w1
  --task r1_llm_state_network
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
echo "[launch] task: r1_llm_state_network (UCNets only)"
echo "[launch] console log: ${CONSOLE_LOG}"
echo "[launch] attach: tmux attach -t ${SESSION}"
echo "[launch] tail:   tail -f ${CONSOLE_LOG}"
