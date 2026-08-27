#!/usr/bin/env bash
# Launch full-sample State generation in a detached tmux session (16 workers).
# Usage: bash Testing/scripts/launch_state_full.sh
set -euo pipefail

ROOT="/home/bld/data/data2/jiaxin/simulation"
COMMON="${ROOT}/Testing/common"
OUT="${ROOT}/Testing/logs/state"
MODEL="llama3.1:latest"
SLUG="llama3.1_latest"
SESSION="phase1_llama31_full"
WORKERS="${WORKERS:-16}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_full_w${WORKERS}"

cd "${ROOT}"
export PYTHONUNBUFFERED=1

LOG_DIR="${OUT}/logs/${SLUG}"
mkdir -p "${LOG_DIR}"
CONSOLE_LOG="${LOG_DIR}/console_${RUN_ID}.log"

PY_CMD=(
  python3 "${COMMON}/run_state_generation.py"
  --model "${MODEL}"
  --limit 0
  --workers "${WORKERS}"
  --resume
  --num-ctx "${NUM_CTX:-4096}"
  --temperature 0.7
  --num-predict 2048
  --timeout 300
  --run-id "${RUN_ID}"
)

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[launch] ERROR: tmux session '${SESSION}' already exists."
  echo "  attach: tmux attach -t ${SESSION}"
  echo "  kill:   tmux kill-session -t ${SESSION}"
  exit 1
fi

# Join command for the tmux shell string.
PY_JOINED="${PY_CMD[*]}"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && export PYTHONUNBUFFERED=1 && \
   nohup ${PY_JOINED} > '${CONSOLE_LOG}' 2>&1; \
   echo EXIT_CODE=\$? >> '${CONSOLE_LOG}'; \
   echo FINISHED_AT=\$(date -u +%Y-%m-%dT%H:%M:%SZ) >> '${CONSOLE_LOG}'; \
   exec bash"

echo "[launch] started tmux session: ${SESSION}"
echo "[launch] run_id: ${RUN_ID}"
echo "[launch] workers: ${WORKERS}"
echo "[launch] resume: on"
echo "[launch] console log: ${CONSOLE_LOG}"
echo "[launch] script log:  ${LOG_DIR}/${RUN_ID}.log"
echo "[launch] attach: tmux attach -t ${SESSION}"
echo "[launch] tail:   tail -f ${CONSOLE_LOG}"
