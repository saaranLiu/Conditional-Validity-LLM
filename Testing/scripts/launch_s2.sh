#!/usr/bin/env bash
# Launch S2 (party/religion/voting conditioned) in a detached tmux session.
# Does not touch S0/S0b/S1/S1b.
#
# Usage:
#   bash Testing/scripts/launch_s2.sh
set -euo pipefail

ROOT="/home/bld/data/data2/jiaxin/simulation"
SCRIPTS="${ROOT}/Testing/scripts"
OUT="${ROOT}/Testing/logs/state"
MODEL="${MODEL:-llama3.1:latest}"
SLUG="${SLUG:-llama3.1_latest}"
SESSION="${SESSION:-phase1_task3_llama31}"
WORKERS="${WORKERS:-16}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_task3_w${WORKERS}"
TASK="s2_politics_religion_conditioned"

cd "${ROOT}"
export PYTHONUNBUFFERED=1

LOG_DIR="${OUT}/logs/${SLUG}"
mkdir -p "${LOG_DIR}"
CONSOLE_LOG="${LOG_DIR}/console_${RUN_ID}.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[launch] ERROR: tmux session '${SESSION}' already exists."
  echo "  attach: tmux attach -t ${SESSION}"
  echo "  kill:   tmux kill-session -t ${SESSION}"
  exit 1
fi

PY_CMD=(
  python3 "${SCRIPTS}/run_state_generation.py"
  --model "${MODEL}"
  --limit 0
  --workers "${WORKERS}"
  --resume
  --num-ctx "${NUM_CTX:-4096}"
  --temperature 0.7
  --num-predict 2048
  --timeout 300
  --run-id "${RUN_ID}"
  --task "${TASK}"
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
echo "[launch] task: ${TASK}"
echo "[launch] workers: ${WORKERS}"
echo "[launch] console log: ${CONSOLE_LOG}"
echo "[launch] script log:  ${LOG_DIR}/${RUN_ID}.log"
echo "[launch] attach: tmux attach -t ${SESSION}"
echo "[launch] tail:   tail -f ${CONSOLE_LOG}"
