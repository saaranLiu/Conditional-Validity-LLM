#!/usr/bin/env bash
# ISSP 2017 T6 country-resource generation on GPU1 (existing ollama :11438).
#
#   80 people x 30 countries = 2400 oneshot resource_summary prompts.
#   Does not use GPU0/2/3 (NSHAP / 1985 T4).
#
# Usage:
#   bash Testing/scripts/launch_t6_issp_gpu1.sh
set -euo pipefail

ROOT="/home/bld/data/data2/jiaxin/simulation"
SCRIPTS="${ROOT}/Testing/scripts"
PY="${PY:-/home/bld/anaconda3/bin/python}"
MODEL="${MODEL:-llama3.1:latest}"
SLUG="${SLUG:-llama3.1_latest}"
HOST="${HOST:-http://127.0.0.1:11438}"
WORKERS="${WORKERS:-16}"
SESSION="${SESSION:-t6_issp_gpu1}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"

cd "${ROOT}"
export PYTHONUNBUFFERED=1

if ! curl -sf --max-time 3 "${HOST}/api/tags" >/dev/null; then
  echo "[launch] ERROR: ollama not reachable at ${HOST}" >&2
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[launch] ERROR: tmux session '${SESSION}' already exists."
  echo "  attach: tmux attach -t ${SESSION}"
  echo "  kill:   tmux kill-session -t ${SESSION}"
  exit 1
fi

echo "[launch] building ISSP T6 inputs (80 per country)..."
"${PY}" "${SCRIPTS}/build_network_inputs.py" --dataset issp_2017

LOG_DIR="${ROOT}/Testing/logs/network/logs/${SLUG}"
mkdir -p "${LOG_DIR}"
CONSOLE="${LOG_DIR}/console_${RUN_TS}_t6_issp_gpu1.log"

CMD="${PY} ${SCRIPTS}/run_network_generation.py --model ${MODEL} --host ${HOST} --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 1024 --timeout 300 --run-id ${RUN_TS}_issp_2017_t6_gpu1 --dataset issp_2017 --task t6_country_resources"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && export PYTHONUNBUFFERED=1 && \
   { echo '[t6] ISSP start' && ${CMD} && echo '[t6] ISSP done' ; \
     echo EXIT_CODE=\$? >> '${CONSOLE}'; echo FINISHED_AT=\$(date -u +%Y-%m-%dT%H:%M:%SZ) >> '${CONSOLE}'; } \
   > '${CONSOLE}' 2>&1; exec bash"

echo "[launch] session=${SESSION} run_ts=${RUN_TS}"
echo "[launch] host=${HOST} workers=${WORKERS}"
echo "[launch] console=${CONSOLE}"
echo "[launch] attach: tmux attach -t ${SESSION}"
echo "[launch] tail:   tail -f ${CONSOLE}"
