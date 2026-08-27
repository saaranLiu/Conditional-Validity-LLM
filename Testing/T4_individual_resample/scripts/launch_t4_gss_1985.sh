#!/usr/bin/env bash
# GSS 1985 T4: S0 then R1 on GPU3 (existing ollama :11436).
#
#   S0  -> Testing/T1_association/gss_1985/output/.../s0_baseline_state/
#   R1  -> Testing/T4_individual_resample/gss_1985/output/.../r1_llm_state_network/
#   R0  = T3 N0 (already generated)
#
# Usage:
#   bash Testing/scripts/launch_t4_gss_1985.sh
set -euo pipefail

ROOT="/home/bld/data/data2/jiaxin/simulation"
COMMON="${ROOT}/Testing/common"
PY="${PY:-/home/bld/anaconda3/bin/python}"
MODEL="${MODEL:-llama3.1:latest}"
SLUG="${SLUG:-llama3.1_latest}"
HOST="${HOST:-http://127.0.0.1:11436}"
WORKERS="${WORKERS:-16}"
SESSION="${SESSION:-t4_gss1985_gpu3}"
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

echo "[launch] building GSS 1985 S0 inputs..."
"${PY}" "${COMMON}/build_state_inputs.py" --dataset gss_1985 --task s0_baseline_state

STATE_LOG_DIR="${ROOT}/Testing/logs/state/logs/${SLUG}"
NET_LOG_DIR="${ROOT}/Testing/logs/network/logs/${SLUG}"
mkdir -p "${STATE_LOG_DIR}" "${NET_LOG_DIR}"
CONSOLE="${NET_LOG_DIR}/console_${RUN_TS}_t4_gss1985_gpu3.log"

S0_CMD="${PY} ${COMMON}/run_state_generation.py --model ${MODEL} --host ${HOST} --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_gss_1985_s0_gpu3 --dataset gss_1985 --task s0_baseline_state"
R1_BUILD="${PY} ${ROOT}/Testing/T4_individual_resample/scripts/build_r1_inputs.py --dataset gss_1985"
R1_CMD="${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host ${HOST} --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_gss_1985_r1_gpu3 --dataset gss_1985 --task r1_llm_state_network"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && export PYTHONUNBUFFERED=1 && \
   { echo '[t4] S0 start' && ${S0_CMD} && echo '[t4] S0 done; building R1' && ${R1_BUILD} && echo '[t4] R1 start' && ${R1_CMD} && echo '[t4] R1 done' ; \
     echo EXIT_CODE=\$? >> '${CONSOLE}'; echo FINISHED_AT=\$(date -u +%Y-%m-%dT%H:%M:%SZ) >> '${CONSOLE}'; } \
   > '${CONSOLE}' 2>&1; exec bash"

echo "[launch] session=${SESSION}  run_ts=${RUN_TS}"
echo "[launch] host=${HOST} workers=${WORKERS}"
echo "[launch] console=${CONSOLE}"
echo "[launch] attach: tmux attach -t ${SESSION}"
echo "[launch] tail:   tail -f ${CONSOLE}"
