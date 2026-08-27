#!/usr/bin/env bash
# NSHAP r1 T4 at full eligible N, plus the T3 N0 remainder so R0 covers all gold.
#
#   GPU2 :11435  S0 (2327 complete X+S) → R1
#   GPU0 :11437  N0 remainder (~1931 not in the T3 1000 subsample)
#   GPU3 is reserved for GSS 1985 T4.
#
# Usage:
#   bash Testing/scripts/launch_t4_nshap_r1.sh
set -euo pipefail

ROOT="/home/bld/data/data2/jiaxin/simulation"
SCRIPTS="${ROOT}/Testing/scripts"
PY="${PY:-/home/bld/anaconda3/bin/python}"
MODEL="${MODEL:-llama3.1:latest}"
SLUG="${SLUG:-llama3.1_latest}"
WORKERS="${WORKERS:-16}"
HOST_S0="${HOST_S0:-http://127.0.0.1:11435}"
HOST_N0="${HOST_N0:-http://127.0.0.1:11437}"
SESSION_S0="${SESSION_S0:-t4_nshap_s0r1_gpu2}"
SESSION_N0="${SESSION_N0:-t4_nshap_n0rem_gpu0}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"

cd "${ROOT}"
export PYTHONUNBUFFERED=1

need_ollama() {
  local host="$1"
  if ! curl -sf --max-time 3 "${host}/api/tags" >/dev/null; then
    echo "[launch] ERROR: ollama not reachable at ${host}" >&2
    exit 1
  fi
}

need_session_free() {
  local session="$1"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[launch] ERROR: tmux session '${session}' already exists."
    echo "  attach: tmux attach -t ${session}"
    echo "  kill:   tmux kill-session -t ${session}"
    exit 1
  fi
}

need_ollama "${HOST_S0}"
need_ollama "${HOST_N0}"
need_session_free "${SESSION_S0}"
need_session_free "${SESSION_N0}"

echo "[launch] building NSHAP S0 gold + inputs..."
"${PY}" "${SCRIPTS}/build_nshap_s0_gold.py"
"${PY}" "${SCRIPTS}/build_state_inputs.py" --dataset nshap_r1 --task s0_baseline_state
echo "[launch] building N0 remainder (does not rewrite the T3 1000)..."
"${PY}" "${SCRIPTS}/build_network_inputs.py" --dataset nshap_r1 --remainder

NET_LOG_DIR="${ROOT}/Testing/logs/network/logs/${SLUG}"
mkdir -p "${NET_LOG_DIR}"
C_S0="${NET_LOG_DIR}/console_${RUN_TS}_t4_nshap_s0r1_gpu2.log"
C_N0="${NET_LOG_DIR}/console_${RUN_TS}_t4_nshap_n0rem_gpu0.log"

S0_CMD="${PY} ${SCRIPTS}/run_state_generation.py --model ${MODEL} --host ${HOST_S0} --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_nshap_r1_s0_gpu2 --dataset nshap_r1 --task s0_baseline_state"
R1_BUILD="${PY} ${SCRIPTS}/build_r1_inputs.py --dataset nshap_r1"
R1_CMD="${PY} ${SCRIPTS}/run_network_generation.py --model ${MODEL} --host ${HOST_S0} --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_nshap_r1_r1_gpu2 --dataset nshap_r1 --task r1_llm_state_network"
N0_CMD="${PY} ${SCRIPTS}/run_network_generation.py --model ${MODEL} --host ${HOST_N0} --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_nshap_r1_n0rem_gpu0 --dataset nshap_r1 --task n0_baseline_network --input-dir ${ROOT}/Testing/T3_interval_following/nshap_r1/input"

tmux new-session -d -s "${SESSION_S0}" \
  "cd '${ROOT}' && export PYTHONUNBUFFERED=1 && \
   { echo '[t4] NSHAP S0 start' && ${S0_CMD} && echo '[t4] S0 done; building R1' && ${R1_BUILD} && echo '[t4] R1 start' && ${R1_CMD} && echo '[t4] R1 done' ; \
     echo EXIT_CODE=\$? >> '${C_S0}'; echo FINISHED_AT=\$(date -u +%Y-%m-%dT%H:%M:%SZ) >> '${C_S0}'; } \
   > '${C_S0}' 2>&1; exec bash"

tmux new-session -d -s "${SESSION_N0}" \
  "cd '${ROOT}' && export PYTHONUNBUFFERED=1 && \
   { echo '[t4] NSHAP N0 remainder start' && ${N0_CMD} && echo '[t4] N0 remainder done' ; \
     echo EXIT_CODE=\$? >> '${C_N0}'; echo FINISHED_AT=\$(date -u +%Y-%m-%dT%H:%M:%SZ) >> '${C_N0}'; } \
   > '${C_N0}' 2>&1; exec bash"

echo "[launch] run_ts=${RUN_TS}"
echo "[launch] ${SESSION_S0}  ${HOST_S0}  S0→R1  console=${C_S0}"
echo "[launch] ${SESSION_N0}  ${HOST_N0}  N0 remainder  console=${C_N0}"
echo "[launch] attach: tmux attach -t ${SESSION_S0}"
echo "[launch] tail:   tail -f ${C_S0} ${C_N0}"
