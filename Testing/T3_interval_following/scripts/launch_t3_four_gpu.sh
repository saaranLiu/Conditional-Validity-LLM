#!/usr/bin/env bash
# T3 supplementary network generation on four A40s (reuse existing ollama).
#
#   GPU3 :11436  gss_1985 N0 → nshap_r1 N0 → gss_2024 N0
#   GPU2 :11435  gss_1985 N4c → gss_2004 N4c → gss_2024 N4
#   GPU1 :11438  gss_1985 N4s → nshap_r1 N4c
#   GPU0 :11437  gss_2004 N0 → gss_2004 N4s → signos N0/N4o/N4r (w=4)
#
# Usage:
#   bash Testing/scripts/launch_t3_four_gpu.sh
set -euo pipefail

ROOT="/home/bld/data/data2/jiaxin/simulation"
COMMON="${ROOT}/Testing/common"
OUT="${ROOT}/Testing/logs/network"
PY="${PY:-/home/bld/anaconda3/bin/python}"
MODEL="${MODEL:-llama3.1:latest}"
SLUG="${SLUG:-llama3.1_latest}"
WORKERS="${WORKERS:-16}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"

cd "${ROOT}"
export PYTHONUNBUFFERED=1
LOG_DIR="${OUT}/logs/${SLUG}"
mkdir -p "${LOG_DIR}"

need_ollama() {
  local port="$1"
  if ! curl -sf --max-time 3 "http://127.0.0.1:${port}/api/tags" >/dev/null; then
    echo "[launch] ERROR: ollama not reachable on :${port}" >&2
    exit 1
  fi
}

run_one() {
  local dataset="$1"
  local task="$2"
  local port="$3"
  local workers="$4"
  local timeout="$5"
  local npred="$6"
  local run_id="${RUN_TS}_${dataset}_${task}_gpu_port${port}_w${workers}"
  "${PY}" "${COMMON}/run_network_generation.py" \
    --model "${MODEL}" \
    --host "http://127.0.0.1:${port}" \
    --limit 0 \
    --workers "${workers}" \
    --resume \
    --num-ctx "${NUM_CTX:-4096}" \
    --temperature 0.7 \
    --num-predict "${npred}" \
    --timeout "${timeout}" \
    --run-id "${run_id}" \
    --dataset "${dataset}" \
    --task "${task}"
}

launch_queue() {
  local session="$1"
  local console="$2"
  shift 2
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[launch] ERROR: tmux session '${session}' already exists."
    echo "  attach: tmux attach -t ${session}"
    echo "  kill:   tmux kill-session -t ${session}"
    exit 1
  fi
  # Remaining args are a bash snippet run inside tmux.
  local body="$*"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && export PYTHONUNBUFFERED=1 && \
     { ${body} ; echo EXIT_CODE=\$? >> '${console}'; echo FINISHED_AT=\$(date -u +%Y-%m-%dT%H:%M:%SZ) >> '${console}'; } \
     > '${console}' 2>&1; exec bash"
  echo "[launch] ${session}  console=${console}"
}

need_ollama 11436
need_ollama 11435
need_ollama 11438
need_ollama 11437

C3="${LOG_DIR}/console_${RUN_TS}_t3_gpu3.log"
C2="${LOG_DIR}/console_${RUN_TS}_t3_gpu2.log"
C1="${LOG_DIR}/console_${RUN_TS}_t3_gpu1.log"
C0="${LOG_DIR}/console_${RUN_TS}_t3_gpu0.log"

# Source run_one in the tmux shells via exporting a wrapper file would be heavy;
# inline the python invocations so each queue is explicit.

launch_queue t3_gpu3 "${C3}" \
  "${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host http://127.0.0.1:11436 --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_gss_1985_n0_gpu3 --dataset gss_1985 --task n0_baseline_network && \
   ${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host http://127.0.0.1:11436 --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_nshap_r1_n0_gpu3 --dataset nshap_r1 --task n0_baseline_network && \
   ${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host http://127.0.0.1:11436 --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_gss_2024_n0_gpu3 --dataset gss_2024 --task n0_baseline_network"

launch_queue t3_gpu2 "${C2}" \
  "${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host http://127.0.0.1:11435 --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_gss_1985_n4c_gpu2 --dataset gss_1985 --task n4c_closure && \
   ${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host http://127.0.0.1:11435 --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_gss_2004_n4c_gpu2 --dataset gss_2004 --task n4c_closure && \
   ${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host http://127.0.0.1:11435 --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_gss_2024_n4_gpu2 --dataset gss_2024 --task n4_contact"

launch_queue t3_gpu1 "${C1}" \
  "${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host http://127.0.0.1:11438 --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_gss_1985_n4s_gpu1 --dataset gss_1985 --task n4s_sex_homophily && \
   ${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host http://127.0.0.1:11438 --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_nshap_r1_n4c_gpu1 --dataset nshap_r1 --task n4c_closure"

launch_queue t3_gpu0 "${C0}" \
  "${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host http://127.0.0.1:11437 --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_gss_2004_n0_gpu0 --dataset gss_2004 --task n0_baseline_network && \
   ${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host http://127.0.0.1:11437 --limit 0 --workers ${WORKERS} --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 2048 --timeout 300 --run-id ${RUN_TS}_gss_2004_n4s_gpu0 --dataset gss_2004 --task n4s_sex_homophily && \
   ${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host http://127.0.0.1:11437 --limit 0 --workers 4 --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 4096 --timeout 900 --run-id ${RUN_TS}_signos_n0_gpu0 --dataset signos --task n0_baseline_network && \
   ${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host http://127.0.0.1:11437 --limit 0 --workers 4 --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 4096 --timeout 900 --run-id ${RUN_TS}_signos_n4o_gpu0 --dataset signos --task n4o_same_origin && \
   ${PY} ${COMMON}/run_network_generation.py --model ${MODEL} --host http://127.0.0.1:11437 --limit 0 --workers 4 --resume --num-ctx ${NUM_CTX:-4096} --temperature 0.7 --num-predict 4096 --timeout 900 --run-id ${RUN_TS}_signos_n4r_gpu0 --dataset signos --task n4r_resides_spain"

echo "[launch] run_ts=${RUN_TS}"
echo "[launch] tmux: t3_gpu3 t3_gpu2 t3_gpu1 t3_gpu0"
echo "[launch] tail -f ${C3} ${C2} ${C1} ${C0}"
tmux ls | grep t3_gpu || true
