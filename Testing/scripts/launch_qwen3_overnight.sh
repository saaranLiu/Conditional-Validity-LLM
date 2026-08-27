#!/usr/bin/env bash
# Overnight replication on two idle GPUs (reuse existing ollama, do not restart).
#
#   GPU2 :11435  qwen3:8b
#   GPU3 :11436  qwen3:14b
#
# Each GPU runs the full T1–T6 generation queue in identification-first order.
# Resume is on; a failed arm does not stop the queue.
set -euo pipefail

ROOT="/home/bld/data/data2/jiaxin/simulation"
SCRIPTS="${ROOT}/Testing/scripts"
PY="${PY:-/home/bld/anaconda3/bin/python}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"

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

need_ollama "http://127.0.0.1:11435"
need_ollama "http://127.0.0.1:11436"
need_session_free qwen3_8b_gpu2
need_session_free qwen3_14b_gpu3

launch_one() {
  local session="$1"
  local model="$2"
  local slug="$3"
  local host="$4"
  local workers="$5"
  local status_dir="${ROOT}/Testing/logs/overnight/${slug}"
  local console="${status_dir}/console_${RUN_TS}.log"
  mkdir -p "${status_dir}" "${ROOT}/Testing/logs/state/logs/${slug}" "${ROOT}/Testing/logs/network/logs/${slug}"

  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && export PYTHONUNBUFFERED=1 \
      MODEL='${model}' SLUG='${slug}' HOST='${host}' \
      WORKERS='${workers}' PY='${PY}' RUN_TS='${RUN_TS}' \
      STATUS_DIR='${status_dir}' \
    && bash '${SCRIPTS}/run_overnight_queue.sh' \
    > '${console}' 2>&1; \
    echo EXIT_CODE=\$? >> '${console}'; \
    echo FINISHED_AT=\$(date -u +%Y-%m-%dT%H:%M:%SZ) >> '${console}'; \
    exec bash"

  echo "[launch] ${session}  ${model}  ${host}  workers=${workers}"
  echo "         console=${console}"
  echo "         status=${status_dir}/queue_status.txt"
}

launch_one qwen3_8b_gpu2  qwen3:8b  qwen3_8b  http://127.0.0.1:11435 16
launch_one qwen3_14b_gpu3 qwen3:14b qwen3_14b http://127.0.0.1:11436 8

echo "[launch] run_ts=${RUN_TS}"
echo "[launch] attach: tmux attach -t qwen3_8b_gpu2"
echo "[launch] tail:   tail -f ${ROOT}/Testing/logs/overnight/qwen3_8b/console_${RUN_TS}.log ${ROOT}/Testing/logs/overnight/qwen3_14b/console_${RUN_TS}.log"
tmux ls | grep qwen3 || true
