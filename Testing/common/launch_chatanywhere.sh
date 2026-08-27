#!/usr/bin/env bash
# Launch one commercial-model queue via ChatAnywhere (OpenAI-compatible API).
# Does not touch local Ollama / tmux queues.
#
# Setup:
#   cp .env.example .env   # set CHATANYWHERE_API_KEY=sk-...
#
# Usage:
#   MODEL=qwen3.5-plus SLUG=qwen35_plus ./Testing/scripts/launch_chatanywhere.sh
#   MODEL=deepseek-v3-2 SLUG=deepseek_v3_2 WORKERS=8 ./Testing/scripts/launch_chatanywhere.sh
#
# Optional env: API_BASE WORKERS TIMEOUT STATUS_DIR PY
set -u
set +e

ROOT="/home/bld/data/data2/jiaxin/simulation"
COMMON="${ROOT}/Testing/common"
PY="${PY:-/home/bld/anaconda3/bin/python}"

MODEL="${MODEL:-qwen3.5-plus}"
SLUG="${SLUG:-$(echo "${MODEL}" | tr ':/.' '_')}"
API_BASE="${API_BASE:-https://api.chatanywhere.tech/v1}"
WORKERS="${WORKERS:-8}"
TIMEOUT="${TIMEOUT:-300}"
STATUS_DIR="${STATUS_DIR:-${ROOT}/Testing/logs/overnight/${SLUG}_ca}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

if [[ -z "${CHATANYWHERE_API_KEY:-}" ]]; then
  echo "ERROR: set CHATANYWHERE_API_KEY in ${ROOT}/.env (see .env.example)" >&2
  exit 1
fi

export BACKEND=chatanywhere
export API_BASE
export MODEL SLUG WORKERS TIMEOUT STATUS_DIR PY
export HOST="${API_BASE}"  # unused when BACKEND=chatanywhere; kept for queue script compat

SESSION="chatanywhere_${SLUG}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session ${SESSION} already exists. Attach: tmux attach -t ${SESSION}"
  exit 0
fi

mkdir -p "${STATUS_DIR}"
LOG="${STATUS_DIR}/console.log"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && BACKEND='${BACKEND}' API_BASE='${API_BASE}' MODEL='${MODEL}' SLUG='${SLUG}' HOST='${HOST}' WORKERS='${WORKERS}' TIMEOUT='${TIMEOUT}' STATUS_DIR='${STATUS_DIR}' PY='${PY}' bash '${COMMON}/run_overnight_queue.sh' 2>&1 | tee '${LOG}'"

echo "Started tmux session: ${SESSION}"
echo "  model=${MODEL} slug=${SLUG} workers=${WORKERS}"
echo "  log=${LOG}"
echo "  attach: tmux attach -t ${SESSION}"
