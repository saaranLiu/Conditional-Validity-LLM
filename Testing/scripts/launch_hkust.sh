#!/usr/bin/env bash
# Launch generation queue via HKUST-GZ AIGC API (OpenAI-compatible).
#
# Setup:
#   cp .env.example .env
#   # set HKUST_AIGC_API_KEY=...
#
# Usage:
#   MODEL=Qwen SLUG=hkust_qwen ./Testing/scripts/launch_hkust.sh
#   MODEL=DeepSeek-V4-Flash SLUG=hkust_dsv4_flash WORKERS=8 ./Testing/scripts/launch_hkust.sh
#   MODEL=gpt-5.3-chat bash Testing/scripts/launch_hkust_gpt53.sh   # gen + eval pipeline
#
# SLUG is for logs/tmux only. OUTPUT_SLUG=model_slug(MODEL) for all output/R1/eval paths.
#
# Supported models (per HKUST docs):
#   Qwen, gpt-4-turbo, gpt-5.3-chat, DeepSeek-V4-Pro, DeepSeek-V4-Flash
set -u
set +e

ROOT="/home/bld/data/data2/jiaxin/simulation"
SCRIPTS="${ROOT}/Testing/scripts"
PY="${PY:-/home/bld/anaconda3/bin/python}"

MODEL="${MODEL:-Qwen}"
SLUG="${SLUG:-$(echo "${MODEL}" | tr ':/.' '_' | tr '[:upper:]' '[:lower:]')}"
API_BASE="${API_BASE:-https://aigc-api.hkust-gz.edu.cn/v1}"
WORKERS="${WORKERS:-16}"
TIMEOUT="${TIMEOUT:-300}"
STATUS_DIR="${STATUS_DIR:-${ROOT}/Testing/logs/overnight/${SLUG}}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

if [[ -z "${HKUST_AIGC_API_KEY:-}" ]]; then
  echo "ERROR: set HKUST_AIGC_API_KEY in ${ROOT}/.env" >&2
  exit 1
fi

export BACKEND=hkust
export API_BASE MODEL SLUG WORKERS TIMEOUT STATUS_DIR PY
export HOST="${API_BASE}"
OUTPUT_SLUG="$("${PY}" -c "import sys; sys.path.insert(0,'${SCRIPTS}'); from paths import model_slug; print(model_slug('${MODEL}'))")"
export OUTPUT_SLUG

SESSION="hkust_${SLUG}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session ${SESSION} already exists. Attach: tmux attach -t ${SESSION}"
  exit 0
fi

mkdir -p "${STATUS_DIR}"
LOG="${STATUS_DIR}/console.log"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && set -a && source '${ROOT}/.env' 2>/dev/null; set +a && \
   export BACKEND='${BACKEND}' API_BASE='${API_BASE}' MODEL='${MODEL}' SLUG='${SLUG}' HOST='${HOST}' \
   WORKERS='${WORKERS}' TIMEOUT='${TIMEOUT}' STATUS_DIR='${STATUS_DIR}' PY='${PY}' && \
   bash '${SCRIPTS}/run_overnight_queue.sh' 2>&1 | tee '${LOG}'"

echo "Started tmux session: ${SESSION}"
echo "  model=${MODEL} slug=${SLUG} output_slug=${OUTPUT_SLUG} workers=${WORKERS}"
echo "  api=${API_BASE}"
echo "  log=${LOG}"
echo "  attach: tmux attach -t ${SESSION}"
