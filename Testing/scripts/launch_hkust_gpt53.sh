#!/usr/bin/env bash
# Launch gpt-5.3-chat on HKUST: full generation queue + evaluation (tmux).
#
# R1 paths use output slug gpt-5.3-chat (from model name), NOT hkust_gpt53.
#
# Usage:
#   bash Testing/scripts/launch_hkust_gpt53.sh
#   WORKERS=12 bash Testing/scripts/launch_hkust_gpt53.sh
set -u
set +e

ROOT="/home/bld/data/data2/jiaxin/simulation"
SCRIPTS="${ROOT}/Testing/scripts"

MODEL="${MODEL:-gpt-5.3-chat}"
SLUG="${SLUG:-gpt53}"
WORKERS="${WORKERS:-16}"
TIMEOUT="${TIMEOUT:-300}"

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

SESSION="hkust_${SLUG}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session ${SESSION} already exists. Attach: tmux attach -t ${SESSION}"
  exit 0
fi

OUTPUT_SLUG="$("${PY:-/home/bld/anaconda3/bin/python}" -c "import sys; sys.path.insert(0,'${SCRIPTS}'); from paths import model_slug; print(model_slug('${MODEL}'))")"
PIPELINE_DIR="${ROOT}/Testing/logs/pipeline/${OUTPUT_SLUG}"
mkdir -p "${PIPELINE_DIR}"
LOG="${PIPELINE_DIR}/tmux_console.log"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && set -a && source '${ROOT}/.env' 2>/dev/null; set +a && \
   export MODEL='${MODEL}' SLUG='${SLUG}' WORKERS='${WORKERS}' TIMEOUT='${TIMEOUT}' BACKEND=hkust && \
   bash '${SCRIPTS}/run_hkust_gen_and_eval.sh' 2>&1 | tee -a '${LOG}'"

echo "Started tmux session: ${SESSION}"
echo "  model=${MODEL}"
echo "  output_slug=${OUTPUT_SLUG}  (generation + eval paths)"
echo "  log_slug=${SLUG}  (tmux/logs only)"
echo "  workers=${WORKERS} timeout=${TIMEOUT}s"
echo "  pipeline log: ${LOG}"
echo "  attach: tmux attach -t ${SESSION}"
