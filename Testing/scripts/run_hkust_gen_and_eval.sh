#!/usr/bin/env bash
# HKUST generation queue + full evaluation for one model.
#
# Ensures R1 uses output/<model_slug(MODEL)>/, not the tmux log SLUG.
#
# Usage:
#   MODEL=gpt-5.3-chat bash Testing/scripts/run_hkust_gen_and_eval.sh
#   MODEL=Qwen SLUG=hkust_qwen WORKERS=16 bash Testing/scripts/run_hkust_gen_and_eval.sh
#
# Optional: SKIP_GENERATION=1 SLUG=gpt-5.3-chat bash ...  (eval only)
# Optional: SKIP_EVAL=1 bash ...  (generation only)
set -u
set -e

ROOT="/home/bld/data/data2/jiaxin/simulation"
SCRIPTS="${ROOT}/Testing/scripts"
PY="${PY:-/home/bld/anaconda3/bin/python}"

MODEL="${MODEL:?set MODEL (e.g. gpt-5.3-chat, Qwen)}"
BACKEND="${BACKEND:-hkust}"
API_BASE="${API_BASE:-https://aigc-api.hkust-gz.edu.cn/v1}"
WORKERS="${WORKERS:-16}"
TIMEOUT="${TIMEOUT:-300}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

if [[ "${BACKEND}" == "hkust" && -z "${HKUST_AIGC_API_KEY:-}" ]]; then
  echo "ERROR: set HKUST_AIGC_API_KEY in ${ROOT}/.env" >&2
  exit 1
fi

OUTPUT_SLUG="$("${PY}" -c "import sys; sys.path.insert(0,'${SCRIPTS}'); from paths import model_slug; print(model_slug('${MODEL}'))")"
SLUG="${SLUG:-hkust_$(echo "${MODEL}" | tr ':/.' '_' | tr '[:upper:]' '[:lower:]')}"
PIPELINE_DIR="${PIPELINE_DIR:-${ROOT}/Testing/logs/pipeline/${OUTPUT_SLUG}}"
mkdir -p "${PIPELINE_DIR}"
PIPELINE_LOG="${PIPELINE_DIR}/gen_and_eval_$(date -u +%Y%m%dT%H%M%SZ).log"

log() {
  echo "[pipeline $(date -u +%H:%M:%SZ)] $*"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" >> "${PIPELINE_DIR}/status.txt"
}

exec > >(tee -a "${PIPELINE_LOG}") 2>&1

log "START model=${MODEL} output_slug=${OUTPUT_SLUG} log_slug=${SLUG} backend=${BACKEND}"

"${PY}" "${SCRIPTS}/verify_model_paths.py" --model "${MODEL}" --log-slug "${SLUG}"

if [[ "${SKIP_GENERATION}" != "1" ]]; then
  log "GENERATION queue (OUTPUT_SLUG=${OUTPUT_SLUG})"
  export BACKEND API_BASE MODEL SLUG OUTPUT_SLUG WORKERS TIMEOUT PY
  export HOST="${API_BASE}"
  export STATUS_DIR="${ROOT}/Testing/logs/overnight/${SLUG}"
  bash "${SCRIPTS}/run_overnight_queue.sh"
  gen_rc=$?
  log "GENERATION finished rc=${gen_rc}"
else
  log "SKIP_GENERATION=1"
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  log "EVALUATION slug=${OUTPUT_SLUG} (must match output dirs, not log slug ${SLUG})"
  SLUG="${OUTPUT_SLUG}" bash "${SCRIPTS}/run_model_evaluation.sh"
  log "EVALUATION done"
else
  log "SKIP_EVAL=1"
fi

log "ALL DONE model=${MODEL} output_slug=${OUTPUT_SLUG} log=${PIPELINE_LOG}"
