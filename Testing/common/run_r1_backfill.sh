#!/usr/bin/env bash
# Build + generate missing T4 R1 arms for one model.
#
# Usage:
#   MODEL=DeepSeek-V4-Flash BACKEND=hkust bash Testing/scripts/run_r1_backfill.sh
#   MODEL=Qwen BACKEND=hkust bash Testing/scripts/run_r1_backfill.sh
#   MODEL=qwen3:8b HOST=http://127.0.0.1:11435 bash Testing/scripts/run_r1_backfill.sh
#
# Optional: DATASETS="ucnets_w1 gss_1985 nshap_r1"
set -u
set -e

ROOT="/home/bld/data/data2/jiaxin/simulation"
COMMON="${ROOT}/Testing/common"
T4="${ROOT}/Testing/T4_individual_resample/scripts"
PY="${PY:-/home/bld/anaconda3/bin/python}"
MODEL="${MODEL:?set MODEL}"
BACKEND="${BACKEND:-ollama}"
WORKERS="${WORKERS:-16}"
NUM_CTX="${NUM_CTX:-4096}"
TIMEOUT="${TIMEOUT:-300}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASETS="${DATASETS:-ucnets_w1 gss_1985 nshap_r1}"

case "${BACKEND}" in
  hkust) API_BASE="${API_BASE:-https://aigc-api.hkust-gz.edu.cn/v1}" ;;
  chatanywhere) API_BASE="${API_BASE:-https://api.chatanywhere.tech/v1}" ;;
  *) API_BASE="${API_BASE:-}" ;;
esac

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

OUTPUT_SLUG="$("${PY}" -c "import sys; sys.path.insert(0,'${COMMON}'); from paths import model_slug; print(model_slug('${MODEL}'))")"
STATUS_DIR="${STATUS_DIR:-${ROOT}/Testing/logs/r1_backfill/${OUTPUT_SLUG}}"
mkdir -p "${STATUS_DIR}"

LLM_FLAGS=(--backend "${BACKEND}")
if [[ "${BACKEND}" == "chatanywhere" || "${BACKEND}" == "hkust" ]]; then
  LLM_FLAGS+=(--api-base "${API_BASE}")
else
  HOST="${HOST:?set HOST for ollama backend}"
  LLM_FLAGS+=(--host "${HOST}")
fi

cd "${ROOT}"
export PYTHONUNBUFFERED=1

note() {
  echo "[r1-backfill $(date -u +%H:%M:%SZ)] $*"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" >> "${STATUS_DIR}/status.txt"
}

note "START model=${MODEL} output_slug=${OUTPUT_SLUG} backend=${BACKEND} datasets=${DATASETS}"

for ds in ${DATASETS}; do
  in_dir="${ROOT}/Testing/T4_individual_resample/${ds}/input/${OUTPUT_SLUG}"
  rid="${RUN_TS}_${ds}_r1_llm_state_network"
  note "BUILD R1 inputs ${ds}"
  "${PY}" "${T4}/build_r1_inputs.py" --dataset "${ds}" --model "${MODEL}" \
    >> "${STATUS_DIR}/build_${ds}.log" 2>&1
  note "GENERATE R1 ${ds}"
  "${PY}" "${COMMON}/run_network_generation.py" \
    --model "${MODEL}" \
    "${LLM_FLAGS[@]}" \
    --input-dir "${in_dir}" \
    --limit 0 \
    --workers "${WORKERS}" \
    --resume \
    --num-ctx "${NUM_CTX}" \
    --temperature 0.7 \
    --num-predict 2048 \
    --timeout "${TIMEOUT}" \
    --run-id "${rid}" \
    --dataset "${ds}" \
    --task r1_llm_state_network \
    >> "${STATUS_DIR}/gen_${ds}.log" 2>&1
  note "DONE R1 ${ds}"
done

note "ALL DONE model=${MODEL} output_slug=${OUTPUT_SLUG}"
