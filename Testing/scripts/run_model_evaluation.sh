#!/usr/bin/env bash
# Run full evaluation suite for one completed generation model slug.
# Usage: SLUG=qwen3_8b bash Testing/scripts/run_model_evaluation.sh
set -u

ROOT="/home/bld/data/data2/jiaxin/simulation"
PY="${PY:-/home/bld/anaconda3/bin/python}"
SCRIPTS="${ROOT}/Testing/scripts"
SLUG="${SLUG:?set SLUG (e.g. Qwen, qwen3_8b, qwen3_14b)}"
LOG_DIR="${ROOT}/Testing/logs/eval/${SLUG}"
mkdir -p "${LOG_DIR}"

note() {
  echo "[eval $(date -u +%H:%M:%SZ)] $*"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" >> "${LOG_DIR}/eval_status.txt"
}

cd "${ROOT}"

note "START slug=${SLUG}"

note "state eval1"
"${PY}" "${SCRIPTS}/evaluate_state_eval1.py" \
  --model-slug "${SLUG}" \
  --output-dir "${ROOT}/Testing/T1_association/eval/${SLUG}" \
  >> "${LOG_DIR}/state_eval1.log" 2>&1

note "state eval2"
"${PY}" "${SCRIPTS}/evaluate_state_eval2.py" \
  --model-slug "${SLUG}" \
  --output-dir "${ROOT}/Testing/T1_association/eval/${SLUG}" \
  >> "${LOG_DIR}/state_eval2.log" 2>&1

note "state eval3"
"${PY}" "${SCRIPTS}/evaluate_state_eval3.py" \
  --model-slug "${SLUG}" \
  --output-dir "${ROOT}/Testing/T1_association/eval/${SLUG}" \
  >> "${LOG_DIR}/state_eval3.log" 2>&1

note "network ucnets"
"${PY}" "${SCRIPTS}/evaluate_network.py" \
  --model-slug "${SLUG}" \
  --output-dir "${ROOT}/Testing/T1_association/ucnets_w1/eval/${SLUG}" \
  >> "${LOG_DIR}/network_ucnets.log" 2>&1

for ds in gss_1985 gss_2004 gss_2024 nshap_r1 signos; do
  note "network supplementary ${ds}"
  "${PY}" "${SCRIPTS}/evaluate_network.py" \
    --model-slug "${SLUG}" \
    --dataset "${ds}" \
    --output-dir "${ROOT}/Testing/T3_interval_following/${ds}/eval/${SLUG}" \
    >> "${LOG_DIR}/network_${ds}.log" 2>&1
done

note "t4 all datasets"
"${PY}" "${SCRIPTS}/evaluate_t4.py" \
  --all-t4 \
  --slug "${SLUG}" \
  >> "${LOG_DIR}/t4.log" 2>&1

note "t6"
"${PY}" "${SCRIPTS}/evaluate_t6.py" \
  --model-slug "${SLUG}" \
  >> "${LOG_DIR}/t6.log" 2>&1

note "DONE slug=${SLUG}"
