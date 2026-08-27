#!/usr/bin/env bash
# Run full evaluation suite for one completed generation model slug.
# Usage: SLUG=qwen3_8b bash Testing/common/run_model_evaluation.sh
set -u

ROOT="/home/bld/data/data2/jiaxin/simulation"
PY="${PY:-/home/bld/anaconda3/bin/python}"
COMMON="${ROOT}/Testing/common"
T1="${ROOT}/Testing/T1_association/scripts"
T3="${ROOT}/Testing/T3_interval_following"
T4="${ROOT}/Testing/T4_individual_resample/scripts"
T6="${ROOT}/Testing/T6_cross_population/scripts"
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
"${PY}" "${T1}/evaluate_state_eval1.py" \
  --model-slug "${SLUG}" \
  --output-dir "${ROOT}/Testing/T1_association/eval/${SLUG}" \
  >> "${LOG_DIR}/state_eval1.log" 2>&1

note "state eval2"
"${PY}" "${T1}/evaluate_state_eval2.py" \
  --model-slug "${SLUG}" \
  --output-dir "${ROOT}/Testing/T1_association/eval/${SLUG}" \
  >> "${LOG_DIR}/state_eval2.log" 2>&1

note "state eval3"
"${PY}" "${T1}/evaluate_state_eval3.py" \
  --model-slug "${SLUG}" \
  --output-dir "${ROOT}/Testing/T1_association/eval/${SLUG}" \
  >> "${LOG_DIR}/state_eval3.log" 2>&1

note "network ucnets"
"${PY}" "${T1}/evaluate_network.py" \
  --model-slug "${SLUG}" \
  --output-dir "${ROOT}/Testing/T1_association/ucnets_w1/eval/${SLUG}" \
  >> "${LOG_DIR}/network_ucnets.log" 2>&1

for ds in gss_1985 gss_2004 gss_2024 nshap_r1 signos; do
  note "network supplementary ${ds}"
  "${PY}" "${T1}/evaluate_network.py" \
    --model-slug "${SLUG}" \
    --dataset "${ds}" \
    --output-dir "${T3}/${ds}/eval/${SLUG}" \
    >> "${LOG_DIR}/network_${ds}.log" 2>&1
done

note "t4 all datasets"
"${PY}" "${T4}/evaluate_t4.py" \
  --all-t4 \
  --slug "${SLUG}" \
  >> "${LOG_DIR}/t4.log" 2>&1

note "t6"
"${PY}" "${T6}/evaluate_t6.py" \
  --model-slug "${SLUG}" \
  >> "${LOG_DIR}/t6.log" 2>&1

note "DONE slug=${SLUG}"
