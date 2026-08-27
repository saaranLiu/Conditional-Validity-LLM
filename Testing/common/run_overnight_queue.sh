#!/usr/bin/env bash
# Sequential generation queue for one local model on one existing Ollama host.
# Does not start/stop ollama. Continues past a failed arm.
#
# Required env: MODEL HOST SLUG
# SLUG = log/tmux id (e.g. hkust_qwen). OUTPUT_SLUG = generation output dir (from MODEL).
# Optional: PY WORKERS NUM_CTX TIMEOUT RUN_TS STATUS_DIR BACKEND API_BASE
set -u
set +e

ROOT="/home/bld/data/data2/jiaxin/simulation"
COMMON="${ROOT}/Testing/common"
T4="${ROOT}/Testing/T4_individual_resample/scripts"
PY="${PY:-/home/bld/anaconda3/bin/python}"
WORKERS="${WORKERS:-16}"
NUM_CTX="${NUM_CTX:-4096}"
TIMEOUT="${TIMEOUT:-300}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
STATUS_DIR="${STATUS_DIR:-${ROOT}/Testing/logs/overnight/${SLUG}}"
BACKEND="${BACKEND:-ollama}"
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

LLM_FLAGS=(--backend "${BACKEND}")
if [[ "${BACKEND}" == "chatanywhere" || "${BACKEND}" == "hkust" ]]; then
  LLM_FLAGS+=(--api-base "${API_BASE}")
else
  LLM_FLAGS+=(--host "${HOST}")
fi

cd "${ROOT}"
export PYTHONUNBUFFERED=1
mkdir -p "${STATUS_DIR}"

OUTPUT_SLUG="${OUTPUT_SLUG:-$("${PY}" -c "import sys; sys.path.insert(0,'${COMMON}'); from paths import model_slug; print(model_slug('${MODEL}'))")}"

note() {
  local msg="$1"
  echo "[queue $(date -u +%H:%M:%SZ)] ${msg}"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  ${msg}" >> "${STATUS_DIR}/queue_status.txt"
}

run_state() {
  local ds="$1"
  local task="$2"
  local rid="${RUN_TS}_${ds}_${task}"
  note "START state ${ds} ${task}"
  "${PY}" "${COMMON}/run_state_generation.py" \
    --model "${MODEL}" \
    "${LLM_FLAGS[@]}" \
    --limit 0 \
    --workers "${WORKERS}" \
    --resume \
    --num-ctx "${NUM_CTX}" \
    --temperature 0.7 \
    --num-predict 2048 \
    --timeout "${TIMEOUT}" \
    --run-id "${rid}" \
    --dataset "${ds}" \
    --task "${task}"
  local rc=$?
  note "END state ${ds} ${task} rc=${rc}"
  return 0
}

run_net() {
  local ds="$1"
  local task="$2"
  local extra_timeout="${3:-${TIMEOUT}}"
  local extra_npred="${4:-2048}"
  local extra_workers="${5:-${WORKERS}}"
  local input_dir="${6:-}"
  local rid="${RUN_TS}_${ds}_${task}"
  note "START net ${ds} ${task}"
  local cmd=(
    "${PY}" "${COMMON}/run_network_generation.py"
    --model "${MODEL}"
    "${LLM_FLAGS[@]}"
    --limit 0
    --workers "${extra_workers}"
    --resume
    --num-ctx "${NUM_CTX}"
    --temperature 0.7
    --num-predict "${extra_npred}"
    --timeout "${extra_timeout}"
    --run-id "${rid}"
    --dataset "${ds}"
    --task "${task}"
  )
  if [[ -n "${input_dir}" ]]; then
    cmd+=(--input-dir "${input_dir}")
  fi
  "${cmd[@]}"
  local rc=$?
  note "END net ${ds} ${task} rc=${rc}"
  return 0
}

run_r1() {
  local ds="$1"
  local in_dir="${ROOT}/Testing/T4_individual_resample/${ds}/input/${OUTPUT_SLUG}"
  note "BUILD R1 inputs ${ds} output_slug=${OUTPUT_SLUG}"
  "${PY}" "${T4}/build_r1_inputs.py" --dataset "${ds}" --model "${MODEL}"
  local rc=$?
  note "BUILD R1 ${ds} rc=${rc}"
  if [[ "${rc}" -ne 0 ]]; then
    note "SKIP R1 generation ${ds} (build failed)"
    return 0
  fi
  run_net "${ds}" "r1_llm_state_network" "${TIMEOUT}" 2048 "${WORKERS}" "${in_dir}"
}

note "QUEUE START model=${MODEL} backend=${BACKEND} workers=${WORKERS} slug=${SLUG} output_slug=${OUTPUT_SLUG} order=ucnets_first"

# --- Tier 0: UCNets W1 (主数据，全部优先) ---
run_state ucnets_w1 s0_baseline_state
run_state ucnets_w1 s1_health_conditioned
run_state ucnets_w1 s2_politics_religion_conditioned
run_state ucnets_w1 s0b_anti_typology
run_state ucnets_w1 s1b_anti_typology
run_net ucnets_w1 n0_baseline_network
run_net ucnets_w1 n1_size_bucket
run_net ucnets_w1 n2_population_prior
run_net ucnets_w1 n4_homophily
run_r1 ucnets_w1
run_net ucnets_w1 n3_roster_multistep
run_net ucnets_w1 n5_size_multistep
run_net ucnets_w1 n6_prior_multistep

# --- Tier 1: 补充 survey — State + 识别核心 Network ---
run_state gss_1987 s0_baseline_state
run_state gss_1987 s1_health_conditioned
run_state gss_1987 s2_politics_religion_conditioned
run_state gss_2010_network s0_baseline_state
run_state gss_2010_network s1_health_conditioned
run_state gss_2010_network s2_politics_religion_conditioned
run_net gss_1985 n0_baseline_network
run_net gss_1985 n4c_closure
run_net gss_1985 n4s_sex_homophily
run_net issp_2017 t6_country_resources "${TIMEOUT}" 1024 "${WORKERS}"

# --- Tier 2: 其余补充 Network ---
run_net gss_2004 n0_baseline_network
run_net gss_2004 n4c_closure
run_net gss_2004 n4s_sex_homophily
run_net nshap_r1 n0_baseline_network
run_net nshap_r1 n4c_closure
run_net gss_2024 n0_baseline_network
run_net gss_2024 n4_contact

# --- Tier 3: T4 (S0 → R1) ---
run_state gss_1985 s0_baseline_state
run_r1 gss_1985
run_state nshap_r1 s0_baseline_state
run_r1 nshap_r1

# --- Tier 4: 反类型学 State + Signos + NSHAP N0 补跑 ---
run_state gss_1987 s0b_anti_typology
run_state gss_1987 s1b_anti_typology
run_state gss_2010_network s0b_anti_typology
run_state gss_2010_network s1b_anti_typology
run_net signos n0_baseline_network 900 4096 4
run_net signos n4o_same_origin 900 4096 4
run_net signos n4r_resides_spain 900 4096 4
run_net nshap_r1 n0_baseline_network "${TIMEOUT}" 2048 "${WORKERS}" \
  "${ROOT}/Testing/T3_interval_following/nshap_r1/input"

note "QUEUE DONE model=${MODEL}"
exit 0
