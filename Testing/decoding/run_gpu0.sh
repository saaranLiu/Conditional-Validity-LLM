#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/bld/data/data2/jiaxin/simulation"
PY="/home/bld/anaconda3/envs/sociosim/bin/python"
RUN_NAME="${RUN_NAME:-main}"
MAX_EGOS="${MAX_EGOS:-400}"
NATIVE_BATCH_SIZE="${NATIVE_BATCH_SIZE:-24}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-4}"
T3_REPS="${T3_REPS:-3}"
SIZE_REPS="${SIZE_REPS:-1}"

cd "${ROOT}"
mkdir -p "Testing/decoding/logs/${RUN_NAME}"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PY}" "Testing/decoding/scripts/run_native_path.py" \
  --run-name "${RUN_NAME}" \
  --max-egos "${MAX_EGOS}" \
  --native-batch-size "${NATIVE_BATCH_SIZE}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  --t3-reps "${T3_REPS}" \
  --size-reps "${SIZE_REPS}" \
  2>&1 | tee -a "Testing/decoding/logs/${RUN_NAME}/console.log"
