#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/bld/data/data2/jiaxin/simulation"
PY="/home/bld/anaconda3/envs/sociosim/bin/python"
GPU_ID="${GPU_ID:?Set GPU_ID to 0, 1, 2, or 3}"
SHARD_INDEX="${SHARD_INDEX:?Set SHARD_INDEX to 0, 1, 2, or 3}"
NUM_SHARDS="${NUM_SHARDS:-4}"
RUN_BASE="${RUN_BASE:-main}"
RUN_NAME="${RUN_BASE}/shard_${SHARD_INDEX}"

cd "${ROOT}"
mkdir -p "Testing/decoding/logs/${RUN_NAME}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PY}" "Testing/decoding/scripts/run_native_path.py" \
  --run-name "${RUN_NAME}" \
  --input-run-name "${RUN_BASE}" \
  --max-egos 400 \
  --shard-index "${SHARD_INDEX}" \
  --num-shards "${NUM_SHARDS}" \
  --native-batch-size 24 \
  --generation-batch-size 4 \
  --t3-reps 3 \
  --size-reps 1 \
  2>&1 | tee -a "Testing/decoding/logs/${RUN_NAME}/console.log"
