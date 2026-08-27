#!/usr/bin/env bash
# Backfill R1 for DeepSeek + Qwen, then re-run evaluation for both.
set -u
set -e

ROOT="/home/bld/data/data2/jiaxin/simulation"
COMMON="${ROOT}/Testing/common"
PY="${PY:-/home/bld/anaconda3/bin/python}"
mkdir -p "${ROOT}/Testing/logs/r1_backfill" "${ROOT}/Testing/logs/eval"

cd "${ROOT}"
export PYTHONUNBUFFERED=1

log="${ROOT}/Testing/logs/r1_backfill/pipeline_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${log}") 2>&1

echo "=== R1 backfill: DeepSeek-V4-Flash ==="
MODEL=DeepSeek-V4-Flash BACKEND=hkust WORKERS=16 bash "${COMMON}/run_r1_backfill.sh"

echo "=== R1 backfill: Qwen ==="
MODEL=Qwen BACKEND=hkust WORKERS=16 bash "${COMMON}/run_r1_backfill.sh"

echo "=== Evaluation: DeepSeek-V4-Flash ==="
SLUG=DeepSeek-V4-Flash bash "${COMMON}/run_model_evaluation.sh"

echo "=== Evaluation: Qwen ==="
SLUG=Qwen bash "${COMMON}/run_model_evaluation.sh"

echo "=== Pipeline complete ==="
