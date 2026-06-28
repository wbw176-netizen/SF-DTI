#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
BATCH_SIZE=${BATCH_SIZE:-64}
PRED_BATCH_SIZE=${PRED_BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
SPLIT=${SPLIT:-random2}
DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-42}
USE_AMP=${USE_AMP:-1}
QUERY_LIST=${QUERY_LIST:-"nadh glucose fad h1r adra1c adra2a bace1 htr7"}

read -r -a QUERIES <<< "${QUERY_LIST}"

OUTPUT_ROOT=${OUTPUT_ROOT:-../output/case_blind_predictions}
FEATURE_ROOT=${FEATURE_ROOT:-../output/case_blind_features}
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS+=(--amp)
fi

echo "Running query-blinded SF-DTI case studies"
echo "Queries: ${QUERIES[*]}"
echo "Split: ${SPLIT}"
echo "Device: ${DEVICE}"
echo "Feature root: ${FEATURE_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"

for query in "${QUERIES[@]}"; do
  data="Drugbank_case_blind_${query}"
  precomputed_dir="${FEATURE_ROOT}/${data}/${SPLIT}"
  train_log="${LOG_DIR}/${query}_train.log"
  predict_log="${LOG_DIR}/${query}_predict.log"

  echo
  echo "===== Training ${query} ====="
  "${PYTHON_BIN}" train_blinded_case_study.py \
    --data "${data}" \
    --split "${SPLIT}" \
    --precomputed_dir "${precomputed_dir}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    "${AMP_ARGS[@]}" 2>&1 | tee "${train_log}"

  echo
  echo "===== Ranking ${query} candidates ====="
  "${PYTHON_BIN}" predict_blinded_case_study.py \
    --data "${data}" \
    --split "${SPLIT}" \
    --precomputed_dir "${precomputed_dir}" \
    --batch_size "${PRED_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --output "${OUTPUT_ROOT}/${query}_case_study.csv" \
    --metrics_output "${OUTPUT_ROOT}/${query}_case_study_metrics.csv" \
    2>&1 | tee "${predict_log}"
done

echo
echo "Completed all requested blinded case studies."
