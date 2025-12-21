#!/usr/bin/env bash
set -euo pipefail

# Submits tokenizer-extension jobs for the common Llama models using CLI args.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_ROOT="${RUN_DIR}/logs/tokenize_extension"
SLURM_SCRIPT="${RUN_DIR}/run_pipeline_slurm.sh"

if [[ ! -x "${SLURM_SCRIPT}" ]]; then
  echo "[run_all_tokenizer_extensions] Missing SLURM script: ${SLURM_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"

BASE_DATA_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset"
EXTENSION_DATA_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset"
TRAIN_DATA_DIR="/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset"

TOTAL_NEW_TOKENS=128000
WEIGHT_PPW=0.5
WEIGHT_CPT=0.5
WEIGHT_UNK=0.0
GAMMA=2.0
TRAIN_VOCAB_SIZE=300000
TRAIN_TEXT_KEY="text"
EXTENSION_TEXT_KEY="text"
EXTENSION_VOCAB_CAP=256000
EXTENSION_NUM_WORKERS=8
EXTENDED_NUM_WORKERS=64
BASE_NUM_WORKERS=64

MODELS=(
  "llama-3.1-8B|meta-llama/Llama-3.1-8B|/scratch/hpc-prf-merlin/shared_cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b|/scratch/hpc-prf-merlin/joel/moe-study/data_prep/tokenizer_extension/outputs/llama-3.1-8B|10000|10000|04:00:00"
  "llama-3.2-1B|meta-llama/Llama-3.2-1B|/scratch/hpc-prf-merlin/shared_cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08|/scratch/hpc-prf-merlin/joel/moe-study/data_prep/tokenizer_extension/outputs/llama-3.2-1B|1000|50|08:00:00"
  "llama-3.2-3B|meta-llama/Llama-3.2-3B|/scratch/hpc-prf-merlin/shared_cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062|/scratch/hpc-prf-merlin/joel/moe-study/data_prep/tokenizer_extension/outputs/llama-3.2-3B|1000|50|12:00:00"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag model base_path output_dir max_lines sample_docs time_limit <<< "${entry}"

  job_name="tokenizer_ext_${tag}"
  log_path="${LOG_ROOT}/${tag}_%j.log"
  echo "[run_all_tokenizer_extensions] Submitting ${tag} job"

  sbatch --job-name "${job_name}" --time "${time_limit}" --output "${log_path}" "${SLURM_SCRIPT}" \
    --output-dir "${output_dir}" \
    --base-data-dir "${BASE_DATA_DIR}" \
    --base-tokenizer "${model}" \
    --base-max-lines "${max_lines}" \
    --base-num-workers "${BASE_NUM_WORKERS}" \
    --total-new-tokens "${TOTAL_NEW_TOKENS}" \
    --weight-ppw "${WEIGHT_PPW}" \
    --weight-cpt "${WEIGHT_CPT}" \
    --weight-unk "${WEIGHT_UNK}" \
    --gamma "${GAMMA}" \
    --train-multilingual \
    --train-data-dir "${TRAIN_DATA_DIR}" \
    --train-base-model "${model}" \
    --train-vocab-size "${TRAIN_VOCAB_SIZE}" \
    --train-text-key "${TRAIN_TEXT_KEY}" \
    --extend \
    --extension-data-dir "${EXTENSION_DATA_DIR}" \
    --extension-base-path "${base_path}" \
    --extension-sample-docs "${sample_docs}" \
    --extension-text-key "${EXTENSION_TEXT_KEY}" \
    --extension-num-workers "${EXTENSION_NUM_WORKERS}" \
    --extension-vocab-cap "${EXTENSION_VOCAB_CAP}" \
    --compute-extended-coverage \
    --extended-data-dir "${EXTENSION_DATA_DIR}" \
    --extended-max-lines "${max_lines}" \
    --extended-num-workers "${EXTENDED_NUM_WORKERS}"
done
