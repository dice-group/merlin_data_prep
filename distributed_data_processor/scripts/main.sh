#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/tokenize_all_tokenizers_lang_combinations.sh" \
  --shard-dir /scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/sharded_samples \
  --output-base /scratch/hpc-prf-merlin/project_data/moe_study/tokenized/fw_l2v_samples_moe_study_comparison \
  --job-prefix tok_moe_study_comparison
