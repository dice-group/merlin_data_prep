#!/usr/bin/env bash
#SBATCH --job-name=lang_count
#SBATCH --array=0-199
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/lang_count_%A_%a.log

set -euo pipefail
mkdir -p logs
python -u /scratch/hpc-prf-merlin/joel/moe-study/data_prep/merlin_data_prep/distributed_data_processor/scripts/count_language_samples.py

# Merge after completion:
cat /scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/lang_counts/part_*.tsv \
  | awk -F'\t' '{sum[$1]+=$2} END{for (l in sum) print l"\t"sum[l]}' \
  | sort -k1 \
  > /scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/lang_counts/language_sample_counts.tsv
