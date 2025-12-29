#!/usr/bin/env bash
#SBATCH --job-name=shard
#SBATCH --output=logs/shard_%A_%a.log
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-214
#SBATCH --partition=normal

set -euo pipefail

INPUT_DIR="/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/samples"
OUTPUT_DIR="/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/sharded_samples"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="/scratch/hpc-prf-merlin/joel/moe-study/data_prep/merlin_data_prep/distributed_data_processor/src"

# select exactly one file per array task
FILE=$(find "$INPUT_DIR" -type f -name "*.gz" | sort | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
# safety check
[ -z "$FILE" ] && echo "No file for task $SLURM_ARRAY_TASK_ID" && exit 1

# run single-file mode
python -u "${CORE_DIR}/shard_python.py" \
  --file "$FILE" \
  --output "$OUTPUT_DIR"
