#!/usr/bin/env bash
#SBATCH --job-name=python_split_fast
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=430G
#SBATCH --time=05:00:00
#SBATCH --output=logs/python_split-%j.out
#SBATCH --error=logs/python_split-%j.err

# This takes all large files (>512 MB)
# and separates them for efficient tokenization.
# Takes ~10 min for 90 GB of compressed data with the compute resources above

echo "Running Python-based parallel split with timing..."

INPUT_DIR="/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/samples"
OUTPUT_DIR="/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/sharded_samples"
CORE_DIR="/scratch/hpc-prf-merlin/joel/moe-study/data_prep/merlin_data_prep/distributed_data_processor/src"

time python -u "${CORE_DIR}/shard_python.py" \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR"