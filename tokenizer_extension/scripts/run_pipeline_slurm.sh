#!/bin/bash
#SBATCH --job-name=tokenizer_extension
#SBATCH --output=logs/tokenize_extension/tokenizer_extension_%j.out
#SBATCH --error=logs/tokenize_extension/tokenizer_extension_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=420G
#SBATCH -p normal

set -euo pipefail

LOG_DIR="logs/tokenize_extension"
mkdir -p "${LOG_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${PROJECT_ROOT}"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [pipeline args...]" >&2
  exit 1
fi

echo "[run_pipeline_slurm] Launching tokenizer extension pipeline"
srun python -m tokenizer_extension.pipeline_slurm "$@"
