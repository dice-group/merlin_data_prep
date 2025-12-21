#!/bin/bash
# Submit each tokenizer-extension pipeline stage as a separate Slurm job.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [pipeline args...]" >&2
  exit 1
fi

PARTITION=${PARTITION:-normal}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
BASE_PYTHONPATH="${PYTHONPATH:-}"
if [[ -n "$BASE_PYTHONPATH" ]]; then
  COMBINED_PYTHONPATH="${PROJECT_ROOT}:${BASE_PYTHONPATH}"
else
  COMBINED_PYTHONPATH="${PROJECT_ROOT}"
fi

OUTPUT_ROOT=${TOKENIZER_EXTENSION_OUTPUT_DIR:-"${SCRIPT_DIR}/outputs"}
RUN_ID=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${OUTPUT_ROOT}/run_${RUN_ID}"
LOG_ROOT="${RUN_DIR}/logs"

export TOKENIZER_EXTENSION_OUTPUT_DIR="${RUN_DIR}"
export PYTHONPATH="${COMBINED_PYTHONPATH}"

mkdir -p "$LOG_ROOT"
MASTER_LOG="${LOG_ROOT}/master_${RUN_ID}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "Run directory: $RUN_DIR"
echo "Master log: $MASTER_LOG"

declare -A CPUS=(
  [base_coverage]=64
  [allocation]=4
  [training]=32
  [extension]=8
  [extended_coverage]=64
  [comparison]=2
)
declare -A MEM=(
  [base_coverage]="64G"
  [allocation]="8G"
  [training]="128G"
  [extension]="96G"
  [extended_coverage]="64G"
  [comparison]="8G"
)
declare -A TIME=(
  [base_coverage]="02:00:00"
  [allocation]="00:30:00"
  [training]="08:00:00"
  [extension]="04:00:00"
  [extended_coverage]="02:00:00"
  [comparison]="00:30:00"
)

STAGES=(base_coverage allocation training extension extended_coverage comparison)
ARGS=("$@")

for stage in "${STAGES[@]}"; do
  job_name="tokext_${stage}"
  job_log_out="${LOG_ROOT}/${stage}_%j.out"
  job_log_err="${LOG_ROOT}/${stage}_%j.err"

  echo "------------------------------------------------------------"
  echo "Preparing stage: ${stage}"
  echo "  job_name   : ${job_name}"
  echo "  cpus       : ${CPUS[$stage]}"
  echo "  memory     : ${MEM[$stage]}"
  echo "  time limit : ${TIME[$stage]}"
  echo "  stdout log : ${job_log_out}"
  echo "  stderr log : ${job_log_err}"

  sbatch_args=(
    --parsable
    --job-name="$job_name"
    --partition="$PARTITION"
    --cpus-per-task="${CPUS[$stage]}"
    --mem="${MEM[$stage]}"
    --time="${TIME[$stage]}"
    --output="$job_log_out"
    --error="$job_log_err"
    --export=ALL
  )

  wrap_cmd=(
    "srun"
    "python" "-u"
    "-m" "tokenizer_extension.pipeline_slurm"
    "${ARGS[@]}"
    "--stage" "$stage"
  )

  echo "  command    : ${wrap_cmd[*]}"
  job_id=$(sbatch "${sbatch_args[@]}" --wrap "${wrap_cmd[*]}")
  echo "Stage '${stage}' submitted as job ${job_id}"

  while squeue -h -j "$job_id" | grep -q "$job_id"; do
    sleep 10
  done

  echo "===== START $stage =====" >> "$MASTER_LOG"
  cat "${LOG_ROOT}/${stage}_${job_id}.out" >> "$MASTER_LOG" 2>/dev/null || true
  echo "===== END $stage =====" >> "$MASTER_LOG"
done

summary_path="${RUN_DIR}/SUMMARY.md"
cat <<EOM > "$summary_path"
# Run Summary

- Output directory: ${RUN_DIR}
- Trained tokenizer: ${RUN_DIR}/trained_tokenizer
- Extended tokenizer: ${RUN_DIR}/extended_tokenizer
- Base metrics: ${RUN_DIR}/base_metrics.csv
- Allocation CSV: ${RUN_DIR}/allocation.csv
- Extended metrics: ${RUN_DIR}/extended_metrics.csv
- Comparison CSV: ${RUN_DIR}/comparison.csv
EOM

echo "Summary written to $summary_path"
echo "------------------------------------------------------------"
echo "All stages completed. Logs under: ${LOG_ROOT}"
echo "Master log: ${MASTER_LOG}"
