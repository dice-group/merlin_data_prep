#!/usr/bin/env bash
# Submit tokenization+merge jobs for all tokenizer/subset combinations defined below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/tokenize_and_merge_pipeline.sh"

if [[ ! -x "${PIPELINE_SCRIPT}" ]]; then
  echo "Missing pipeline script: ${PIPELINE_SCRIPT}" >&2
  exit 1
fi

NUM_RANKS=1
USER_NUM_RANKS=""
NUM_PROC=1
CPUS_PER_TASK=4
MEM_PER_TASK=32G
TIME_LIMIT=02:00:00
MERGE_CPUS=4
MERGE_MEM=64G
MERGE_TIME=04:00:00
TRANSFORMERS_OFFLINE=0
EVAL_FRACTION=0.02
EVAL_SEED=42
ENABLE_RANKING=0
RANK_ALPHA=0.5
RANK_BETA=0.5
RANK_WINDOW=5
COUNT_CPUS=4
COUNT_MEM=32G
COUNT_TIME=02:00:00
RANK_CPUS=8
RANK_MEM=120G
RANK_TIME=01:30:00
TOPK_SIZES=""
TOPK_MIN_LANGUAGE_SIZE=30000
TOPK_CPUS=8
TOPK_MEM=64G
TOPK_TIME=02:00:00
LANGUAGE_RATES_DIR="/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/lang_counts/language_rates"
LANGUAGE_MAP_DIR=""

usage() {
  cat <<EOF
Usage: $0 --shard-dir PATH --output-base PATH [options]

Required arguments:
  --shard-dir PATH      Root directory that holds per-language shard folders.
  --output-base PATH    Base directory for outputs (e.g., /scratch/.../hierarchical_adapter).

Options (propagated to tokenize_and_merge_pipeline.sh):
  --num-ranks INT       Number of SLURM array tasks/ranks (default: ${NUM_RANKS}).
  --num-proc INT        --num_proc for tokenize_slurm.py (default: ${NUM_PROC}).
  --cpus-per-task INT   CPUs per tokenization task (default: ${CPUS_PER_TASK}).
  --mem MEM             Memory per tokenization task (default: ${MEM_PER_TASK}).
  --time HH:MM:SS       Time limit per tokenization task (default: ${TIME_LIMIT}).
  --merge-cpus INT      CPUs for merge job (default: ${MERGE_CPUS}).
  --merge-mem MEM       Memory for merge job (default: ${MERGE_MEM}).
  --merge-time HH:MM:SS Time limit for merge job (default: ${MERGE_TIME}).
  --trans-offline {0,1} Set TRANSFORMERS_OFFLINE flag (default: ${TRANSFORMERS_OFFLINE}).
  --eval-fraction FLOAT Validation fraction per language during tokenization (default: ${EVAL_FRACTION}).
  --eval-seed INT       Seed for eval hashing (default: ${EVAL_SEED}).
  --enable-ranking      Run the ranking pipeline (counts + scores) before merging.
  --rank-alpha FLOAT    Local-popularity weight α (default: ${RANK_ALPHA}).
  --rank-beta FLOAT     Global-importance weight β (default: ${RANK_BETA}).
  --rank-window INT     Co-occurrence window size (default: ${RANK_WINDOW}).
  --count-cpus INT      CPUs per count job (default: ${COUNT_CPUS}).
  --count-mem STR       Memory per count job (default: ${COUNT_MEM}).
  --count-time HH:MM:SS Time per count job (default: ${COUNT_TIME}).
  --rank-cpus INT       CPUs per ranking job (default: ${RANK_CPUS}).
  --rank-mem STR        Memory per ranking job (default: ${RANK_MEM}).
  --rank-time HH:MM:SS  Time per ranking job (default: ${RANK_TIME}).
  --topk-sizes "10k ..." Produce per-language top-K datasets (enables ranking automatically).
  --topk-min-language-size INT  Per-language minimum rows before selecting K (default: ${TOPK_MIN_LANGUAGE_SIZE}).
  --topk-cpus INT       CPUs for the top-K filtering job (default: ${TOPK_CPUS}).
  --topk-mem STR        Memory for the top-K filtering job (default: ${TOPK_MEM}).
  --topk-time HH:MM:SS  Time for the top-K filtering job (default: ${TOPK_TIME}).
  --job-prefix STR      Prefix for SBATCH job names (default derived per combo).
  --language-map-dir PATH  Optional dir with per-subset language maps (expects <subset_slug>.json).
  -h, --help            Show this help message.
EOF
}

SHARD_DIR=""
OUTPUT_BASE=""
JOB_PREFIX_BASE="bulk_tok"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shard-dir) SHARD_DIR="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --num-ranks) NUM_RANKS="$2"; USER_NUM_RANKS="$2"; shift 2 ;;
    --num-proc) NUM_PROC="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --mem) MEM_PER_TASK="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --merge-cpus) MERGE_CPUS="$2"; shift 2 ;;
    --merge-mem) MERGE_MEM="$2"; shift 2 ;;
    --merge-time) MERGE_TIME="$2"; shift 2 ;;
    --trans-offline) TRANSFORMERS_OFFLINE="$2"; shift 2 ;;
    --job-prefix) JOB_PREFIX_BASE="$2"; shift 2 ;;
    --eval-fraction) EVAL_FRACTION="$2"; shift 2 ;;
    --eval-seed) EVAL_SEED="$2"; shift 2 ;;
    --enable-ranking) ENABLE_RANKING=1; shift ;;
    --rank-alpha) RANK_ALPHA="$2"; shift 2 ;;
    --rank-beta) RANK_BETA="$2"; shift 2 ;;
    --rank-window) RANK_WINDOW="$2"; shift 2 ;;
    --count-cpus) COUNT_CPUS="$2"; shift 2 ;;
    --count-mem) COUNT_MEM="$2"; shift 2 ;;
    --count-time) COUNT_TIME="$2"; shift 2 ;;
    --rank-cpus) RANK_CPUS="$2"; shift 2 ;;
    --rank-mem) RANK_MEM="$2"; shift 2 ;;
    --rank-time) RANK_TIME="$2"; shift 2 ;;
    --topk-sizes) TOPK_SIZES="$2"; ENABLE_RANKING=1; shift 2 ;;
    --topk-min-language-size) TOPK_MIN_LANGUAGE_SIZE="$2"; shift 2 ;;
    --topk-cpus) TOPK_CPUS="$2"; shift 2 ;;
    --topk-mem) TOPK_MEM="$2"; shift 2 ;;
    --topk-time) TOPK_TIME="$2"; shift 2 ;;
    --language-map-dir) LANGUAGE_MAP_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${SHARD_DIR}" || -z "${OUTPUT_BASE}" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

LOCAL_MODEL_DIR="/scratch/hpc-prf-merlin/joel/moe-study/data_prep/merlin_data_prep/distributed_data_processor/scripts/.model"
TOKENIZERS=(
  "llama-3.1-8B_tokenizer|${LOCAL_MODEL_DIR}/meta-llama_Llama-3.1-8B"
  "llama-3.2-1B_tokenizer|${LOCAL_MODEL_DIR}/meta-llama_Llama-3.2-1B"
  "llama-3.2-3B_tokenizer|${LOCAL_MODEL_DIR}/meta-llama_Llama-3.2-3B"
)

L2V_SUBSET_DIR="/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fineweb2_metadata/l2v_nested_samples"
SUBSETS=(
  "5_langs|${L2V_SUBSET_DIR}/5.csv"
  "10_langs|${L2V_SUBSET_DIR}/10.csv"
  "22_langs|${L2V_SUBSET_DIR}/22.csv"
  "46_langs|${L2V_SUBSET_DIR}/46.csv"
  "96_langs|${L2V_SUBSET_DIR}/96.csv"
  "200_langs|${L2V_SUBSET_DIR}/200.csv"
)

for tok_entry in "${TOKENIZERS[@]}"; do
  IFS='|' read -r tok_dir tok_name <<<"${tok_entry}"
  for subset_entry in "${SUBSETS[@]}"; do
    IFS='|' read -r subset_dir subset_flag <<<"${subset_entry}"
    run_num_ranks="${NUM_RANKS}"
    if [[ -z "${USER_NUM_RANKS}" ]]; then
      case "${subset_dir}" in
        5_langs) run_num_ranks=5 ;;
        10_langs) run_num_ranks=10 ;;
        22_langs) run_num_ranks=22 ;;
        46_langs) run_num_ranks=46 ;;
        96_langs) run_num_ranks=96 ;;
        200_langs) run_num_ranks=200 ;;
      esac
    fi
    output_dir="${OUTPUT_BASE}/${tok_dir}/${subset_dir}"
    job_prefix="${JOB_PREFIX_BASE}_${tok_dir}_${subset_dir}"
    echo "Submitting ${tok_name} / ${subset_flag} -> ${output_dir}"
    languages_arg=()
    if [[ "${subset_flag}" == *.csv && -f "${subset_flag}" ]]; then
      lang_list=$(awk -F',' 'NR>1 {print $2}' "${subset_flag}" | sort -u | tr '\n' ' ')
      lang_list="${lang_list} english"
      lang_list=$(printf "%s\n" ${lang_list} | sort -u | tr '\n' ' ')
      if [[ -n "${lang_list}" ]]; then
        languages_arg=(--languages "${lang_list}")
      fi
    fi

    pipeline_cmd=(
      "${PIPELINE_SCRIPT}"
      --shard-dir "${SHARD_DIR}"
      --tokenizer "${tok_name}"
      --num-ranks "${run_num_ranks}"
      --num-proc "${NUM_PROC}"
      --cpus-per-task "${CPUS_PER_TASK}"
      --mem "${MEM_PER_TASK}"
      --time "${TIME_LIMIT}"
      --merge-cpus "${MERGE_CPUS}"
      --merge-mem "${MERGE_MEM}"
      --merge-time "${MERGE_TIME}"
      --eval-fraction "${EVAL_FRACTION}"
      --eval-seed "${EVAL_SEED}"
      --job-prefix "${job_prefix}"
      --output-root "${output_dir}"
    )

    if [[ "${#languages_arg[@]}" -gt 0 ]]; then
      pipeline_cmd+=("${languages_arg[@]}")
    else
      pipeline_cmd+=(--language-subset "${subset_flag}")
    fi

    if [[ -n "${LANGUAGE_MAP_DIR}" ]]; then
      map_file="${LANGUAGE_MAP_DIR}/${subset_dir}.json"
      if [[ -f "${map_file}" ]]; then
        pipeline_cmd+=(--language-map "${map_file}")
      else
        echo "[warn] No language map file: ${map_file} (language_ids/family_ids may be padded)"
      fi
    fi

    rate_file="${LANGUAGE_RATES_DIR}/${tok_dir}/${subset_dir}.json"
    if [[ -f "${rate_file}" ]]; then
      pipeline_cmd+=(--language-rates "${rate_file}")
    else
      echo "[warn] No language rates file: ${rate_file} (using full data)"
    fi

    if [[ "${ENABLE_RANKING}" -eq 1 ]]; then
      pipeline_cmd+=(
        --enable-ranking
        --rank-alpha "${RANK_ALPHA}"
        --rank-beta "${RANK_BETA}"
        --rank-window "${RANK_WINDOW}"
        --count-cpus "${COUNT_CPUS}"
        --count-mem "${COUNT_MEM}"
        --count-time "${COUNT_TIME}"
        --rank-cpus "${RANK_CPUS}"
        --rank-mem "${RANK_MEM}"
        --rank-time "${RANK_TIME}"
      )
    fi

    if [[ -n "${TOPK_SIZES}" ]]; then
      pipeline_cmd+=(
        --topk-sizes "${TOPK_SIZES}"
        --topk-min-language-size "${TOPK_MIN_LANGUAGE_SIZE}"
        --topk-cpus "${TOPK_CPUS}"
        --topk-mem "${TOPK_MEM}"
        --topk-time "${TOPK_TIME}"
      )
    fi

    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 "${pipeline_cmd[@]}"
  done
done
