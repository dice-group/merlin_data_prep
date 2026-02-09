#!/usr/bin/env bash
# Tokenize MoE-LPR Stage 1 (new languages) + Stage 2 (original language replay)
# for all tiers (5/10/22/46/96/200) and all tokenizers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/../tokenize_and_merge_pipeline.sh"

SHARD_DIR=${SHARD_DIR:-/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/sharded_samples}
L2V_SUBSET_DIR=${L2V_SUBSET_DIR:-/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fineweb2_metadata/l2v_nested_samples}
LANGUAGE_RATES_DIR=${LANGUAGE_RATES_DIR:-/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/lang_counts/language_rates/moelpr}
OUTPUT_BASE=${OUTPUT_BASE:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/fw_l2v_samples_moe_study_comparison/moelpr}
LANGUAGE_MAP_DIR=${LANGUAGE_MAP_DIR:-}
REQUIRE_LANGUAGE_MAP=${REQUIRE_LANGUAGE_MAP:-1}

LOCAL_MODEL_DIR=${LOCAL_MODEL_DIR:-/scratch/hpc-prf-merlin/joel/moe-study/data_prep/merlin_data_prep/distributed_data_processor/scripts/.model}
TOKENIZERS=(
  "llama-3.1-8B_tokenizer|${LOCAL_MODEL_DIR}/meta-llama_Llama-3.1-8B"
  "llama-3.2-1B_tokenizer|${LOCAL_MODEL_DIR}/meta-llama_Llama-3.2-1B"
  "llama-3.2-3B_tokenizer|${LOCAL_MODEL_DIR}/meta-llama_Llama-3.2-3B"
)

SUBSETS=(
  "10_langs|${L2V_SUBSET_DIR}/10.csv|10"
  "96_langs|${L2V_SUBSET_DIR}/96.csv|96"
  "200_langs|${L2V_SUBSET_DIR}/200.csv|200"
)

NUM_PROC=${NUM_PROC:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
MEM_PER_TASK=${MEM_PER_TASK:-32G}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
MAX_LENGTH=${MAX_LENGTH:-1024}
EVAL_FRACTION=${EVAL_FRACTION:-0.02}
EVAL_SEED=${EVAL_SEED:-42}

STAGE2_LANGUAGE="english"
STAGE2_NUM_RANKS=${STAGE2_NUM_RANKS:-}
STAGE1_NUM_RANKS_OVERRIDE=${STAGE1_NUM_RANKS_OVERRIDE:-}

if [[ ! -x "${PIPELINE_SCRIPT}" ]]; then
  echo "[ERROR] Missing pipeline script at ${PIPELINE_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${SHARD_DIR}" ]]; then
  echo "[ERROR] Shard dir ${SHARD_DIR} missing." >&2
  exit 1
fi

for tok_entry in "${TOKENIZERS[@]}"; do
  IFS='|' read -r tok_slug tok_path <<<"${tok_entry}"

  for subset_entry in "${SUBSETS[@]}"; do
    IFS='|' read -r subset_slug subset_csv subset_size <<<"${subset_entry}"
    if [[ ! -f "${subset_csv}" ]]; then
      echo "[WARN] Missing subset csv ${subset_csv}; skipping."
      continue
    fi

    stage1_langs=$(awk -F',' 'NR>1 {print $2}' "${subset_csv}" | sed 's/^ *//;s/ *$//' | grep -vi '^english$' | sort -u | tr '\n' ' ')
    if [[ -z "${stage1_langs}" ]]; then
      echo "[WARN] No languages found in ${subset_csv}; skipping."
      continue
    fi

    if [[ -n "${STAGE1_NUM_RANKS_OVERRIDE}" ]]; then
      stage1_num_ranks="${STAGE1_NUM_RANKS_OVERRIDE}"
    else
      stage1_num_ranks="${subset_size}"
    fi

    stage1_output="${OUTPUT_BASE}/${tok_slug}/${subset_slug}/stage1_newlangs"
    stage2_output="${OUTPUT_BASE}/${tok_slug}/${subset_slug}/stage2_${STAGE2_LANGUAGE}"
    stage1_job="moelpr_${tok_slug}_${subset_slug}_stage1"
    stage2_job="moelpr_${tok_slug}_${subset_slug}_stage2"

    stage1_rates="${LANGUAGE_RATES_DIR}/${tok_slug}/${subset_slug}_stage1.json"
    stage2_rates="${LANGUAGE_RATES_DIR}/${tok_slug}/${subset_slug}_stage2_english.json"
    language_map_file=""
    if [[ -n "${LANGUAGE_MAP_DIR}" ]]; then
      language_map_file="${LANGUAGE_MAP_DIR}/${subset_slug}.json"
    fi
    if [[ "${REQUIRE_LANGUAGE_MAP}" == "1" ]]; then
      if [[ -z "${language_map_file}" || ! -f "${language_map_file}" ]]; then
        echo "[ERROR] Missing language map for ${subset_slug}. Set LANGUAGE_MAP_DIR with ${subset_slug}.json." >&2
        exit 1
      fi
    elif [[ -n "${language_map_file}" && ! -f "${language_map_file}" ]]; then
      echo "[WARN] language map not found: ${language_map_file} (language_ids/family_ids may be padded)"
      language_map_file=""
    fi

    echo "[INFO] Stage 1 ${tok_slug}/${subset_slug}: ${stage1_langs}"
    stage1_cmd=(
      "${PIPELINE_SCRIPT}"
      --shard-dir "${SHARD_DIR}"
      --tokenizer "${tok_path}"
      --output-root "${stage1_output}"
      --languages "${stage1_langs}"
      --job-prefix "${stage1_job}"
      --num-proc "${NUM_PROC}"
      --num-ranks "${stage1_num_ranks}"
      --cpus-per-task "${CPUS_PER_TASK}"
      --mem "${MEM_PER_TASK}"
      --time "${TIME_LIMIT}"
      --max-length "${MAX_LENGTH}"
      --eval-fraction "${EVAL_FRACTION}"
      --eval-seed "${EVAL_SEED}"
    )
    if [[ -n "${language_map_file}" ]]; then
      stage1_cmd+=(--language-map "${language_map_file}")
    fi
    if [[ -f "${stage1_rates}" ]]; then
      stage1_cmd+=(--language-rates "${stage1_rates}")
    else
      echo "[ERROR] Missing stage1 rates: ${stage1_rates}" >&2
      exit 1
    fi
    "${stage1_cmd[@]}"

    echo "[INFO] Stage 2 ${tok_slug}/${subset_slug}: ${STAGE2_LANGUAGE}"
    if [[ ! -f "${stage2_rates}" ]]; then
      echo "[ERROR] Missing stage2 rates file ${stage2_rates}" >&2
      exit 1
    fi
    if [[ -n "${STAGE2_NUM_RANKS}" ]]; then
      stage2_num_ranks="${STAGE2_NUM_RANKS}"
    else
      stage2_num_ranks="${subset_size}"
    fi
    stage2_cmd=(
      "${PIPELINE_SCRIPT}"
      --shard-dir "${SHARD_DIR}"
      --tokenizer "${tok_path}"
      --output-root "${stage2_output}"
      --languages "${STAGE2_LANGUAGE}"
      --job-prefix "${stage2_job}"
      --num-proc "${NUM_PROC}"
      --num-ranks "${stage2_num_ranks}"
      --cpus-per-task "${CPUS_PER_TASK}"
      --mem "${MEM_PER_TASK}"
      --time "${TIME_LIMIT}"
      --max-length "${MAX_LENGTH}"
      --eval-fraction "${EVAL_FRACTION}"
      --eval-seed "${EVAL_SEED}"
      --language-rates "${stage2_rates}"
    )
    if [[ -n "${language_map_file}" ]]; then
      stage2_cmd+=(--language-map "${language_map_file}")
    fi
    "${stage2_cmd[@]}"
  done
done

echo "[INFO] Submitted MoE-LPR Stage 1/2 tokenization jobs for all tiers and tokenizers."
