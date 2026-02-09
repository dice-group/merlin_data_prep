#!/usr/bin/env bash
# Tokenize sampled small nested-tier data (10/96/200) for Llama-3.2-1B.
# Designed for sampled data produced under:
#   sampled_data/llama-3.2-1B/<tier>_langs/samples/<language>/*.jsonl.gz
#
# Stage 1 (new-language) tokenization is enabled by default.
# Optional Stage 2 (English replay) tokenization can be enabled by setting:
#   RUN_STAGE2_ENGLISH=1 and ENGLISH_SHARD_DIR=<path containing english/*.jsonl(.gz)>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/../tokenize_and_merge_pipeline.sh"

if [[ ! -x "${PIPELINE_SCRIPT}" ]]; then
  echo "[ERROR] Missing pipeline script at ${PIPELINE_SCRIPT}" >&2
  exit 1
fi

SAMPLED_ROOT="${SAMPLED_ROOT:-${REPO_ROOT}/sampled_data/llama-3.2-1B}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/project_data/moe_study/tokenized/sampled_small_fw_l2v/llama-3.2-1B_tokenizer}"
TOKENIZER="${TOKENIZER:-meta-llama/Llama-3.2-1B}"
TIERS="${TIERS:-10,96,200}"
PYTHON_BIN="${PYTHON_BIN:-/upb/users/j/joeldag/profiles/unix/cs/.conda/envs/merlin/bin/python}"

NUM_PROC="${NUM_PROC:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM_PER_TASK="${MEM_PER_TASK:-32G}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
MERGE_CPUS="${MERGE_CPUS:-8}"
MERGE_MEM="${MERGE_MEM:-96G}"
MERGE_TIME="${MERGE_TIME:-06:00:00}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
EVAL_FRACTION="${EVAL_FRACTION:-0.02}"
EVAL_SEED="${EVAL_SEED:-42}"
JOB_PREFIX_BASE="${JOB_PREFIX_BASE:-small1b_tok}"
NUM_RANKS_OVERRIDE="${NUM_RANKS_OVERRIDE:-}"
DRY_RUN="${DRY_RUN:-0}"

RUN_STAGE2_ENGLISH="${RUN_STAGE2_ENGLISH:-0}"
ENGLISH_SHARD_DIR="${ENGLISH_SHARD_DIR:-}"
STAGE2_SAMPLE_FRACTION="${STAGE2_SAMPLE_FRACTION:-1.0}"
STAGE2_MAX_SAMPLES="${STAGE2_MAX_SAMPLES:-}"
STAGE2_LANGUAGE="${STAGE2_LANGUAGE:-english}"

MAP_DIR="${OUTPUT_BASE}/language_maps"
mkdir -p "${MAP_DIR}"

if [[ ! -d "${SAMPLED_ROOT}" ]]; then
  echo "[ERROR] SAMPLED_ROOT does not exist: ${SAMPLED_ROOT}" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] PYTHON_BIN not found: ${PYTHON_BIN}" >&2
  exit 1
fi

make_identity_map() {
  local tier_slug="$1"
  local samples_dir="$2"
  local map_path="$3"
  local include_english="$4"

  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

samples_dir = Path(${samples_dir@Q})
map_path = Path(${map_path@Q})
include_english = ${include_english@Q} == "1"

langs = sorted([p.name for p in samples_dir.iterdir() if p.is_dir()])
if include_english and "english" not in langs:
    langs.append("english")

mapping = {lang: lang for lang in langs}
map_path.parent.mkdir(parents=True, exist_ok=True)
map_path.write_text(json.dumps(mapping, ensure_ascii=True, indent=2), encoding="utf-8")
print(f"[INFO] Wrote language map: {map_path} ({len(mapping)} entries)")
PY
}

run_pipeline() {
  local shard_dir="$1"
  local output_root="$2"
  local job_prefix="$3"
  local num_ranks="$4"
  local langs="$5"
  local language_map="$6"
  local sample_fraction="$7"
  local max_samples="$8"

  local cmd=(
    "${PIPELINE_SCRIPT}"
    --shard-dir "${shard_dir}"
    --tokenizer "${TOKENIZER}"
    --output-root "${output_root}"
    --languages "${langs}"
    --language-map "${language_map}"
    --num-ranks "${num_ranks}"
    --num-proc "${NUM_PROC}"
    --cpus-per-task "${CPUS_PER_TASK}"
    --mem "${MEM_PER_TASK}"
    --time "${TIME_LIMIT}"
    --merge-cpus "${MERGE_CPUS}"
    --merge-mem "${MERGE_MEM}"
    --merge-time "${MERGE_TIME}"
    --max-length "${MAX_LENGTH}"
    --eval-fraction "${EVAL_FRACTION}"
    --eval-seed "${EVAL_SEED}"
    --sample-fraction "${sample_fraction}"
    --job-prefix "${job_prefix}"
  )

  if [[ -n "${max_samples}" ]]; then
    cmd+=(--max-samples "${max_samples}")
  fi

  echo "[INFO] Submitting ${job_prefix}"
  echo "[INFO]   shard_dir=${shard_dir}"
  echo "[INFO]   output_root=${output_root}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] %q ' "${cmd[@]}"
    echo
  else
    "${cmd[@]}"
  fi
}

IFS=',' read -r -a tier_arr <<< "${TIERS}"
for tier in "${tier_arr[@]}"; do
  tier="$(echo "${tier}" | xargs)"
  tier_slug="${tier}_langs"
  samples_dir="${SAMPLED_ROOT}/${tier_slug}/samples"
  if [[ ! -d "${samples_dir}" ]]; then
    echo "[WARN] Missing sampled tier dir: ${samples_dir}; skipping."
    continue
  fi

  langs="$(find "${samples_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -u | tr '\n' ' ')"
  if [[ -z "${langs// }" ]]; then
    echo "[WARN] No language subdirs under ${samples_dir}; skipping."
    continue
  fi
  lang_count="$(echo "${langs}" | wc -w | tr -d ' ')"
  if [[ -n "${NUM_RANKS_OVERRIDE}" ]]; then
    num_ranks="${NUM_RANKS_OVERRIDE}"
  else
    num_ranks="${lang_count}"
  fi

  map_path="${MAP_DIR}/${tier_slug}.json"
  make_identity_map "${tier_slug}" "${samples_dir}" "${map_path}" "0"

  stage1_output="${OUTPUT_BASE}/${tier_slug}/stage1_newlangs"
  stage1_job="${JOB_PREFIX_BASE}_${tier_slug}_stage1"
  run_pipeline "${samples_dir}" "${stage1_output}" "${stage1_job}" "${num_ranks}" "${langs}" "${map_path}" "1.0" ""

  if [[ "${RUN_STAGE2_ENGLISH}" == "1" ]]; then
    if [[ -z "${ENGLISH_SHARD_DIR}" || ! -d "${ENGLISH_SHARD_DIR}" ]]; then
      echo "[ERROR] RUN_STAGE2_ENGLISH=1 but ENGLISH_SHARD_DIR is not a valid directory." >&2
      exit 1
    fi
    if [[ ! -d "${ENGLISH_SHARD_DIR}/${STAGE2_LANGUAGE}" ]]; then
      echo "[ERROR] Expected ${ENGLISH_SHARD_DIR}/${STAGE2_LANGUAGE} for Stage 2." >&2
      exit 1
    fi

    stage2_map_path="${MAP_DIR}/${tier_slug}_with_english.json"
    make_identity_map "${tier_slug}" "${samples_dir}" "${stage2_map_path}" "1"
    stage2_output="${OUTPUT_BASE}/${tier_slug}/stage2_${STAGE2_LANGUAGE}"
    stage2_job="${JOB_PREFIX_BASE}_${tier_slug}_stage2_${STAGE2_LANGUAGE}"
    run_pipeline \
      "${ENGLISH_SHARD_DIR}" \
      "${stage2_output}" \
      "${stage2_job}" \
      "${num_ranks}" \
      "${STAGE2_LANGUAGE}" \
      "${stage2_map_path}" \
      "${STAGE2_SAMPLE_FRACTION}" \
      "${STAGE2_MAX_SAMPLES}"
  fi
done

echo "[INFO] Submitted sampled-data tokenization jobs."
