#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

ROOT="${ROOT:-/scratch/hpc-prf-merlin/nikit/repos/moe-study}"
SUBSET_DIR="${SUBSET_DIR:-${ROOT}/data_dir/fineweb2_metadata/l2v_nested_samples}"
COUNTS="${COUNTS:-${ROOT}/data_dir/fw_l2v_samples/lang_counts/language_sample_counts.tsv}"
OUT_DIR="${OUT_DIR:-${ROOT}/data_dir/fw_l2v_samples/lang_counts/language_rates/moelpr}"
PY_SCRIPT="${SCRIPT_DIR}/build_language_rates_from_examples.py"
STAGE2_FRACTION="${STAGE2_FRACTION:-0.006}"

SUBSETS_DEFAULT=(
  "10_langs|10.csv|10"
  "96_langs|96.csv|96"
  "200_langs|200.csv|200"
)

if [[ -n "${SUBSETS_OVERRIDE:-}" ]]; then
  IFS=' ' read -r -a SUBSETS <<< "${SUBSETS_OVERRIDE}"
else
  SUBSETS=("${SUBSETS_DEFAULT[@]}")
fi

TOKS=(
  "llama-3.2-1B_tokenizer"
  "llama-3.2-3B_tokenizer"
  "llama-3.1-8B_tokenizer"
)

target_examples_for() {
  local tok="$1"
  local tier="$2"
  case "${tok}:${tier}" in
    llama-3.2-1B_tokenizer:10) echo 2939418 ;;
    llama-3.2-1B_tokenizer:96) echo 10951476 ;;
    llama-3.2-1B_tokenizer:200) echo 21627306 ;;
    llama-3.2-3B_tokenizer:10) echo 1471725 ;;
    llama-3.2-3B_tokenizer:96) echo 5110977 ;;
    llama-3.2-3B_tokenizer:200) echo 9374404 ;;
    llama-3.1-8B_tokenizer:10) echo 735116 ;;
    llama-3.1-8B_tokenizer:96) echo 2557048 ;;
    llama-3.1-8B_tokenizer:200) echo 4326736 ;;
    *) echo "" ;;
  esac
}

english_csv="$(mktemp /tmp/moelpr_english_only.XXXXXX.csv)"
trap 'rm -f "${english_csv}"' EXIT
printf "subset\nenglish\n" > "${english_csv}"

for tok_slug in "${TOKS[@]}"; do
  for subset_entry in "${SUBSETS[@]}"; do
    IFS='|' read -r subset_slug subset_csv tier <<<"${subset_entry}"
    subset_path="${SUBSET_DIR}/${subset_csv}"
    if [[ ! -f "${subset_path}" ]]; then
      echo "[WARN] Missing subset csv ${subset_path}; skipping." >&2
      continue
    fi

    target_examples=$(target_examples_for "${tok_slug}" "${tier}")
    if [[ -z "${target_examples}" ]]; then
      echo "[ERROR] Missing target examples for ${tok_slug} tier ${tier}" >&2
      exit 1
    fi

    stage2_examples=$(python - <<PY
print(${target_examples} * float("${STAGE2_FRACTION}"))
PY
)

    out_dir="${OUT_DIR}/${tok_slug}"
    mkdir -p "${out_dir}"

    stage1_out="${out_dir}/${subset_slug}_stage1.json"
    stage2_out="${out_dir}/${subset_slug}_stage2_english.json"

    echo "[INFO] ${tok_slug} ${subset_slug} stage1 examples=${target_examples} (no english)"
    python -u "${PY_SCRIPT}" \
      --subset-csv "${subset_path}" \
      --counts-tsv "${COUNTS}" \
      --target-examples "${target_examples}" \
      --out "${stage1_out}" \
      --english none

    echo "[INFO] ${tok_slug} ${subset_slug} stage2 examples=${stage2_examples} (english only)"
    python -u "${PY_SCRIPT}" \
      --subset-csv "${english_csv}" \
      --counts-tsv "${COUNTS}" \
      --target-examples "${stage2_examples}" \
      --out "${stage2_out}" \
      --english only
  done
done

echo "[OK] Wrote MoE-LPR language rates to ${OUT_DIR}"
