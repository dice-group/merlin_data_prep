#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scratch/hpc-prf-merlin/nikit/repos/moe-study}"
BUDGET="${BUDGET:-24e9}"
SUBSET="${SUBSET:-${ROOT}/data_dir/fineweb2_metadata/l2v_nested_samples/200.csv}"
COUNTS="${COUNTS:-${ROOT}/data_dir/fw_l2v_samples/lang_counts/language_sample_counts.tsv}"
PROBE="${PROBE:-${ROOT}/data_dir/fw_l2v_samples/lang_counts/token_probe_tokens_llama3.1-8b.tsv}"
OUT="${OUT:-${ROOT}/data_dir/fw_l2v_samples/lang_counts/language_rates/llama-3.1-8B_200langs_24b.json}"
ENGLISH="${ENGLISH:-mean}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/build_language_rates_from_budget.py"

exec python -u "${PY_SCRIPT}" \
  --subset-csv "${SUBSET}" \
  --counts-tsv "${COUNTS}" \
  --probe-tsv "${PROBE}" \
  --budget "${BUDGET}" \
  --out "${OUT}" \
  --english "${ENGLISH}"
