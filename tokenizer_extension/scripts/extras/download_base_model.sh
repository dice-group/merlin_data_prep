#!/usr/bin/env bash
set -euo pipefail

# Downloads the base Llama tokenizer/model snapshot into the repo-local cache
# at data_prep/tokenizer_extension/.models/llama-3.1-8b/ using huggingface-cli.

MODEL_ID="meta-llama/Llama-3.1-8B"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/.models/llama-3.1-8b"

mkdir -p "${TARGET_DIR}"
echo "[download_base_model] Downloading ${MODEL_ID} to ${TARGET_DIR}"
huggingface-cli download "${MODEL_ID}" --local-dir "${TARGET_DIR}" --local-dir-use-symlinks False
echo "[download_base_model] Done"
