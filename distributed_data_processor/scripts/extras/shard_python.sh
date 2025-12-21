#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input_dir> <output_dir>" >&2
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "${SCRIPT_DIR}/../../src" && pwd)"

python -u "${CORE_DIR}/shard_python.py" \
  --input "${INPUT_DIR}" \
  --output "${OUTPUT_DIR}"
