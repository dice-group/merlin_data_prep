#!/usr/bin/env bash
set -euo pipefail

mkdir -p .model

python <<'PY'
from huggingface_hub import snapshot_download

models = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B",
]

for model in models:
    path = snapshot_download(
        repo_id=model,
        local_dir=f".model/{model.replace('/', '_')}",
        local_dir_use_symlinks=False,
        allow_patterns=[
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
            "*.json",
            "*.model",
        ],
    )
    print(f"{model} -> {path}")
PY
