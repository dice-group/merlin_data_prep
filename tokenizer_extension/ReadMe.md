## Tokenizer extension (SLURM-first)

Core logic lives in `src/`. Main entrypoints are in `scripts/`.

### Run on SLURM (single job)
```bash
srun python -m tokenizer_extension.pipeline_slurm \
  --base-data-dir /path/to/base_shards \
  --base-tokenizer /path/to/base_tokenizer \
  --total-new-tokens 128000 \
  --extend \
  --extension-data-dir /path/to/extension_shards \
  --extension-base-path /path/to/base_tokenizer
```

### Run staged on SLURM
```bash
bash scripts/run_pipeline_staged.sh \
  --base-data-dir /path/to/base_shards \
  --base-tokenizer /path/to/base_tokenizer \
  --total-new-tokens 128000 \
  --extend \
  --extension-data-dir /path/to/extension_shards \
  --extension-base-path /path/to/base_tokenizer
```

Notes
- Config files are not used; everything is CLI-driven.
- Optional scripts live in `scripts/extras/`.
