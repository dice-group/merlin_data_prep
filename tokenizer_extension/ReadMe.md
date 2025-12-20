## Tokenizer extension core

This is the shared tokenizer-extension core (coverage, allocation, optional training, extension, and comparison).
Keep repo-specific configs and cluster scripts outside this folder.

### Run locally
```bash
python -m tokenizer_extension.pipeline --config /path/to/config.yaml
```

### Run staged on SLURM
```bash
python -m tokenizer_extension.pipeline_slurm --config /path/to/config.yaml
```

Notes
- Configs should live in the repo that owns the dataset (e.g., CoLA tiers or MoE study configs).
- Wrapper scripts (SLURM submissions, tier presets, etc.) should stay outside this core.
