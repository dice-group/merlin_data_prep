from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .allocation import AllocationConfig, compute_allocation
from .coverage import CoverageConfig, compute_coverage
from .extension import ExtensionConfig, ExtensionResult, extend_tokenizer
from .training import TrainingConfig, TrainingResult, train_tokenizer
from .utils import compare_metrics


@dataclass
class ComparisonResult:
    dataframe: pd.DataFrame
    output_csv: Optional[Path]


@dataclass
class PipelineConfig:
    output_dir: Path
    allocation: AllocationConfig
    training: Optional[TrainingConfig] = None
    base_coverage: CoverageConfig | None = None
    extension: Optional[ExtensionConfig] = None
    extended_coverage: Optional[CoverageConfig] = None


@dataclass
class PipelineResult:
    base_metrics: Optional[pd.DataFrame]
    allocation: Optional[pd.DataFrame]
    allocation_csv: Path
    training: Optional[TrainingResult]
    extension: Optional[ExtensionResult]
    extended_metrics: Optional[pd.DataFrame]
    comparison: Optional[ComparisonResult]


STAGE_ORDER = (
    "base_coverage",
    "allocation",
    "training",
    "extension",
    "extended_coverage",
    "comparison",
)


def _resolve_stages(stages: Optional[Sequence[str]]) -> list[str]:
    if stages is None:
        return list(STAGE_ORDER)

    invalid = [stage for stage in stages if stage not in STAGE_ORDER]
    if invalid:
        raise ValueError(f"Unknown stage(s): {', '.join(invalid)}")

    selected = set(stages)
    return [stage for stage in STAGE_ORDER if stage in selected]


def run_base_coverage(config: PipelineConfig) -> pd.DataFrame:
    if config.base_coverage is None:
        raise ValueError("Base coverage configuration is missing.")

    base_metrics_path = config.output_dir / "base_metrics.csv"
    cov_cfg = config.base_coverage
    print(
        "[pipeline] base_coverage starting\n"
        f"  data_dir    : {cov_cfg.data_dir}\n"
        f"  tokenizer   : {cov_cfg.tokenizer}\n"
        f"  max_lines   : {cov_cfg.max_lines}\n"
        f"  num_workers : {cov_cfg.num_workers or 'auto'}\n"
        f"  output_csv  : {base_metrics_path}"
    )
    config.base_coverage.output_csv = base_metrics_path
    df = compute_coverage(config.base_coverage)
    print(f"[pipeline] Base metrics saved to {base_metrics_path}")
    return df


def run_allocation(
    config: PipelineConfig,
    base_metrics: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_metrics_df = base_metrics
    if base_metrics_df is None:
        base_metrics_path = config.output_dir / "base_metrics.csv"
        if not base_metrics_path.exists():
            raise FileNotFoundError(
                f"Base metrics not found at {base_metrics_path}. "
                "Run the base_coverage stage first."
            )
        print(f"[pipeline] allocation: loading base metrics from {base_metrics_path}")
        base_metrics_df = pd.read_csv(base_metrics_path)

    allocation_df = compute_allocation(base_metrics_df, config.allocation)
    allocation_path = config.output_dir / "allocation.csv"
    allocation_df.to_csv(allocation_path, index=False)
    print(
        "[pipeline] allocation summary\n"
        f"  languages        : {len(allocation_df)}\n"
        f"  total_new_tokens : {config.allocation.total_new_tokens}\n"
        f"  output_csv       : {allocation_path}"
    )
    print(f"[pipeline] Allocation saved to {allocation_path}")
    return allocation_df, base_metrics_df


def run_training_stage(config: PipelineConfig) -> TrainingResult:
    if config.training is None:
        raise ValueError("Training configuration is missing.")
    train_cfg = config.training
    print(
        "[pipeline] training starting\n"
        f"  data_dir    : {train_cfg.data_dir}\n"
        f"  output_dir  : {train_cfg.output_dir}\n"
        f"  base_model  : {train_cfg.base_model}\n"
        f"  vocab_size  : {train_cfg.vocab_size}\n"
        f"  max_samples : {train_cfg.max_samples or 'all'}"
    )
    return train_tokenizer(config.training)


def run_extension_stage(
    config: PipelineConfig,
    allocation: Optional[pd.DataFrame],
) -> ExtensionResult:
    if config.extension is None:
        raise ValueError("Extension configuration is missing.")

    allocation_df = allocation
    if allocation_df is None:
        allocation_path = config.output_dir / "allocation.csv"
        if not allocation_path.exists():
            raise FileNotFoundError(
                f"Allocation results not found at {allocation_path}. "
                "Run the allocation stage first."
            )
        print(f"[pipeline] extension: loading allocation from {allocation_path}")
        allocation_df = pd.read_csv(allocation_path)

    ext_cfg = config.extension
    print(
        "[pipeline] extension starting\n"
        f"  base_tokenizer       : {ext_cfg.base_tokenizer_path}\n"
        f"  multilingual_source  : {ext_cfg.multilingual_tokenizer_path}\n"
        f"  data_dir             : {ext_cfg.data_dir}\n"
        f"  output_dir           : {ext_cfg.output_dir}\n"
        f"  vocab_cap            : {ext_cfg.vocab_cap or 'none'}\n"
        f"  sample_docs          : {ext_cfg.sample_docs}\n"
        f"  allocation_languages : {allocation_df.shape[0]}"
    )
    return extend_tokenizer(config.extension, allocation_df)


def run_extended_coverage(config: PipelineConfig) -> pd.DataFrame:
    if config.extended_coverage is None:
        raise ValueError("Extended coverage configuration is missing.")

    extended_metrics_path = config.output_dir / "extended_metrics.csv"
    ext_cov = config.extended_coverage
    print(
        "[pipeline] extended_coverage starting\n"
        f"  data_dir    : {ext_cov.data_dir}\n"
        f"  tokenizer   : {ext_cov.tokenizer}\n"
        f"  max_lines   : {ext_cov.max_lines}\n"
        f"  num_workers : {ext_cov.num_workers or 'auto'}\n"
        f"  output_csv  : {extended_metrics_path}"
    )
    config.extended_coverage.output_csv = extended_metrics_path
    df = compute_coverage(config.extended_coverage)
    print(f"[pipeline] Extended metrics saved to {extended_metrics_path}")
    return df


def run_comparison_stage(
    config: PipelineConfig,
    base_metrics: Optional[pd.DataFrame],
    extended_metrics: Optional[pd.DataFrame],
) -> ComparisonResult:
    base_df = base_metrics
    if base_df is None:
        base_metrics_path = config.output_dir / "base_metrics.csv"
        if not base_metrics_path.exists():
            raise FileNotFoundError(
                f"Base metrics not found at {base_metrics_path}. "
                "Run the base_coverage stage first."
            )
        print(f"[pipeline] comparison: loading base metrics from {base_metrics_path}")
        base_df = pd.read_csv(base_metrics_path)

    extended_df = extended_metrics
    if extended_df is None:
        extended_metrics_path = config.output_dir / "extended_metrics.csv"
        if not extended_metrics_path.exists():
            raise FileNotFoundError(
                f"Extended metrics not found at {extended_metrics_path}. "
                "Run the extended_coverage stage first."
            )
        print(f"[pipeline] comparison: loading extended metrics from {extended_metrics_path}")
        extended_df = pd.read_csv(extended_metrics_path)

    comparison_path = config.output_dir / "comparison.csv"
    print(
        "[pipeline] comparison starting\n"
        f"  base_metrics_rows    : {len(base_df)}\n"
        f"  extended_metrics_rows: {len(extended_df)}\n"
        f"  output_csv           : {comparison_path}"
    )
    comparison_df = compare_metrics(base_df, extended_df)
    comparison_df.to_csv(comparison_path, index=False)
    print(f"[pipeline] Comparison saved to {comparison_path}")

    return ComparisonResult(comparison_df, comparison_path)


def run_pipeline(
    config: PipelineConfig,
    stages: Optional[Sequence[str]] = None,
) -> PipelineResult:
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_stages = _resolve_stages(stages)
    print("[pipeline] execution plan:", ", ".join(resolved_stages))

    base_metrics_df: Optional[pd.DataFrame] = None
    allocation_df: Optional[pd.DataFrame] = None
    training_result: Optional[TrainingResult] = None
    extension_result: Optional[ExtensionResult] = None
    extended_metrics_df: Optional[pd.DataFrame] = None
    comparison_result: Optional[ComparisonResult] = None

    for stage in STAGE_ORDER:
        if stage not in resolved_stages:
            continue
        print(f"[pipeline] >>> entering stage '{stage}'")

        if stage == "base_coverage":
            base_metrics_df = run_base_coverage(config)
        elif stage == "allocation":
            allocation_df, base_metrics_df = run_allocation(config, base_metrics_df)
        elif stage == "training":
            if config.training is None:
                print("[pipeline] Training skipped (no configuration).")
                continue
            training_result = run_training_stage(config)
        elif stage == "extension":
            if config.extension is None:
                print("[pipeline] Extension skipped (no configuration).")
                continue
            allocation_for_extension = allocation_df
            extension_result = run_extension_stage(config, allocation_for_extension)
        elif stage == "extended_coverage":
            if config.extended_coverage is None:
                print("[pipeline] Extended coverage skipped (no configuration).")
                continue
            extended_metrics_df = run_extended_coverage(config)
        elif stage == "comparison":
            comparison_result = run_comparison_stage(
                config,
                base_metrics=base_metrics_df,
                extended_metrics=extended_metrics_df,
            )

        print(f"[pipeline] <<< finished stage '{stage}'")

    return PipelineResult(
        base_metrics=base_metrics_df,
        allocation=allocation_df,
        allocation_csv=out_dir / "allocation.csv",
        training=training_result,
        extension=extension_result,
        extended_metrics=extended_metrics_df,
        comparison=comparison_result,
    )


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    output_dir = Path(
        args.output_dir
        or os.environ.get("TOKENIZER_EXTENSION_OUTPUT_DIR")
        or "results"
    ).expanduser().resolve()

    base_coverage = CoverageConfig(
        data_dir=Path(args.base_data_dir).expanduser().resolve(),
        tokenizer=args.base_tokenizer,
        max_lines=args.base_max_lines,
        num_workers=args.base_num_workers,
        output_csv=None,
    )

    allocation = AllocationConfig(
        total_new_tokens=args.total_new_tokens,
        weight_ppw=args.weight_ppw,
        weight_cpt=args.weight_cpt,
        weight_unk=args.weight_unk,
        gamma=args.gamma,
    )

    training = None
    if args.train_multilingual:
        if not args.train_data_dir:
            raise ValueError("--train-data-dir is required when --train-multilingual is set.")
        train_output_dir = Path(
            args.train_output_dir or (output_dir / "trained_tokenizer")
        ).expanduser().resolve()
        training = TrainingConfig(
            data_dir=Path(args.train_data_dir).expanduser().resolve(),
            output_dir=train_output_dir,
            vocab_size=args.train_vocab_size,
            base_model=args.train_base_model,
            text_key=args.train_text_key,
            max_samples=args.train_max_samples,
        )

    extension = None
    if args.extend:
        if not args.extension_data_dir or not args.extension_base_path:
            raise ValueError("--extension-data-dir and --extension-base-path are required when --extend is set.")
        multilingual_path = args.multilingual_tokenizer_path
        if multilingual_path is None:
            multilingual_path = (output_dir / "trained_tokenizer")
        extension_output_dir = Path(
            args.extension_output_dir or (output_dir / "extended_tokenizer")
        ).expanduser().resolve()
        init_model_dir = (
            Path(args.init_model_output_dir).expanduser().resolve()
            if args.init_model_output_dir
            else (output_dir / "initialized_model")
        )
        extension = ExtensionConfig(
            base_tokenizer_path=Path(args.extension_base_path).expanduser().resolve(),
            multilingual_tokenizer_path=Path(multilingual_path).expanduser().resolve(),
            data_dir=Path(args.extension_data_dir).expanduser().resolve(),
            output_dir=extension_output_dir,
            sample_docs=args.extension_sample_docs,
            text_key=args.extension_text_key,
            vocab_cap=args.extension_vocab_cap,
            num_workers=args.extension_num_workers,
            init_embeddings=args.init_embeddings,
            model_path=args.init_model_path,
            initialized_model_dir=init_model_dir,
        )

    extended_coverage = None
    if args.compute_extended_coverage:
        extended_data_dir = Path(
            args.extended_data_dir or args.extension_data_dir
        ).expanduser().resolve()
        extended_tokenizer_path = args.extended_tokenizer_path or (output_dir / "extended_tokenizer")
        extended_coverage = CoverageConfig(
            data_dir=extended_data_dir,
            tokenizer=str(extended_tokenizer_path),
            max_lines=args.extended_max_lines,
            num_workers=args.extended_num_workers,
        )

    return PipelineConfig(
        output_dir=output_dir,
        allocation=allocation,
        training=training,
        base_coverage=base_coverage,
        extension=extension,
        extended_coverage=extended_coverage,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokenizer extension pipeline (SLURM-oriented).")

    parser.add_argument("--output-dir", default=None, help="Output root (defaults to ./results or TOKENIZER_EXTENSION_OUTPUT_DIR).")
    parser.add_argument("--stage", choices=list(STAGE_ORDER), nargs="+", help="Limit execution to selected stages.")

    parser.add_argument("--base-data-dir", required=True, help="Directory with base shards (*.jsonl.gz).")
    parser.add_argument("--base-tokenizer", required=True, help="Base tokenizer name or path.")
    parser.add_argument("--base-max-lines", type=int, default=10_000, help="Max lines per shard for base coverage.")
    parser.add_argument("--base-num-workers", type=int, default=None, help="Workers for base coverage.")

    parser.add_argument("--total-new-tokens", type=int, default=128_000, help="Total number of new tokens to allocate.")
    parser.add_argument("--weight-ppw", type=float, default=0.5, help="Weight for pieces/word.")
    parser.add_argument("--weight-cpt", type=float, default=0.5, help="Weight for chars/token.")
    parser.add_argument("--weight-unk", type=float, default=0.0, help="Weight for unknown rate.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Non-linear scaling for inefficiency.")

    parser.add_argument("--train-multilingual", action="store_true", help="Enable multilingual tokenizer training.")
    parser.add_argument("--train-data-dir", default=None, help="Directory with training shards (*.jsonl.gz).")
    parser.add_argument("--train-output-dir", default=None, help="Output directory for trained tokenizer.")
    parser.add_argument("--train-vocab-size", type=int, default=250_000, help="Vocab size for training.")
    parser.add_argument("--train-base-model", default="meta-llama/Llama-3.2-1B", help="Base model for training.")
    parser.add_argument("--train-text-key", default="text", help="JSON key for text in training data.")
    parser.add_argument("--train-max-samples", type=int, default=None, help="Max samples for training.")

    parser.add_argument("--extend", action="store_true", help="Enable tokenizer extension.")
    parser.add_argument("--extension-data-dir", default=None, help="Directory with extension shards (*.jsonl.gz).")
    parser.add_argument("--extension-base-path", default=None, help="Path to base tokenizer directory.")
    parser.add_argument("--multilingual-tokenizer-path", default=None, help="Path to trained multilingual tokenizer.")
    parser.add_argument("--extension-output-dir", default=None, help="Output directory for extended tokenizer.")
    parser.add_argument("--extension-sample-docs", type=int, default=50, help="Documents per language for extension.")
    parser.add_argument("--extension-text-key", default="text", help="JSON key for text in extension data.")
    parser.add_argument("--extension-vocab-cap", type=int, default=256_000, help="Cap on total vocab size.")
    parser.add_argument("--extension-num-workers", type=int, default=None, help="Workers for extension.")

    parser.add_argument("--init-embeddings", action="store_true", help="Initialize new embeddings from base model.")
    parser.add_argument("--init-model-path", default=None, help="Model path for embedding init.")
    parser.add_argument("--init-model-output-dir", default=None, help="Output dir for initialized model.")

    parser.add_argument("--compute-extended-coverage", action="store_true", help="Compute coverage for extended tokenizer.")
    parser.add_argument("--extended-data-dir", default=None, help="Directory with extended shards (*.jsonl.gz).")
    parser.add_argument("--extended-tokenizer-path", default=None, help="Tokenizer path for extended coverage.")
    parser.add_argument("--extended-max-lines", type=int, default=10_000, help="Max lines per shard for extended coverage.")
    parser.add_argument("--extended-num-workers", type=int, default=None, help="Workers for extended coverage.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> Optional[PipelineResult]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    pipeline_config = _build_config(args)
    return run_pipeline(pipeline_config, stages=args.stage)


if __name__ == "__main__":
    main()
