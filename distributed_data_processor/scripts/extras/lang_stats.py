#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool

from datasets import DatasetDict, load_from_disk


def _iter_rows(dataset):
    for row in dataset:
        yield row


def _list_rank_dirs(path):
    return sorted(
        p for p in path.iterdir()
        if p.is_dir() and p.name.startswith("rank_") and (p / "dataset_info.json").exists()
    )


def _count_dataset(path, split_name, lang_col, token_col):
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        dataset = dataset[split_name]
    counts = Counter()
    tokens = Counter()
    for row in _iter_rows(dataset):
        lang = row.get(lang_col)
        if lang is None:
            continue
        if isinstance(lang, bytes):
            lang = lang.decode()
        ids = row.get(token_col)
        if isinstance(ids, bytes):
            ids = json.loads(ids)
        counts[lang] += 1
        if ids is not None:
            tokens[lang] += len(ids)
    return counts, tokens


def _count_shard(args):
    path, split_name, shard_idx, num_shards, lang_col, token_col = args
    dataset = load_from_disk(path)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[split_name]
    if num_shards > 1:
        dataset = dataset.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
    counts = Counter()
    tokens = Counter()
    for row in _iter_rows(dataset):
        lang = row.get(lang_col)
        if lang is None:
            continue
        if isinstance(lang, bytes):
            lang = lang.decode()
        ids = row.get(token_col)
        if isinstance(ids, bytes):
            ids = json.loads(ids)
        counts[lang] += 1
        if ids is not None:
            tokens[lang] += len(ids)
    return counts, tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute exact per-language sample/token counts from a tokenized dataset."
    )
    parser.add_argument("--path", required=True, help="Path to tokenized dataset (load_from_disk).")
    parser.add_argument("--lang-col", default="language", help="Language column name.")
    parser.add_argument("--token-col", default="input_ids", help="Token id column name.")
    parser.add_argument("--out", default="lang_stats.tsv", help="Output TSV path.")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of worker processes.")
    args = parser.parse_args()

    counts: Counter = Counter()
    tokens: Counter = Counter()

    num_proc = max(1, args.num_proc)
    rank_dirs = _list_rank_dirs(Path(args.path))
    if rank_dirs and not (Path(args.path) / "dataset_info.json").exists():
        tasks = [(rank_dir, None, args.lang_col, args.token_col) for rank_dir in rank_dirs]
        if num_proc == 1:
            results = [_count_dataset(*tasks[0])]
        else:
            with Pool(processes=num_proc) as pool:
                results = pool.starmap(_count_dataset, tasks)
        for shard_counts, shard_tokens in results:
            counts.update(shard_counts)
            tokens.update(shard_tokens)
    else:
        dataset = load_from_disk(args.path)
        if isinstance(dataset, DatasetDict):
            splits = list(dataset.keys())
        else:
            splits = [None]
        for split_name in splits:
            tasks = [
                (args.path, split_name, shard_idx, num_proc, args.lang_col, args.token_col)
                for shard_idx in range(num_proc)
            ]
            if num_proc == 1:
                results = [_count_shard(tasks[0])]
            else:
                with Pool(processes=num_proc) as pool:
                    results = pool.map(_count_shard, tasks)
            for shard_counts, shard_tokens in results:
                counts.update(shard_counts)
                tokens.update(shard_tokens)

    total_samples = sum(counts.values())
    total_tokens = sum(tokens.values())
    total_avg = (total_tokens / total_samples) if total_samples else 0.0

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("lang\tsamples\ttokens\tavg_tokens\n")
        for lang in sorted(counts):
            s = counts[lang]
            t = tokens[lang]
            avg = (t / s) if s else 0.0
            f.write(f"{lang}\t{s}\t{t}\t{avg:.3f}\n")
        f.write(f"TOTAL\t{total_samples}\t{total_tokens}\t{total_avg:.3f}\n")

    print(f"[lang_stats] total_samples={total_samples} total_tokens={total_tokens} avg_tokens={total_avg:.3f}")


if __name__ == "__main__":
    main()
