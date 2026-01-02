#!/usr/bin/env python3
import gzip
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

SHARD_ROOT = Path("/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/sharded_samples")
OUT_DIR = Path("/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/lang_counts")
MODELS = [
    ("llama3.2-1b", "meta-llama/Llama-3.2-1B"),
    ("llama3.2-3b", "meta-llama/Llama-3.2-3B"),
    ("llama3.1-8b", "meta-llama/Llama-3.1-8B"),
]
SAMPLES_PER_LANG = 1_000
MAX_FILES_PER_LANG = 50
SEED = 42
WORKERS = 8


def iter_lang_files(lang_dir: Path, max_files: int) -> list[Path]:
    files = sorted(lang_dir.glob("*.jsonl*"))
    if max_files and len(files) > max_files:
        return files[:max_files]
    return files


def iter_texts(lang_dir: Path, max_files: int):
    for path in iter_lang_files(lang_dir, max_files):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        text = obj.get("text")
                        if isinstance(text, str) and text.strip():
                            yield text
                            continue
                except json.JSONDecodeError:
                    pass
                yield line


def _probe_language(args):
    lang_dir, tokenizer = args
    total = 0
    count = 0
    for text in iter_texts(lang_dir, MAX_FILES_PER_LANG):
        if count >= SAMPLES_PER_LANG:
            break
        total += len(tokenizer.encode(text))
        count += 1
    if count == 0:
        return lang_dir.name, 0, 0.0
    return lang_dir.name, count, total / count


def main() -> None:
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lang_dirs = sorted(p for p in SHARD_ROOT.iterdir() if p.is_dir())

    for tag, model_name in MODELS:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        stats = {}
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for lang, count, avg in ex.map(_probe_language, [(p, tok) for p in lang_dirs]):
                if count > 0:
                    stats[lang] = (count, avg)

        out_path = OUT_DIR / f"token_probe_tokens_{tag}.tsv"
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("lang\tsamples\tavg_tokens\n")
            for lang in sorted(stats):
                samples, avg = stats[lang]
                fh.write(f"{lang}\t{samples}\t{avg:.3f}\n")


if __name__ == "__main__":
    main()
