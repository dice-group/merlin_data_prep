#!/usr/bin/env python3
import os
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SHARD_ROOT = Path("/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/sharded_samples")
OUT_DIR = Path("/scratch/hpc-prf-merlin/nikit/repos/moe-study/data_dir/fw_l2v_samples/lang_counts")
TASKS = 200
WORKERS = 8

PIGZ = shutil.which("pigz")


def _is_shard(path: Path) -> bool:
    name = path.name
    return name.endswith(".jsonl") or name.endswith(".jsonl.gz")


def _count_lines(path: Path) -> int:
    if path.name.endswith(".jsonl.gz"):
        if PIGZ:
            proc = subprocess.Popen([PIGZ, "-dc", str(path)], stdout=subprocess.PIPE)
            assert proc.stdout is not None
            data = proc.stdout.read()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"pigz failed on {path} with code {rc}")
            return data.count(b"\n")
        with path.open("rb") as fh:
            return fh.read().count(b"\n")

    with path.open("rb") as fh:
        return fh.read().count(b"\n")


def _count_one(path: Path) -> tuple[str, int]:
    lang = path.parent.name
    return lang, _count_lines(path)


def main() -> None:
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    if task_id < 0 or task_id >= TASKS:
        return

    files = sorted(p for p in SHARD_ROOT.rglob("*") if p.is_file() and _is_shard(p))
    files = [p for i, p in enumerate(files) if i % TASKS == task_id]

    counts: dict[str, int] = defaultdict(int)
    if files:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for lang, n in ex.map(_count_one, files):
                counts[lang] += n

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"part_{task_id:03d}.tsv"
    with out_path.open("w", encoding="utf-8") as fh:
        for lang in sorted(counts):
            fh.write(f"{lang}\t{counts[lang]}\n")


if __name__ == "__main__":
    main()
