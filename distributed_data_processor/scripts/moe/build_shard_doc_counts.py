import argparse
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple


def _list_shards(shard_dir: Path, languages: List[str] | None) -> List[Tuple[str, Path, str]]:
    valid_suffixes = (".jsonl", ".jsonl.gz")
    roots = [shard_dir] if not languages else [shard_dir / language for language in languages]
    shards: List[Tuple[str, Path, str]] = []
    for root in roots:
        if not root.exists():
            raise RuntimeError(f"Requested language directory is missing: {root}")
        for path in root.rglob("*"):
            if not path.is_file() or not any(path.name.endswith(suffix) for suffix in valid_suffixes):
                continue
            rel = path.relative_to(shard_dir)
            language = rel.parts[0] if rel.parts else path.parent.name
            shards.append((language, path, str(rel)))
    shards.sort(key=lambda item: item[2])
    if not shards:
        raise RuntimeError(f"No .jsonl or .jsonl.gz files found under {shard_dir}")
    return shards


def _count_newlines(path: Path) -> int:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as handle:
        return handle.read().count(b"\n")


def _count_one(shard: Tuple[str, Path, str]) -> Tuple[str, str, int]:
    language, path, rel = shard
    return language, rel, _count_newlines(path)


def _write_tsv(output_path: Path, rows: Iterable[Tuple[str, str, int]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("language\tshard\tdoc_count\n")
        for language, shard, doc_count in rows:
            handle.write(f"{language}\t{shard}\t{doc_count}\n")
    tmp_path.replace(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build persistent per-shard document counts.")
    parser.add_argument("--shard_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument("--scan_processes", type=int, default=8)
    args = parser.parse_args()

    languages = sorted(set(args.languages or [])) or None
    shards = _list_shards(args.shard_dir, languages)
    workers = max(1, min(int(args.scan_processes), len(shards)))

    if workers == 1:
        rows = []
        total = len(shards)
        for done, shard in enumerate(shards, start=1):
            rows.append(_count_one(shard))
            print(f"[COUNT] {done}/{total} ({100.0 * done / total:.1f}%)", flush=True)
    else:
        rows = []
        total = len(shards)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_count_one, shard) for shard in shards]
            for done, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                print(f"[COUNT] {done}/{total} ({100.0 * done / total:.1f}%)", flush=True)

    rows.sort(key=lambda row: (row[0], row[1]))
    _write_tsv(args.output_path, rows)

    total_docs = sum(doc_count for _language, _shard, doc_count in rows)
    print(
        f"[OK] shards={len(rows)} total_docs={total_docs} output={args.output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
