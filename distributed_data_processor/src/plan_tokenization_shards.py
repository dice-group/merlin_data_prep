import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


def _load_rates(path: Path | None) -> Dict[str, float]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError("language_rates_json must contain a JSON object mapping language -> rate.")
    rates: Dict[str, float] = {}
    for language, value in data.items():
        rates[str(language)] = max(0.0, min(1.0, float(value)))
    return rates


def _load_shard_counts(path: Path, languages: List[str]) -> List[dict]:
    wanted = set(languages)
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        if header != ["language", "shard", "doc_count"]:
            raise RuntimeError(f"Unexpected shard counts header in {path}: {header}")
        for line_no, line in enumerate(handle, start=2):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 3:
                raise RuntimeError(f"Invalid row in {path}:{line_no}")
            language, shard, doc_count = parts
            if language not in wanted:
                continue
            rows.append(
                {
                    "language": language,
                    "shard": shard,
                    "doc_count": int(doc_count),
                }
            )
    if not rows:
        raise RuntimeError(f"No shard rows found for requested languages in {path}")
    rows.sort(key=lambda row: (row["language"], row["shard"]))
    return rows


def _distribute_exact(total: int, rows: List[dict], weight_key: str, out_key: str, tie_key: str) -> None:
    if total <= 0 or not rows:
        for row in rows:
            row[out_key] = 0
        return
    weight_sum = sum(int(row[weight_key]) for row in rows)
    if weight_sum <= 0:
        for row in rows:
            row[out_key] = 0
        return
    floors = []
    for row in rows:
        ideal = total * int(row[weight_key]) / weight_sum
        base = int(math.floor(ideal))
        row[out_key] = base
        floors.append((ideal - base, str(row[tie_key]), row))
    remaining = total - sum(int(row[out_key]) for row in rows)
    for _frac, _name, row in sorted(floors, key=lambda item: (-item[0], item[1]))[:remaining]:
        row[out_key] += 1


def _target_counts(language_totals: Dict[str, int], rates: Dict[str, float], default_rate: float) -> Dict[str, int]:
    targets: Dict[str, int] = {}
    for language, total_docs in sorted(language_totals.items()):
        rate = rates.get(language, default_rate)
        target = int(round(total_docs * rate))
        if total_docs > 0 and rate > 0.0:
            target = max(1, target)
        targets[language] = min(total_docs, target)
    return targets


def _validation_targets(selected_targets: Dict[str, int], eval_fraction: float) -> Dict[str, int]:
    val_targets: Dict[str, int] = {}
    for language, selected_docs in sorted(selected_targets.items()):
        if selected_docs <= 0 or eval_fraction <= 0.0:
            val_targets[language] = 0
            continue
        if selected_docs == 1:
            val_targets[language] = 0
            continue
        target = int(round(selected_docs * eval_fraction))
        val_targets[language] = min(selected_docs - 1, max(1, target))
    return val_targets


def _write_rank_plans(output_dir: Path, rows: List[dict], world_size: int) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("rank_*.jsonl"):
        stale.unlink()
    rank_rows: List[List[dict]] = [[] for _ in range(world_size)]
    rank_counts = [0 for _ in range(world_size)]
    active_rows = [row for row in rows if int(row["selected_quota"]) > 0]
    active_rows.sort(key=lambda row: (-int(row["selected_quota"]), str(row["shard"])))
    for row in active_rows:
        rank = min(range(world_size), key=lambda idx: (rank_counts[idx], idx))
        rank_rows[rank].append(row)
        rank_counts[rank] += int(row["selected_quota"])
    for rank, plan_rows in enumerate(rank_rows):
        path = output_dir / f"rank_{rank:05d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in sorted(plan_rows, key=lambda item: (item["language"], item["shard"])):
                out = {
                    "language": row["language"],
                    "shard": row["shard"],
                    "doc_count": int(row["doc_count"]),
                    "selected_quota": int(row["selected_quota"]),
                    "validation_quota": int(row["validation_quota"]),
                    "train_quota": int(row["train_quota"]),
                }
                handle.write(json.dumps(out, sort_keys=True) + "\n")
    return {f"rank_{idx:05d}": count for idx, count in enumerate(rank_counts)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan exact per-shard tokenization quotas for Slurm ranks.")
    parser.add_argument("--shard_counts_tsv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--eval_fraction", type=float, default=0.02)
    parser.add_argument("--language_rates_json", type=Path, default=None)
    args = parser.parse_args()

    if args.world_size <= 0:
        raise RuntimeError("--world_size must be positive")

    languages = sorted(set(args.languages))
    rates = _load_rates(args.language_rates_json)
    default_rate = max(0.0, min(1.0, float(args.sample_fraction)))
    rows = _load_shard_counts(args.shard_counts_tsv, languages)

    by_language: Dict[str, List[dict]] = {}
    for row in rows:
        by_language.setdefault(str(row["language"]), []).append(row)

    language_totals = {language: sum(int(row["doc_count"]) for row in lang_rows) for language, lang_rows in by_language.items()}
    selected_targets = _target_counts(language_totals, rates, default_rate)
    validation_targets = _validation_targets(selected_targets, max(0.0, min(1.0, float(args.eval_fraction))))

    for language, lang_rows in sorted(by_language.items()):
        _distribute_exact(int(selected_targets.get(language, 0)), lang_rows, "doc_count", "selected_quota", "shard")
        selected_rows = [row for row in lang_rows if int(row["selected_quota"]) > 0]
        _distribute_exact(
            int(validation_targets.get(language, 0)),
            selected_rows,
            "selected_quota",
            "validation_quota",
            "shard",
        )
        for row in lang_rows:
            row["validation_quota"] = int(row.get("validation_quota", 0))
            row["train_quota"] = int(row["selected_quota"]) - int(row["validation_quota"])

    rank_counts = _write_rank_plans(args.output_dir, rows, int(args.world_size))
    summary = {
        "world_size": int(args.world_size),
        "sample_fraction": default_rate,
        "eval_fraction": max(0.0, min(1.0, float(args.eval_fraction))),
        "language_totals": language_totals,
        "selected_targets": selected_targets,
        "validation_targets": validation_targets,
        "rank_counts": rank_counts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[OK] languages={len(languages)} shards={len(rows)} output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
