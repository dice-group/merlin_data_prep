#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_budget(raw: str) -> float:
    value = raw.strip().lower().replace("_", "")
    mult = 1.0
    if value.endswith("b"):
        mult = 1e9
        value = value[:-1]
    elif value.endswith("m"):
        mult = 1e6
        value = value[:-1]
    elif value.endswith("k"):
        mult = 1e3
        value = value[:-1]
    return float(value) * mult


def load_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in csv.reader(path.open(), delimiter="\t"):
        if row and row[0] != "lang":
            counts[row[0]] = int(row[1])
    return counts


def load_avgs(path: Path) -> dict[str, float]:
    avgs: dict[str, float] = {}
    for row in csv.reader(path.open(), delimiter="\t"):
        if row and row[0] != "lang":
            avgs[row[0]] = float(row[2])
    return avgs


def load_langs(path: Path) -> list[str]:
    reader = csv.reader(path.open())
    header = next(reader, [])
    idx = header.index("subset") if "subset" in header else 1
    langs = [row[idx].strip() for row in reader if len(row) > idx and row[idx].strip()]
    return sorted(set(langs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build language keep-rates for a token budget.")
    ap.add_argument("--subset-csv", type=Path, required=True)
    ap.add_argument("--counts-tsv", type=Path, required=True)
    ap.add_argument("--probe-tsv", type=Path, required=True)
    ap.add_argument("--budget", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--english", choices=("mean", "same", "none"), default="mean")
    args = ap.parse_args()

    budget = parse_budget(args.budget)
    counts = load_counts(args.counts_tsv)
    avgs = load_avgs(args.probe_tsv)
    langs = load_langs(args.subset_csv)

    if args.english != "none" and "english" not in langs:
        langs.append("english")
    if args.english == "none" and "english" in langs:
        langs = [l for l in langs if l != "english"]

    missing = [l for l in langs if l not in counts or l not in avgs]
    if missing:
        print(f"[warn] missing {len(missing)} languages, dropping them")
    langs = [l for l in langs if l not in missing]
    non_eng = [l for l in langs if l != "english"]

    rates: dict[str, float] = {}
    if not non_eng:
        denom = counts.get("english", 0) * avgs.get("english", 0.0)
        rates["english"] = min(1.0, budget / denom) if denom else 0.0
    else:
        mean_count = sum(counts[l] for l in non_eng) / len(non_eng)
        tokens_ne = sum(counts[l] * avgs[l] for l in non_eng)
        denom = tokens_ne
        if args.english == "mean" and "english" in langs:
            denom += mean_count * avgs.get("english", 0.0)
        rate_ne = min(1.0, budget / denom) if denom else 0.0
        for lang in non_eng:
            rates[lang] = rate_ne
        if "english" in langs:
            if args.english == "same":
                rates["english"] = rate_ne
            elif args.english == "mean":
                count_en = counts.get("english", 0)
                rates["english"] = min(1.0, (rate_ne * mean_count) / count_en) if count_en else 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rates, indent=2, sort_keys=True))
    print(f"[ok] wrote {args.out} langs={len(langs)}")


if __name__ == "__main__":
    main()
