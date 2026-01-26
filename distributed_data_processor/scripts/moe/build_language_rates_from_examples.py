#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if not row or row[0] == "lang":
                continue
            counts[row[0]] = int(row[1])
    return counts


def load_subset_langs(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        if "subset" in header:
            idx = header.index("subset")
        elif len(header) > 1:
            idx = 1
        else:
            idx = 0
        langs = [row[idx].strip() for row in reader if len(row) > idx and row[idx].strip()]
    return sorted(set(langs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build keep-rates to hit a target example count.")
    ap.add_argument("--subset-csv", type=Path, required=True)
    ap.add_argument("--counts-tsv", type=Path, required=True)
    ap.add_argument("--target-examples", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--english", choices=("none", "only", "include"), default="include")
    args = ap.parse_args()

    counts = load_counts(args.counts_tsv)
    langs = load_subset_langs(args.subset_csv)
    if args.english == "only":
        langs = ["english"]
    elif args.english == "none":
        langs = [l for l in langs if l != "english"]
    else:
        if "english" not in langs:
            langs.append("english")

    missing = [l for l in langs if l not in counts]
    if missing:
        print(f"[warn] missing {len(missing)} languages, dropping them")
    langs = [l for l in langs if l in counts]
    if not langs:
        raise SystemExit("No languages available after filtering.")

    total = sum(counts[l] for l in langs)
    if total <= 0:
        raise SystemExit("Total example count is zero.")

    keep_rate = args.target_examples / total
    if keep_rate > 1.0:
        print(f"[warn] target exceeds total ({args.target_examples} > {total}); clamping keep_rate to 1.0")
        keep_rate = 1.0
    if keep_rate < 0.0:
        keep_rate = 0.0

    rates = {lang: keep_rate for lang in langs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rates, indent=2, sort_keys=True))
    print(f"[ok] wrote {args.out} langs={len(langs)} keep_rate={keep_rate:.6f}")


if __name__ == "__main__":
    main()
