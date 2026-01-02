#!/usr/bin/env python3
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/scratch/hpc-prf-merlin/nikit/repos/moe-study")
COUNTS_TSV = ROOT / "data_dir/fw_l2v_samples/lang_counts/language_sample_counts.tsv"
PROBE_DIR = ROOT / "data_dir/fw_l2v_samples/lang_counts"
OUT_DIR = ROOT / "data_dir/fw_l2v_samples/lang_counts/language_rates"
SUBSET_CSV_DIR = ROOT / "data_dir/fineweb2_metadata/l2v_nested_samples"

MODELS = {
    "llama-3.2-1B_tokenizer": {
        "probe": "token_probe_tokens_llama3.2-1b.tsv",
        "budgets_b": {"5": 0.746, "10": 0.860, "22": 1.136, "46": 1.686, "96": 2.833, "200": 5.219},
    },
    "llama-3.2-3B_tokenizer": {
        "probe": "token_probe_tokens_llama3.2-3b.tsv",
        "budgets_b": {"5": 0.821, "10": 0.947, "22": 1.250, "46": 1.856, "96": 3.119, "200": 5.746},
    },
    "llama-3.1-8B_tokenizer": {
        "probe": "token_probe_tokens_llama3.1-8b.tsv",
        "budgets_b": {"5": 0.959, "10": 1.106, "22": 1.461, "46": 2.169, "96": 3.644, "200": 6.713},
    },
}

SUBSET_CONFIG = {
    "5_langs": {"csv": "5.csv", "tier": "5"},
    "10_langs": {"csv": "10.csv", "tier": "10"},
    "22_langs": {"csv": "22.csv", "tier": "22"},
    "46_langs": {"csv": "46.csv", "tier": "46"},
    "96_langs": {"csv": "96.csv", "tier": "96"},
    "200_langs": {"csv": "200.csv", "tier": "200"},
}


def _load_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if not row or row[0] == "lang":
                continue
            counts[row[0]] = int(row[1])
    return counts


def _load_avg_tokens(path: Path) -> dict[str, float]:
    avgs: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if not row or row[0] == "lang":
                continue
            avgs[row[0]] = float(row[2])
    return avgs


def _load_subset_languages(path: Path) -> list[str]:
    langs: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lang = row.get("subset")
            if lang:
                langs.append(lang)
    return sorted(set(langs))


def main() -> None:
    counts = _load_counts(COUNTS_TSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for tok_dir, meta in MODELS.items():
        avg_tokens = _load_avg_tokens(PROBE_DIR / meta["probe"])
        tok_out = OUT_DIR / tok_dir
        tok_out.mkdir(parents=True, exist_ok=True)

        for subset_name, cfg in SUBSET_CONFIG.items():
            subset_csv = SUBSET_CSV_DIR / cfg["csv"]
            langs = _load_subset_languages(subset_csv)
            if "english" not in langs:
                langs.append("english")
            if not langs:
                continue

            missing = []
            raw_tokens: dict[str, float] = {}
            for lang in langs:
                if lang not in counts or lang not in avg_tokens:
                    missing.append(lang)
                    continue
                raw_tokens[lang] = counts[lang] * avg_tokens[lang]

            if not raw_tokens:
                continue

            budget_tokens = meta["budgets_b"][cfg["tier"]] * 1e9
            non_eng_raw = [tok for lang, tok in raw_tokens.items() if lang != "english"]
            english_raw = raw_tokens.get("english")

            if not non_eng_raw:
                keep_rate = min(1.0, budget_tokens / english_raw) if english_raw else 0.0
                rates = {"english": keep_rate} if english_raw else {}
            else:
                total_non_eng = sum(non_eng_raw)
                median_non_eng = statistics.median(non_eng_raw)
                cap_tokens = median_non_eng

                keep_rate = budget_tokens / (total_non_eng + cap_tokens)
                if keep_rate > 1.0:
                    keep_rate = 1.0

                english_keep_rate = None
                if english_raw:
                    english_keep_rate = min(1.0, (keep_rate * cap_tokens) / english_raw)
                    if english_keep_rate >= 1.0:
                        remaining = max(0.0, budget_tokens - english_raw)
                        keep_rate = min(1.0, remaining / total_non_eng) if total_non_eng else 0.0
                        english_keep_rate = 1.0

                rates = {lang: keep_rate for lang in langs if lang in raw_tokens and lang != "english"}
                if english_keep_rate is not None:
                    rates["english"] = english_keep_rate

            out_path = tok_out / f"{subset_name}.json"
            out_path.write_text(json.dumps(rates, indent=2, sort_keys=True))

            if missing:
                miss_path = tok_out / f"{subset_name}.missing.txt"
                miss_path.write_text("\n".join(sorted(missing)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
