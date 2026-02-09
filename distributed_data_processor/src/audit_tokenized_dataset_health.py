from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

from datasets import load_from_disk


def _iter_splits(dataset_obj):
    if hasattr(dataset_obj, "keys"):
        for split in dataset_obj.keys():
            yield split, dataset_obj[split]
    else:
        yield "train", dataset_obj


def _fmt_ratio(num: int, den: int) -> str:
    if den <= 0:
        return "n/a"
    return f"{num / den:.4f}"


def audit_path(path: Path, sample_rows: int, require_language_ids: bool) -> tuple[bool, list[str]]:
    logs: list[str] = []
    ok = True

    data = load_from_disk(str(path))
    logs.append(f"[AUDIT] Dataset: {path}")

    for split, ds in _iter_splits(data):
        rows = len(ds)
        cols = list(ds.column_names)
        logs.append(f"  - split={split} rows={rows} columns={cols}")
        if rows == 0:
            logs.append(f"  [FAIL] split={split} has zero rows.")
            ok = False
            continue

        needed = {"input_ids", "attention_mask", "labels"}
        missing = needed.difference(cols)
        if missing:
            logs.append(f"  [FAIL] split={split} missing columns: {sorted(missing)}")
            ok = False
            continue

        inspect_n = min(sample_rows, rows)
        mask0_total = 0
        mask0_nonignored = 0
        neg100_total = 0
        lengths = []
        lang_vals = []
        fam_vals = []

        for i in range(inspect_n):
            ex = ds[i]
            input_ids = ex["input_ids"]
            attn = ex["attention_mask"]
            labels = ex["labels"]

            lengths.append(len(input_ids))
            for a, l in zip(attn, labels):
                if l == -100:
                    neg100_total += 1
                if a == 0:
                    mask0_total += 1
                    if l != -100:
                        mask0_nonignored += 1

            if "language_ids" in ex:
                lang_vals.append(ex["language_ids"])
            if "family_ids" in ex:
                fam_vals.append(ex["family_ids"])

        if mask0_total > 0 and mask0_nonignored > 0:
            logs.append(
                "  [FAIL] split={} has non-ignored pad labels: {} / {} (ratio={})".format(
                    split,
                    mask0_nonignored,
                    mask0_total,
                    _fmt_ratio(mask0_nonignored, mask0_total),
                )
            )
            ok = False
        else:
            logs.append(
                "  [OK] split={} pad-label masking healthy: nonignored={} / {}".format(
                    split, mask0_nonignored, mask0_total
                )
            )

        logs.append(
            "  [INFO] split={} sampled_rows={} -100_labels={} len(min/avg/max)={}/{:.2f}/{}".format(
                split,
                inspect_n,
                neg100_total,
                min(lengths) if lengths else 0,
                (sum(lengths) / len(lengths)) if lengths else 0.0,
                max(lengths) if lengths else 0,
            )
        )

        if require_language_ids:
            if "language_ids" not in cols:
                logs.append(f"  [FAIL] split={split} missing `language_ids` column.")
                ok = False
            else:
                pad = sum(1 for x in lang_vals if x == -1)
                nonpad = len(lang_vals) - pad
                if nonpad == 0:
                    logs.append(f"  [FAIL] split={split} has only padded `language_ids` (-1).")
                    ok = False
                else:
                    logs.append(
                        f"  [OK] split={split} non-pad language_ids in sample: {nonpad}/{len(lang_vals)}"
                    )

        if "language_ids" in cols and lang_vals:
            logs.append(f"  [INFO] split={split} top language_ids: {Counter(lang_vals).most_common(8)}")
        if "family_ids" in cols and fam_vals:
            logs.append(f"  [INFO] split={split} top family_ids: {Counter(fam_vals).most_common(8)}")

    return ok, logs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit tokenized HF datasets for training-safety checks.")
    ap.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="One or more load_from_disk dataset paths.",
    )
    ap.add_argument(
        "--sample-rows",
        type=int,
        default=3000,
        help="Rows to inspect per split.",
    )
    ap.add_argument(
        "--require-language-ids",
        action="store_true",
        help="Fail if language_ids are missing or all -1 in inspected rows.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    overall_ok = True
    for raw in args.paths:
        ok, logs = audit_path(
            path=Path(raw),
            sample_rows=args.sample_rows,
            require_language_ids=args.require_language_ids,
        )
        for line in logs:
            print(line)
        overall_ok = overall_ok and ok

    if not overall_ok:
        raise SystemExit(1)
    print("[AUDIT] All checks passed.")


if __name__ == "__main__":
    main()
