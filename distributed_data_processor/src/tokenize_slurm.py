import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datasets import Dataset
from transformers import AutoTokenizer

from validation import ValidationError, verify_tokenized_dataset

LANGUAGE_PAD_ID = -1


def load_language_map(spec: Optional[str]) -> Optional[Dict[str, str]]:
    if spec is None:
        return None

    path = Path(spec)
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            data = json.loads(spec)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse language_map '{spec}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("language_map must decode to a dict mapping language -> family.")

    normalized: Dict[str, str] = {}
    if all(isinstance(value, str) or value is None for value in data.values()):
        for lang, family in data.items():
            if lang is None or family is None:
                continue
            normalized[str(lang)] = str(family)
        return normalized if normalized else None

    flattened = _flatten_groupings_payload(data)
    if flattened:
        return flattened

    raise ValueError("language_map must decode to language->family or groupings JSON.")


def _flatten_groupings_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for group_id, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("group") or group_id)
        languages = set(entry.get("languages") or entry.get("language") or [])
        subgroups = entry.get("subgroups") or {}
        if isinstance(subgroups, dict):
            for members in subgroups.values():
                if isinstance(members, list):
                    languages.update(members)
        metadata = entry.get("metadata") or {}
        if isinstance(metadata, dict):
            languages.update(metadata.keys())
        for lang in languages:
            lang_key = str(lang)
            normalized.setdefault(lang_key, label)
    return normalized


def build_language_vocab(language_map: Dict[str, str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    languages = sorted(language_map.keys())
    families = sorted(set(language_map.values()))
    language_vocab = {lang: idx for idx, lang in enumerate(languages)}
    family_vocab = {fam: idx for idx, fam in enumerate(families)}
    return language_vocab, family_vocab


def language_to_ids(
    language_value: Optional[str],
    language_map: Dict[str, str],
    language_vocab: Dict[str, int],
    family_vocab: Dict[str, int],
) -> Tuple[int, int]:
    if language_value is None:
        return LANGUAGE_PAD_ID, LANGUAGE_PAD_ID

    lang = str(language_value)
    lang_id = language_vocab.get(lang, LANGUAGE_PAD_ID)
    family = language_map.get(lang)
    family_id = family_vocab.get(family, LANGUAGE_PAD_ID) if family is not None else LANGUAGE_PAD_ID
    return lang_id, family_id


def _resolve_rank_world(rank: Optional[int], world_size: Optional[int]) -> tuple[int, int]:
    if rank is not None:
        resolved_rank = rank
    else:
        env_rank = (
            os.getenv("SLURM_ARRAY_TASK_ID")
            or os.getenv("SLURM_PROCID")
            or os.getenv("RANK")
            or os.getenv("LOCAL_RANK")
            or 0
        )
        resolved_rank = int(env_rank)

    if world_size is not None:
        resolved_world = world_size
    else:
        env_world = (
            os.getenv("SLURM_ARRAY_TASK_COUNT")
            or os.getenv("SLURM_NTASKS")
            or os.getenv("WORLD_SIZE")
            or 1
        )
        resolved_world = int(env_world)

    return resolved_rank, max(1, resolved_world)


def _load_rank_shard_plan(shard_plan_dir: str, rank: int) -> List[dict]:
    plan_path = Path(shard_plan_dir) / f"rank_{rank:05d}.jsonl"
    if not plan_path.exists():
        raise RuntimeError(f"Shard plan file missing for rank {rank}: {plan_path}")
    rows: List[dict] = []
    with plan_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {plan_path}:{line_no}") from exc
            if not {"language", "shard", "doc_count", "selected_quota", "validation_quota", "train_quota"} <= row.keys():
                raise RuntimeError(f"Shard plan row missing required fields in {plan_path}:{line_no}")
            rows.append(row)
    return rows


def _hash_score(seed: int, language: str, shard_rel: str, line_idx: int) -> float:
    key = f"{language}\u0000{shard_rel}\u0000{seed}\u0000{line_idx}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _iter_shard_plan_lines(
    shard_dir: Path,
    plan_rows: List[dict],
    sample_seed: int,
    eval_seed: int,
    max_samples: Optional[int],
) -> Iterator[dict]:
    emitted = 0
    for row in sorted(plan_rows, key=lambda item: (str(item["language"]), str(item["shard"]))):
        selected_quota = int(row["selected_quota"])
        validation_quota = int(row["validation_quota"])
        if selected_quota <= 0:
            continue
        language = str(row["language"])
        shard_rel = str(row["shard"])
        shard_path = shard_dir / shard_rel
        if not shard_path.exists():
            raise RuntimeError(f"Shard plan references missing shard: {shard_path}")

        scored_lines: List[Tuple[float, int]] = []
        opener = gzip.open if shard_path.name.endswith(".gz") else open
        with opener(shard_path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line_idx, line in enumerate(handle):
                text = _extract_text(line)
                if text is None:
                    continue
                scored_lines.append((_hash_score(sample_seed, language, shard_rel, line_idx), line_idx))
        expected_doc_count = int(row["doc_count"])
        if len(scored_lines) != expected_doc_count:
            raise RuntimeError(
                f"Shard doc count mismatch for {shard_rel}: planned={expected_doc_count} actual={len(scored_lines)}"
            )
        selected_lines = sorted(scored_lines, key=lambda item: (item[0], item[1]))[:selected_quota]
        selected_line_ids = {line_idx for _score, line_idx in selected_lines}
        validation_ranks = sorted(
            (
                (_hash_score(eval_seed, language, shard_rel, line_idx), line_idx)
                for _score, line_idx in selected_lines
            ),
            key=lambda item: (item[0], item[1]),
        )[:validation_quota]
        validation_lines = {line_idx for _score, line_idx in validation_ranks}

        with opener(shard_path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line_idx, line in enumerate(handle):
                if line_idx not in selected_line_ids:
                    continue
                if max_samples is not None and emitted >= max_samples:
                    return
                text = _extract_text(line)
                if text is None:
                    continue
                doc_id = _document_id(language, shard_path, line_idx, text)
                yield {
                    "text": text,
                    "language": language,
                    "doc_id": doc_id,
                    "split": "validation" if line_idx in validation_lines else "train",
                }
                emitted += 1


def _document_id(language: str, shard_path: Path, line_idx: int, text: str) -> str:
    key = f"{language}\u0000{shard_path}\u0000{line_idx}\u0000{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(key).hexdigest()


def _extract_text(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            text = obj.get("text")
            if isinstance(text, str) and text.strip():
                return text
    except json.JSONDecodeError:
        pass
    return stripped


LanguageMetadata = Optional[Tuple[Dict[str, str], Dict[str, int], Dict[str, int]]]


def _resolve_language_metadata(language_map_spec: Optional[str]) -> LanguageMetadata:
    if language_map_spec is None:
        return None
    language_map = load_language_map(language_map_spec)
    if language_map is None:
        raise RuntimeError(f"language_map {language_map_spec} contains no languages.")
    language_vocab, family_vocab = build_language_vocab(language_map)
    return language_map, language_vocab, family_vocab


def _language_id_columns(
    languages: List[Optional[str]], language_metadata: LanguageMetadata
) -> Tuple[List[int], List[int]]:
    length = len(languages)
    pad = LANGUAGE_PAD_ID
    if language_metadata is None:
        return [pad] * length, [pad] * length
    language_map, language_vocab, family_vocab = language_metadata
    language_ids: List[int] = []
    family_ids: List[int] = []
    for language in languages:
        lang_id, fam_id = language_to_ids(language, language_map, language_vocab, family_vocab)
        language_ids.append(lang_id)
        family_ids.append(fam_id)
    return language_ids, family_ids


def tokenize_fn(
    batch,
    tokenizer,
    keep_text: bool,
    language_metadata: LanguageMetadata,
    max_length: int,
    mask_pad_labels: bool,
):
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_overflowing_tokens=True,
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping", list(range(len(batch["text"]))))
    if mask_pad_labels:
        # Ensure padded tokens do not contribute to LM loss.
        labels = []
        for input_ids, attention_mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
            labels.append([tok if mask == 1 else -100 for tok, mask in zip(input_ids, attention_mask)])
        tokenized["labels"] = labels
    else:
        # Legacy behavior kept for controlled ablations.
        tokenized["labels"] = tokenized["input_ids"].copy()
    languages = batch.get("language")
    if languages is None:
        languages = [None] * len(batch["text"])
    chunk_languages = [languages[idx] for idx in sample_mapping]
    language_ids, family_ids = _language_id_columns(chunk_languages, language_metadata)
    for column in ("language", "split", "doc_id", "language_loss_weight"):
        values = batch.get(column)
        if values is not None:
            tokenized[column] = [values[idx] for idx in sample_mapping]
    chunk_counts: Dict[int, int] = {}
    chunk_idx = []
    for idx in sample_mapping:
        current = chunk_counts.get(idx, 0)
        chunk_idx.append(current)
        chunk_counts[idx] = current + 1
    doc_token_totals = {
        idx: 0 for idx in chunk_counts
    }
    for idx, attention_mask in zip(sample_mapping, tokenized["attention_mask"]):
        doc_token_totals[idx] += int(sum(attention_mask))
    tokenized["chunk_idx"] = chunk_idx
    tokenized["doc_token_count"] = [doc_token_totals[idx] for idx in sample_mapping]
    tokenized["language_ids"] = language_ids
    tokenized["family_ids"] = family_ids
    if keep_text:
        tokenized["text"] = [batch["text"][idx] for idx in sample_mapping]
    return tokenized

def main(args):
    shard_dir = Path(args.shard_dir)
    print(f"Tokenizer used for tokenization: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    rank, world_size = _resolve_rank_world(args.rank, args.world_size)
    shard_plan_rows = _load_rank_shard_plan(args.shard_plan_dir, rank)

    language_metadata = _resolve_language_metadata(getattr(args, "language_map", None))

    with tempfile.TemporaryDirectory(prefix="tok_simple_") as cache_dir:
        data_iter = lambda: _iter_shard_plan_lines(
            shard_dir,
            shard_plan_rows,
            args.sample_seed,
            args.eval_seed,
            args.max_samples,
        )
        assigned_count = len([row for row in shard_plan_rows if int(row["selected_quota"]) > 0])
        raw_dataset = Dataset.from_generator(
            data_iter,
            cache_dir=cache_dir,
        )

        print(f"Rank {rank}: start tokenizing {assigned_count} part(s)")
        keep_text = bool(args.keep_text)
        remove_columns = None if keep_text else ["text"]
        tokenized_dataset = raw_dataset.map(
            lambda batch: tokenize_fn(
                batch,
                tokenizer,
                keep_text,
                language_metadata,
                args.max_length,
                args.mask_pad_labels,
            ),
            batched=True,
            remove_columns=remove_columns,
            num_proc=args.num_proc,
        )

        try:
            verify_tokenized_dataset(
                tokenized_dataset,
                require_language_metadata=language_metadata is not None,
            )
        except ValidationError as exc:
            raise RuntimeError("Tokenized dataset validation failed") from exc

        output_dir = Path(args.save_tokenized_data_dir)
        if world_size > 1:
            output_dir = output_dir / f"rank_{rank:05d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenized_dataset.save_to_disk(str(output_dir))
        print(f"Rank {rank}: tokenized data saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize pre-split corpus parts.")
    parser.add_argument("--shard_dir", type=str, required=True, help="Directory containing <lang>_part_*.jsonl(.gz) files.")
    parser.add_argument("--save_tokenized_data_dir", type=str, required=True, help="Directory to save tokenized shards.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Tokenizer model name or path.")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of workers for tokenizer.map.")
    parser.add_argument("--rank", type=int, default=None, help="Rank override (defaults to SLURM env vars).")
    parser.add_argument("--world_size", type=int, default=None, help="World size override (defaults to SLURM env vars).")
    parser.add_argument("--eval_seed", type=int, default=42, help="Seed for the eval split hash.")
    parser.add_argument("--keep_text", action="store_true", help="Keep the raw text column in the tokenized dataset.")
    parser.add_argument("--sample_seed", type=int, default=0, help="Seed controlling deterministic sampling.")
    parser.add_argument("--shard-plan-dir", type=str, required=True, help="Rank shard-plan directory for exact tokenization quotas.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max number of records to emit per rank.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for tokenization.")
    parser.add_argument("--language-map", type=str, default=None, help="Optional path or JSON string mapping languages to families.")
    parser.add_argument(
        "--mask_pad_labels",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Mask padded label positions to -100 (true) or keep legacy labels (false).",
    )
    args = parser.parse_args()
    main(args)
