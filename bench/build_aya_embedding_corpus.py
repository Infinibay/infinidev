"""Build a multilingual instruction/QA embedding corpus from human Aya data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from bench.language_id import detect_target_language
except ModuleNotFoundError:  # direct ``python bench/<script>.py`` execution
    from language_id import detect_target_language


DATASET = "CohereLabs/aya_dataset"
REVISION = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"
SOURCE_FILE = "data/train-00000-of-00001.parquet"
LANGUAGE_CODES = {
    "eng": "en",
    "spa": "es",
    "por": "pt",
    "fra": "fr",
    "ita": "it",
}


def _clean(value: object, maximum: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:maximum].strip()


def _split(group: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(group.encode()).digest()[:4], "big") % 100
    if bucket < 90:
        return "train"
    return "validation" if bucket < 95 else "test"


def record_from_row(
    row: dict[str, Any], *, max_instruction_chars: int, max_response_chars: int
) -> dict[str, Any] | None:
    """Convert one labeled Aya row without retaining contributor identity."""
    source_code = str(row.get("language_code") or "")
    language = LANGUAGE_CODES.get(source_code)
    if language is None:
        return None
    instruction = _clean(row.get("inputs"), max_instruction_chars)
    response = _clean(row.get("targets"), max_response_chars)
    if len(instruction) < 8 or len(response) < 8:
        return None
    detected_language, confidence = detect_target_language(instruction)
    if detected_language != language:
        return None
    contributor = str(row.get("user_id") or "unknown")
    source_group = hashlib.sha256(contributor.encode()).hexdigest()[:16]
    identity = hashlib.sha256(
        f"{source_code}\0{source_group}\0{instruction}\0{response}".encode()
    ).hexdigest()[:24]
    return {
        "id": identity,
        "source": f"aya_human_{language}",
        "source_dataset": DATASET,
        "source_revision": REVISION,
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "source_group": source_group,
        "licenses": ["Apache-2.0"],
        "kind": "instruction_to_response",
        "annotation_type": str(row.get("annotation_type") or "unknown"),
        "language": language,
        "language_confidence": confidence,
        "parallel_language": language,
        "text": instruction,
        "parallel_text": response,
        "split": _split(source_group),
    }


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise SystemExit("Aya extraction requires pyarrow") from exc
    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=2_048):
        yield from batch.to_pylist()


def _download(cache_dir: Path | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=DATASET,
        filename=SOURCE_FILE,
        repo_type="dataset",
        revision=REVISION,
        cache_dir=str(cache_dir) if cache_dir else None,
    ))


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = _download(args.cache_dir)
    reservoirs: dict[str, list[tuple[bytes, str, dict[str, Any]]]] = {
        language: [] for language in LANGUAGE_CODES.values()
    }
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    for row in _rows(source):
        record = record_from_row(
            row,
            max_instruction_chars=args.max_instruction_chars,
            max_response_chars=args.max_response_chars,
        )
        if record is None:
            rejected["other_language_or_invalid"] += 1
            continue
        identity = str(record["id"])
        if identity in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(identity)
        language = str(record["language"])
        priority = hashlib.sha256(f"{args.seed}\0{identity}".encode()).digest()
        item = (priority, identity, record)
        reservoir = reservoirs[language]
        if len(reservoir) < args.pairs_per_language:
            heapq.heappush(reservoir, item)
        elif item[:2] > reservoir[0][:2]:
            heapq.heapreplace(reservoir, item)

    counts: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    annotations: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for language in sorted(reservoirs):
            for _, _, record in sorted(reservoirs[language], reverse=True):
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                counts[language] += 1
                splits[str(record["split"])] += 1
                annotations[str(record["annotation_type"])] += 1
    temporary.replace(args.output)
    return {
        "output": str(args.output),
        "source_bytes": source.stat().st_size,
        "output_bytes": args.output.stat().st_size,
        "records": sum(counts.values()),
        "records_by_language": dict(sorted(counts.items())),
        "records_by_split": dict(sorted(splits.items())),
        "records_by_annotation_type": dict(sorted(annotations.items())),
        "rejections": dict(sorted(rejected.items())),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-per-language", type=int, default=20_000)
    parser.add_argument("--max-instruction-chars", type=int, default=750)
    parser.add_argument("--max-response-chars", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs_per_language <= 0:
        parser.error("pairs per language must be positive")
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
