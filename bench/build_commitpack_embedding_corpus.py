"""Build balanced instruction-to-code-change pairs from CommitPackFT.

CommitPackFT contains real commit messages paired with before/after files.  This
builder pins the source revision, keeps only explicitly permissive row licenses,
turns each edit into a compact unified diff, and samples every programming
language independently with a deterministic weighted reservoir.  Non-English
instructions receive more sampling weight without inventing translations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import difflib
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

try:
    from bench.language_id import detect_target_language
except ModuleNotFoundError:  # direct ``python bench/<script>.py`` execution
    from language_id import detect_target_language


DATASET = "bigcode/commitpackft"
REVISION = "fc56fe33c030c6daa414c2b112c932b8eed085e6"
DEFAULT_LANGUAGES = (
    "java",
    "javascript",
    "typescript",
    "c",
    "c++",
    "c#",
    "go",
    "rust",
    "assembly",
    "python",
    "ruby",
    "perl",
    "shell",
    "powershell",
    "php",
    "kotlin",
    "dart",
    "lua",
    "sql",
    "zig",
)
CANONICAL_LANGUAGE = {"c++": "cpp", "c#": "csharp", "shell": "bash"}
PERMISSIVE_LICENSES = {
    "0bsd",
    "apache-2.0",
    "artistic-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc0-1.0",
    "isc",
    "mit",
    "mit-0",
    "unlicense",
    "zlib",
}
_LOW_INFORMATION = re.compile(
    r"^(?:merge(?: branch| pull request)?|update|updates|fix|fixes|changes?|cleanup|"
    r"wip|bump version|initial commit)[\s.!#\d_-]*$",
    re.IGNORECASE,
)


def _canonical(language: str) -> str:
    return CANONICAL_LANGUAGE.get(language, language)


def _clean_instruction(value: object, maximum: int) -> str:
    if not isinstance(value, str):
        return ""
    first = " ".join(value.splitlines()[0].split())[:maximum].strip()
    if (
        len(first) < 10
        or sum(character.isalpha() for character in first) < 5
        or _LOW_INFORMATION.fullmatch(first)
    ):
        return ""
    return first


def _detect_language(text: str) -> tuple[str, float]:
    return detect_target_language(text)


def _compact_diff(
    old_contents: object,
    new_contents: object,
    old_file: str,
    new_file: str,
    maximum: int,
) -> str:
    if not isinstance(old_contents, str) or not isinstance(new_contents, str):
        return ""
    if old_contents == new_contents:
        return ""
    lines = difflib.unified_diff(
        old_contents.splitlines(),
        new_contents.splitlines(),
        fromfile=old_file or "before",
        tofile=new_file or "after",
        n=3,
        lineterm="",
    )
    output: list[str] = []
    length = 0
    for line in lines:
        if length + len(line) + 1 > maximum:
            break
        output.append(line)
        length += len(line) + 1
    text = "\n".join(output).strip()
    return text if "@@" in text and any(
        line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        for line in output
    ) else ""


def _split(repository: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(repository.encode()).digest()[:4], "big") % 100
    if bucket < 90:
        return "train"
    return "validation" if bucket < 95 else "test"


def record_from_row(
    row: dict[str, Any],
    dataset_language: str,
    *,
    max_instruction_chars: int,
    max_diff_chars: int,
) -> dict[str, Any] | None:
    """Convert one CommitPackFT row into a provenance-rich retrieval pair."""
    license_name = str(row.get("license", "")).casefold()
    if license_name not in PERMISSIVE_LICENSES:
        return None
    instruction = _clean_instruction(row.get("subject"), max_instruction_chars)
    if not instruction:
        return None
    old_file = str(row.get("old_file") or "")
    new_file = str(row.get("new_file") or "")
    change = _compact_diff(
        row.get("old_contents"), row.get("new_contents"),
        old_file, new_file, max_diff_chars,
    )
    if not change:
        return None
    repositories = str(row.get("repos") or "").split(",")
    repository = repositories[0].strip()
    commit = str(row.get("commit") or "").strip()
    if not repository or not commit:
        return None
    natural_language, confidence = _detect_language(instruction)
    programming_language = _canonical(dataset_language)
    identity = hashlib.sha256(
        f"{repository}\0{commit}\0{new_file}\0{instruction}\0{change}".encode()
    ).hexdigest()[:24]
    return {
        "id": identity,
        "source": f"commitpackft_{programming_language}",
        "source_dataset": DATASET,
        "source_revision": REVISION,
        "source_url": f"https://github.com/{repository}/commit/{commit}",
        "repository": repository,
        "revision": commit,
        "path": new_file or old_file,
        "licenses": [license_name],
        "kind": "instruction_to_code_change",
        "language": natural_language,
        "language_confidence": confidence,
        "programming_language": programming_language,
        "text": instruction,
        "parallel_language": programming_language,
        "parallel_text": change,
        "split": _split(repository),
    }


def _sampling_key(record: dict[str, Any], seed: int) -> float:
    digest = hashlib.sha256(f"{seed}\0{record['id']}".encode()).digest()
    uniform = (int.from_bytes(digest[:8], "big") + 1) / (2**64 + 1)
    target_language = (
        record["language"] in {"es", "pt", "fr", "it"}
        and float(record["language_confidence"]) >= 0.80
    )
    weight = 4.0 if target_language else 1.0
    return math.log(uniform) / weight


def sample_records(
    rows: Iterable[dict[str, Any]],
    dataset_language: str,
    *,
    limit: int,
    max_instruction_chars: int,
    max_diff_chars: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Select a deterministic weighted reservoir without loading a shard."""
    heap: list[tuple[float, str, dict[str, Any]]] = []
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        record = record_from_row(
            row,
            dataset_language,
            max_instruction_chars=max_instruction_chars,
            max_diff_chars=max_diff_chars,
        )
        if record is None:
            rejected["invalid_or_incompatible"] += 1
            continue
        identity = str(record["id"])
        if identity in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(identity)
        item = (_sampling_key(record, seed), identity, record)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    records = [item[2] for item in sorted(heap, reverse=True)]
    return records, rejected


def _download_shard(language: str, cache_dir: Path | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("CommitPackFT download requires huggingface_hub") from exc
    return Path(hf_hub_download(
        repo_id=DATASET,
        filename=f"data/{language}/data.jsonl",
        repo_type="dataset",
        revision=REVISION,
        cache_dir=str(cache_dir) if cache_dir else None,
    ))


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def build(args: argparse.Namespace) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    natural: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    downloaded_bytes = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for dataset_language in args.language:
            shard = _download_shard(dataset_language, args.cache_dir)
            downloaded_bytes += shard.stat().st_size
            selected, reasons = sample_records(
                _rows(shard),
                dataset_language,
                limit=args.pairs_per_language,
                max_instruction_chars=args.max_instruction_chars,
                max_diff_chars=args.max_diff_chars,
                seed=args.seed,
            )
            rejected.update(reasons)
            for record in selected:
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                counts[str(record["programming_language"])] += 1
                natural[str(record["language"])] += 1
                splits[str(record["split"])] += 1
            print(json.dumps({
                "language": _canonical(dataset_language),
                "records": len(selected),
                "source_bytes": shard.stat().st_size,
            }, sort_keys=True), flush=True)
    temporary.replace(args.output)
    return {
        "output": str(args.output),
        "records": sum(counts.values()),
        "records_by_programming_language": dict(sorted(counts.items())),
        "records_by_natural_language": dict(sorted(natural.items())),
        "records_by_split": dict(sorted(splits.items())),
        "rejections": dict(sorted(rejected.items())),
        "downloaded_source_bytes": downloaded_bytes,
        "output_bytes": args.output.stat().st_size,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", action="append", choices=DEFAULT_LANGUAGES)
    parser.add_argument("--pairs-per-language", type=int, default=7_500)
    parser.add_argument("--max-instruction-chars", type=int, default=500)
    parser.add_argument("--max-diff-chars", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.language = args.language or list(DEFAULT_LANGUAGES)
    if args.pairs_per_language <= 0:
        parser.error("pairs per language must be positive")
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
