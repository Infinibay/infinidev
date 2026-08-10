"""Build a leakage-resistant multilingual corpus for embedding distillation.

The corpus combines aligned Spanish/English programming documentation with
English query/code pairs from CodeSearchNet validation splits.  Every source is
sampled independently with a stable hash so one large language cannot dominate
the student.  Files whose name looks like a test split are rejected: external
retrieval gates must remain untouched by training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def _stable_key(record: dict[str, Any], seed: int) -> bytes:
    identity = str(record.get("id", record.get("text", "")))
    return hashlib.sha256(f"{seed}\0{identity}".encode()).digest()


def select_stable(
    records: Sequence[dict[str, Any]], maximum: int | None, seed: int
) -> list[dict[str, Any]]:
    """Select a deterministic hash-uniform subset."""
    ordered = sorted(records, key=lambda record: _stable_key(record, seed))
    return ordered if maximum is None else ordered[:maximum]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load validated JSONL records without changing their provenance."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("text"), str) or not record["text"].strip():
                raise ValueError(f"{path}:{line_number}: missing non-empty text")
            records.append(dict(record))
    return records


def _looks_like_test_split(path: Path) -> bool:
    parts = path.stem.casefold().replace("-", "_").split("_")
    return "test" in parts


def load_codesearchnet(
    path: Path,
    *,
    language: str,
    max_chars: int,
) -> list[dict[str, Any]]:
    """Load aligned documentation/code records from a non-test parquet split."""
    if _looks_like_test_split(path):
        raise ValueError(
            f"refusing CodeSearchNet test split as distillation data: {path}"
        )
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("CodeSearchNet extraction requires pandas and pyarrow") from exc

    frame = pd.read_parquet(
        path,
        columns=["func_documentation_string", "func_code_string", "func_code_url"],
    )
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        query = " ".join(str(row["func_documentation_string"]).split())[:max_chars]
        code = str(row["func_code_string"]).strip()[:max_chars]
        if len(query) < 16 or len(code) < 24:
            continue
        identity = str(row["func_code_url"])
        digest = hashlib.sha256(
            f"codesearchnet\0{language}\0{identity}".encode()
        ).hexdigest()[:24]
        records.append({
            "id": digest,
            "source": f"codesearchnet_{language}",
            "source_url": identity,
            "license": "dataset-source-metadata",
            "license_class": "upstream-record",
            "path": identity,
            "kind": "text_to_code_retrieval",
            "language": "en",
            "text": query,
            "parallel_language": language,
            "parallel_text": code,
            "characters": len(query),
            "parallel_characters": len(code),
        })
    return records


def _deduplicate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for record in records:
        key = (
            " ".join(str(record["text"]).casefold().split()),
            " ".join(str(record.get("parallel_text", "")).casefold().split()),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def parse_codesearchnet_specs(values: Sequence[str]) -> list[tuple[str, Path]]:
    """Parse repeatable ``LANGUAGE=PATH`` CLI values."""
    result: list[tuple[str, Path]] = []
    for value in values:
        language, separator, raw_path = value.partition("=")
        if not separator or not language.strip() or not raw_path.strip():
            raise ValueError(
                f"invalid --codesearchnet value {value!r}; expected LANGUAGE=PATH"
            )
        result.append((language.strip().casefold(), Path(raw_path).expanduser()))
    return result


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build a balanced corpus from the requested independent sources."""
    groups: list[list[dict[str, Any]]] = []
    if args.spanish_jsonl:
        spanish = load_jsonl(args.spanish_jsonl)
        if not args.include_sharealike:
            spanish = [
                record for record in spanish
                if record.get("license_class") != "sharealike"
            ]
        if not args.include_monolingual_spanish:
            spanish = [
                record for record in spanish
                if isinstance(record.get("parallel_text"), str)
                and record["parallel_text"].strip()
            ]
        groups.append(select_stable(spanish, args.spanish_records, args.seed))
    for language, path in parse_codesearchnet_specs(args.codesearchnet):
        records = load_codesearchnet(
            path,
            language=language,
            max_chars=args.max_chars,
        )
        groups.append(select_stable(records, args.records_per_language, args.seed))
    if not groups:
        raise SystemExit("provide --spanish-jsonl and/or --codesearchnet")
    return sorted(
        _deduplicate(record for group in groups for record in group),
        key=lambda record: (str(record.get("source", "")), str(record["id"])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spanish-jsonl", type=Path)
    parser.add_argument("--spanish-records", type=int, default=4_000)
    parser.add_argument(
        "--include-sharealike",
        action="store_true",
        help="explicitly admit ShareAlike Spanish sources",
    )
    parser.add_argument(
        "--include-monolingual-spanish",
        action="store_true",
        help="admit Spanish rows without an aligned English translation",
    )
    parser.add_argument(
        "--codesearchnet",
        action="append",
        default=[],
        metavar="LANGUAGE=PATH",
    )
    parser.add_argument("--records-per-language", type=int, default=2_000)
    parser.add_argument("--max-chars", type=int, default=700)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.spanish_records < 0 or args.records_per_language < 0:
        parser.error("record limits cannot be negative")
    if args.max_chars < 128:
        parser.error("--max-chars must be at least 128")

    records = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "records": len(records),
        "sources": dict(sorted(Counter(
            str(record.get("source", "unknown")) for record in records
        ).items())),
        "languages": dict(sorted(Counter(
            str(record.get("language", "unknown")) for record in records
        ).items())),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
