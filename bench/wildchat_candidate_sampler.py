"""Build a manual-review queue from natural WildChat user requests.

The lexical rules in this module are acquisition hints, not task-policy labels.
They find programming-related conversations and diversify the queue; every
selected request remains unlabeled until an individual human review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


SOURCE_DATASET = "allenai/WildChat"
SOURCE_CONFIG = "default"
SOURCE_SPLIT = "train"
SOURCE_REVISION = "f66566ceaaeb619dd98ffb0f3bf3ce1f86775ac4"
SOURCE_LICENSE = "ODC-BY-1.0"
SOURCE_URL = "https://huggingface.co/datasets/allenai/WildChat"
DEFAULT_OUTPUT = Path(".infinidev/external-data/wildchat/candidates.jsonl")

_PROGRAMMING_SIGNAL = re.compile(
    r"(?:```|\b(?:api|backend|bug|cli|code|coding|compile|compiler|css|database|debug|"
    r"docker|endpoint|frontend|function|git|github|html|javascript|kotlin|library|"
    r"linux|method|npm|package|php|program|python|query|react|repository|ruby|rust|"
    r"schema|script|sdk|shell|sql|test|typescript|ui|unit test|web app)\b)",
    flags=re.IGNORECASE,
)
_SELECTION_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("review_signal", re.compile(
        r"\b(?:audit|critique|inspect|review|security check|spot (?:issues|problems))\b",
        flags=re.IGNORECASE,
    )),
    ("research_signal", re.compile(
        r"\b(?:benchmark|compare|evaluate (?:options|libraries|frameworks)|find (?:a |the )?"
        r"documentation|investigate|research|survey)\b",
        flags=re.IGNORECASE,
    )),
    ("repair_signal", re.compile(
        r"\b(?:bug|broken|crash(?:es|ed|ing)?|debug|error|fail(?:s|ed|ing)?|fix|regression)\b",
        flags=re.IGNORECASE,
    )),
    ("refactor_signal", re.compile(
        r"\b(?:clean up|decouple|extract (?:a |the )?(?:class|function|method)|refactor|"
        r"reorganize|restructure|simplify (?:the )?code)\b",
        flags=re.IGNORECASE,
    )),
    ("performance_signal", re.compile(
        r"\b(?:benchmark|latency|memory usage|optimi[sz]e|performance|profil(?:e|ing)|"
        r"slow|speed up|throughput)\b",
        flags=re.IGNORECASE,
    )),
    ("implementation_signal", re.compile(
        r"\b(?:add|build|create|implement|make|write)\b",
        flags=re.IGNORECASE,
    )),
    ("question_signal", re.compile(
        r"^(?:can|could|do|does|explain|how|is|should|what|when|where|why|would)\b|\?$",
        flags=re.IGNORECASE,
    )),
)


def first_user_utterance(conversation: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the first non-empty user utterance in a conversation."""
    for message in conversation:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return message
    raise ValueError("conversation has no non-empty user utterance")


def selection_hint(text: str) -> str:
    """Return a coarse acquisition bucket without assigning a policy."""
    for hint, pattern in _SELECTION_HINTS:
        if pattern.search(text):
            return hint
    return "general_programming_signal"


def candidate_from_row(
    row: dict[str, Any],
    *,
    revision: str = SOURCE_REVISION,
) -> dict[str, Any]:
    """Normalize one safe programming request without assigning a policy."""
    conversation = row["conversation"]
    if not isinstance(conversation, list):
        raise ValueError("conversation must be a list")
    utterance = first_user_utterance(conversation)
    text = str(utterance["content"]).strip()
    if row.get("toxic") is True or utterance.get("toxic") is True:
        raise ValueError("toxic conversation")
    if row.get("redacted") is True or utterance.get("redacted") is True:
        raise ValueError("redacted conversation")
    if not _PROGRAMMING_SIGNAL.search(text):
        raise ValueError("request has no programming signal")
    conversation_id = str(row["conversation_id"])
    request_language = str(utterance.get("language") or row.get("language") or "unknown")
    return {
        "candidate_id": f"wildchat:{conversation_id}:0",
        "source": {
            "dataset": SOURCE_DATASET,
            "config": SOURCE_CONFIG,
            "split": SOURCE_SPLIT,
            "dataset_revision": revision,
            "dataset_license": SOURCE_LICENSE,
            "dataset_url": SOURCE_URL,
            "conversation_id": conversation_id,
            "repo": f"wildchat/{conversation_id}",
            "programming_language": "unknown",
            "request_language": request_language,
            "model": str(row.get("model", "unknown")),
            "timestamp": str(row.get("timestamp", "")),
            "turns": int(row.get("turn", 0)),
            "upstream_selection_hint": selection_hint(text),
        },
        "issue_text": text,
        "manual_review": {
            "status": "unreviewed",
            "include": None,
            "policies": None,
            "uncategorized_reason": None,
            "notes": None,
        },
    }


def _rank(candidate: dict[str, Any], seed: int) -> bytes:
    value = f"{seed}:{candidate['candidate_id']}".encode()
    return hashlib.sha256(value).digest()


def _language_bucket(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized and normalized != "unknown" else "unknown"


def select_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    limit: int,
    max_per_language: int,
    seed: int,
    excluded_candidate_ids: frozenset[str] = frozenset(),
    excluded_conversation_ids: frozenset[str] = frozenset(),
    revision: str = SOURCE_REVISION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic, multilingual and hint-diverse review queue."""
    if limit < 1 or max_per_language < 1:
        raise ValueError("limit and max_per_language must be positive")
    candidates: dict[str, dict[str, Any]] = {}
    text_hashes: set[str] = set()
    rejected = Counter()
    for row in rows:
        try:
            candidate = candidate_from_row(row, revision=revision)
        except (KeyError, TypeError, ValueError):
            rejected["invalid_or_unsafe_request"] += 1
            continue
        candidate_id = candidate["candidate_id"]
        conversation_id = candidate["source"]["conversation_id"]
        if conversation_id in excluded_conversation_ids:
            rejected["excluded_conversation"] += 1
            continue
        if candidate_id in excluded_candidate_ids:
            rejected["excluded_candidate_id"] += 1
            continue
        word_count = len(candidate["issue_text"].split())
        if not 4 <= word_count <= 1200:
            rejected["length_outside_review_budget"] += 1
            continue
        text_hash = hashlib.sha256(candidate["issue_text"].casefold().encode()).hexdigest()
        if text_hash in text_hashes:
            rejected["duplicate_text"] += 1
            continue
        text_hashes.add(text_hash)
        candidates[candidate_id] = candidate

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates.values():
        source = candidate["source"]
        buckets[(
            str(source["upstream_selection_hint"]),
            _language_bucket(str(source["request_language"])),
        )].append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: _rank(item, seed))

    selected: list[dict[str, Any]] = []
    language_counts: Counter[str] = Counter()
    positions: Counter[tuple[str, str]] = Counter()
    names = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for name in names:
            bucket = buckets[name]
            language = name[1]
            if language_counts[language] >= max_per_language:
                continue
            if positions[name] >= len(bucket):
                continue
            selected.append(bucket[positions[name]])
            positions[name] += 1
            language_counts[language] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    selected.sort(key=lambda item: item["candidate_id"])
    return selected, {
        "source_rows": sum(rejected.values()) + len(candidates),
        "unique_valid_requests": len(candidates),
        "selected": len(selected),
        "selected_languages": dict(sorted(language_counts.items())),
        "selected_hints": dict(sorted(Counter(
            str(item["source"]["upstream_selection_hint"])
            for item in selected
        ).items())),
        "rejected": dict(sorted(rejected.items())),
        "interpretation": (
            "Selection hints only diversify the manual queue; they are not Infinidev labels."
        ),
    }


def exclusion_sets(paths: Iterable[Path]) -> tuple[frozenset[str], frozenset[str]]:
    """Read candidate and conversation IDs that a new queue must not reuse."""
    candidate_ids: set[str] = set()
    conversation_ids: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                candidate_ids.add(str(row["candidate_id"]))
                conversation_ids.add(str(row["source"]["conversation_id"]))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid candidate row") from exc
    return frozenset(candidate_ids), frozenset(conversation_ids)


def stream_rows(
    *,
    scan_limit: int,
    revision: str = SOURCE_REVISION,
) -> Iterable[dict[str, Any]]:
    """Stream a bounded upstream prefix from the pinned dataset revision."""
    from datasets import load_dataset

    dataset = load_dataset(
        SOURCE_DATASET,
        SOURCE_CONFIG,
        split=SOURCE_SPLIT,
        revision=revision,
        streaming=True,
    )
    yield from dataset.take(scan_limit)


def write_candidate_queue(
    output: Path,
    candidates: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    scan_limit: int,
    selection_limit: int,
    max_per_language: int,
    seed: int,
    revision: str = SOURCE_REVISION,
) -> Path:
    """Write ignored source data and a provenance manifest beside it."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n"
        for candidate in candidates
    )
    output.write_text(payload, encoding="utf-8")
    manifest_path = output.with_suffix(output.suffix + ".provenance.json")
    manifest = {
        "artifact": {
            "path": output.name,
            "rows": len(candidates),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "distribution_notice": (
                "This downloaded artifact is not part of Infinidev's MIT-licensed source. "
                "ODC-By governs the database, while content may carry independent rights."
            ),
        },
        "source": {
            "dataset": SOURCE_DATASET,
            "config": SOURCE_CONFIG,
            "split": SOURCE_SPLIT,
            "revision": revision,
            "license": SOURCE_LICENSE,
            "url": SOURCE_URL,
        },
        "selection": {
            "scan_limit": scan_limit,
            "selection_limit": selection_limit,
            "max_per_language": max_per_language,
            "seed": seed,
            "report": report,
        },
        "privacy_filters": {
            "conversation_toxic_must_be_false": True,
            "conversation_redacted_must_be_false": True,
            "utterance_toxic_must_be_false": True,
            "utterance_redacted_must_be_false": True,
        },
        "review_contract": {
            "selection_hint_is_policy_label": False,
            "individual_manual_review_required": True,
            "source_text_remains_external": True,
            "reviewed_annotations_path": "data/task-policy-reviews/wildchat",
            "reviewed_annotations_license": "CC-BY-4.0 AND ODC-By-1.0",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--scan-limit", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=2_000)
    parser.add_argument("--max-per-language", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument("--exclude-candidates", type=Path, action="append", default=[])
    args = parser.parse_args()
    excluded_ids, excluded_conversations = exclusion_sets(args.exclude_candidates)
    candidates, report = select_candidates(
        stream_rows(scan_limit=args.scan_limit, revision=args.revision),
        limit=args.limit,
        max_per_language=args.max_per_language,
        seed=args.seed,
        excluded_candidate_ids=excluded_ids,
        excluded_conversation_ids=excluded_conversations,
        revision=args.revision,
    )
    manifest_path = write_candidate_queue(
        args.output,
        candidates,
        report,
        scan_limit=args.scan_limit,
        selection_limit=args.limit,
        max_per_language=args.max_per_language,
        seed=args.seed,
        revision=args.revision,
    )
    report["output"] = str(args.output)
    report["provenance_manifest"] = str(manifest_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "SOURCE_DATASET",
    "SOURCE_LICENSE",
    "SOURCE_REVISION",
    "candidate_from_row",
    "exclusion_sets",
    "first_user_utterance",
    "select_candidates",
    "selection_hint",
    "stream_rows",
    "write_candidate_queue",
]
