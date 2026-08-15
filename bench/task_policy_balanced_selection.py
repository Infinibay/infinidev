"""Select a diverse, label-balanced subset from audited model proposals.

Targets are coverage goals rather than permission to fabricate rows.  The
selector fails closed when there are too few independently reviewed positives
and records every shortfall in a manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from bench.external_candidate_family_split import candidate_families, read_jsonl
from bench.task_policy_teacher_proposals import POLICIES


SELECTION_VERSION = "task-policy-balanced-selection-v1"


def _unique(rows: Iterable[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get(field, ""))
        if not identifier or identifier in result:
            raise ValueError(f"missing or duplicate {field}: {identifier}")
        result[identifier] = row
    return result


def _length_bucket(text: str) -> str:
    words = len(text.split())
    if words < 80:
        return "short"
    if words < 300:
        return "medium"
    return "long"


def _source_value(candidate: dict[str, Any], name: str, fallback: str) -> str:
    source = candidate.get("source")
    value = source.get(name) if isinstance(source, dict) else None
    return str(value or fallback)


def _selection_key(
    candidate: dict[str, Any],
    proposal: dict[str, Any],
    *,
    deficits: dict[str, int],
    repo_counts: Counter[str],
    language_counts: Counter[str],
    length_counts: Counter[str],
    seed: int,
) -> tuple[object, ...]:
    policies = set(map(str, proposal.get("policies") or ()))
    coverage = sum(max(deficits.get(label, 0), 0) for label in policies)
    candidate_id = str(candidate["candidate_id"])
    repo = _source_value(candidate, "repo", candidate_id).casefold()
    language = _source_value(candidate, "programming_language", "unknown")
    length = _length_bucket(str(candidate["issue_text"]))
    rank = hashlib.sha256(f"{seed}:{candidate_id}".encode()).digest()
    return (
        -coverage,
        repo_counts[repo],
        length_counts[length],
        language_counts[language],
        -float(proposal.get("confidence", 0.0)),
        rank,
    )


def select_balanced(
    candidates: list[dict[str, Any]],
    proposals: list[dict[str, Any]],
    *,
    targets: dict[str, int],
    excluded_family_candidates: list[dict[str, Any]] | None = None,
    minimum_confidence: float = 0.0,
    max_per_lexical_family: int = 1,
    seed: int = 20260816,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Greedily meet label goals while spreading sources, lengths, and languages."""
    if not targets or any(label not in POLICIES for label in targets):
        raise ValueError("targets must use known task-policy labels")
    if any(value < 1 for value in targets.values()):
        raise ValueError("target counts must be positive")
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be in [0, 1]")
    if max_per_lexical_family < 1:
        raise ValueError("max_per_lexical_family must be positive")
    candidate_by_id = _unique(candidates, "candidate_id")
    proposal_by_id = _unique(proposals, "candidate_id")
    unknown = set(proposal_by_id) - set(candidate_by_id)
    if unknown:
        raise ValueError(f"proposals reference unknown candidates: {sorted(unknown)[:3]}")

    excluded_family_candidates = excluded_family_candidates or []
    excluded_by_id = _unique(excluded_family_candidates, "candidate_id")
    directly_excluded = set(candidate_by_id) & set(excluded_by_id)
    combined_rows = [
        *candidates,
        *(row for candidate_id, row in excluded_by_id.items() if candidate_id not in candidate_by_id),
    ]
    blocked_ids = set(directly_excluded)
    excluded_ids = set(excluded_by_id)
    for family in candidate_families(combined_rows):
        identifiers = {str(row["candidate_id"]) for row in family}
        if identifiers & excluded_ids:
            blocked_ids.update(identifiers & set(candidate_by_id))

    eligible: dict[str, dict[str, Any]] = {}
    for candidate_id, proposal in proposal_by_id.items():
        if candidate_id in blocked_ids:
            continue
        if proposal.get("proposal_status") != "model_reviewed":
            continue
        confidence = float(proposal.get("confidence", -1.0))
        if confidence < minimum_confidence:
            continue
        policies = set(map(str, proposal.get("policies") or ()))
        if policies & set(targets):
            eligible[candidate_id] = proposal

    content_only = [
        {
            **candidate_by_id[candidate_id],
            "source": {"conversation_id": candidate_id},
        }
        for candidate_id in eligible
    ]
    lexical_family_by_id: dict[str, int] = {}
    for family_index, family in enumerate(candidate_families(content_only)):
        for row in family:
            lexical_family_by_id[str(row["candidate_id"])] = family_index

    selected_ids: set[str] = set()
    counts: Counter[str] = Counter()
    repo_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    lexical_family_counts: Counter[int] = Counter()
    exhausted: set[str] = set()
    while True:
        deficits = {label: target - counts[label] for label, target in targets.items()}
        active = [label for label, deficit in deficits.items() if deficit > 0 and label not in exhausted]
        if not active:
            break
        label = max(active, key=lambda item: (deficits[item] / targets[item], item))
        options = [
            candidate_id for candidate_id, proposal in eligible.items()
            if candidate_id not in selected_ids
            and label in proposal.get("policies", ())
            and lexical_family_counts[lexical_family_by_id[candidate_id]]
            < max_per_lexical_family
        ]
        if not options:
            exhausted.add(label)
            continue
        chosen_id = min(
            options,
            key=lambda candidate_id: _selection_key(
                candidate_by_id[candidate_id], eligible[candidate_id],
                deficits=deficits, repo_counts=repo_counts,
                language_counts=language_counts, length_counts=length_counts,
                seed=seed,
            ),
        )
        selected_ids.add(chosen_id)
        lexical_family_counts[lexical_family_by_id[chosen_id]] += 1
        proposal = eligible[chosen_id]
        counts.update(
            label for label in map(str, proposal.get("policies") or ()) if label in targets
        )
        candidate = candidate_by_id[chosen_id]
        repo_counts[_source_value(candidate, "repo", chosen_id).casefold()] += 1
        language_counts[_source_value(candidate, "programming_language", "unknown")] += 1
        length_counts[_length_bucket(str(candidate["issue_text"]))] += 1

    selected_candidates = sorted(
        (candidate_by_id[item] for item in selected_ids),
        key=lambda row: str(row["candidate_id"]),
    )
    selected_proposals = [
        eligible[str(candidate["candidate_id"])] for candidate in selected_candidates
    ]
    selected_content_only = [
        {
            **candidate,
            "source": {"conversation_id": str(candidate["candidate_id"])},
        }
        for candidate in selected_candidates
    ]
    lexical_families = (
        candidate_families(selected_content_only) if selected_content_only else []
    )
    duplicate_texts = len(selected_candidates) - len({
        " ".join(str(row["issue_text"]).casefold().split()) for row in selected_candidates
    })
    shortfalls = {
        label: max(target - counts[label], 0) for label, target in targets.items()
        if counts[label] < target
    }
    report = {
        "targets": dict(sorted(targets.items())),
        "selected_rows": len(selected_candidates),
        "selected_labels": dict(sorted(counts.items())),
        "shortfalls": dict(sorted(shortfalls.items())),
        "eligible_rows": len(eligible),
        "family_excluded_rows": len(blocked_ids),
        "minimum_confidence": minimum_confidence,
        "max_per_lexical_family": max_per_lexical_family,
        "repositories": len(repo_counts),
        "languages": dict(sorted(language_counts.items())),
        "length_buckets": dict(sorted(length_counts.items())),
        "exact_normalized_text_duplicates": duplicate_texts,
        "lexical_near_duplicate_families": sum(len(family) > 1 for family in lexical_families),
        "lexical_near_duplicate_rows": sum(
            len(family) for family in lexical_families if len(family) > 1
        ),
    }
    return selected_candidates, selected_proposals, report


def _payload(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()


def write_selection(
    candidate_paths: Iterable[Path],
    proposal_paths: Iterable[Path],
    output_prefix: Path,
    *,
    targets: dict[str, int],
    excluded_family_paths: Iterable[Path] = (),
    minimum_confidence: float,
    seed: int,
) -> Path:
    """Write selected candidates/proposals and their reproducibility manifest."""
    candidate_paths = list(candidate_paths)
    proposal_paths = list(proposal_paths)
    excluded_family_paths = list(excluded_family_paths)
    candidates = [row for path in candidate_paths for row in read_jsonl(path)]
    proposals = [row for path in proposal_paths for row in read_jsonl(path)]
    excluded_family_candidates = [
        row for path in excluded_family_paths for row in read_jsonl(path)
    ]
    selected_candidates, selected_proposals, report = select_balanced(
        candidates, proposals, targets=targets,
        excluded_family_candidates=excluded_family_candidates,
        minimum_confidence=minimum_confidence, seed=seed,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    candidate_output = output_prefix.with_name(f"{output_prefix.name}_candidates.jsonl")
    proposal_output = output_prefix.with_name(f"{output_prefix.name}_proposals.jsonl")
    candidate_payload = _payload(selected_candidates)
    proposal_payload = _payload(selected_proposals)
    candidate_output.write_bytes(candidate_payload)
    proposal_output.write_bytes(proposal_payload)
    manifest = {
        "version": SELECTION_VERSION,
        "seed": seed,
        "inputs": {
            "candidates": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in candidate_paths
            ],
            "proposals": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in proposal_paths
            ],
            "excluded_families": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in excluded_family_paths
            ],
        },
        "outputs": {
            "candidates": {
                "path": candidate_output.name, "rows": len(selected_candidates),
                "sha256": hashlib.sha256(candidate_payload).hexdigest(),
            },
            "proposals": {
                "path": proposal_output.name, "rows": len(selected_proposals),
                "sha256": hashlib.sha256(proposal_payload).hexdigest(),
            },
        },
        "report": report,
    }
    manifest_path = output_prefix.with_name(f"{output_prefix.name}_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _target(value: str) -> tuple[str, int]:
    try:
        label, count = value.rsplit("=", 1)
        parsed = int(count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target must be LABEL=COUNT") from exc
    if label not in POLICIES or parsed < 1:
        raise argparse.ArgumentTypeError("target uses an unknown label or invalid count")
    return label, parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--proposal", action="append", type=Path, required=True)
    parser.add_argument("--exclude-family-with", action="append", type=Path, default=[])
    parser.add_argument("--target", action="append", type=_target, required=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    targets = dict(args.target)
    if len(targets) != len(args.target):
        parser.error("each target label may be specified only once")
    manifest = write_selection(
        args.candidate, args.proposal, args.output_prefix,
        targets=targets, excluded_family_paths=args.exclude_family_with,
        minimum_confidence=args.minimum_confidence, seed=args.seed,
    )
    print(manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()


__all__ = ["SELECTION_VERSION", "select_balanced", "write_selection"]
