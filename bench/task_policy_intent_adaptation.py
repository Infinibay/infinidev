"""Adapt distinct real issues into explicit rare task-policy requests.

Each output preserves one real issue verbatim and adds a bounded instruction
that makes the requested workflow explicit.  The intended category is merely a
generation constraint: every adapted request remains unreviewed until an
independent human or model decision is recorded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from bench.external_candidate_family_split import read_jsonl


ADAPTATION_VERSION = "task-policy-real-issue-intent-adaptation-v1"
SUPPORTED_TARGETS = ("performance", "refactor", "research")
_HINTS = {
    "performance": ("performance", "optimization", "benchmark"),
    "refactor": ("refactor", "cleanup", "tech-debt", "architecture"),
    "research": ("investigation", "question", "research", "discussion", "rfc"),
}
_TEMPLATES = {
    "performance": (
        "Use the issue below as the concrete workload. Measure it under representative conditions, locate the dominant cost, and improve it without changing functional output.",
        "The behavior is functionally acceptable, but its resource profile is not. Establish a reproducible benchmark for this case, optimize the measured bottleneck, and report before/after results.",
        "Treat this as a performance task. Profile the scenario described below first; then reduce latency or resource consumption while preserving its contract.",
        "Before changing implementation details, quantify the slowdown in this report. Make the smallest optimization supported by the measurements and verify that outputs remain equivalent.",
        "Turn this report into a representative benchmark, identify where time or memory is spent, and improve the result. Include measurements rather than relying on intuition.",
        "Investigate the runtime cost described here with profiling data, then implement and verify a behavior-preserving optimization.",
        "Mide este caso con una carga representativa, identifica el cuello de botella y reduce su coste sin alterar los resultados funcionales.",
        "Convierte el escenario siguiente en un benchmark reproducible y optimiza únicamente lo que las mediciones demuestren que domina el coste.",
        "Profile this scenario across the relevant hot path and allocations. Improve the observed p95 while keeping the public behavior unchanged.",
        "The goal is not a speculative cleanup: demonstrate the current cost, optimize the responsible path, and show the measured delta for this exact scenario.",
        "Use tracing or profiling appropriate to this stack to explain the regression below, then recover performance without weakening correctness checks.",
        "Measure throughput and tail latency for the reported workflow, isolate the limiting stage, and make a verified optimization that preserves compatibility.",
    ),
    "refactor": (
        "Restructure the implementation described below while deliberately preserving every observable behavior. Characterize the existing contract first and verify equivalence afterward.",
        "Treat this as an internal refactor, not a feature request. Improve the separation of responsibilities in the affected code without changing its API or outputs.",
        "Simplify the structure behind this issue while keeping callers, error behavior, and externally visible results identical. Add focused characterization coverage where needed.",
        "Reorganize the relevant components so ownership and dependencies are clearer, but do not introduce a new capability or alter established behavior.",
        "Extract the tangled responsibilities in this case into coherent units. Preserve compatibility and prove the refactor with the existing and targeted tests.",
        "Reduce the implementation complexity exposed by this report without changing semantics. Keep the public contract stable and avoid opportunistic feature work.",
        "Refactoriza la estructura interna relacionada con este caso, manteniendo exactamente la API, los errores y los resultados observables.",
        "Separa las responsabilidades que aparecen en el siguiente issue sin añadir capacidades nuevas; caracteriza antes el comportamiento que debe conservarse.",
        "Make this area easier to maintain by removing the structural coupling described below. The acceptance criterion is behavior preservation, not new output.",
        "Consolidate the duplicated or fragmented internal path behind this issue while retaining all supported inputs and lifecycle guarantees.",
        "Move the affected logic behind a clearer boundary and preserve its existing callers. Verify no observable behavior drifts across the restructuring.",
        "Clean up the architecture implicated here with a narrow, behavior-preserving change; do not fold unrelated fixes or product changes into it.",
    ),
    "research": (
        "Investigate the question below and return an evidence-backed recommendation. Compare the relevant alternatives using primary documentation or a focused experiment; do not modify the repository.",
        "Before anyone implements a change for this issue, determine what the external evidence supports. Report sources, tradeoffs, and remaining uncertainty without editing code.",
        "Research this concrete project question. Validate the plausible options against current authoritative sources and, where useful, a small reproducible experiment.",
        "Use the issue below as the research brief: gather evidence, compare viable approaches, and recommend one with explicit assumptions. This is read-only analysis.",
        "We need a decision, not an implementation. Examine the documented behavior and ecosystem alternatives relevant to this case, then produce a sourced recommendation.",
        "Resolve the uncertainty in this report by consulting primary references and testing the key disputed assumption. Summarize findings without changing files.",
        "Investiga esta pregunta concreta, contrasta las alternativas con documentación primaria o un experimento acotado y entrega una recomendación sin modificar código.",
        "Antes de implementar nada, reúne evidencia actual sobre el caso siguiente, explica los tradeoffs y deja claras las incertidumbres restantes.",
        "Assess which approach best fits this scenario by comparing current specifications, upstream behavior, and a minimal experiment. Return a report only.",
        "Find out whether the premise in this issue holds across the relevant versions or platforms. Cite authoritative evidence and recommend the next step without applying it.",
        "Map the available solutions to the constraints described below, verify any consequential claims, and explain which option is best supported.",
        "Conduct a bounded technical investigation of this case: identify competing explanations, test the discriminating evidence, and report the conclusion read-only.",
    ),
}


def _length_bucket(text: str) -> str:
    words = len(text.split())
    if words < 80:
        return "short"
    if words < 300:
        return "medium"
    return "long"


def _hints(candidate: dict[str, Any]) -> tuple[str, ...]:
    source = candidate.get("source")
    if not isinstance(source, dict):
        return ()
    raw = source.get("selection_query_hints") or source.get("selection_query_hint") or ()
    if isinstance(raw, str):
        return (raw.casefold(),)
    if isinstance(raw, list):
        return tuple(str(item).casefold() for item in raw)
    return ()


def _matches(candidate: dict[str, Any], target: str) -> bool:
    return any(token in hint for token in _HINTS[target] for hint in _hints(candidate))


def _adapt(candidate: dict[str, Any], target: str) -> dict[str, Any]:
    parent_id = str(candidate["candidate_id"])
    text = str(candidate["issue_text"]).strip()
    digest = hashlib.sha256(f"{target}:{parent_id}".encode()).digest()
    templates = _TEMPLATES[target]
    instruction = templates[digest[0] % len(templates)]
    language = str(candidate.get("source", {}).get("programming_language") or "")
    context = f" The repository's primary language is {language}." if language else ""
    if digest[1] % 3 == 0:
        issue_text = f"Project context:\n\n{text}\n\nRequested outcome: {instruction}{context}"
    else:
        issue_text = f"{instruction}{context}\n\nProject context:\n\n{text}"
    source = dict(candidate["source"])
    source.update({
        "parent_candidate_id": parent_id,
        "parent_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "intent_adaptation_version": ADAPTATION_VERSION,
        "intended_workflow": target,
        "length_bucket": _length_bucket(issue_text),
    })
    return {
        "candidate_id": f"adapted-{target}:{parent_id}",
        "source": source,
        "issue_text": issue_text,
        "manual_review": {
            "status": "unreviewed", "include": None, "policies": None,
            "uncategorized_reason": None, "notes": None,
        },
    }


def adapt_candidates(
    candidates: list[dict[str, Any]],
    *,
    targets: dict[str, int],
    excluded_parent_ids: frozenset[str] = frozenset(),
    max_per_repo: int = 1,
    seed: int = 20260816,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select source-diverse parents and create one auditable adaptation each."""
    if not targets or any(target not in SUPPORTED_TARGETS for target in targets):
        raise ValueError("unsupported or empty target set")
    if any(count < 1 for count in targets.values()) or max_per_repo < 1:
        raise ValueError("counts and max_per_repo must be positive")
    identifiers = [str(row.get("candidate_id", "")) for row in candidates]
    if any(not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("source candidates need unique candidate IDs")

    parent_ids: set[str] = set()
    adapted: list[dict[str, Any]] = []
    target_reports: dict[str, Any] = {}
    for target, requested in targets.items():
        pool = [
            row for row in candidates
            if str(row["candidate_id"]) not in excluded_parent_ids
            and str(row["candidate_id"]) not in parent_ids
            and _matches(row, target)
        ]
        repo_counts: Counter[str] = Counter()
        language_counts: Counter[str] = Counter()
        length_counts: Counter[str] = Counter()
        chosen: list[dict[str, Any]] = []
        while len(chosen) < requested:
            options = []
            for candidate in pool:
                parent_id = str(candidate["candidate_id"])
                if parent_id in parent_ids:
                    continue
                source = candidate.get("source") or {}
                repo = str(source.get("repo") or parent_id).casefold()
                if repo_counts[repo] >= max_per_repo:
                    continue
                language = str(source.get("programming_language") or "unknown")
                length = _length_bucket(str(candidate["issue_text"]))
                rank = hashlib.sha256(f"{seed}:{target}:{parent_id}".encode()).digest()
                options.append((
                    repo_counts[repo], length_counts[length], language_counts[language],
                    rank, candidate, repo, language, length,
                ))
            if not options:
                break
            _, _, _, _, candidate, repo, language, length = min(options)
            parent_ids.add(str(candidate["candidate_id"]))
            repo_counts[repo] += 1
            language_counts[language] += 1
            length_counts[length] += 1
            chosen.append(_adapt(candidate, target))
        adapted.extend(chosen)
        target_reports[target] = {
            "requested": requested,
            "created": len(chosen),
            "shortfall": requested - len(chosen),
            "repositories": len(repo_counts),
            "languages": dict(sorted(language_counts.items())),
            "source_length_buckets": dict(sorted(length_counts.items())),
        }
    adapted.sort(key=lambda row: str(row["candidate_id"]))
    normalized = [" ".join(str(row["issue_text"]).casefold().split()) for row in adapted]
    report = {
        "source_rows": len(candidates),
        "created": len(adapted),
        "excluded_parent_ids": len(excluded_parent_ids),
        "exact_normalized_text_duplicates": len(normalized) - len(set(normalized)),
        "targets": target_reports,
    }
    return adapted, report


def _excluded_positive_parents(paths: Iterable[Path], targets: set[str]) -> frozenset[str]:
    identifiers: set[str] = set()
    for path in paths:
        for proposal in read_jsonl(path):
            if set(map(str, proposal.get("policies") or ())) & targets:
                identifiers.add(str(proposal["candidate_id"]))
    return frozenset(identifiers)


def _used_parent_ids(paths: Iterable[Path]) -> frozenset[str]:
    identifiers: set[str] = set()
    for path in paths:
        for candidate in read_jsonl(path):
            source = candidate.get("source")
            if not isinstance(source, dict):
                raise ValueError(f"{path}: candidate source must be an object")
            parent_id = source.get("parent_candidate_id")
            identifiers.add(str(parent_id or candidate.get("candidate_id") or ""))
    if "" in identifiers:
        raise ValueError("excluded candidates must have a candidate or parent ID")
    return frozenset(identifiers)


def _target(value: str) -> tuple[str, int]:
    try:
        label, raw_count = value.rsplit("=", 1)
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target must be LABEL=COUNT") from exc
    if label not in SUPPORTED_TARGETS or count < 1:
        raise argparse.ArgumentTypeError("unsupported target or invalid count")
    return label, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target", action="append", type=_target, required=True)
    parser.add_argument("--exclude-positive-proposals", action="append", type=Path, default=[])
    parser.add_argument("--exclude-parent-candidate", action="append", type=Path, default=[])
    parser.add_argument("--max-per-repo", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    targets = dict(args.target)
    if len(targets) != len(args.target):
        parser.error("each target label may be specified only once")
    excluded = _excluded_positive_parents(
        args.exclude_positive_proposals, set(targets),
    ) | _used_parent_ids(args.exclude_parent_candidate)
    adapted, report = adapt_candidates(
        read_jsonl(args.source), targets=targets, excluded_parent_ids=excluded,
        max_per_repo=args.max_per_repo, seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in adapted
    )
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "version": ADAPTATION_VERSION,
        "source": {
            "path": str(args.source),
            "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        },
        "output": {
            "path": str(args.output), "rows": len(adapted),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        },
        "generation_constraint_is_training_label": False,
        "independent_review_required": True,
        "report": report,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["ADAPTATION_VERSION", "adapt_candidates"]
