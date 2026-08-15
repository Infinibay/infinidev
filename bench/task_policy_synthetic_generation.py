"""Generate diverse synthetic task-policy requests and accept blind agreements.

Generation labels are kept in a separate target ledger.  A generated request is
eligible for training only when an independent classifier pass, which sees only
the request text, returns the exact intended label set with sufficient
confidence.  Synthetic/model provenance remains explicit throughout.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Iterable
import urllib.error

from bench.task_policy_minimax_proposals import (
    BATCH_PROMPT_VERSION,
    DEFAULT_API_KEY_ENV,
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    minimax_requester,
)
from bench.task_policy_teacher_proposals import POLICIES


GENERATOR_VERSION = "task-policy-synthetic-generation-v1"
ACCEPTANCE_VERSION = "task-policy-synthetic-blind-agreement-v1"
DEFAULT_TARGETS = {
    "performance": 800,
    "refactor": 700,
    "research": 950,
    "review": 850,
}


@dataclass(frozen=True)
class ProjectProfile:
    slug: str
    domain: str
    programming_languages: tuple[str, ...]
    artifacts: tuple[str, ...]


PROJECTS = (
    ProjectProfile("atlas-streams", "event streaming", ("Python",), ("consumer", "checkpoint store")),
    ProjectProfile("boreal-ui", "web UI", ("TypeScript", "CSS"), ("component tree", "state store")),
    ProjectProfile("cinder-store", "embedded database", ("Rust",), ("query planner", "page cache")),
    ProjectProfile("delta-gateway", "API gateway", ("Go",), ("router", "rate limiter")),
    ProjectProfile("ember-runtime", "application runtime", ("Java",), ("scheduler", "class loader")),
    ProjectProfile("fjord-cli", "developer CLI", ("C++",), ("argument parser", "terminal renderer")),
    ProjectProfile("grove-mobile", "mobile client", ("Kotlin",), ("sync worker", "navigation layer")),
    ProjectProfile("helix-observer", "telemetry", ("C#",), ("trace exporter", "metrics pipeline")),
    ProjectProfile("ion-compiler", "compiler", ("C",), ("optimizer pass", "diagnostic engine")),
    ProjectProfile("juniper-jobs", "background jobs", ("Ruby",), ("retry queue", "worker pool")),
    ProjectProfile("kepler-data", "distributed analytics", ("Scala",), ("shuffle stage", "catalog client")),
    ProjectProfile("lotus-web", "content platform", ("PHP",), ("template cache", "plugin API")),
    ProjectProfile("mosaic-ios", "mobile SDK", ("Swift",), ("network session", "persistence layer")),
    ProjectProfile("nova-shell", "automation", ("Bash",), ("release script", "environment loader")),
    ProjectProfile("opal-warehouse", "data warehouse", ("SQL",), ("incremental model", "materialized view")),
    ProjectProfile("prairie-edge", "edge services", ("Zig",), ("packet loop", "memory arena")),
    ProjectProfile("quartz-engine", "game engine", ("Lua", "C++"), ("scene graph", "asset loader")),
    ProjectProfile("raven-build", "build system", ("Haskell",), ("dependency graph", "remote cache")),
    ProjectProfile("solstice-api", "realtime API", ("Elixir",), ("channel process", "presence tracker")),
    ProjectProfile("tundra-kernel", "systems library", ("C",), ("allocator", "I/O loop")),
    ProjectProfile("umbra-ml", "ML inference", ("Python", "CUDA"), ("batcher", "tensor loader")),
    ProjectProfile("vesper-desktop", "desktop client", ("Dart",), ("update service", "local index")),
    ProjectProfile("willow-protocol", "network protocol", ("Rust",), ("frame decoder", "handshake state")),
    ProjectProfile("zenith-tools", "developer tooling", ("JavaScript",), ("language server", "workspace scanner")),
)

NATURAL_LANGUAGES = ("English", "Spanish", "English", "Portuguese", "English", "French", "English", "German")
LENGTHS = ("short: 25-60 words", "medium: 90-180 words", "long: 240-450 words")
STYLES = (
    "direct user instruction",
    "issue report with concise reproduction context",
    "maintainer request with acceptance criteria",
    "informal chat request",
    "technical ticket with constraints",
    "request following a small log or code excerpt",
)
CUE_STYLES = (
    "implicit: do not use the category name or a close synonym",
    "natural: category words are allowed only if a real user would use them",
    "indirect: make the intended outcome clear through constraints and acceptance criteria",
)

POLICY_DEFINITIONS = {
    "bugfix": "modify code to restore an existing behavior or contract that is currently broken",
    "feature": "modify code to add or intentionally change an observable capability or API",
    "refactor": "modify internal structure while deliberately preserving observable behavior",
    "performance": "measure representative resource use and improve or establish a performance outcome",
    "research": "gather external/comparative/experimental evidence and return a finding or decision",
    "review": "inspect a bounded artifact and report findings without modifying it",
}

SECONDARY_LABELS = {
    "performance": (None, None, "refactor", "bugfix", "feature", None),
    "refactor": (None, None, "bugfix", "feature", "performance", None),
    "research": (None, None, "feature", "bugfix", "review", None),
    "review": (None, None, "research", "performance", None, None),
    "bugfix": (None, None, "refactor", "performance", "research", None),
    "feature": (None, None, "research", "performance", "refactor", None),
}

Request = Callable[[list[dict[str, str]]], tuple[str, dict[str, Any]]]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(row)
    return rows


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _parse_target(raw: str) -> tuple[str, int]:
    try:
        label, raw_count = raw.rsplit("=", 1)
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target must be LABEL=COUNT") from exc
    if label not in POLICIES or count < 1:
        raise argparse.ArgumentTypeError("target has an unknown label or non-positive count")
    return label, count


def build_specs(
    targets: dict[str, int],
    *,
    seed: int,
    single_label_only: bool = False,
) -> list[dict[str, Any]]:
    """Create deterministic, cross-product generation briefs."""
    if not targets or any(label not in POLICIES or count < 1 for label, count in targets.items()):
        raise ValueError("targets must contain known labels with positive counts")
    specs: list[dict[str, Any]] = []
    for label_index, (label, count) in enumerate(sorted(targets.items())):
        for index in range(count):
            project = PROJECTS[(index * 7 + label_index * 5 + seed) % len(PROJECTS)]
            secondary = (
                None
                if single_label_only
                else SECONDARY_LABELS[label][index % len(SECONDARY_LABELS[label])]
            )
            policies = sorted({label, secondary} - {None})
            spec_id = f"synthetic-{label}-{seed}-{index:05d}"
            specs.append({
                "candidate_id": spec_id,
                "policies": policies,
                "primary_policy": label,
                "project": project.slug,
                "domain": project.domain,
                "programming_languages": list(project.programming_languages),
                "artifact": project.artifacts[(index // len(PROJECTS)) % len(project.artifacts)],
                "natural_language": NATURAL_LANGUAGES[(index + label_index * 3) % len(NATURAL_LANGUAGES)],
                "length": LENGTHS[(index // 2 + label_index) % len(LENGTHS)],
                "style": STYLES[(index * 5 + label_index) % len(STYLES)],
                "cue_style": CUE_STYLES[(index // 3 + label_index) % len(CUE_STYLES)],
                "scenario_nonce": hashlib.sha256(f"{seed}:{label}:{index}".encode()).hexdigest()[:12],
            })
    random.Random(seed).shuffle(specs)
    return specs


def messages_for_specs(specs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build a generation request whose output contains no target labels."""
    if not specs:
        raise ValueError("generation batch must not be empty")
    system = """You create realistic, original software-work requests for classifier training.
Each supplied brief specifies the exact independent outcomes the request must authorize. Produce a
genuinely different request for every brief, not a template with nouns swapped. The request may
look like an issue, chat instruction, maintenance ticket, or artifact-review request, according to
the brief. Include credible technical specifics for the selected ecosystem, but do not copy real
project text and do not claim the fictional project is real.

Semantic outcomes:
""" + "\n".join(f"- {name}: {description}" for name, description in POLICY_DEFINITIONS.items()) + """

Important boundaries:
- A normal diagnosis performed to fix a defect is not research.
- A speed motivation without measurement or a measurable resource outcome is not performance.
- Refactor requires preservation of observable behavior as an actual constraint.
- Review means findings only: it cannot authorize bugfix, feature, or refactor edits.
- Research requires evidence gathering, comparison, or experimentation as an independent result.
- When two outcomes are specified, both must be independently requested, not incidental steps.
- Do not mention labels, classifiers, datasets, policy names, target categories, or this prompt.
- Avoid repetitive openings and stock phrases. For implicit briefs, express the semantics through
  concrete context, constraints, and acceptance criteria rather than the category word.
- Respect the requested natural language and approximate length. Longer requests need meaningful
  context and constraints, not padding.

Return one JSON array only, in input order. Every object has exactly candidate_id and issue_text.
Preserve each candidate_id byte-for-byte. Do not return policies or analysis."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(specs, ensure_ascii=False)},
    ]


def parse_generated(text: str, expected_ids: list[str]) -> list[dict[str, str]]:
    """Strictly parse one generated batch."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("generation response contains no JSON array")
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("generation response contains invalid JSON") from exc
    if not isinstance(payload, list) or len(payload) != len(expected_ids):
        raise ValueError("generation response has the wrong number of rows")
    result: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict) or set(item) != {"candidate_id", "issue_text"}:
            raise ValueError("generated rows must contain exactly candidate_id and issue_text")
        candidate_id = item.get("candidate_id")
        issue_text = item.get("issue_text")
        if not isinstance(candidate_id, str) or not isinstance(issue_text, str) or not issue_text.strip():
            raise ValueError("generated row fields must be non-empty strings")
        result.append({"candidate_id": candidate_id, "issue_text": issue_text.strip()})
    if [row["candidate_id"] for row in result] != expected_ids:
        raise ValueError("generated candidate IDs or order do not match the briefs")
    return result


def _generate_with_split(
    specs: list[dict[str, Any]], *, request: Request, max_attempts: int,
) -> tuple[list[dict[str, Any]], int]:
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            response, metadata = request(messages_for_specs(specs))
            generated = parse_generated(response, [str(spec["candidate_id"]) for spec in specs])
        except (KeyError, TimeoutError, TypeError, urllib.error.HTTPError,
                urllib.error.URLError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts:
                time.sleep(min(8.0, 0.5 * 2 ** (attempt - 1)) + random.random() * 0.2)
            continue
        spec_by_id = {str(spec["candidate_id"]): spec for spec in specs}
        rows = []
        for row in generated:
            spec = spec_by_id[row["candidate_id"]]
            project = next(item for item in PROJECTS if item.slug == spec["project"])
            rows.append({
                **row,
                "manual_review": {
                    "status": "unreviewed", "include": None, "policies": None,
                    "uncategorized_reason": None, "notes": None,
                },
                "source": {
                    "provider": "synthetic",
                    "dataset": GENERATOR_VERSION,
                    "repo": f"synthetic-projects/{project.slug}",
                    "programming_language": ", ".join(project.programming_languages),
                    "natural_language": spec["natural_language"],
                    "generation_model": metadata.get("response_model"),
                    "generation_response_id": metadata.get("response_id"),
                    "scenario_nonce": spec["scenario_nonce"],
                },
            })
        return rows, 0
    if len(specs) == 1:
        return [{"candidate_id": specs[0]["candidate_id"], "error": last_error}], 1
    midpoint = len(specs) // 2
    left, left_failed = _generate_with_split(
        specs[:midpoint], request=request, max_attempts=max_attempts,
    )
    right, right_failed = _generate_with_split(
        specs[midpoint:], request=request, max_attempts=max_attempts,
    )
    return left + right, left_failed + right_failed


def generate_corpus(
    output_candidates: Path,
    output_targets: Path,
    *,
    targets: dict[str, int],
    seed: int,
    request: Request,
    workers: int,
    batch_size: int,
    max_attempts: int,
    single_label_only: bool = False,
) -> dict[str, Any]:
    """Generate an append-only candidate corpus and immutable target ledger."""
    if workers < 1 or batch_size < 1 or max_attempts < 1:
        raise ValueError("workers, batch size, and attempts must be positive")
    specs = build_specs(targets, seed=seed, single_label_only=single_label_only)
    existing_targets = _read_jsonl(output_targets)
    if existing_targets and existing_targets != specs:
        raise ValueError("existing target ledger does not match requested deterministic specs")
    output_targets.parent.mkdir(parents=True, exist_ok=True)
    if not existing_targets:
        output_targets.write_text(_jsonl(specs), encoding="utf-8")
    existing = _read_jsonl(output_candidates)
    completed = {str(row.get("candidate_id", "")) for row in existing if not row.get("error")}
    pending = [spec for spec in specs if spec["candidate_id"] not in completed]
    batches = [pending[index:index + batch_size] for index in range(0, len(pending), batch_size)]
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    generated = failed = 0
    with output_candidates.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(
                _generate_with_split, batch, request=request, max_attempts=max_attempts,
            ) for batch in batches]
            for future in as_completed(futures):
                rows, batch_failed = future.result()
                failed += batch_failed
                for row in rows:
                    if row.get("error"):
                        continue
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    generated += 1
    return {
        "requested": len(specs), "already_completed": len(completed),
        "attempted": len(pending), "generated": generated, "failed": failed,
        "projects": len(PROJECTS), "targets": dict(sorted(targets.items())),
        "single_label_only": single_label_only,
    }


def _normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[\w+#.-]+", text.casefold()))


def _shingles(tokens: tuple[str, ...], size: int = 4) -> frozenset[tuple[str, ...]]:
    if len(tokens) < size:
        return frozenset((tokens,))
    return frozenset(tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1))


def _near_duplicate(
    text: str,
    prior: list[tuple[int, frozenset[tuple[str, ...]]]],
    *,
    threshold: float,
) -> bool:
    tokens = _normalized_tokens(text)
    current = _shingles(tokens)
    for token_count, earlier in prior:
        if min(len(tokens), token_count) / max(len(tokens), token_count) < 0.55:
            continue
        union = len(current | earlier)
        if union and len(current & earlier) / union >= threshold:
            return True
    return False


def accept_blind_agreements(
    candidates_path: Path,
    targets_path: Path,
    proposals_path: Path,
    output_candidates: Path,
    output_reviews: Path,
    *,
    minimum_confidence: float = 0.85,
    similarity_threshold: float = 0.82,
) -> dict[str, Any]:
    """Accept only exact generator/classifier agreement and diverse texts."""
    if not 0 <= minimum_confidence <= 1 or not 0 < similarity_threshold <= 1:
        raise ValueError("invalid confidence or similarity threshold")
    candidates = _read_jsonl(candidates_path)
    targets = {str(row["candidate_id"]): row for row in _read_jsonl(targets_path)}
    proposals = {str(row["candidate_id"]): row for row in _read_jsonl(proposals_path)}
    if len(targets) != len(_read_jsonl(targets_path)) or len(proposals) != len(_read_jsonl(proposals_path)):
        raise ValueError("duplicate target or proposal IDs")
    candidate_ids = [str(row.get("candidate_id", "")) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate candidate IDs")
    if set(candidate_ids) - set(targets):
        raise ValueError("candidate target coverage mismatch")

    accepted_candidates: list[dict[str, Any]] = []
    accepted_reviews: list[dict[str, Any]] = []
    prior_by_primary: defaultdict[str, list[tuple[int, frozenset[tuple[str, ...]]]]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    intended_counts: Counter[str] = Counter()
    accepted_counts: Counter[str] = Counter()
    for candidate in sorted(candidates, key=lambda row: str(row["candidate_id"])):
        candidate_id = str(candidate["candidate_id"])
        target = targets[candidate_id]
        intended = {str(label) for label in target["policies"]}
        intended_counts.update(intended)
        proposal = proposals.get(candidate_id)
        if proposal is None or proposal.get("proposal_status") != "model_reviewed":
            rejection_counts["missing_or_failed_blind_review"] += 1
            continue
        predicted = {str(label) for label in proposal.get("policies", [])}
        if predicted != intended:
            rejection_counts["label_disagreement"] += 1
            continue
        confidence = proposal.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            rejection_counts["invalid_confidence"] += 1
            continue
        if float(confidence) < minimum_confidence:
            rejection_counts["low_confidence"] += 1
            continue
        primary = str(target["primary_policy"])
        text = str(candidate["issue_text"])
        if _near_duplicate(text, prior_by_primary[primary], threshold=similarity_threshold):
            rejection_counts["near_duplicate"] += 1
            continue
        tokens = _normalized_tokens(text)
        prior_by_primary[primary].append((len(tokens), _shingles(tokens)))
        accepted_counts.update(intended)
        accepted_candidates.append(candidate)
        accepted_reviews.append({
            "candidate_id": candidate_id,
            "include": True,
            "policies": sorted(intended),
            "uncategorized_reason": None,
            "notes": "Exact agreement between the generation brief and blind MiniMax classification.",
            "annotation": {
                "kind": "model",
                "provenance": "synthetic_generation_plus_blind_model_agreement",
                "generator_version": GENERATOR_VERSION,
                "acceptance_version": ACCEPTANCE_VERSION,
                "reviewer_model": proposal.get("reviewer_model"),
                "reviewer_version": proposal.get("reviewer_version"),
                "prompt_version": proposal.get("prompt_version"),
                "confidence": float(confidence),
                "response_id": proposal.get("response_id"),
            },
        })
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    output_reviews.parent.mkdir(parents=True, exist_ok=True)
    output_candidates.write_text(_jsonl(accepted_candidates), encoding="utf-8")
    output_reviews.write_text(_jsonl(accepted_reviews), encoding="utf-8")
    report = {
        "version": ACCEPTANCE_VERSION,
        "generated_candidates": len(candidates),
        "accepted": len(accepted_candidates),
        "rejected": len(candidates) - len(accepted_candidates),
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "intended_labels": dict(sorted(intended_counts.items())),
        "accepted_labels": dict(sorted(accepted_counts.items())),
        "minimum_confidence": minimum_confidence,
        "similarity_threshold": similarity_threshold,
        "projects": len({row["source"]["repo"] for row in accepted_candidates}),
        "natural_languages": dict(sorted(Counter(
            str(row["source"]["natural_language"]) for row in accepted_candidates
        ).items())),
        "programming_languages": dict(sorted(Counter(
            str(row["source"]["programming_language"]) for row in accepted_candidates
        ).items())),
        "artifacts": {
            "candidates": str(output_candidates),
            "candidates_sha256": hashlib.sha256(output_candidates.read_bytes()).hexdigest(),
            "reviews": str(output_reviews),
            "reviews_sha256": hashlib.sha256(output_reviews.read_bytes()).hexdigest(),
        },
    }
    manifest_path = output_reviews.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("candidates", type=Path)
    generate.add_argument("targets", type=Path)
    generate.add_argument("--target", action="append", type=_parse_target)
    generate.add_argument("--seed", type=int, default=20260813)
    generate.add_argument("--model", default=DEFAULT_MODEL)
    generate.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    generate.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    generate.add_argument("--workers", type=int, default=8)
    generate.add_argument("--batch-size", type=int, default=8)
    generate.add_argument("--requests-per-second", type=float, default=4.0)
    generate.add_argument("--timeout", type=float, default=120.0)
    generate.add_argument("--max-completion-tokens", type=int, default=8192)
    generate.add_argument("--max-attempts", type=int, default=4)
    generate.add_argument("--single-label-only", action="store_true")

    accept = subparsers.add_parser("accept")
    accept.add_argument("candidates", type=Path)
    accept.add_argument("targets", type=Path)
    accept.add_argument("proposals", type=Path)
    accept.add_argument("output_candidates", type=Path)
    accept.add_argument("output_reviews", type=Path)
    accept.add_argument("--minimum-confidence", type=float, default=0.85)
    accept.add_argument("--similarity-threshold", type=float, default=0.82)
    args = parser.parse_args()

    if args.command == "generate":
        targets = dict(args.target) if args.target else DEFAULT_TARGETS
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            parser.error(f"missing API key in environment variable {args.api_key_env}")
        request = minimax_requester(
            endpoint=args.endpoint,
            api_key=api_key,
            model=args.model,
            timeout=args.timeout,
            max_completion_tokens=args.max_completion_tokens,
            requests_per_second=args.requests_per_second,
        )
        report = generate_corpus(
            args.candidates, args.targets, targets=targets, seed=args.seed,
            request=request, workers=args.workers, batch_size=args.batch_size,
            max_attempts=args.max_attempts,
            single_label_only=args.single_label_only,
        )
    else:
        report = accept_blind_agreements(
            args.candidates, args.targets, args.proposals,
            args.output_candidates, args.output_reviews,
            minimum_confidence=args.minimum_confidence,
            similarity_threshold=args.similarity_threshold,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ACCEPTANCE_VERSION",
    "GENERATOR_VERSION",
    "PROJECTS",
    "accept_blind_agreements",
    "build_specs",
    "generate_corpus",
    "messages_for_specs",
    "parse_generated",
]
