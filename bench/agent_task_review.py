#!/usr/bin/env python3
"""Blind-review agent task fixtures and materialize an approved immutable dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from bench.agent_task_eval import file_sha256, load_tasks


VERDICTS = frozenset({"approve", "revise", "reject"})
_IGNORED_FIXTURE_PARTS = frozenset(
    {".git", ".infinidev", ".pytest_cache", "__pycache__", ".venv"}
)


def fixture_sha256(path: Path) -> str:
    """Hash relative paths and bytes for every non-runtime fixture file."""
    digest = hashlib.sha256()
    files = sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file()
            and not (_IGNORED_FIXTURE_PARTS & set(item.relative_to(path).parts))
        ),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    if not files:
        raise ValueError(f"fixture is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = item.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _fixture_contents(path: Path) -> dict[str, str]:
    contents: dict[str, str] = {}
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path)
        if not item.is_file() or _IGNORED_FIXTURE_PARTS & set(relative.parts):
            continue
        if item.stat().st_size > 100_000:
            contents[relative.as_posix()] = "<file omitted: larger than 100000 bytes>"
            continue
        try:
            contents[relative.as_posix()] = item.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            contents[relative.as_posix()] = "<binary file omitted>"
    return contents


def export_packet(
    tasks_path: Path,
    fixture_root: Path,
    preflight: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Export tasks and fixture hashes without exposing any candidate prompt."""
    tasks = load_tasks(tasks_path)
    preflight_records: dict[str, Mapping[str, object]] = {}
    if preflight is not None:
        raw_records = preflight.get("records")
        if preflight.get("all_passed") is not True or not isinstance(raw_records, list):
            raise ValueError("task review preflight is missing or failed")
        preflight_records = {
            str(record["task_id"]): record
            for record in raw_records
            if isinstance(record, dict) and record.get("task_id")
        }
        if set(preflight_records) != {task.id for task in tasks}:
            raise ValueError("task review preflight does not cover the frozen dataset")
        for task in tasks:
            expected_fixture_sha = fixture_sha256(fixture_root / task.repository_fixture)
            if preflight_records[task.id].get("fixture_sha256") != expected_fixture_sha:
                raise ValueError(f"task review preflight fixture hash mismatch: {task.id}")
    return {
        "schema_version": 1,
        "dataset_sha256": file_sha256(tasks_path),
        "candidate_blind": True,
        "review_contract": [
            "Confirm the task represents its named category and does not favor a model or provider.",
            "Run or inspect the pristine and reference-solution verifier preflight.",
            "Confirm deterministic checks cannot be passed by weakening tests or editing forbidden files.",
            "Confirm every human rubric item is observable from preserved artifacts and matches quality-and-control.",
            "Request revision for ambiguous success criteria, leaked candidate behavior, or an unrealistic fixture.",
        ],
        "tasks": [
            {
                "task": json.loads(line),
                "fixture_sha256": fixture_sha256(fixture_root / task.repository_fixture),
                "fixture_files": _fixture_contents(fixture_root / task.repository_fixture),
                "preflight": dict(preflight_records[task.id]) if preflight_records else None,
            }
            for task, line in zip(
                tasks,
                (line for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()),
                strict=True,
            )
        ],
    }


def render_packet_markdown(packet: Mapping[str, object]) -> str:
    """Render a self-contained candidate-blind dossier for a human reviewer."""
    raw_tasks = packet.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("review packet has no tasks")
    lines = [
        "# Candidate-blind agent task review",
        "",
        f"Dataset SHA-256: `{packet.get('dataset_sha256')}`",
        "",
        "This dossier intentionally contains no baseline or candidate prompt guidance.",
        "",
        "## Review contract",
        "",
    ]
    for item in packet.get("review_contract", []):
        lines.append(f"- {item}")
    for wrapped in raw_tasks:
        if not isinstance(wrapped, dict) or not isinstance(wrapped.get("task"), dict):
            raise ValueError("review packet task is invalid")
        task = wrapped["task"]
        lines.extend(
            [
                "",
                f"## `{task.get('id')}` — {task.get('category')}",
                "",
                f"Fixture SHA-256: `{wrapped.get('fixture_sha256')}`",
                "",
                "### User request",
                "",
                str(task.get("request")),
                "",
                "### Execution boundaries",
                "",
                f"- Verifier: `{task.get('verify_command')}`",
                f"- Expected changes: `{json.dumps(task.get('expected_changed_paths'))}`",
                f"- Forbidden changes: `{json.dumps(task.get('forbidden_changed_paths'))}`",
                f"- Required final patterns: `{json.dumps(task.get('required_final_patterns'))}`",
                f"- Required action patterns: `{json.dumps(task.get('required_action_patterns'))}`",
                "",
                "### Rubric",
                "",
                "| ID | Kind | Weight | Observable | Evidence |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for rubric in task.get("rubric", []):
            lines.append(
                f"| `{rubric.get('id')}` | {rubric.get('kind')} | {rubric.get('weight')} | "
                f"{rubric.get('description')} | {rubric.get('evidence_source')} |"
            )
        lines.extend(["", "### Fixture files", ""])
        fixture_files = wrapped.get("fixture_files")
        if not isinstance(fixture_files, dict):
            raise ValueError("review packet fixture files are missing")
        for path, content in fixture_files.items():
            lines.extend([f"#### `{path}`", "", "````text", str(content).rstrip(), "````", ""])
        preflight = wrapped.get("preflight")
        if isinstance(preflight, dict):
            lines.extend(
                [
                    "### Verifier controls",
                    "",
                    f"- Passed: `{preflight.get('passed')}`",
                    f"- Pristine exit: `{preflight.get('pristine_verify_exit_code')}`",
                    f"- Reference exit: `{preflight.get('reference_verify_exit_code')}`",
                    f"- Reference changes: `{json.dumps(preflight.get('reference_changed_paths'))}`",
                    f"- Forbidden reference changes: "
                    f"`{json.dumps(preflight.get('forbidden_reference_changes'))}`",
                    f"- Missing expected reference changes: "
                    f"`{json.dumps(preflight.get('missing_expected_reference_changes'))}`",
                ]
            )
        lines.extend(
            [
                "",
                "### Reviewer decision",
                "",
                "- Verdict: approve / revise / reject",
                "- Rubric valid: yes / no",
                "- Verifier valid: yes / no",
                "- Held-out valid: yes / no",
                "- Provider-neutral: yes / no",
                "- Evidence-based rationale:",
            ]
        )
    return "\n".join(lines) + "\n"


def render_review_template(packet: Mapping[str, object]) -> str:
    """Render one intentionally incomplete JSONL decision per frozen task."""
    tasks = packet.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("review packet has no tasks")
    rows = []
    for item in tasks:
        if not isinstance(item, dict) or not isinstance(item.get("task"), dict):
            raise ValueError("review packet task is invalid")
        rows.append(
            {
                "dataset_sha256": packet["dataset_sha256"],
                "task_id": item["task"]["id"],
                "reviewer_identity": "REPLACE_WITH_REVIEWER_IDENTITY",
                "verdict": "REPLACE_WITH_approve_revise_or_reject",
                "rationale": "REPLACE_WITH_EVIDENCE_BASED_RATIONALE",
                "rubric_valid": False,
                "verifier_valid": False,
                "held_out_valid": False,
                "provider_neutral": False,
            }
        )
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )


def build_review_report(
    packet: Mapping[str, object], reviews: list[Mapping[str, object]]
) -> dict[str, object]:
    """Require one complete documented decision for every frozen task."""
    raw_tasks = packet.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("review packet has no tasks")
    task_ids = {
        str(item["task"]["id"])
        for item in raw_tasks
        if isinstance(item, dict) and isinstance(item.get("task"), dict)
    }
    indexed: dict[str, Mapping[str, object]] = {}
    for review in reviews:
        if review.get("dataset_sha256") != packet.get("dataset_sha256"):
            raise ValueError("agent task review dataset hash mismatch")
        task_id = str(review.get("task_id", "")).strip()
        reviewer = str(review.get("reviewer_identity", "")).strip()
        verdict = str(review.get("verdict", "")).strip()
        rationale = str(review.get("rationale", "")).strip()
        if task_id not in task_ids or task_id in indexed:
            raise ValueError(f"review task id is unknown or duplicated: {task_id}")
        if not reviewer or not rationale or verdict not in VERDICTS:
            raise ValueError(f"review is missing identity, verdict, or rationale: {task_id}")
        for field in ("rubric_valid", "verifier_valid", "held_out_valid", "provider_neutral"):
            if not isinstance(review.get(field), bool):
                raise ValueError(f"review field must be boolean: {task_id}/{field}")
        indexed[task_id] = review
    missing = sorted(task_ids - set(indexed))
    if missing:
        raise ValueError(f"agent task reviews are incomplete: {missing}")
    records = []
    for task_id in sorted(task_ids):
        review = indexed[task_id]
        gates = all(
            bool(review[field])
            for field in ("rubric_valid", "verifier_valid", "held_out_valid", "provider_neutral")
        )
        approved = review.get("verdict") == "approve" and gates
        records.append({**dict(review), "approved": approved})
    return {
        "schema_version": 1,
        "dataset_sha256": packet.get("dataset_sha256"),
        "candidate_blind": packet.get("candidate_blind") is True,
        "task_count": len(task_ids),
        "approved_count": sum(bool(record["approved"]) for record in records),
        "all_approved": all(bool(record["approved"]) for record in records),
        "records": records,
    }


def apply_review_report(
    tasks_path: Path, report: Mapping[str, object]
) -> tuple[str, str]:
    """Write nothing; return source and approved JSONL for an explicit caller to persist."""
    if report.get("dataset_sha256") != file_sha256(tasks_path):
        raise ValueError("agent task report does not bind the current dataset")
    if report.get("candidate_blind") is not True or report.get("all_approved") is not True:
        raise ValueError("agent task report is not blind and fully approved")
    raw_records = report.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("agent task report records are missing")
    reviewers = {
        str(item["task_id"]): str(item["reviewer_identity"])
        for item in raw_records
        if isinstance(item, dict) and item.get("approved") is True
    }
    source_lines = [
        line for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    output_lines = []
    for line in source_lines:
        value = json.loads(line)
        task_id = str(value["id"])
        if task_id not in reviewers:
            raise ValueError(f"approved review is missing for task: {task_id}")
        value["review_status"] = "approved"
        value["reviewer"] = reviewers[task_id]
        output_lines.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(source_lines) + "\n", "\n".join(output_lines) + "\n"


def _load_reviews(path: Path) -> list[Mapping[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("tasks", type=Path)
    export.add_argument("fixture_root", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--reviews-template", type=Path)
    export.add_argument("--preflight", type=Path)
    export.add_argument("--markdown", type=Path)
    report = subparsers.add_parser("report")
    report.add_argument("packet", type=Path)
    report.add_argument("reviews", type=Path)
    report.add_argument("output", type=Path)
    apply = subparsers.add_parser("apply")
    apply.add_argument("tasks", type=Path)
    apply.add_argument("report", type=Path)
    apply.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        preflight = (
            json.loads(args.preflight.read_text(encoding="utf-8")) if args.preflight else None
        )
        value = export_packet(args.tasks, args.fixture_root, preflight)
    elif args.command == "report":
        value = build_review_report(
            json.loads(args.packet.read_text(encoding="utf-8")), _load_reviews(args.reviews)
        )
    else:
        _, rendered = apply_review_report(
            args.tasks, json.loads(args.report.read_text(encoding="utf-8"))
        )
        args.output.write_text(rendered, encoding="utf-8")
        return
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.command == "export" and args.reviews_template:
        args.reviews_template.write_text(render_review_template(value), encoding="utf-8")
    if args.command == "export" and args.markdown:
        args.markdown.write_text(render_packet_markdown(value), encoding="utf-8")


if __name__ == "__main__":
    main()
