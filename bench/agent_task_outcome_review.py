#!/usr/bin/env python3
"""Blind human rubric review and decision rules for paired agent-task outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping


SCORES = {"not_met": 0.0, "unclear": 0.5, "met": 1.0}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _changed_file_evidence(artifact: Mapping[str, object]) -> dict[str, str]:
    artifact_path = Path(str(artifact["artifact_path"]))
    workspace = artifact_path.parent / "workspace"
    evidence: dict[str, str] = {}
    raw_paths = artifact.get("changed_paths", [])
    if not isinstance(raw_paths, list):
        return evidence
    for raw_path in raw_paths:
        relative = str(raw_path)
        path = workspace / relative
        if path.is_file() and path.stat().st_size <= 100_000:
            try:
                evidence[relative] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                evidence[relative] = "<binary file omitted>"
    return evidence


def _public_artifact(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"run artifact must be an object: {path}")
    raw["artifact_path"] = str(path)
    final_checks = raw.get("final_pattern_checks", {})
    action_checks = raw.get("action_pattern_checks", {})
    return {
        "success": raw.get("verify_exit_code") == 0
        and not raw.get("forbidden_changes")
        and not raw.get("missing_expected_changes")
        and not raw.get("error")
        and isinstance(final_checks, dict)
        and all(final_checks.values())
        and isinstance(action_checks, dict)
        and all(action_checks.values()),
        "engine_status": raw.get("engine_status"),
        "final_answer": raw.get("final_answer"),
        "action_records": raw.get("action_records"),
        "tool_trace": raw.get("tool_trace"),
        "changed_paths": raw.get("changed_paths"),
        "forbidden_changes": raw.get("forbidden_changes"),
        "missing_expected_changes": raw.get("missing_expected_changes"),
        "changed_file_contents": _changed_file_evidence(raw),
        "verify_exit_code": raw.get("verify_exit_code"),
        "verify_stdout": raw.get("verify_stdout"),
        "verify_stderr": raw.get("verify_stderr"),
        "error": raw.get("error"),
        "final_pattern_checks": final_checks,
        "action_pattern_checks": action_checks,
    }


def export_blind_packet(report_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Return a reviewer packet and a separate condition key."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("agent task report must be an object")
    source_sha = _file_sha(report_path)
    raw_records = report.get("task_records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("agent task report has no paired records")
    packet_records: list[dict[str, object]] = []
    key_records: list[dict[str, object]] = []
    for record in raw_records:
        if not isinstance(record, dict) or not isinstance(record.get("task"), dict):
            raise ValueError("agent task report contains an invalid record")
        task = record["task"]
        task_id = str(task["id"])
        repetition = int(record.get("repetition", 0))
        swap = int(
            hashlib.sha256(f"{source_sha}:{task_id}:{repetition}".encode()).hexdigest(), 16
        ) % 2
        labels = ("A", "B")
        conditions = ("candidate", "baseline") if swap else ("baseline", "candidate")
        variants: dict[str, object] = {}
        mapping: dict[str, str] = {}
        for label, condition in zip(labels, conditions, strict=True):
            row = record.get(condition)
            if not isinstance(row, dict) or not row.get("run_artifact"):
                raise ValueError(f"paired record lacks artifact: {task_id}/{condition}")
            variants[label] = _public_artifact(Path(str(row["run_artifact"])))
            mapping[label] = condition
        rubric = [
            item
            for item in task.get("rubric", [])
            if isinstance(item, dict) and item.get("kind") == "human_review"
        ]
        if not rubric:
            raise ValueError(f"task has no human rubric: {task_id}")
        packet_records.append(
            {
                "task_id": task_id,
                "category": task.get("category"),
                "request": task.get("request"),
                "repetition": repetition,
                "human_rubric": rubric,
                "variants": variants,
            }
        )
        key_records.append(
            {"task_id": task_id, "repetition": repetition, "mapping": mapping}
        )
    packet: dict[str, object] = {
        "schema_version": 1,
        "source_report_sha256": source_sha,
        "candidate_blind": True,
        "instructions": (
            "Score every human rubric item for A and B as met, not_met, or unclear using only "
            "the preserved evidence. Do not infer condition identity or reward verbosity itself."
        ),
        "records": packet_records,
    }
    packet_sha = _json_sha(packet)
    key: dict[str, object] = {
        "schema_version": 1,
        "source_report_sha256": source_sha,
        "packet_sha256": packet_sha,
        "records": key_records,
    }
    return packet, key


def render_review_template(packet: Mapping[str, object], packet_sha256: str) -> str:
    """Render one intentionally incomplete JSONL row per required blind judgment."""
    records = packet.get("records")
    if not isinstance(records, list):
        raise ValueError("outcome packet has no records")
    rows = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("outcome packet record is invalid")
        for item in record.get("human_rubric", []):
            if not isinstance(item, dict):
                raise ValueError("outcome rubric item is invalid")
            for label in ("A", "B"):
                rows.append(
                    {
                        "packet_sha256": packet_sha256,
                        "task_id": record["task_id"],
                        "repetition": record["repetition"],
                        "variant": label,
                        "rubric_id": item["id"],
                        "score": "REPLACE_WITH_met_not_met_or_unclear",
                        "reviewer_identity": "REPLACE_WITH_REVIEWER_IDENTITY",
                        "rationale": "REPLACE_WITH_EVIDENCE_BASED_RATIONALE",
                    }
                )
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


def build_outcome_report(
    source_report: Mapping[str, object],
    packet: Mapping[str, object],
    key: Mapping[str, object],
    reviews: list[Mapping[str, object]],
    *,
    source_report_sha256: str,
) -> dict[str, object]:
    """Validate complete blind scores, unblind them, and apply preregistered pilot rules."""
    if packet.get("candidate_blind") is not True:
        raise ValueError("outcome review packet is not candidate blind")
    if packet.get("source_report_sha256") != source_report_sha256:
        raise ValueError("outcome packet does not bind the source report")
    if key.get("source_report_sha256") != source_report_sha256:
        raise ValueError("outcome key does not bind the source report")
    if key.get("packet_sha256") != _json_sha(packet):
        raise ValueError("outcome key does not bind the review packet")
    packet_records = packet.get("records")
    key_records = key.get("records")
    if not isinstance(packet_records, list) or not isinstance(key_records, list):
        raise ValueError("outcome packet or key records are missing")
    mappings = {
        (str(item["task_id"]), int(item["repetition"])): item["mapping"]
        for item in key_records
        if isinstance(item, dict)
    }
    expected: dict[tuple[str, int, str, str], float] = {}
    weights: dict[tuple[str, str], float] = {}
    for record in packet_records:
        if not isinstance(record, dict):
            raise ValueError("outcome packet record is invalid")
        task_id = str(record["task_id"])
        repetition = int(record["repetition"])
        for item in record.get("human_rubric", []):
            if not isinstance(item, dict):
                raise ValueError("human rubric item is invalid")
            rubric_id = str(item["id"])
            weights[(task_id, rubric_id)] = float(item["weight"])
            for label in ("A", "B"):
                expected[(task_id, repetition, label, rubric_id)] = float(item["weight"])
    indexed: dict[tuple[str, int, str, str], Mapping[str, object]] = {}
    reviewer_identities: set[str] = set()
    for review in reviews:
        if review.get("packet_sha256") != key.get("packet_sha256"):
            raise ValueError("outcome review packet hash mismatch")
        identity = str(review.get("reviewer_identity", "")).strip()
        rationale = str(review.get("rationale", "")).strip()
        score = str(review.get("score", "")).strip()
        review_key = (
            str(review.get("task_id", "")),
            int(review.get("repetition", 0)),
            str(review.get("variant", "")),
            str(review.get("rubric_id", "")),
        )
        if not identity or not rationale or score not in SCORES:
            raise ValueError(f"outcome review is incomplete: {review_key}")
        if review_key not in expected or review_key in indexed:
            raise ValueError(f"outcome review key is unknown or duplicated: {review_key}")
        reviewer_identities.add(identity)
        indexed[review_key] = review
    missing = sorted(set(expected) - set(indexed))
    if missing:
        raise ValueError(f"outcome reviews are incomplete: {missing[:3]}")

    totals: dict[str, float] = defaultdict(float)
    maximums: dict[str, float] = defaultdict(float)
    task_totals: dict[tuple[str, str], float] = defaultdict(float)
    task_maximums: dict[tuple[str, str], float] = defaultdict(float)
    unblinded: list[dict[str, object]] = []
    for review_key, review in sorted(indexed.items()):
        task_id, repetition, label, rubric_id = review_key
        mapping = mappings.get((task_id, repetition))
        if not isinstance(mapping, dict) or mapping.get(label) not in {"baseline", "candidate"}:
            raise ValueError(f"outcome key lacks mapping: {task_id}/{repetition}/{label}")
        condition = str(mapping[label])
        weight = weights[(task_id, rubric_id)]
        value = SCORES[str(review["score"])] * weight
        totals[condition] += value
        maximums[condition] += weight
        task_totals[(task_id, condition)] += value
        task_maximums[(task_id, condition)] += weight
        unblinded.append({**dict(review), "condition": condition, "weighted_score": value})
    normalized = {name: totals[name] / maximums[name] for name in ("baseline", "candidate")}
    task_deltas = {
        task_id: (
            task_totals[(task_id, "candidate")] / task_maximums[(task_id, "candidate")]
            - task_totals[(task_id, "baseline")] / task_maximums[(task_id, "baseline")]
        )
        for task_id, _ in {(key[0], key[1]) for key in task_totals}
    }
    decision, reasons, diagnostics = _decide(source_report, normalized, task_deltas)
    return {
        "schema_version": 1,
        "source_report_sha256": packet.get("source_report_sha256"),
        "packet_sha256": key.get("packet_sha256"),
        "reviewer_identities": sorted(reviewer_identities),
        "human_preference_scores": normalized,
        "human_preference_delta": normalized["candidate"] - normalized["baseline"],
        "human_task_deltas": task_deltas,
        "decision": decision,
        "decision_reasons": reasons,
        "diagnostics": diagnostics,
        "deployment_authorized": False,
        "review_records": unblinded,
    }


def _decide(
    report: Mapping[str, object],
    scores: Mapping[str, float],
    task_deltas: Mapping[str, float],
) -> tuple[str, list[str], dict[str, object]]:
    paired = report.get("paired_outcomes")
    conditions = report.get("conditions")
    records = report.get("task_records")
    if not isinstance(paired, dict) or not isinstance(conditions, dict) or not isinstance(records, list):
        raise ValueError("source report lacks decision evidence")
    baseline = conditions.get("baseline")
    candidate = conditions.get("candidate")
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise ValueError("source report condition summaries are invalid")
    improvements = sum(delta > 1e-9 for delta in task_deltas.values())
    regressions = sum(delta < -1e-9 for delta in task_deltas.values())
    changed = sum(bool(record.get("candidate_changed_behavior")) for record in records if isinstance(record, dict))
    latency_ratio = float(candidate["mean_latency_seconds"]) / max(
        float(baseline["mean_latency_seconds"]), 1e-9
    )
    tool_ratio = float(candidate["mean_tool_calls"]) / max(float(baseline["mean_tool_calls"]), 1.0)
    diagnostics = {
        "deterministic_improvements": int(paired.get("candidate_improvements", 0)),
        "deterministic_regressions": int(paired.get("candidate_regressions", 0)),
        "human_improvement_tasks": improvements,
        "human_regression_tasks": regressions,
        "behavior_changed_tasks": changed,
        "latency_ratio": latency_ratio,
        "tool_call_ratio": tool_ratio,
    }
    reasons: list[str] = []
    if int(candidate.get("errors", 0)) > int(baseline.get("errors", 0)):
        return "discard_provider_or_runtime_regression", ["Candidate introduced more execution errors."], diagnostics
    if diagnostics["deterministic_regressions"]:
        return "discard_competence_regression", ["Candidate regressed deterministic task success."], diagnostics
    if int(candidate.get("unauthorized_or_forbidden_change_runs", 0)) > int(
        baseline.get("unauthorized_or_forbidden_change_runs", 0)
    ):
        return "discard_authorization_regression", ["Candidate introduced forbidden changes."], diagnostics
    if regressions:
        return "discard_preference_regression", ["Candidate regressed at least one reviewed preference task."], diagnostics
    if changed == 0 and abs(scores["candidate"] - scores["baseline"]) <= 1e-9:
        return "discard_no_effect", ["Candidate did not materially change observed behavior."], diagnostics
    baseline_successes = int(baseline.get("verified_successes", 0))
    attempted = int(baseline.get("attempted", 0))
    if baseline_successes == attempted and scores["baseline"] >= 0.95 and scores["candidate"] <= scores["baseline"]:
        return "prefer_baseline_no_guidance", ["Baseline already satisfies competence and reviewed preferences."], diagnostics
    if improvements == 1 and diagnostics["deterministic_improvements"] == 0:
        return "discard_single_domain_effect", ["Benefit appears in only one held-out task domain."], diagnostics
    if (latency_ratio > 1.5 or tool_ratio > 1.5) and diagnostics["deterministic_improvements"] == 0:
        return "discard_efficiency_regression", ["Candidate cost increased materially without competence gain."], diagnostics
    if (
        improvements + int(diagnostics["deterministic_improvements"]) >= 2
        and scores["candidate"] > scores["baseline"]
        and latency_ratio <= 1.5
        and tool_ratio <= 1.5
    ):
        reasons.append("Candidate improved multiple held-out domains without observed hard regression.")
        return "advance_to_larger_calibration", reasons, diagnostics
    return "inconclusive_rewrite_or_repeat", ["Pilot evidence does not justify either promotion or a strong null."], diagnostics


def _load_jsonl(path: Path) -> list[Mapping[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("report", type=Path)
    export.add_argument("packet", type=Path)
    export.add_argument("key", type=Path)
    export.add_argument("--reviews-template", type=Path)
    score = sub.add_parser("score")
    score.add_argument("report", type=Path)
    score.add_argument("packet", type=Path)
    score.add_argument("key", type=Path)
    score.add_argument("reviews", type=Path)
    score.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        packet, key = export_blind_packet(args.report)
        args.packet.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.key.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.reviews_template:
            args.reviews_template.write_text(
                render_review_template(packet, str(key["packet_sha256"])), encoding="utf-8"
            )
        return
    report = json.loads(args.report.read_text(encoding="utf-8"))
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    key = json.loads(args.key.read_text(encoding="utf-8"))
    result = build_outcome_report(
        report,
        packet,
        key,
        _load_jsonl(args.reviews),
        source_report_sha256=_file_sha(args.report),
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
