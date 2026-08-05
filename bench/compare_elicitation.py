#!/usr/bin/env python3
"""Compare isolated choice-only and self-report observations without merging them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from bench.model_behavior import Observation, load_observations


def compare_protocols(
    choice_rows: Iterable[Observation],
    self_report_rows: Iterable[Observation],
) -> dict[str, object]:
    """Pair protocol observations and retain every changed decision as raw evidence."""
    choices = _index(choice_rows, "choice_only")
    reports = _index(self_report_rows, "self_report")
    shared = sorted(set(choices) & set(reports))
    records: list[dict[str, object]] = []
    successful_pairs = agreements = 0
    for key in shared:
        choice = choices[key]
        report = reports[key]
        _validate_pair(choice, report)
        both_successful = choice.error is None and report.error is None
        agrees = both_successful and choice.answer == report.answer
        if both_successful:
            successful_pairs += 1
            agreements += int(agrees)
        records.append(
            {
                "probe_id": choice.probe_id,
                "condition": choice.condition,
                "repetition": choice.repetition,
                "choice_only_answer": choice.answer,
                "self_report_answer": report.answer,
                "answer_agrees": agrees if both_successful else None,
                "choice_only_response": choice.response_text,
                "self_report_response": report.response_text,
                "expressed_decision_criterion": report.decision_criterion,
                "stated_missing_context": report.missing_context,
                "verbal_confidence": report.confidence,
                "choice_only_error": choice.error,
                "self_report_error": report.error,
            }
        )
    return {
        "choice_only_n": len(choices),
        "self_report_n": len(reports),
        "paired_n": len(shared),
        "successful_paired_n": successful_pairs,
        "answer_agreement": agreements / successful_pairs if successful_pairs else None,
        "choice_only_unpaired": [_key_record(key) for key in sorted(set(choices) - set(reports))],
        "self_report_unpaired": [_key_record(key) for key in sorted(set(reports) - set(choices))],
        "records": records,
    }


def _index(
    rows: Iterable[Observation], expected_protocol: str
) -> dict[tuple[str, str, int], Observation]:
    indexed: dict[tuple[str, str, int], Observation] = {}
    for row in rows:
        if row.elicitation_protocol != expected_protocol:
            raise ValueError(
                f"expected only {expected_protocol} observations, got {row.elicitation_protocol}"
            )
        key = (row.condition, row.probe_id, row.repetition)
        if key in indexed:
            raise ValueError(f"duplicate observation key: {key}")
        indexed[key] = row
    return indexed


def _validate_pair(choice: Observation, report: Observation) -> None:
    if choice.model_identity != report.model_identity:
        raise ValueError(f"model identity mismatch for probe {choice.probe_id}")
    if choice.condition_sha256 != report.condition_sha256:
        raise ValueError(f"condition hash mismatch for probe {choice.probe_id}")
    if choice.utility_profile_sha256 != report.utility_profile_sha256:
        raise ValueError(f"utility profile mismatch for probe {choice.probe_id}")


def _key_record(key: tuple[str, str, int]) -> dict[str, object]:
    condition, probe_id, repetition = key
    return {"condition": condition, "probe_id": probe_id, "repetition": repetition}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("choice_only", type=Path)
    parser.add_argument("self_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = compare_protocols(
        load_observations(args.choice_only), load_observations(args.self_report)
    )
    report["choice_only_sha256"] = hashlib.sha256(args.choice_only.read_bytes()).hexdigest()
    report["self_report_sha256"] = hashlib.sha256(args.self_report.read_bytes()).hexdigest()
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
