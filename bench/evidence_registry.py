#!/usr/bin/env python3
"""Audit methodological claims and their bounded evidence sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


STATUSES = {"supported", "partially_supported", "internal_evidence", "hypothesis"}
REQUIRED_DECISIONS = {
    "retain_raw_events",
    "multi_category_multi_metric",
    "isolated_fresh_calls",
    "counterbalance_mcq_options",
    "paired_perturbation_families",
    "repeat_behavioral_observations",
    "separate_self_report_protocol",
    "condition_on_user_preferences",
    "automatic_prompt_search",
    "heldout_regression_gates",
    "minimal_guidance_fragments",
}


def audit_registry(registry: Mapping[str, object], *, root: Path) -> dict[str, object]:
    """Reject uncited support claims, missing local evidence, and scope-free citations."""
    errors: list[str] = []
    raw_sources = registry.get("sources")
    raw_decisions = registry.get("decisions")
    if registry.get("schema_version") != 1:
        errors.append("unsupported schema_version")
    if not isinstance(raw_sources, list) or not isinstance(raw_decisions, list):
        return {"passes": False, "errors": errors + ["sources and decisions must be arrays"]}

    sources: dict[str, Mapping[str, object]] = {}
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            errors.append(f"source[{index}] must be an object")
            continue
        source_id = str(raw_source.get("id", ""))
        if not source_id or source_id in sources:
            errors.append(f"source[{index}] has missing or duplicate id")
            continue
        sources[source_id] = raw_source
        kind = raw_source.get("kind")
        if kind == "paper":
            if not str(raw_source.get("url", "")).startswith("https://"):
                errors.append(f"{source_id} paper needs an https URL")
        elif kind == "internal_experiment":
            path = root / str(raw_source.get("path", ""))
            if not path.is_file():
                errors.append(f"{source_id} internal evidence is missing: {path}")
        else:
            errors.append(f"{source_id} has unsupported kind {kind!r}")
        for field in ("title", "primary_finding_used", "limits"):
            if not str(raw_source.get(field, "")).strip():
                errors.append(f"{source_id} needs {field}")

    decision_ids: set[str] = set()
    status_counts: dict[str, int] = {status: 0 for status in sorted(STATUSES)}
    missing_implementation: list[str] = []
    for index, raw_decision in enumerate(raw_decisions):
        if not isinstance(raw_decision, dict):
            errors.append(f"decision[{index}] must be an object")
            continue
        decision_id = str(raw_decision.get("id", ""))
        if not decision_id or decision_id in decision_ids:
            errors.append(f"decision[{index}] has missing or duplicate id")
            continue
        decision_ids.add(decision_id)
        status = str(raw_decision.get("status", ""))
        if status not in STATUSES:
            errors.append(f"{decision_id} has unsupported status {status!r}")
            continue
        status_counts[status] += 1
        cited = raw_decision.get("sources")
        if not isinstance(cited, list):
            errors.append(f"{decision_id} sources must be an array")
            cited = []
        unknown = [source_id for source_id in cited if source_id not in sources]
        if unknown:
            errors.append(f"{decision_id} cites unknown sources: {unknown}")
        if status != "hypothesis" and not cited:
            errors.append(f"{decision_id} claims {status} without evidence")
        if status == "hypothesis" and cited:
            errors.append(f"{decision_id} hypothesis must not masquerade as sourced support")
        for field in ("claim", "supported_scope", "does_not_establish"):
            if not str(raw_decision.get(field, "")).strip():
                errors.append(f"{decision_id} needs {field}")
        controls = raw_decision.get("controls")
        if not isinstance(controls, list) or not controls:
            errors.append(f"{decision_id} needs at least one control")
        if raw_decision.get("implementation_status") == "missing":
            missing_implementation.append(decision_id)

    missing_decisions = sorted(REQUIRED_DECISIONS - decision_ids)
    if missing_decisions:
        errors.append(f"required decisions missing: {missing_decisions}")
    return {
        "passes": not errors,
        "errors": errors,
        "source_count": len(sources),
        "decision_count": len(decision_ids),
        "status_counts": status_counts,
        "missing_implementation": sorted(missing_implementation),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = audit_registry(
        json.loads(args.registry.read_text(encoding="utf-8")), root=args.root
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passes"] else 1)


if __name__ == "__main__":
    main()
