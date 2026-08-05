#!/usr/bin/env python3
"""Materialize curated preference-family seeds into controlled draft probes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from bench.generate_probe_drafts import split_for_group
from bench.model_behavior import Probe, load_probes

_VARIANT_ORDERS = ((0, 1, 2, 3), (2, 0, 3, 1))


def materialize(value: object) -> list[Probe]:
    """Validate curated seeds and create two option-reordered variants per family."""
    if not isinstance(value, dict) or not isinstance(value.get("families"), list):
        raise ValueError("seed file needs a families array")
    generator = str(value.get("generator", "")).strip()
    if not generator:
        raise ValueError("seed file needs an immutable generator identity")
    probes: list[Probe] = []
    seen_families: set[str] = set()
    for raw in value["families"]:
        if not isinstance(raw, dict):
            raise ValueError("every family seed must be an object")
        family = str(raw.get("family", "")).strip()
        category = str(raw.get("category", "")).strip()
        if not family or not category or family in seen_families:
            raise ValueError("family and category are required and family must be unique")
        seen_families.add(family)
        scenarios = raw.get("scenarios")
        requests = raw.get("user_requests")
        actions = raw.get("actions")
        if not isinstance(scenarios, list) or len(scenarios) != 2:
            raise ValueError(f"{family}: exactly two scenario variants are required")
        if not isinstance(requests, list) or len(requests) != 2:
            raise ValueError(f"{family}: exactly two user-request variants are required")
        if not isinstance(actions, list) or len(actions) != 4:
            raise ValueError(f"{family}: exactly four actions are required")
        action_ids = [str(action.get("id", "")) for action in actions if isinstance(action, dict)]
        if len(action_ids) != 4 or len(set(action_ids)) != 4 or not all(action_ids):
            raise ValueError(f"{family}: action ids must be four unique non-empty strings")
        for variant_index, order in enumerate(_VARIANT_ORDERS, 1):
            labels = "ABCD"
            ordered_actions = [actions[index] for index in order]
            choices = {
                label: str(action["text"]).strip()
                for label, action in zip(labels, ordered_actions, strict=True)
            }
            effects = {
                label: action["effects"]
                for label, action in zip(labels, ordered_actions, strict=True)
            }
            rationales = {
                label: str(action["rationale"]).strip()
                for label, action in zip(labels, ordered_actions, strict=True)
            }
            probe = Probe.from_dict(
                {
                    "id": f"{family}-v{variant_index}",
                    "category": category,
                    "scenario": str(scenarios[variant_index - 1]),
                    "user_request": str(requests[variant_index - 1]),
                    "choices": choices,
                    "evaluation_mode": "preference",
                    "choice_effects": effects,
                    "group": family,
                    "tags": [str(tag) for tag in raw.get("tags", [])],
                    "split": split_for_group(family),
                    "review_status": "draft",
                    "gold_rationale": str(raw.get("gold_rationale", "")),
                    "reviewer": "",
                    "generator": generator,
                    "analysis": {
                        "hypothesis": str(raw.get("hypothesis", "")),
                        "decisive_information": str(raw.get("decisive_information", "")),
                        "variant_axis": str(raw.get("variant_axis", "")),
                        "failure_signal": str(raw.get("failure_signal", "")),
                        "calibration_use": str(raw.get("calibration_use", "")),
                        "preference_tradeoff": str(raw.get("preference_tradeoff", "")),
                        "choice_rationales": rationales,
                    },
                }
            )
            probes.append(probe)
    return probes


def append_materialized(path: Path, probes: list[Probe]) -> None:
    """Append validated probes without replacing any existing authored evidence."""
    existing = load_probes(path) if path.exists() else {}
    overlap = sorted(existing.keys() & {probe.id for probe in probes})
    if overlap:
        raise ValueError(f"probe ids already exist: {', '.join(overlap)}")
    with path.open("a", encoding="utf-8") as stream:
        for probe in probes:
            stream.write(json.dumps(asdict(probe), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    append_materialized(
        args.output,
        materialize(json.loads(args.seeds.read_text(encoding="utf-8"))),
    )


if __name__ == "__main__":
    main()
