#!/usr/bin/env python3
"""Audit coverage and leakage in a behavioral probe dataset."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from bench.model_behavior import UTILITY_AXES, Probe, load_probes


def audit_dataset(
    probes: dict[str, Probe],
    targets: dict[str, int],
    preference_axis_targets: dict[str, int] | None = None,
    preference_category_targets: dict[str, int] | None = None,
) -> dict[str, object]:
    """Return category coverage and split-family leakage violations."""
    approved = [probe for probe in probes.values() if probe.review_status == "approved"]
    counts = Counter(probe.category for probe in approved)
    all_counts = Counter(probe.category for probe in probes.values())
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_splits: dict[str, set[str]] = defaultdict(set)
    normalized_prompts: dict[str, list[str]] = defaultdict(list)
    approval_issues: list[str] = []
    analysis_issues: dict[str, list[str]] = {}
    for probe in probes.values():
        if probe.review_status == "approved":
            split_counts[probe.category][probe.split] += 1
        if probe.group:
            group_splits[probe.group].add(probe.split)
        normalized_prompts[_normalized_question(probe)].append(probe.id)
        if probe.review_status == "approved" and (
            not probe.gold_rationale.strip() or not probe.reviewer.strip()
        ):
            approval_issues.append(probe.id)
        if probe.review_status != "rejected":
            missing_analysis = _missing_analysis(probe)
            if missing_analysis:
                analysis_issues[probe.id] = missing_analysis

    missing = {
        category: target - counts[category]
        for category, target in targets.items()
        if counts[category] < target
    }
    unknown = sorted(set(all_counts) - set(targets))
    leakage = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    duplicates = sorted(
        sorted(ids) for ids in normalized_prompts.values() if len(ids) > 1
    )
    split_shortfalls: dict[str, dict[str, int]] = {}
    answer_balance_issues: list[str] = []
    choice_count_issues: list[str] = []
    approved_group_counts = Counter(
        probe.group for probe in approved if probe.group
    )
    group_size_issues: list[str] = []
    preference_axis_counts = Counter(
        axis
        for probe in probes.values()
        if probe.evaluation_mode == "preference"
        for axis in {
            axis
            for effects in probe.choice_effects.values()
            for axis, effect in effects.items()
            if effect
        }
    )
    approved_preference_axis_counts = Counter(
        axis
        for probe in approved
        if probe.evaluation_mode == "preference"
        for axis in {
            axis
            for effects in probe.choice_effects.values()
            for axis, effect in effects.items()
            if effect
        }
    )
    axis_targets = preference_axis_targets or {}
    category_preference_targets = preference_category_targets or {}
    preference_axis_shortfalls = {
        axis: target - approved_preference_axis_counts[axis]
        for axis, target in axis_targets.items()
        if approved_preference_axis_counts[axis] < target
    }
    preference_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for probe in probes.values():
        if probe.evaluation_mode != "preference":
            continue
        preference_by_category[probe.category].update(
            {
                axis
                for effects in probe.choice_effects.values()
                for axis, effect in effects.items()
                if effect
            }
        )
    authored_preference_counts = Counter(
        probe.category
        for probe in probes.values()
        if probe.evaluation_mode == "preference"
    )
    approved_preference_counts = Counter(
        probe.category
        for probe in approved
        if probe.evaluation_mode == "preference"
    )
    authored_preference_category_shortfalls = {
        category: target - authored_preference_counts[category]
        for category, target in category_preference_targets.items()
        if authored_preference_counts[category] < target
    }
    approved_preference_category_shortfalls = {
        category: target - approved_preference_counts[category]
        for category, target in category_preference_targets.items()
        if approved_preference_counts[category] < target
    }
    for category, target in targets.items():
        if target < 10:
            continue
        minimums = {
            "calibration": math.ceil(target * 0.7),
            "validation": math.ceil(target * 0.2),
        }
        short = {
            split: minimum - split_counts[category][split]
            for split, minimum in minimums.items()
            if split_counts[category][split] < minimum
        }
        if short:
            split_shortfalls[category] = short
        category_probes = [probe for probe in approved if probe.category == category]
        choice_count_issues.extend(
            probe.id for probe in category_probes if len(probe.choices) != 4
        )
        normative_probes = [
            probe for probe in category_probes
            if probe.evaluation_mode == "normative"
        ]
        if len(normative_probes) >= target:
            answers = Counter(probe.answer for probe in normative_probes)
            if max(answers.values(), default=0) / len(normative_probes) > 0.4:
                answer_balance_issues.append(category)
    for group, count in approved_group_counts.items():
        group_categories = {
            probe.category for probe in approved if probe.group == group
        }
        if any(targets.get(category, 0) >= 10 for category in group_categories):
            if count not in {2, 3}:
                group_size_issues.append(group)
    return {
        "total": len(probes),
        "approved": len(approved),
        "draft": sum(probe.review_status == "draft" for probe in probes.values()),
        "rejected": sum(probe.review_status == "rejected" for probe in probes.values()),
        "normative": sum(
            probe.evaluation_mode == "normative" for probe in probes.values()
        ),
        "preference": sum(
            probe.evaluation_mode == "preference" for probe in probes.values()
        ),
        "preference_axis_counts": {
            axis: preference_axis_counts[axis] for axis in sorted(UTILITY_AXES)
        },
        "approved_preference_axis_counts": {
            axis: approved_preference_axis_counts[axis]
            for axis in sorted(UTILITY_AXES)
        },
        "preference_axis_targets": dict(sorted(axis_targets.items())),
        "preference_axis_shortfalls": dict(sorted(preference_axis_shortfalls.items())),
        "preference_category_targets": dict(sorted(category_preference_targets.items())),
        "authored_preference_category_shortfalls": dict(
            sorted(authored_preference_category_shortfalls.items())
        ),
        "approved_preference_category_shortfalls": dict(
            sorted(approved_preference_category_shortfalls.items())
        ),
        "preference_axes_by_category": {
            category: {
                axis: counts[axis]
                for axis in sorted(UTILITY_AXES)
                if counts[axis]
            }
            for category, counts in sorted(preference_by_category.items())
        },
        "categories": {
            category: {
                "total": counts[category],
                "authored_total": all_counts[category],
                "authored_normative": sum(
                    probe.category == category
                    and probe.evaluation_mode == "normative"
                    for probe in probes.values()
                ),
                "authored_preference": sum(
                    probe.category == category
                    and probe.evaluation_mode == "preference"
                    for probe in probes.values()
                ),
                "calibration": split_counts[category]["calibration"],
                "validation": split_counts[category]["validation"],
                "target": target,
            }
            for category, target in sorted(targets.items())
        },
        "missing_to_target": missing,
        "unknown_categories": unknown,
        "group_split_leakage": leakage,
        "duplicate_questions": duplicates,
        "approval_metadata_issues": sorted(approval_issues),
        "analysis_metadata_issues": analysis_issues,
        "split_shortfalls": split_shortfalls,
        "choice_count_issues": sorted(choice_count_issues),
        "answer_balance_issues": sorted(answer_balance_issues),
        "group_size_issues": sorted(group_size_issues),
        "passes": not any(
            (
                missing,
                unknown,
                leakage,
                duplicates,
                approval_issues,
                {
                    probe_id: issues
                    for probe_id, issues in analysis_issues.items()
                    if probes[probe_id].review_status == "approved"
                },
                split_shortfalls,
                choice_count_issues,
                answer_balance_issues,
                group_size_issues,
                preference_axis_shortfalls,
                approved_preference_category_shortfalls,
            )
        ),
    }


def _missing_analysis(probe: Probe) -> list[str]:
    common = (
        "hypothesis",
        "decisive_information",
        "variant_axis",
        "failure_signal",
        "calibration_use",
    )
    required = common + (
        ("distractor_rationales",)
        if probe.evaluation_mode == "normative"
        else ("preference_tradeoff", "choice_rationales")
    )
    missing = [key for key in required if not probe.analysis.get(key)]
    rationales_key = (
        "distractor_rationales"
        if probe.evaluation_mode == "normative"
        else "choice_rationales"
    )
    distractors = probe.analysis.get(rationales_key)
    if isinstance(distractors, dict):
        expected = set(probe.choices) - {probe.answer}
        if probe.evaluation_mode == "preference":
            expected = set(probe.choices)
        absent = sorted(expected - set(distractors))
        if absent:
            missing.append(rationales_key + ":" + ",".join(absent))
    return missing


def _normalized_question(probe: Probe) -> str:
    text = f"{probe.scenario or ''} {probe.user_request or probe.prompt}"
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_targets(path: Path) -> dict[str, int]:
    """Load positive per-category targets from JSON."""
    value = json.loads(path.read_text(encoding="utf-8"))
    categories = value.get("categories") if isinstance(value, dict) else None
    if not isinstance(categories, dict) or not categories:
        raise ValueError("taxonomy needs a non-empty categories object")
    targets = {
        str(name): int(
            target.get("target", 0) if isinstance(target, dict) else target
        )
        for name, target in categories.items()
    }
    if any(target <= 0 for target in targets.values()):
        raise ValueError("all category targets must be positive")
    return targets


def load_preference_axis_targets(path: Path) -> dict[str, int]:
    """Load optional release targets for approved preference-axis coverage."""
    value = json.loads(path.read_text(encoding="utf-8"))
    raw_targets = value.get("preference_axis_targets") if isinstance(value, dict) else None
    if raw_targets is None:
        return {}
    if not isinstance(raw_targets, dict):
        raise ValueError("preference_axis_targets must be an object")
    targets = {str(axis): int(target) for axis, target in raw_targets.items()}
    unknown = sorted(set(targets) - UTILITY_AXES)
    if unknown:
        raise ValueError(f"unknown preference utility axes: {', '.join(unknown)}")
    if any(target <= 0 for target in targets.values()):
        raise ValueError("all preference axis targets must be positive")
    return targets


def load_preference_category_targets(path: Path) -> dict[str, int]:
    """Load per-category minimums for approved preference-sensitive probes."""
    value = json.loads(path.read_text(encoding="utf-8"))
    categories = value.get("categories") if isinstance(value, dict) else None
    if not isinstance(categories, dict):
        return {}
    raw_default = value.get("preference_per_category_target")
    default = int(raw_default) if raw_default is not None else None
    targets: dict[str, int] = {}
    for raw_category, raw_config in categories.items():
        if not isinstance(raw_config, dict):
            continue
        raw_target = raw_config.get("preference_target", default)
        if raw_target is None:
            continue
        target = int(raw_target)
        overall = int(raw_config.get("target", 0))
        if target <= 0 or target > overall:
            raise ValueError(
                "category preference_target must be positive and no greater than target"
            )
        targets[str(raw_category)] = target
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path)
    parser.add_argument("taxonomy", type=Path)
    args = parser.parse_args()
    report = audit_dataset(
        load_probes(args.probes),
        load_targets(args.taxonomy),
        load_preference_axis_targets(args.taxonomy),
        load_preference_category_targets(args.taxonomy),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passes"] else 1)


if __name__ == "__main__":
    main()
