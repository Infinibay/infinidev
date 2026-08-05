#!/usr/bin/env python3
"""Turn counterbalanced raw priors into a user-profile-conditioned prompt brief."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from bench.model_behavior import UtilityProfile, choice_utility, load_probes


def _best_keys(probe: object, profile: UtilityProfile) -> tuple[list[str], dict[str, float]]:
    choices = getattr(probe, "choices")
    utilities = {key: choice_utility(probe, key, profile) for key in choices}
    maximum = max(utilities.values())
    return sorted(key for key, value in utilities.items() if abs(value - maximum) < 1e-12), utilities


def build_prompt_brief(
    analysis: Mapping[str, object],
    probes: Mapping[str, object],
    *,
    model: str,
    profile: UtilityProfile,
) -> dict[str, object]:
    """Classify concrete raw actions as preserve or compensate hypotheses for one profile."""
    models = analysis.get("models")
    if not isinstance(models, dict) or model not in models:
        raise ValueError(f"model is missing from counterbalanced analysis: {model}")
    records = analysis.get("records")
    if not isinstance(records, list):
        raise ValueError("counterbalanced analysis needs records")

    preserve: list[dict[str, object]] = []
    compensate: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    normative: list[dict[str, object]] = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            continue
        probe_id = str(raw_record.get("probe_id", ""))
        probe = probes.get(probe_id)
        raw_models = raw_record.get("models")
        model_record = raw_models.get(model) if isinstance(raw_models, dict) else None
        if probe is None or not isinstance(model_record, dict):
            raise ValueError(f"missing probe/model evidence for {probe_id}/{model}")
        modes = [str(key) for key in model_record.get("balanced_modal_keys", [])]
        exactly_stable = bool(model_record.get("exactly_stable"))
        base = {
            "probe_id": probe_id,
            "category": raw_record.get("category"),
            "scenario": raw_record.get("scenario"),
            "observed_counts": model_record.get("balanced_counts"),
            "observed_modal_keys": modes,
            "observed_modal_actions": model_record.get("balanced_modal_actions"),
            "exactly_stable": exactly_stable,
            "fixed_to_balanced_relation": model_record.get("fixed_to_balanced_relation"),
        }
        if getattr(probe, "evaluation_mode") == "normative":
            expected = getattr(probe, "answer")
            normative.append(
                {
                    **base,
                    "draft_expected_key": expected,
                    "draft_expected_action": getattr(probe, "choices").get(expected),
                    "status": (
                        "stable_match"
                        if exactly_stable and modes == [expected]
                        else "requires_independent_review_or_more_evidence"
                    ),
                }
            )
            continue

        best_keys, utilities = _best_keys(probe, profile)
        modal_utilities = {key: utilities[key] for key in modes}
        aligned = bool(modes) and all(key in best_keys for key in modes)
        item = {
            **base,
            "profile_best_keys": best_keys,
            "profile_best_actions": [getattr(probe, "choices")[key] for key in best_keys],
            "choice_utilities": utilities,
            "modal_utilities": modal_utilities,
            "inference": (
                "raw prior already selects a profile-optimal action"
                if aligned
                else "raw prior differs from the profile-optimal action"
            ),
        }
        if exactly_stable and aligned:
            preserve.append(item)
        elif exactly_stable and not aligned:
            compensate.append({**item, "evidence_strength": "stable_raw_prior"})
        else:
            unresolved.append(
                {
                    **item,
                    "evidence_strength": "position_sensitive_or_tied",
                    "candidate_direction": "preserve" if aligned else "compensate",
                }
            )
    return {
        "model": model,
        "utility_profile": {
            "name": profile.name,
            "description": profile.description,
            "weights": profile.weights,
            "sha256": profile.sha256,
        },
        "interpretation_boundary": (
            "This brief compares unprofiled raw actions with an explicit user's scoring objective. "
            "It proposes what to preserve or test compensating guidance for; it does not establish "
            "that a prompt improves task outcomes. Numeric utility routes records, while concrete "
            "actions remain the prompt-authoring evidence."
        ),
        "authoring_contract": [
            "Do not add guidance for stable behavior that already serves the active profile unless it is needed to prevent a measured regression.",
            "A stable conflicting raw prior is a candidate for small compensating guidance, not proof that the model is defective.",
            "An unstable or tied raw prior requires profile-conditioned replication before authoring a strong candidate.",
            "Cite probe IDs and concrete actions in every candidate rationale; do not write prompts from aggregate scores alone.",
            "Validate every candidate against the unchanged baseline on held-out repository tasks and reject normative regressions.",
        ],
        "stable_profile_aligned_actions_to_preserve": preserve,
        "stable_profile_conflicts_to_test": compensate,
        "unstable_profile_hypotheses": unresolved,
        "normative_evidence": normative,
    }


def render_markdown(brief: Mapping[str, object]) -> str:
    """Render concrete action evidence before calibration instructions."""
    profile = brief.get("utility_profile", {})
    lines = [
        f"# Counterbalanced prompt brief: {brief.get('model')}",
        "",
        f"Profile: **{profile.get('name') if isinstance(profile, dict) else ''}** — "
        f"{profile.get('description') if isinstance(profile, dict) else ''}",
        "",
        str(brief.get("interpretation_boundary")),
        "",
        "## Authoring contract",
        "",
    ]
    contract = brief.get("authoring_contract", [])
    if isinstance(contract, list):
        lines.extend(f"- {item}" for item in contract)
    sections = (
        ("Stable profile-aligned actions to preserve", "stable_profile_aligned_actions_to_preserve"),
        ("Stable profile conflicts to test", "stable_profile_conflicts_to_test"),
        ("Unstable profile hypotheses", "unstable_profile_hypotheses"),
        ("Normative evidence", "normative_evidence"),
    )
    for title, key in sections:
        lines.extend(["", f"## {title}", ""])
        records = brief.get(key, [])
        if not isinstance(records, list) or not records:
            lines.append("None in this selected follow-up.")
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            lines.append(
                f"- `{record.get('probe_id')}`: observed "
                f"{record.get('observed_modal_actions')} with counts "
                f"`{json.dumps(record.get('observed_counts'), sort_keys=True)}`."
            )
            if "profile_best_actions" in record:
                lines.append(f"  - Profile-optimal action(s): {record.get('profile_best_actions')}")
            if record.get("inference"):
                lines.append(f"  - Inference: {record.get('inference')}.")
            if record.get("status"):
                lines.append(f"  - Status: `{record.get('status')}`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("probes", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("model")
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    profile = UtilityProfile.from_dict(json.loads(args.profile.read_text(encoding="utf-8")))
    brief = build_prompt_brief(analysis, load_probes(args.probes), model=args.model, profile=profile)
    args.output_markdown.write_text(render_markdown(brief), encoding="utf-8")
    args.output_json.write_text(
        json.dumps(brief, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
