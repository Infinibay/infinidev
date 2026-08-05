#!/usr/bin/env python3
"""Freeze a provider-neutral paired agent-task pilot from a compiled prompt candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from bench.agent_task_eval import file_sha256, load_tasks


def _profile_sha(profile: Mapping[str, object]) -> str:
    weights = {
        str(key): float(value)
        for key, value in dict(profile.get("weights", {})).items()
        if float(value)
    }
    encoded = json.dumps(
        {
            "name": str(profile.get("name", "")).strip(),
            "weights": weights,
            "description": str(profile.get("description", "")).strip(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    tasks_path: Path,
    compiled: Mapping[str, object],
    profile: Mapping[str, object],
    *,
    candidate_name: str,
    split: str = "validation",
) -> dict[str, object]:
    """Bind task bytes, concrete guidance, model identity, profile, and exact call count."""
    tasks = [task for task in load_tasks(tasks_path) if task.split == split]
    if not tasks:
        raise ValueError(f"no tasks exist for split: {split}")
    if compiled.get("deployment_approved") is not False:
        raise ValueError("agent task pilot requires inert compiled candidates")
    raw_candidates = compiled.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("compiled candidate artifact has no candidates")
    matches = [
        item
        for item in raw_candidates
        if isinstance(item, dict) and item.get("name") == candidate_name
    ]
    if len(matches) != 1:
        raise ValueError(f"compiled candidate is missing or ambiguous: {candidate_name}")
    candidate = matches[0]
    guidance = str(candidate.get("guidance", "")).strip()
    if not guidance:
        raise ValueError("compiled candidate guidance is empty")
    profile_sha = _profile_sha(profile)
    raw_compiled_profile = compiled.get("utility_profile")
    if not isinstance(raw_compiled_profile, dict) or raw_compiled_profile.get("sha256") != profile_sha:
        raise ValueError("compiled candidate and explicit user profile differ")
    if profile.get("schema_version") != 1 or profile.get("provenance") != "explicit_user":
        raise ValueError("pilot user profile must have explicit_user provenance")
    model_identity = str(compiled.get("model_identity", "")).strip()
    if not model_identity:
        raise ValueError("compiled candidate needs immutable model identity")
    return {
        "schema_version": 1,
        "purpose": "Small paired falsification pilot; not calibration or deployment evidence",
        "dataset_sha256": file_sha256(tasks_path),
        "task_split": split,
        "task_ids": [task.id for task in tasks],
        "task_count": len(tasks),
        "conditions_per_task": 2,
        "repetitions": 1,
        "planned_executions": len(tasks) * 2,
        "model": compiled.get("model"),
        "model_identity": model_identity,
        "calibration_role": compiled.get("calibration_role"),
        "candidate_name": candidate_name,
        "candidate_guidance_sha256": candidate.get("guidance_sha256"),
        "candidate_evidence_probe_ids": candidate.get("evidence_probe_ids"),
        "utility_profile_sha256": profile_sha,
        "utility_profile": dict(profile),
        "conditions": {
            "baseline": None,
            "candidate": {"system_prompt": guidance},
        },
        "execution_contract": {
            "fresh_workspace_per_execution": True,
            "fresh_agent_session_per_execution": True,
            "parallel_requests": False,
            "minimum_llm_request_interval_seconds": 2.0,
            "automatic_llm_retries": False,
            "stop_on_first_runtime_or_provider_error": True,
        },
        "promotion_boundary": (
            "This manifest can only falsify a candidate cheaply. It cannot authorize a runtime "
            "profile or substitute for calibration and independent held-out validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("compiled_candidates", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("candidate_name")
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("calibration", "validation"), default="validation")
    args = parser.parse_args()
    compiled = json.loads(args.compiled_candidates.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    manifest = build_manifest(
        args.tasks,
        compiled,
        profile,
        candidate_name=args.candidate_name,
        split=args.split,
    )
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
