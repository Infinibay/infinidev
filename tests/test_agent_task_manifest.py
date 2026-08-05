from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.agent_task_manifest import build_manifest
from bench.model_behavior import UtilityProfile


def _compiled(profile_sha: str) -> dict[str, object]:
    return {
        "deployment_approved": False,
        "model": "Kimi",
        "model_identity": "kimi:k2@revision",
        "calibration_role": "developer",
        "utility_profile": {"name": "quality", "sha256": profile_sha},
        "candidates": [
            {
                "name": "candidate",
                "guidance": "Prefer visible verification.",
                "guidance_sha256": "guidance-sha",
                "evidence_probe_ids": ["p1"],
            }
        ],
    }


def test_builds_exact_two_condition_call_plan() -> None:
    profile = {
        "schema_version": 1,
        "provenance": "explicit_user",
        "name": "quality",
        "description": "Prefer quality.",
        "weights": {"quality": 1.0},
    }
    profile_sha = UtilityProfile.from_dict(profile).sha256
    manifest = build_manifest(
        Path("bench/agent_task_pilot.tasks.jsonl"),
        _compiled(profile_sha),
        profile,
        candidate_name="candidate",
    )
    assert manifest["task_count"] == 6
    assert manifest["planned_executions"] == 12
    assert manifest["conditions"] == {
        "baseline": None,
        "candidate": {"system_prompt": "Prefer visible verification."},
    }
    assert manifest["execution_contract"]["parallel_requests"] is False


def test_rejects_a_deployable_or_profile_mismatched_source() -> None:
    profile = {
        "schema_version": 1,
        "provenance": "explicit_user",
        "name": "quality",
        "description": "Prefer quality.",
        "weights": {"quality": 1.0},
    }
    profile_sha = UtilityProfile.from_dict(profile).sha256
    compiled = _compiled(profile_sha)
    compiled["deployment_approved"] = True
    with pytest.raises(ValueError, match="inert"):
        build_manifest(
            Path("bench/agent_task_pilot.tasks.jsonl"),
            compiled,
            profile,
            candidate_name="candidate",
        )
