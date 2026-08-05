from __future__ import annotations

import hashlib
import json

from infinidev.engine.prompt_profile import apply_calibrated_guidance, load_calibrated_guidance
from infinidev.engine.user_preferences import _profile_sha


def _write_profile(tmp_path, **updates):
    guidance = "Inspect evidence before acting."
    value = {
        "schema_version": 2,
        "prompt_layer": "behavior",
        "evidence_kind": "preference_behavior",
        "deployment_approved": True,
        "provider": "openai",
        "model": "openai/model-v1",
        "model_identity": "openai/model-v1@revision-1",
        "validation": {"utility_profile": {"name": "quality-control", "sha256": "abc"}},
        "roles": {
            "developer": {
                "prompt_layer": "behavior",
                "guidance": guidance,
                "sha256": hashlib.sha256(guidance.encode()).hexdigest(),
                "utf8_bytes": len(guidance.encode()),
            }
        },
    }
    value.update(updates)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(value))
    return path


def test_loads_guidance_only_for_exact_route_and_role(tmp_path) -> None:
    path = _write_profile(tmp_path)
    assert load_calibrated_guidance(
        path,
        provider="openai",
        model="openai/model-v1",
        model_identity="openai/model-v1@revision-1",
        utility_profile="quality-control",
        utility_profile_sha256="abc",
        role="developer",
    ) == "Inspect evidence before acting."
    assert load_calibrated_guidance(
        path,
        provider="openai",
        model="openai/other",
        model_identity="openai/model-v1@revision-1",
        utility_profile="quality-control",
        utility_profile_sha256="abc",
        role="developer",
    ) == ""
    assert load_calibrated_guidance(
        path,
        provider="openai",
        model="openai/model-v1",
        model_identity="openai/model-v1@revision-1",
        utility_profile="quality-control",
        utility_profile_sha256="abc",
        role="planner",
    ) == ""


def test_rejects_wrong_model_revision_or_user_objective(tmp_path) -> None:
    path = _write_profile(tmp_path)
    common = {"provider": "openai", "model": "openai/model-v1", "role": "developer"}

    assert load_calibrated_guidance(
        path,
        model_identity="openai/model-v1@revision-2",
        utility_profile="quality-control",
        utility_profile_sha256="abc",
        **common,
    ) == ""
    assert load_calibrated_guidance(
        path,
        model_identity="openai/model-v1@revision-1",
        utility_profile="fast-autonomy",
        utility_profile_sha256="abc",
        **common,
    ) == ""
    assert load_calibrated_guidance(
        path,
        model_identity="",
        utility_profile="quality-control",
        utility_profile_sha256="abc",
        **common,
    ) == ""
    assert load_calibrated_guidance(
        path,
        model_identity="openai/model-v1@revision-1",
        utility_profile="quality-control",
        utility_profile_sha256="different-profile",
        **common,
    ) == ""


def test_rejects_unapproved_or_tampered_profile(tmp_path) -> None:
    assert load_calibrated_guidance(
        _write_profile(tmp_path, deployment_approved=False),
        provider="openai",
        model="openai/model-v1",
        model_identity="openai/model-v1@revision-1",
        utility_profile="quality-control",
        utility_profile_sha256="abc",
        role="developer",
    ) == ""
    path = _write_profile(tmp_path)
    value = json.loads(path.read_text())
    value["roles"]["developer"]["guidance"] = "Tampered"
    path.write_text(json.dumps(value))
    assert load_calibrated_guidance(
        path,
        provider="openai",
        model="openai/model-v1",
        model_identity="openai/model-v1@revision-1",
        utility_profile="quality-control",
        utility_profile_sha256="abc",
        role="developer",
    ) == ""


def test_rejects_oversized_profile(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text("x" * (64 * 1024 + 1))
    assert load_calibrated_guidance(
        path,
        provider="openai",
        model="openai/model-v1",
        model_identity="openai/model-v1@revision-1",
        utility_profile="quality-control",
        utility_profile_sha256="abc",
        role="developer",
    ) == ""


def test_rejects_oversized_or_miscounted_guidance(tmp_path) -> None:
    guidance = "x" * (4 * 1024 + 1)
    oversized = _write_profile(
        tmp_path,
        roles={
            "developer": {
                "prompt_layer": "behavior",
                "guidance": guidance,
                "sha256": hashlib.sha256(guidance.encode()).hexdigest(),
                "utf8_bytes": len(guidance.encode()),
            }
        },
    )
    args = {
        "provider": "openai",
        "model": "openai/model-v1",
        "model_identity": "openai/model-v1@revision-1",
        "utility_profile": "quality-control",
        "utility_profile_sha256": "abc",
        "role": "developer",
    }
    assert load_calibrated_guidance(oversized, **args) == ""

    miscounted = _write_profile(tmp_path)
    value = json.loads(miscounted.read_text())
    value["roles"]["developer"]["utf8_bytes"] += 1
    miscounted.write_text(json.dumps(value))
    assert load_calibrated_guidance(miscounted, **args) == ""


def test_active_user_profile_selects_matching_calibrated_guidance(
    tmp_path, monkeypatch
) -> None:
    preference = {
        "schema_version": 1,
        "provenance": "explicit_user",
        "name": "quality-control",
        "description": "Prefer evidence and control for consequential trade-offs.",
        "weights": {"quality": 1.0, "user_control": 0.8},
    }
    preference_sha = _profile_sha(
        preference["name"], preference["description"], preference["weights"]
    )
    preference_path = tmp_path / "preference.json"
    preference_path.write_text(json.dumps(preference))
    calibration_path = _write_profile(
        tmp_path,
        validation={
            "utility_profile": {
                "name": preference["name"],
                "sha256": preference_sha,
            }
        },
    )
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "LLM_MODEL", "openai/model-v1")
    monkeypatch.setattr(
        settings, "PROMPT_CALIBRATION_MODEL_IDENTITY", "openai/model-v1@revision-1"
    )
    monkeypatch.setattr(settings, "PROMPT_CALIBRATION_PROFILE", str(calibration_path))
    monkeypatch.setattr(settings, "PROMPT_CALIBRATION_UTILITY_PROFILE", "stale-legacy")
    monkeypatch.setattr(settings, "PROMPT_CALIBRATION_UTILITY_PROFILE_SHA256", "stale")
    monkeypatch.setattr(settings, "USER_PREFERENCE_PROFILE", str(preference_path))
    monkeypatch.setattr(settings, "USER_PREFERENCE_PROFILE_SHA256", preference_sha)

    prompt = apply_calibrated_guidance("base", "developer")

    assert "Prefer evidence and control" in prompt
    assert "Inspect evidence before acting" in prompt
    assert "quality\"" not in prompt
