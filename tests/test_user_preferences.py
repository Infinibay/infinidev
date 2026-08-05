"""Tests for explicit hash-bound runtime user preferences."""

from __future__ import annotations

import json
from pathlib import Path

from infinidev.engine.user_preferences import (
    _profile_sha,
    apply_active_user_preferences,
    load_active_user_preferences,
)


def _value() -> dict:
    return {
        "schema_version": 1,
        "provenance": "explicit_user",
        "name": "fast-autonomy",
        "description": "Prefer fast reversible progress and few checkpoints.",
        "weights": {"speed": 1.0, "autonomy": 0.8, "interaction": -0.5},
    }


def _write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return _profile_sha(value["name"], value["description"], value["weights"])


def test_profile_is_hash_bound_and_keeps_weights_out_of_rendered_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "preferences.json"
    sha = _write(path, _value())
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "USER_PREFERENCE_PROFILE", str(path))
    monkeypatch.setattr(settings, "USER_PREFERENCE_PROFILE_SHA256", sha)

    prompt, profile = apply_active_user_preferences("base prompt")

    assert profile is not None
    assert profile.name == "fast-autonomy"
    assert "Prefer fast reversible progress" in prompt
    assert "speed" not in prompt
    assert "1.0" not in prompt
    assert "never override explicit task requirements" in prompt


def test_profile_fails_closed_on_hash_drift(tmp_path: Path) -> None:
    path = tmp_path / "preferences.json"
    _write(path, _value())

    assert load_active_user_preferences(path, expected_sha256="wrong") is None


def test_profile_requires_explicit_user_provenance(tmp_path: Path) -> None:
    path = tmp_path / "preferences.json"
    value = _value()
    value["provenance"] = "model_inferred"
    sha = _write(path, value)

    assert load_active_user_preferences(path, expected_sha256=sha) is None
