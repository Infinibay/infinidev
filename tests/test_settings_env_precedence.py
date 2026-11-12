"""Environment overrides for persisted user settings."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from infinidev.config.settings import Settings


@pytest.mark.parametrize(
    ("field", "saved", "environment", "expected"),
    [
        ("LLM_PROVIDER", "ollama", "openai", "openai"),
        ("LLM_BASE_URL", "http://localhost:11434", "http://localhost:8787/v1",
         "http://localhost:8787/v1"),
        ("COMMAND_TIMEOUT", 120, "42", 42),
        ("SANDBOX_ENABLED", True, "false", False),
        ("ALLOWED_BASE_DIRS", ["/saved"], '["/environment", "/other"]',
         ["/environment", "/other"]),
    ],
)
def test_environment_overrides_saved_settings(
    tmp_path, monkeypatch, field, saved, environment, expected,
):
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({field: saved}))
    monkeypatch.setenv(f"INFINIDEV_{field}", environment)

    with patch("infinidev.config.settings.SETTINGS_FILE", settings_file):
        loaded = Settings.load_user_settings()

    assert getattr(loaded, field) == expected
    assert json.loads(settings_file.read_text()) == {field: saved}


def test_environment_override_preserves_file_values_and_defaults(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"LLM_PROVIDER": "ollama", "COMMAND_TIMEOUT": 73}))
    monkeypatch.setenv("INFINIDEV_LLM_PROVIDER", "openai")
    monkeypatch.delenv("INFINIDEV_COMMAND_TIMEOUT", raising=False)
    monkeypatch.delenv("INFINIDEV_WEB_TIMEOUT", raising=False)

    with patch("infinidev.config.settings.SETTINGS_FILE", settings_file):
        loaded = Settings.load_user_settings()

    assert loaded.LLM_PROVIDER == "openai"
    assert loaded.COMMAND_TIMEOUT == 73
    assert loaded.WEB_TIMEOUT == 30


def test_explicit_constructor_values_still_override_environment(monkeypatch):
    monkeypatch.setenv("INFINIDEV_LLM_PROVIDER", "openai")

    assert Settings(LLM_PROVIDER="ollama").LLM_PROVIDER == "ollama"
