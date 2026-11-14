"""Engine selection reports the effective setting after environment overrides."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("surface", ["classic", "tui"])
@pytest.mark.parametrize(("override", "effective"), [
    (None, "react"),
    ("task", "task"),
    ("REACT", "react"),
    ("invalid", "task"),
])
def test_engine_command_reports_effective_mode(
    surface, override, effective, tmp_path, monkeypatch, capsys,
):
    settings_module = importlib.import_module("infinidev.config.settings")
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(settings_module, "SETTINGS_FILE", settings_file)
    # reload_all updates the shared singleton; replace it for this test.
    monkeypatch.setattr(settings_module, "settings", settings_module.Settings())
    if override:
        monkeypatch.setenv("INFINIDEV_TASK_ENGINE_MODE", override)
    else:
        monkeypatch.delenv("INFINIDEV_TASK_ENGINE_MODE", raising=False)

    if surface == "classic":
        from infinidev.cli.commands import handle_engine_command

        handle_engine_command(["/engine", "react"])
        output = capsys.readouterr().out
    else:
        from infinidev.ui.handlers.commands import handle_engine

        messages = []
        app = SimpleNamespace(
            add_message=lambda speaker, message, kind: messages.append(message),
            _update_status_bar=lambda: None,
        )
        handle_engine(app, ["/engine", "react"])
        output = "\n".join(messages)

    assert json.loads(settings_file.read_text())["TASK_ENGINE_MODE"] == "react"
    assert settings_module.settings.TASK_ENGINE_MODE == (override or "react")
    if effective != "react":
        assert "INFINIDEV_TASK_ENGINE_MODE" in output
        assert effective in output
        assert "saved" in output.lower()
        assert "set to react" not in output
    else:
        assert "set to react" in output
