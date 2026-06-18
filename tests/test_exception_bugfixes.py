"""Regression tests for the Phase-1 broad-`except Exception` bug audit.

Each test locks in a fix for a confirmed bug where a broad exception handler
silently masked a failure into wrong/lossy behavior. Grouped by the file the
bug lived in. See the Phase-1 review roadmap for the full list.
"""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from infinidev.config.settings import Settings


SRC = Path(__file__).resolve().parents[1] / "src" / "infinidev"


# ─────────────────────────────────────────────────────────────────────────
# settings.py — destructive-overwrite / silent-defaults (HIGH)
# ─────────────────────────────────────────────────────────────────────────
class TestSettingsDataLoss:
    def test_save_aborts_instead_of_destroying_on_unreadable_existing(self, tmp_path):
        """save_user_settings must NOT clobber an existing-but-unreadable file.

        The bug: a read failure defaulted current_data to {} and then dumped,
        wiping every previously-saved key (model, base_url, API keys). The fix
        aborts the save and leaves the existing file untouched.
        """
        sf = tmp_path / "settings.json"
        corrupt = "{ this is NOT valid json -- simulates a corrupt/partial file"
        sf.write_text(corrupt)
        with patch("infinidev.config.settings.SETTINGS_FILE", sf):
            Settings().save_user_settings({"LLM_MODEL": "new/model"})
        # File must be left exactly as it was — nothing overwritten.
        assert sf.read_text() == corrupt

    def test_save_still_merges_on_valid_existing(self, tmp_path):
        """The fix must not regress the normal merge path."""
        sf = tmp_path / "settings.json"
        sf.write_text(json.dumps({"LLM_MODEL": "old", "LLM_BASE_URL": "http://x"}))
        with patch("infinidev.config.settings.SETTINGS_FILE", sf):
            Settings().save_user_settings({"LLM_MODEL": "new"})
        data = json.loads(sf.read_text())
        assert data["LLM_MODEL"] == "new"
        assert data["LLM_BASE_URL"] == "http://x"  # preserved

    def test_load_logs_error_on_corrupt(self, tmp_path, caplog):
        """load_user_settings must log via the logger (not print) on corrupt JSON."""
        sf = tmp_path / "settings.json"
        sf.write_text("{not valid json!!!")
        with patch("infinidev.config.settings.SETTINGS_FILE", sf):
            with caplog.at_level(logging.ERROR, logger="infinidev.config.settings"):
                s = Settings.load_user_settings()
        assert s.LOOP_MAX_ITERATIONS == 50  # defaults, no crash
        assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────
# ui/controls/file_editor.py — read-failure → empty-buffer → overwrite (MEDIUM)
# ─────────────────────────────────────────────────────────────────────────
class TestFileEditorReadFailure:
    def test_read_failure_sets_flag(self, tmp_path):
        """A failed read must be remembered, not silently treated as empty."""
        from infinidev.ui.controls.file_editor import FileEditor

        # Opening a directory for read raises IsADirectoryError (an OSError).
        editor = FileEditor(str(tmp_path))
        assert editor._read_failed is True

    def test_save_refuses_after_read_failure(self, tmp_path):
        """save() must refuse to overwrite when the initial read failed.

        Otherwise a transient read error opens an empty buffer that, on save,
        truncates the real file to nothing.
        """
        from infinidev.ui.controls.file_editor import FileEditor

        real = tmp_path / "real.py"
        real.write_text("important = True\n")
        editor = FileEditor(str(real))
        # Simulate "the initial read had failed".
        editor._read_failed = True
        assert editor.save() is False
        # File must be untouched.
        assert real.read_text() == "important = True\n"


# ─────────────────────────────────────────────────────────────────────────
# git tools — undefined FlowEvent killed the EventBus (HIGH)
# ─────────────────────────────────────────────────────────────────────────
class TestGitEventEmission:
    GIT_TOOLS = [
        SRC / "tools" / "git" / "git_branch_tool.py",
        SRC / "tools" / "git" / "git_commit_tool.py",
        SRC / "tools" / "git" / "git_push_tool.py",
    ]

    def test_no_flowevent_references_remain(self):
        """The undefined `FlowEvent` symbol must be gone from every git tool."""
        for f in self.GIT_TOOLS:
            assert "FlowEvent" not in f.read_text(), f"FlowEvent still referenced in {f}"

    def test_event_bus_emit_delivers_with_4arg_signature(self):
        """Confirm the 4-arg signature the git tools now call actually delivers."""
        from infinidev.flows.event_listeners import event_bus

        received = []

        def cb(event_type, project_id, agent_id, data):
            received.append((event_type, project_id, agent_id, data))

        event_bus.subscribe(cb)
        try:
            event_bus.emit("git_pushed", 7, "agent-x", {"branch": "main"})
        finally:
            event_bus.unsubscribe(cb)

        assert received == [("git_pushed", 7, "agent-x", {"branch": "main"})]


# ─────────────────────────────────────────────────────────────────────────
# loop/schema_sanitizer.py — empty schema = the security boundary (MEDIUM)
# ─────────────────────────────────────────────────────────────────────────
class TestSchemaSanitizerWarns:
    def test_warns_when_schema_extraction_fails(self, caplog):
        """A tool whose schema cannot be extracted must warn, not register silently."""
        from infinidev.engine.schema_sanitizer import tool_to_openai_schema

        class _BadSchema:
            def model_json_schema(self):
                raise RuntimeError("boom")

            def schema(self):
                raise RuntimeError("boom2")

        class _Tool:
            name = "bad_tool"
            description = "a tool whose schema cannot be extracted"
            args_schema = _BadSchema()

        with caplog.at_level(logging.WARNING, logger="infinidev.engine.schema_sanitizer"):
            result = tool_to_openai_schema(_Tool())

        # Still registers (empty params), but the failure is now logged.
        assert result["function"]["parameters"]["properties"] == {}
        assert any("bad_tool" in r.getMessage() for r in caplog.records)
