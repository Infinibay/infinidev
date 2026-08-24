"""Tests for prompt-capability slash commands shared by classic and TUI modes."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from threading import Thread
import time

import pytest

from infinidev.cli.commands import handle_command as handle_classic_command
from infinidev.prompts import commands, profiles
from infinidev.prompts.catalog import STARTER_PROMPT_PROFILES
from infinidev.ui.controls.autocomplete import COMMANDS
from infinidev.ui.handlers.commands import handle_command as handle_tui_command


def _write_prompt_override_in_process(
    path: str,
    capability: str,
    start: multiprocessing.synchronize.Event,
) -> None:
    """Wait for sibling writers, then exercise the real cross-process update path."""
    start.wait()
    commands._write_user_override(Path(path), capability, True)


def _use_temporary_profiles(monkeypatch, tmp_path) -> None:
    catalog = tmp_path / "user-prompts"
    project = tmp_path / "project-prompts.json"
    monkeypatch.setattr(commands, "get_prompt_catalog_path", lambda: catalog)
    monkeypatch.setattr(profiles, "get_prompt_catalog_path", lambda: catalog)
    monkeypatch.setattr(profiles, "get_prompt_profile_path", lambda: project)


def test_prompts_command_lists_and_persists_capability(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    listing = commands.handle_prompts_command(["/prompts"])
    assert "debugging" in listing
    assert "off  debugging — the current task requires diagnosing incorrect behavior" in listing

    result = commands.handle_prompts_command(["/prompts", "enable", "debugging"])

    assert result == "Prompt capability debugging enabled. Applies from the next task."
    override = json.loads(
        (tmp_path / "user-prompts" / "99-user-overrides.json").read_text(encoding="utf-8")
    )
    assert override["develop"]["capability.debugging"]["enabled"] is True
    assert "on   debugging" in commands.handle_prompts_command(["/prompts"])


def test_prompts_command_shows_guidance_before_enabling(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    result = commands.handle_prompts_command(
        ["/prompts", "show", "Capability.Debugging"]
    )

    assert "Prompt capability: debugging (disabled)" in result
    assert "ID: capability.debugging" in result
    assert "Applies when: the current task requires diagnosing incorrect behavior" in result
    assert "Reproduce the symptom or establish the failing contract" in result
    assert "<if" not in result
    assert "</if>" not in result


def test_prompts_command_show_reports_unknown_capability(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    result = commands.handle_prompts_command(["/prompts", "show", "unknown"])

    assert result == (
        "Unknown prompt capability: unknown\nUse /prompts to list available names."
    )


def test_prompts_command_accepts_case_insensitive_capability_name(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    result = commands.handle_prompts_command(
        ["/prompts", "ENABLE", "Capability.Debugging"]
    )

    assert result == "Prompt capability debugging enabled. Applies from the next task."
    override = json.loads(
        (tmp_path / "user-prompts" / "99-user-overrides.json").read_text(encoding="utf-8")
    )
    assert override["develop"]["capability.debugging"]["enabled"] is True


def test_prompts_listing_flattens_multiline_activation_conditions(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    listing = commands.handle_prompts_command(["/prompts"])

    assert (
        "off  security — the current task touches trust boundaries, credentials, permissions, "
        "or untrusted input"
    ) in listing
    assert "\\\n" not in listing


def test_prompts_search_matches_name_and_activation_condition_case_insensitively(
    monkeypatch, tmp_path
) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    name_result = commands.handle_prompts_command(["/prompts", "search", "DeBuG"])
    condition_result = commands.handle_prompts_command(
        ["/prompts", "SEARCH", "CrEdEnTiAlS"]
    )

    assert "Prompt capabilities matching 'DeBuG':" in name_result
    assert "off  debugging — the current task requires diagnosing incorrect behavior" in name_result
    assert "security —" not in name_result
    assert "Prompt capabilities matching 'CrEdEnTiAlS':" in condition_result
    assert "off  security —" in condition_result


def test_prompts_search_matches_guidance_text(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    result = commands.handle_prompts_command(["/prompts", "search", "rollback"])

    assert "off  release_readiness —" in result
    assert "debugging —" not in result


def test_prompts_search_reports_when_nothing_matches(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    result = commands.handle_prompts_command(["/prompts", "search", "no-such-capability"])

    assert result == "No prompt capabilities match: no-such-capability"


def test_prompts_search_rejects_empty_term(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    for parts in (["/prompts", "search"], ["/prompts", "search", "   "]):
        result = commands.handle_prompts_command(parts)
        assert result.startswith("Usage: /prompts")
        assert "Optional prompt capabilities:" not in result


def test_prompts_search_accepts_multi_word_activation_condition(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    result = commands.handle_prompts_command(
        ["/prompts", "search", "trust", "boundaries"]
    )

    assert "Prompt capabilities matching 'trust boundaries':" in result
    assert "off  security —" in result


def test_prompts_non_search_actions_reject_trailing_arguments(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    for action in ("show", "enable", "disable", "reset"):
        result = commands.handle_prompts_command(
            ["/prompts", action, "debugging", "unexpected"]
        )
        assert result.startswith("Usage: /prompts")


def test_prompts_reset_removes_only_selected_override_and_reports_inherited_state(
    monkeypatch, tmp_path
) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)
    commands.handle_prompts_command(["/prompts", "enable", "debugging"])
    commands.handle_prompts_command(["/prompts", "enable", "testing"])

    result = commands.handle_prompts_command(["/prompts", "reset", "debugging"])

    assert result == (
        "Prompt capability debugging reset to its inherited state (disabled). "
        "Applies from the next task."
    )
    override = json.loads(
        (tmp_path / "user-prompts" / "99-user-overrides.json").read_text(encoding="utf-8")
    )
    assert set(override["develop"]) == {"capability.testing"}
    assert "off  debugging" in commands.handle_prompts_command(["/prompts"])
    assert "on   testing" in commands.handle_prompts_command(["/prompts"])


def test_prompts_reset_without_override_reports_inherited_state(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    result = commands.handle_prompts_command(["/prompts", "reset", "debugging"])

    assert result == (
        "Prompt capability debugging already uses its inherited state (disabled)."
    )
    assert not (tmp_path / "user-prompts" / "99-user-overrides.json").exists()


def test_prompts_reset_reports_inherited_state_from_later_profile(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)
    commands.handle_prompts_command(["/prompts", "disable", "debugging"])
    catalog = tmp_path / "user-prompts"
    (catalog / "zz-team-policy.json").write_text(
        json.dumps({"develop": {"capability.debugging": {"enabled": True}}}),
        encoding="utf-8",
    )

    result = commands.handle_prompts_command(["/prompts", "reset", "debugging"])

    assert result == (
        "Prompt capability debugging reset to its inherited state (enabled). "
        "Applies from the next task."
    )


def test_first_toggle_materializes_starter_catalog_before_override(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    commands.handle_prompts_command(["/prompts", "enable", "testing"])

    catalog = tmp_path / "user-prompts"
    assert sorted(path.name for path in catalog.iterdir()) == sorted(
        [filename for filename, _content in STARTER_PROMPT_PROFILES]
        + ["99-user-overrides.json"]
    )


def test_project_override_remains_higher_precedence(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)
    (tmp_path / "project-prompts.json").write_text(
        json.dumps({"develop": {"capability.debugging": {"enabled": False}}}),
        encoding="utf-8",
    )

    result = commands.handle_prompts_command(["/prompts", "enable", "debugging"])

    assert "higher-precedence prompt profile" in result
    assert "project override" in result
    assert "off  debugging" in commands.handle_prompts_command(["/prompts"])


def test_later_shared_profile_mismatch_is_reported_without_false_attribution(
    monkeypatch, tmp_path
) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)
    catalog = tmp_path / "user-prompts"
    catalog.mkdir()
    (catalog / "zz-team-policy.json").write_text(
        json.dumps({"develop": {"capability.debugging": {"enabled": False}}}),
        encoding="utf-8",
    )

    result = commands.handle_prompts_command(["/prompts", "enable", "debugging"])

    assert "higher-precedence prompt profile" in result
    assert "later shared file" in result
    assert "off  debugging" in commands.handle_prompts_command(["/prompts"])


def test_prompts_command_rejects_trailing_list_arguments(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)

    result = commands.handle_prompts_command(["/prompts", "list", "unexpected"])

    assert result.startswith("Usage: /prompts")


def test_prompts_command_reports_malformed_user_override(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)
    catalog = tmp_path / "user-prompts"
    catalog.mkdir()
    (catalog / "99-user-overrides.json").write_text("[]", encoding="utf-8")

    result = commands.handle_prompts_command(["/prompts", "enable", "debugging"])

    assert result.startswith("Prompt configuration error:")
    assert "must be an object" in result


def test_prompt_toggle_flushes_override_and_directory(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)
    monkeypatch.setattr(commands, "materialize_starter_prompt_profiles", lambda _path: ())
    monkeypatch.setattr(profiles, "materialize_starter_prompt_profiles", lambda _path: ())
    fsync_calls: list[int] = []
    monkeypatch.setattr(commands.os, "fsync", fsync_calls.append)

    result = commands.handle_prompts_command(["/prompts", "enable", "debugging"])

    assert result == "Prompt capability debugging enabled. Applies from the next task."
    assert len(fsync_calls) == 2


def test_prompt_toggle_cleans_temporary_file_when_directory_sync_fails(
    monkeypatch, tmp_path
) -> None:
    override = tmp_path / "prompts" / "99-user-overrides.json"

    def fail_directory_sync(_path: Path) -> None:
        raise OSError("directory sync failed")

    monkeypatch.setattr(commands, "fsync_directory", fail_directory_sync)

    with pytest.raises(OSError, match="directory sync failed"):
        commands._write_user_override(override, "capability.debugging", True)

    document = json.loads(override.read_text(encoding="utf-8"))
    assert document["develop"]["capability.debugging"]["enabled"] is True
    assert not list(override.parent.glob(".99-user-overrides.json.*.tmp"))


def test_prompts_command_does_not_reuse_stale_temporary_file(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)
    catalog = tmp_path / "user-prompts"
    catalog.mkdir()
    stale = catalog / "99-user-overrides.tmp"
    stale.write_text("unfinished write", encoding="utf-8")

    result = commands.handle_prompts_command(["/prompts", "enable", "debugging"])

    assert result == "Prompt capability debugging enabled. Applies from the next task."
    assert stale.read_text(encoding="utf-8") == "unfinished write"
    assert not list(catalog.glob(".99-user-overrides.json.*.tmp"))


def test_concurrent_prompt_toggles_preserve_both_updates(monkeypatch, tmp_path) -> None:
    """Parallel in-process commands must not lose a read-modify-write update."""
    _use_temporary_profiles(monkeypatch, tmp_path)
    original_loads = commands.json.loads

    def delayed_loads(content):
        loaded = original_loads(content)
        time.sleep(0.1)
        return loaded

    catalog = tmp_path / "user-prompts"
    catalog.mkdir()
    override = catalog / "99-user-overrides.json"
    override.write_text('{"develop": {}}\n', encoding="utf-8")
    monkeypatch.setattr(commands.json, "loads", delayed_loads)

    workers = [
        Thread(
            target=commands._write_user_override,
            args=(override, capability, True),
        )
        for capability in ("capability.debugging", "capability.testing")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    document = original_loads(override.read_text(encoding="utf-8"))
    assert set(document["develop"]) == {
        "capability.debugging",
        "capability.testing",
    }


def test_concurrent_process_prompt_toggles_preserve_every_update(tmp_path) -> None:
    """Separate CLI processes must not overwrite one another's capability changes."""
    context = multiprocessing.get_context("spawn")
    override = tmp_path / "prompts" / "99-user-overrides.json"
    start = context.Event()
    capabilities = [
        "capability.debugging",
        "capability.testing",
        "capability.review",
        "capability.security",
    ]
    workers = [
        context.Process(
            target=_write_prompt_override_in_process,
            args=(str(override), capability, start),
        )
        for capability in capabilities
    ]

    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=10)

    assert [worker.exitcode for worker in workers] == [0] * len(workers)
    document = json.loads(override.read_text(encoding="utf-8"))
    assert set(document["develop"]) == set(capabilities)


def test_prompts_command_reports_invalid_catalog_when_listing(monkeypatch, tmp_path) -> None:
    _use_temporary_profiles(monkeypatch, tmp_path)
    catalog = tmp_path / "user-prompts"
    catalog.mkdir()
    (catalog / "broken.json").write_text("[]", encoding="utf-8")

    result = commands.handle_prompts_command(["/prompts"])

    assert result.startswith("Prompt configuration error:")
    assert "broken.json" in result


def test_classic_prompts_command_uses_shared_handler(monkeypatch, capsys) -> None:
    monkeypatch.setattr(commands, "handle_prompts_command", lambda parts: "prompt report")

    assert handle_classic_command("/prompts") is True
    assert capsys.readouterr().out.strip() == "prompt report"


def test_tui_prompts_command_uses_shared_handler(monkeypatch) -> None:
    messages: list[tuple[str, str, str]] = []
    app = type("App", (), {"add_message": lambda self, *args: messages.append(args)})()
    monkeypatch.setattr(commands, "handle_prompts_command", lambda parts: "prompt report")

    handle_tui_command(app, "/prompts")

    assert messages == [("System", "prompt report", "system")]
    assert any(command == "/prompts" for command, _description in COMMANDS)


def test_prompts_discovery_commands_are_visible_in_help_and_autocomplete(capsys) -> None:
    assert handle_classic_command("/help") is True
    classic_help = capsys.readouterr().out
    assert "/prompts search <term>" in classic_help
    assert "/prompts show <name>" in classic_help
    assert "/prompts reset <name>" in classic_help

    messages: list[tuple[str, str, str]] = []
    app = type("App", (), {"add_message": lambda self, *args: messages.append(args)})()
    handle_tui_command(app, "/help")

    tui_help = messages[0][1]
    assert "/prompts search <term>" in tui_help
    assert "/prompts show <name>" in tui_help
    assert "/prompts reset <name>" in tui_help
    assert any(command == "/prompts search" for command, _description in COMMANDS)
    assert any(command == "/prompts show" for command, _description in COMMANDS)
    assert any(command == "/prompts reset" for command, _description in COMMANDS)
