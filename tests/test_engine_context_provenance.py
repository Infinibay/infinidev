"""Resume must preserve instructions and identify historical observations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine.loop.context import _render_opened_files
from infinidev.engine.loop.context_builder import _restore_or_start
from infinidev.engine.loop.loop_state import LoopState
from infinidev.engine.loop.resume_checkpoint import (
    CHECKPOINT_MAX_UTF8_BYTES,
    build_loop_resume_checkpoint,
    resume_state_for_task,
)
from infinidev.engine.loop.tool_runner import ToolRunner
from infinidev.engine.loop.user_guidance import UserGuidance


def test_essential_checkpoint_keeps_literal_guidance_without_clipping():
    instruction = "Keep every tenant's original identifier. " * 200
    state = LoopState(
        user_guidance=[UserGuidance(text=instruction, attachment_paths=("schema.png",))],
        read_delivery_revisions={str(index): "x" * 100_000 for index in range(30)},
    )

    checkpoint = build_loop_resume_checkpoint(state, ("Implement tenants", "Verified"))
    restored = LoopState.model_validate(checkpoint["state"])

    assert restored.user_guidance == state.user_guidance
    assert restored.user_guidance[0].text == instruction
    assert "read_delivery_revisions" not in checkpoint["state"]


def test_irreducible_guidance_rejects_checkpoint_instead_of_dropping_instructions():
    state = LoopState(user_guidance=[
        UserGuidance(text="User scope: " + "x" * (CHECKPOINT_MAX_UTF8_BYTES + 1)),
    ])

    with pytest.raises(ValueError, match="checkpoint.*limit"):
        build_loop_resume_checkpoint(state, ("Implement tenants", "Verified"))


def test_guidance_roundtrip_is_task_bound_and_old_states_still_load():
    prompt = ("Implement tenants", "Verified")
    state = LoopState(user_guidance=[UserGuidance(text="Keep legacy identifiers.")])
    checkpoint = build_loop_resume_checkpoint(state, prompt)

    restored = _restore_or_start(resume_state_for_task(checkpoint, prompt))

    assert restored.user_guidance == state.user_guidance
    assert resume_state_for_task(checkpoint, ("Different task", "Verified")) is None
    assert _restore_or_start({"notes": ["old format"]}).user_guidance == []


def test_restored_cache_retains_bodies_and_marks_all_observations_for_revalidation():
    state = LoopState()
    state.cache_file("src/tenant.py", "old source body", pinned=True)
    state.cache_file("[symbol] Tenant", "old symbol body")
    state.cache_file("[search] tenant", "old search result")

    restored = _restore_or_start(state.model_dump(mode="json"))
    rendered = _render_opened_files(restored)

    assert restored.opened_files.keys() == state.opened_files.keys()
    for key, opened in restored.opened_files.items():
        assert opened.content == state.opened_files[key].content
        assert opened.ttl == state.opened_files[key].ttl
        assert opened.pinned == state.opened_files[key].pinned
        assert opened.requires_revalidation is True
        assert state.opened_files[key].requires_revalidation is False
    assert "old source body" in rendered
    assert "requires revalidation" in rendered
    assert "up-to-date" not in rendered
    assert "Do NOT call read_file" not in rendered


@pytest.mark.parametrize("method", ["cache_file", "refresh_file"])
def test_actual_cache_update_clears_only_its_own_revalidation_marker(method):
    state = LoopState()
    state.cache_file("updated.py", "before")
    state.cache_file("untouched.py", "other observation")
    restored = _restore_or_start(state.model_dump(mode="json"))

    getattr(restored, method)("updated.py", "current source")

    assert restored.opened_files["updated.py"].content == "current source"
    assert restored.opened_files["updated.py"].requires_revalidation is False
    assert restored.opened_files["untouched.py"].requires_revalidation is True


def test_real_shell_cache_refresh_revalidates_the_read_source(tmp_path):
    path = tmp_path / "tenant.py"
    path.write_text("CURRENT = 1\n")
    state = LoopState()
    state.cache_file(str(path), "stale checkpoint content", pinned=True)
    restored = _restore_or_start(state.model_dump(mode="json"))
    ctx = SimpleNamespace(state=restored, workspace_path=str(tmp_path))

    ToolRunner._refresh_opened_files_after_shell(ctx, {"command": "generate tenant.py"})

    assert "CURRENT = 1" in restored.opened_files[str(path)].content
    assert restored.opened_files[str(path)].requires_revalidation is False


def test_shell_cache_refresh_resolves_relative_sources_in_the_agent_workspace(
    tmp_path, monkeypatch,
):
    workspace = tmp_path / "target"
    workspace.mkdir()
    (workspace / "tenant.py").write_text("RIGHT_WORKSPACE = 1\n")
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    (other_workspace / "tenant.py").write_text("WRONG_WORKSPACE = 1\n")
    monkeypatch.chdir(other_workspace)
    state = LoopState()
    state.cache_file("tenant.py", "old snapshot", pinned=True)
    ctx = SimpleNamespace(state=state, agent=SimpleNamespace(workspace_path=str(workspace)))

    ToolRunner._refresh_opened_files_after_shell(ctx, {"command": "generate tenant.py"})

    assert "RIGHT_WORKSPACE" in state.opened_files["tenant.py"].content
    assert "WRONG_WORKSPACE" not in state.opened_files["tenant.py"].content
