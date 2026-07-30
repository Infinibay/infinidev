"""``AGENTS.md`` — the project's standing instructions to the agent.

The file is the one place a user can say "in this repo, do it *this* way"
without editing infinidev's own prompts, so the tests here are mostly about
the ways that promise breaks: instructions that never reach the model,
instructions that reach it stale, and instructions large enough to spend the
context window before the agent has done anything.
"""

from __future__ import annotations

import pytest

from infinidev.prompts.project_instructions import (
    MAX_CHARS,
    load_project_instructions,
    render_project_instructions,
)


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


# ── discovery ────────────────────────────────────────────────────────────


def test_a_project_without_instructions_adds_nothing(workspace):
    assert load_project_instructions(workspace) is None
    assert render_project_instructions(workspace) is None


def test_agents_md_is_read(workspace):
    (workspace / "AGENTS.md").write_text("Use pnpm, never npm.")
    assert load_project_instructions(workspace) == "Use pnpm, never npm."


def test_claude_md_is_read_when_that_is_what_the_project_has(workspace):
    (workspace / "CLAUDE.md").write_text("Run the suite before committing.")
    assert "Run the suite" in load_project_instructions(workspace)


def test_agents_md_wins_when_a_project_carries_both(workspace):
    """A repo with both is saying AGENTS.md is the one written for any agent
    and CLAUDE.md the one written for a specific host."""
    (workspace / "AGENTS.md").write_text("the agent file")
    (workspace / "CLAUDE.md").write_text("the claude file")
    assert load_project_instructions(workspace) == "the agent file"


def test_an_empty_file_is_treated_as_no_instructions(workspace):
    (workspace / "AGENTS.md").write_text("   \n\n  ")
    assert load_project_instructions(workspace) is None


# ── it must not be able to break a run ───────────────────────────────────


def test_a_huge_file_is_capped(workspace):
    """This lands in the cacheable prefix of every request for the whole
    session. Unbounded, a project could spend the context window before the
    agent does anything."""
    (workspace / "AGENTS.md").write_text("x" * (MAX_CHARS * 3))
    out = load_project_instructions(workspace)
    assert len(out) <= MAX_CHARS + 200          # the cap, plus its own notice
    assert "truncated here" in out


def test_truncation_says_how_much_was_dropped(workspace):
    """Silently sending half the instructions is worse than saying so."""
    body = ("a paragraph of project rules. " * 20 + "\n\n") * 40
    (workspace / "AGENTS.md").write_text(body)
    out = load_project_instructions(workspace)
    assert "more characters" in out


def test_truncation_breaks_on_a_paragraph(workspace):
    (workspace / "AGENTS.md").write_text(("rule text here.\n\n") * 2000)
    out = load_project_instructions(workspace)
    body = out.split("[AGENTS.md truncated")[0]
    assert body.rstrip().endswith(".")


def test_undecodable_bytes_do_not_raise(workspace):
    (workspace / "AGENTS.md").write_bytes(b"valid \xff\xfe then more")
    assert load_project_instructions(workspace) is not None


def test_a_directory_named_agents_md_is_not_read(workspace):
    (workspace / "AGENTS.md").mkdir()
    assert load_project_instructions(workspace) is None


def test_no_workspace_means_no_instructions():
    assert load_project_instructions(None) is None


# ── it has to reach the model ────────────────────────────────────────────


def test_the_block_names_its_source_and_its_authority(workspace):
    (workspace / "AGENTS.md").write_text("Never force-push.")
    block = render_project_instructions(workspace)
    assert 'source="AGENTS.md"' in block
    assert "Never force-push." in block
    # The model has to be able to tell the project's words from ours.
    assert block.startswith("<project-instructions")
    assert block.rstrip().endswith("</project-instructions>")


def test_the_developer_prompt_carries_them(workspace):
    from infinidev.engine.loop.context import build_system_prompt

    (workspace / "AGENTS.md").write_text("Never force-push.")
    prompt = build_system_prompt("backstory", workspace_path=str(workspace))
    assert "Never force-push." in prompt


def test_they_sit_in_the_cacheable_prefix(workspace):
    """Stable for the whole session, so paying to re-send it every iteration
    would be waste — it belongs above the breakpoint, with the identity."""
    from infinidev.engine.loop.context import CACHE_BREAKPOINT_MARKER, build_system_prompt

    (workspace / "AGENTS.md").write_text("Never force-push.")
    prompt = build_system_prompt(
        "backstory", workspace_path=str(workspace),
        session_summaries=["did a thing"],
    )
    assert prompt.index("Never force-push.") < prompt.index(CACHE_BREAKPOINT_MARKER)


def test_the_chat_agent_gets_them_without_being_handed_a_path(workspace, monkeypatch):
    """It answers questions about this project and decides what to escalate,
    so the project's instructions apply to it too — resolved from the bound
    tool context rather than threaded through every call site."""
    from infinidev.prompts.chat_agent import build_chat_agent_system_prompt
    from infinidev.tools.base.context import clear_agent_context, set_context

    (workspace / "AGENTS.md").write_text("Never force-push.")
    set_context(
        project_id=1, agent_id="test-agent", session_id="s",
        workspace_path=str(workspace),
    )
    try:
        assert "Never force-push." in build_chat_agent_system_prompt()
    finally:
        clear_agent_context("test-agent")


def test_edits_take_effect_without_a_restart(workspace):
    """The file is a live control surface, like settings.json — reading it
    once at import would make editing it look broken."""
    (workspace / "AGENTS.md").write_text("first rule")
    assert "first rule" in load_project_instructions(workspace)
    (workspace / "AGENTS.md").write_text("second rule")
    assert "second rule" in load_project_instructions(workspace)
