"""Tests for the principal-facing ``consult_assistant`` tool and the
underlying ``AssistantCritic.consult`` method.

The tool is a thin bridge: validate input, find the active critic via
``get_active_critic``, call ``critic.consult``, return the answer (or
a clear diagnostic on failure). The harder logic lives inside
``consult()`` — its read sub-loop is exercised here.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from infinidev.engine.loop import critic as critic_mod
from infinidev.engine.loop.critic import (
    AssistantCritic,
    CONSULT_ASSISTANT_SCHEMA,
    get_active_critic,
    set_active_critic,
)
from infinidev.tools.consult_assistant_tool import (
    ConsultAssistantInput,
    ConsultAssistantTool,
)


# --- Helpers ----------------------------------------------------------------


def _text_response(content: str):
    msg = SimpleNamespace(content=content, tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _read_response(tool_name: str, args_json: str = "{}", call_id: str = "t1"):
    fc = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=tool_name, arguments=args_json),
    )
    msg = SimpleNamespace(content="", tool_calls=[fc])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


@pytest.fixture
def critic(monkeypatch):
    monkeypatch.setattr(
        critic_mod,
        "get_litellm_params_for_assistant",
        lambda: {"model": "ollama/test-critic"},
    )
    return AssistantCritic(tool_descriptions={"read_file": "Read a file"})


@pytest.fixture(autouse=True)
def _clear_active_critic():
    """Reset the global registry between tests so they don't interfere."""
    set_active_critic(None)
    yield
    set_active_critic(None)


# --- Schema and input validation -------------------------------------------


def test_schema_has_required_question_field():
    schema = CONSULT_ASSISTANT_SCHEMA["function"]["parameters"]
    assert schema["required"] == ["question"]
    assert schema["properties"]["question"]["minLength"] == 20


def test_input_validates_min_length():
    with pytest.raises(ValidationError):
        ConsultAssistantInput(question="too short")


def test_input_accepts_valid_question():
    inp = ConsultAssistantInput(
        question="Should I use snake_case or camelCase for new field names?"
    )
    assert inp.context_hint == ""


# --- AssistantCritic.consult -----------------------------------------------


def test_consult_returns_text_answer(critic):
    """Single-turn consult: model returns text, that's the answer."""
    with patch(
        "infinidev.engine.llm_client.call_llm",
        lambda *a, **kw: _text_response("Use snake_case — see CLAUDE.md."),
    ):
        ans = critic.consult("Should I use snake_case for new fields?")
    assert ans == "Use snake_case — see CLAUDE.md."


def test_consult_empty_question_short_circuits(critic):
    """An empty question never reaches the LLM."""
    ans = critic.consult("")
    assert ans.startswith("[consult-error]")


def test_consult_handles_llm_failure(critic):
    def boom(*a, **kw):
        raise RuntimeError("network down")
    with patch("infinidev.engine.llm_client.call_llm", boom):
        ans = critic.consult("Test question that is long enough to validate.")
    assert "consult-error" in ans
    assert "network down" in ans


def test_consult_with_read_call_then_text(critic, monkeypatch):
    """Consult uses one read tool, then answers with text."""
    monkeypatch.setattr(
        critic, "_execute_read_tool", lambda n, a: "stub-read-result"
    )
    responses = iter([
        _read_response("read_file", '{"file_path": "x.py"}'),
        _text_response("Looked at x.py — here's the answer."),
    ])
    with patch(
        "infinidev.engine.llm_client.call_llm",
        lambda *a, **kw: next(responses),
    ):
        ans = critic.consult("What does foo do in x.py?")
    assert ans == "Looked at x.py — here's the answer."


def test_consult_subloop_exhaustion(critic, monkeypatch):
    """If the model only ever calls reads and never answers, exhausting
    the loop returns a clear error string, not None or an exception."""
    monkeypatch.setattr(critic, "_execute_read_tool", lambda n, a: "stub")
    with patch(
        "infinidev.engine.llm_client.call_llm",
        lambda *a, **kw: _read_response("read_file", '{"file_path": "x.py"}'),
    ):
        ans = critic.consult("Some question we'll never get an answer to.")
    assert ans.startswith("[consult-error]")


# --- ConsultAssistantTool ---------------------------------------------------


def test_tool_returns_diagnostic_when_no_critic_active():
    """No critic registered → tool returns a clear error, not raises."""
    set_active_critic(None)
    tool = ConsultAssistantTool()
    out = tool._run(question="What does this function do exactly?")
    assert out.startswith("[consult-error]")
    assert "No assistant critic" in out


def test_tool_routes_question_to_active_critic(critic, monkeypatch):
    """When a critic is registered, the tool routes the question and
    returns the critic's answer verbatim."""
    set_active_critic(critic)
    monkeypatch.setattr(
        critic, "consult",
        lambda question, *, context_hint="", principal_messages=None: f"answered: {question[:40]}",
    )
    tool = ConsultAssistantTool()
    out = tool._run(question="Should we keep the cache layer behind a flag?")
    assert out.startswith("answered: Should we keep the cache layer behind a")


def test_tool_handles_critic_exception(critic, monkeypatch):
    """If consult() raises, the tool returns a diagnostic — never propagates."""
    set_active_critic(critic)
    def boom(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(critic, "consult", boom)
    tool = ConsultAssistantTool()
    out = tool._run(question="Will this approach handle concurrent writes?")
    assert out.startswith("[consult-error]")
    assert "boom" in out


def test_tool_handles_empty_critic_answer(critic, monkeypatch):
    """If the critic returns empty / whitespace, the tool says so explicitly."""
    set_active_critic(critic)
    monkeypatch.setattr(
        critic, "consult",
        lambda *a, **kw: "   ",
    )
    tool = ConsultAssistantTool()
    out = tool._run(question="A reasonably long question for validation.")
    assert "empty answer" in out


# --- set_active_critic / get_active_critic registry ------------------------


def test_active_critic_registry_round_trip():
    assert get_active_critic() is None
    sentinel = object()
    set_active_critic(sentinel)  # type: ignore[arg-type]
    assert get_active_critic() is sentinel
    set_active_critic(None)
    assert get_active_critic() is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
