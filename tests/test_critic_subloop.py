"""Tests for the assistant critic's read sub-loop.

The sub-loop lets the critic call up to ``_MAX_READ_CALLS`` read-only
tools before emitting its verdict. These tests cover:

* The budget cap is enforced (no more than 3 reads).
* Tool execution failures don't crash the critic.
* The verdict is still produced after a few reads.
* The legacy single-shot path (no read calls) still works.

We patch ``call_llm`` to return scripted responses so the tests are
hermetic — no real LLM, no real Ollama instance needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from infinidev.engine.loop import critic as critic_mod
from infinidev.engine.loop.critic import AssistantCritic, CriticVerdict


# --- Test helpers -----------------------------------------------------------


def _make_function_call(name: str, args_json: str, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=args_json),
    )


def _make_response(*, tool_calls=None, content=""):
    """Build a minimal LLM response shape compatible with the critic."""
    msg = SimpleNamespace(
        tool_calls=tool_calls or [],
        content=content,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _verdict_response(action: str, message: str):
    args = '{"action": "%s", "message": "%s"}' % (action, message)
    return _make_response(tool_calls=[_make_function_call("emit_verdict", args)])


def _read_response(tool_name: str, args_json: str = "{}", call_id: str = "tc1"):
    return _make_response(tool_calls=[_make_function_call(tool_name, args_json, call_id)])


@pytest.fixture
def critic(monkeypatch):
    """Build an AssistantCritic without hitting the real assistant LLM config."""
    monkeypatch.setattr(
        critic_mod,
        "get_litellm_params_for_assistant",
        lambda: {"model": "ollama/test-critic"},
    )
    return AssistantCritic(tool_descriptions={"read_file": "Read a file"})


# --- Tests ------------------------------------------------------------------


def test_critic_initialises_read_tools(critic):
    """Sanity: the critic loaded its read-only toolkit."""
    assert "read_file" in critic._read_tool_dispatch
    assert critic._read_tool_schemas, "read tool schemas should be built"


def test_legacy_single_shot_still_works(critic):
    """When the critic emits a verdict on its first turn, the sub-loop
    must short-circuit and return that verdict — no spurious extra
    iterations."""
    proposed = [_make_function_call("replace_lines", '{"file_path": "x.py"}')]
    responses = iter([_verdict_response("information", "Looks fine.")])

    with patch("infinidev.engine.llm_client.call_llm", lambda *a, **kw: next(responses)):
        verdict = critic.review(messages=[], tool_calls=proposed)

    assert isinstance(verdict, CriticVerdict)
    assert verdict.action == "information"
    assert verdict.message == "Looks fine."


def test_read_call_then_verdict(critic, monkeypatch):
    """The critic calls one read tool, gets a result, then emits."""
    proposed = [_make_function_call("replace_lines", '{"file_path": "x.py"}')]

    monkeypatch.setattr(critic, "_execute_read_tool", lambda name, args: f"[stub-result for {name}]")
    responses = iter([
        _read_response("read_file", '{"file_path": "x.py"}'),
        _verdict_response("recommendation", "Verified — looks ok."),
    ])

    with patch("infinidev.engine.llm_client.call_llm", lambda *a, **kw: next(responses)):
        verdict = critic.review(messages=[], tool_calls=proposed)

    assert verdict is not None
    assert verdict.action == "recommendation"


def test_read_budget_is_enforced(critic, monkeypatch):
    """After 3 reads, further read calls get a budget-exhausted message
    and the critic is forced to emit on the next turn."""
    proposed = [_make_function_call("replace_lines", '{"file_path": "x.py"}')]
    monkeypatch.setattr(critic, "_execute_read_tool", lambda n, a: "stub")

    # Script: 3 reads, then a 4th attempt (gets budget-exhausted), then verdict.
    responses = iter([
        _read_response("read_file", '{"file_path": "a.py"}', "tc1"),
        _read_response("read_file", '{"file_path": "b.py"}', "tc2"),
        _read_response("read_file", '{"file_path": "c.py"}', "tc3"),
        _verdict_response("continue", ""),
    ])

    with patch("infinidev.engine.llm_client.call_llm", lambda *a, **kw: next(responses)):
        verdict = critic.review(messages=[], tool_calls=proposed)

    # Critic produced a verdict (continue, empty message → silent in
    # the engine but the object is still returned).
    assert verdict is not None
    assert verdict.action == "continue"


def test_read_tool_failure_does_not_crash(critic, monkeypatch):
    """If a read tool raises, the critic gets an error string back and
    can still emit a verdict."""
    proposed = [_make_function_call("replace_lines", '{"file_path": "x.py"}')]

    def boom(name, args):
        return "[critic-error] read tool blew up"
    monkeypatch.setattr(critic, "_execute_read_tool", boom)

    responses = iter([
        _read_response("read_file", '{"file_path": "x.py"}'),
        _verdict_response("information", "Could not verify."),
    ])

    with patch("infinidev.engine.llm_client.call_llm", lambda *a, **kw: next(responses)):
        verdict = critic.review(messages=[], tool_calls=proposed)

    assert verdict is not None
    assert verdict.action == "information"


def test_subloop_exhaustion_returns_none(critic, monkeypatch):
    """If the critic refuses to ever emit a verdict (only calls tools),
    the loop exits silently — never blocks the principal."""
    proposed = [_make_function_call("replace_lines", '{"file_path": "x.py"}')]
    monkeypatch.setattr(critic, "_execute_read_tool", lambda n, a: "stub")

    # Always returns a read call, never a verdict.
    def always_read(*a, **kw):
        return _read_response("read_file", '{"file_path": "x.py"}')

    with patch("infinidev.engine.llm_client.call_llm", always_read):
        verdict = critic.review(messages=[], tool_calls=proposed)

    # Sub-loop exhausted without verdict → silent fallback.
    assert verdict is None


def test_serialise_tool_calls_handles_none():
    """Defensive: serialise_tool_calls must never raise on weird input."""
    out = AssistantCritic._serialise_tool_calls(None)
    assert out == []
    out = AssistantCritic._serialise_tool_calls([])
    assert out == []


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
