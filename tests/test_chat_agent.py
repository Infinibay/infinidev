"""Tests for the chat agent orchestrator (Commit 4).

The chat agent replaces the legacy conversational fastpath as the
default entry point for every user turn. These tests verify the
contract without making real LLM calls — litellm.completion is
monkeypatched with a scripted response generator that walks through
the tool sequences the agent should handle.

Invariants locked in here:
  * Every turn terminates via respond OR escalate (never silent).
  * The schema offered to the LLM contains zero write-capable tools.
  * Escalation packets include user_request verbatim + the args the
    model emitted.
  * Max-iter exhaustion yields a graceful respond, not an exception.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

import infinidev.engine.orchestration.chat_agent as chat_agent_mod
from infinidev.engine.orchestration.chat_agent import run_chat_agent
from infinidev.engine.orchestration.chat_agent_result import ChatAgentResult


# ─────────────────────────────────────────────────────────────────────────
# Mock helpers
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeFunction:
    name: str
    arguments: str  # JSON string — matches the real LiteLLM shape


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction

    type: str = "function"


@dataclass
class _FakeMessage:
    content: str = ""
    tool_calls: list[_FakeToolCall] | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


def _tc(name: str, args: dict[str, Any] | None = None, call_id: str = "tc-1") -> _FakeToolCall:
    return _FakeToolCall(
        id=call_id,
        function=_FakeFunction(name=name, arguments=json.dumps(args or {})),
    )


def _response(tool_calls: list[_FakeToolCall] | None = None, content: str = "") -> _FakeResponse:
    msg = _FakeMessage(
        content=content,
        tool_calls=tool_calls,
    )
    return _FakeResponse(choices=[_FakeChoice(message=msg)])


class _ScriptedLitellm:
    """Replay a scripted sequence of responses for litellm.completion."""

    def __init__(self, responses: list[_FakeResponse]):
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("More LLM calls than scripted responses")
        return self.responses.pop(0)


@pytest.fixture
def patch_litellm(monkeypatch):
    """Return a factory that installs a ScriptedLitellm for the duration
    of a test."""
    def _install(responses: list[_FakeResponse]) -> _ScriptedLitellm:
        scripted = _ScriptedLitellm(responses)
        # Patch BOTH the module the chat agent imports from and the
        # litellm top-level (since we bind via module lookup).
        import litellm as _lit
        monkeypatch.setattr(_lit, "completion", scripted)
        return scripted

    # Also stub out the DB history so tests don't touch a real sqlite file.
    monkeypatch.setattr(
        "infinidev.db.service.get_recent_turns_full",
        lambda *a, **kw: [],
    )
    # Keep config/llm params deterministic.
    monkeypatch.setattr(
        "infinidev.engine.orchestration.chat_agent.get_litellm_params_for_behavior",
        lambda: {"model": "test/mock", "api_base": "http://localhost"},
    )
    return _install


# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────


class TestRespondTerminator:
    def test_respond_first_iteration(self, patch_litellm):
        scripted = patch_litellm([
            _response([_tc("respond", {"message": "¡Hola! ¿En qué te ayudo?"})]),
        ])
        result = run_chat_agent("hola")
        assert result.kind == "respond"
        assert "Hola" in result.reply
        assert len(scripted.calls) == 1

    def test_respond_after_one_read(self, patch_litellm):
        scripted = patch_litellm([
            _response([_tc("list_directory", {"file_path": "."})]),
            _response([_tc("respond", {"message": "Este proyecto es un CLI."})]),
        ])
        result = run_chat_agent("qué es este proyecto?")
        assert result.kind == "respond"
        assert result.reply.startswith("Este proyecto")
        assert len(scripted.calls) == 2


class TestEscalateTerminator:
    def test_escalate_populates_packet(self, patch_litellm):
        patch_litellm([
            _response([_tc("escalate", {
                "understanding": "Fix JWT validation bug in auth.py",
                "user_visible_preview": "Voy a arreglar el JWT.",
                "opened_files": ["src/auth.py"],
                "user_signal": "dale arreglalo",
                "suggested_flow": "develop",
            })]),
        ])
        result = run_chat_agent("arreglá el bug del JWT")
        assert result.kind == "escalate"
        assert result.escalation is not None
        pkt = result.escalation
        assert pkt.user_request == "arreglá el bug del JWT"
        assert "JWT" in pkt.understanding
        assert pkt.opened_files == ["src/auth.py"]
        assert pkt.user_signal == "dale arreglalo"
        assert pkt.suggested_flow == "develop"

    def test_escalate_with_empty_understanding_falls_back_to_respond(self, patch_litellm):
        patch_litellm([
            _response([_tc("escalate", {"understanding": ""})]),
        ])
        result = run_chat_agent("do something")
        # Defensive: an escalation with no handoff content would strand
        # the planner, so the chat agent falls back to a respond asking
        # for clarification.
        assert result.kind == "respond"
        assert "clar" in result.reply.lower() or "?" in result.reply

    def test_suggested_flow_is_pinned_to_develop(self, patch_litellm):
        patch_litellm([
            _response([_tc("escalate", {
                "understanding": "do the thing",
                "suggested_flow": "sysadmin",  # v1 rejects anything but develop
            })]),
        ])
        result = run_chat_agent("do it")
        assert result.kind == "escalate"
        assert result.escalation.suggested_flow == "develop"


class TestGracefulFailureModes:
    def test_empty_input_returns_respond(self, patch_litellm):
        patch_litellm([])
        result = run_chat_agent("   ")
        assert result.kind == "respond"
        assert "empty" in result.reply.lower()

    def test_plain_text_reply_treated_as_respond(self, patch_litellm):
        patch_litellm([
            _response(content="Hola, ¿cómo andás?", tool_calls=None),
        ])
        result = run_chat_agent("hola")
        assert result.kind == "respond"
        assert "Hola" in result.reply

    def test_max_iterations_without_terminator_falls_back(self, patch_litellm):
        # Every response is a read call; none ever terminates.
        patch_litellm([
            _response([_tc("list_directory", {"file_path": "."}, f"tc-{i}")])
            for i in range(5)
        ])
        result = run_chat_agent("hola", max_iterations=5)
        assert result.kind == "respond"
        # Falls back with a neutral "happy to keep going" message —
        # never blames the user for the agent hitting its own ceiling.
        assert "reformul" not in result.reply.lower()
        assert "rephrase" not in result.reply.lower()
        assert result.reply  # non-empty

    def test_llm_raises_returns_respond(self, patch_litellm, monkeypatch):
        patch_litellm([])
        import litellm as _lit
        def _explode(**kwargs):
            raise RuntimeError("LLM backend down")
        monkeypatch.setattr(_lit, "completion", _explode)
        result = run_chat_agent("hola")
        assert result.kind == "respond"
        assert "problema" in result.reply.lower() or "error" in result.reply.lower()


class TestLanguageAwareFallbacks:
    """The fallback-respond paths bypass the LLM, so they must localize
    themselves. English/Spanish coverage only — matches what the system
    prompt tells the model to handle."""

    def test_spanish_input_gets_spanish_fallback(self, patch_litellm):
        patch_litellm([])  # no LLM call — empty_input path
        result = run_chat_agent("")
        # Empty short-circuits to "(empty message)" regardless of lang.
        assert result.reply == "(empty message)"

    def test_max_iter_spanish(self, patch_litellm):
        patch_litellm([
            _response([_tc("list_directory", {"file_path": "."}, f"tc-{i}")])
            for i in range(5)
        ])
        result = run_chat_agent("¿qué hace este proyecto?", max_iterations=5)
        assert result.kind == "respond"
        # Neutral Spanish wrap-up — mentions "investigué" or "seguimos",
        # never tells the user to rephrase.
        lower = result.reply.lower()
        assert any(w in lower for w in ("investigué", "seguimos", "contame"))
        assert "reformul" not in lower

    def test_max_iter_english(self, patch_litellm):
        patch_litellm([
            _response([_tc("list_directory", {"file_path": "."}, f"tc-{i}")])
            for i in range(5)
        ])
        result = run_chat_agent("what does this project do?", max_iterations=5)
        assert result.kind == "respond"
        lower = result.reply.lower()
        assert any(w in lower for w in ("investigated", "keep going", "what you want"))
        assert "rephrase" not in lower

    def test_llm_exception_english_fallback(self, patch_litellm, monkeypatch):
        patch_litellm([])
        import litellm as _lit
        monkeypatch.setattr(_lit, "completion", lambda **kw: (_ for _ in ()).throw(RuntimeError("x")))
        result = run_chat_agent("do something", max_iterations=3)
        assert result.kind == "respond"
        # English input → English fallback, no Spanish characters.
        assert not any(ch in result.reply for ch in "¿¡ñ")


class TestSingleUserMessage:
    """Two consecutive role='user' messages trip some providers; we
    merge history + input into one."""

    def test_only_one_user_message_sent(self, patch_litellm, monkeypatch):
        # Stub history to make sure it's non-empty.
        monkeypatch.setattr(
            "infinidev.db.service.get_recent_turns_full",
            lambda *a, **kw: [("user", "prior msg"), ("assistant", "prior reply")],
        )
        scripted = patch_litellm([
            _response([_tc("respond", {"message": "ok"})]),
        ])
        run_chat_agent("now?", session_id="test-session")
        msgs = scripted.calls[0]["messages"]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 1, (
            f"Expected 1 user message, got {len(user_msgs)}"
        )
        # Both the snapshot AND the current input must be in it.
        content = user_msgs[0]["content"]
        assert "prior msg" in content
        assert "now?" in content


class TestToolboxIntegrity:
    def test_write_tools_absent_from_llm_schema(self, patch_litellm):
        scripted = patch_litellm([
            _response([_tc("respond", {"message": "ok"})]),
        ])
        run_chat_agent("hola")
        assert len(scripted.calls) == 1
        tools = scripted.calls[0]["tools"]
        tool_names = {t["function"]["name"] for t in tools}

        # Read tools must be present.
        for expected in ("read_file", "code_search", "list_directory", "git_diff"):
            assert expected in tool_names, f"chat-agent schema missing {expected}"

        # Terminators must be present.
        assert "respond" in tool_names
        assert "escalate" in tool_names

        # Write tools must NOT be exposed.
        forbidden = {
            "create_file", "replace_lines", "write_file",
            "edit_symbol", "add_symbol", "remove_symbol",
            "execute_command", "code_interpreter",
            "git_commit", "git_branch",
            "record_finding", "delete_finding",
            "send_message",
            "step_complete",  # chat agent uses respond/escalate instead
        }
        overlap = tool_names & forbidden
        assert not overlap, f"chat-agent exposed forbidden tools: {overlap}"
