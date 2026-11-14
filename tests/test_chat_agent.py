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
from infinidev.config.settings import settings
from infinidev.engine.orchestration.chat_agent import (
    run_chat_agent,
)
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
    content: Any = ""
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


def _response(tool_calls: list[_FakeToolCall] | None = None, content: Any = "") -> _FakeResponse:
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


@pytest.fixture(autouse=True)
def stub_semantic_routing(monkeypatch):
    monkeypatch.setattr(chat_agent_mod, "classify_chat_response", lambda *a, **kw: None)


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
    @pytest.mark.parametrize("user_message", [
        "¿Cómo puedo crear app.py y ejecutar tests?",
        "¿Cómo crear app.py y ejecutar tests?",
        "¿Me explicás cómo crear app.py y ejecutar tests?",
        "¿Podés explicarme cómo crear app.py y ejecutar tests?",
        "Could you explain how to create app.py and run tests?",
        "Can you show me how to create app.py and run tests?",
        "Tell me how to fix app.py and run tests.",
        "Explícame cómo crear app.py y ejecutar tests.",
        "¿Qué significa 'create app.py and run tests'?",
        'The example says "create app.py and run tests". What does it mean?',
        "The example says 'Don't create app.py and run tests'. What does it mean?",
        "The example says ‘Don’t create app.py and run tests’. What does it mean?",
        "What does `create app.py and run tests` mean?",
        "What does 'Don't create app.py and run tests' mean?",
        "Continue explaining PLAN.md without changing any files.",
        "Continua explicando PLAN.md, no modifiques nada.",
        "Resume el contenido de PLAN.md.",
        "Puedes leer Continuar.md?",
        "Lee CONTINUE.md y dime el estado actual.",
        "Read run.py and tests.py.",
        "Can you read create.py and explain the tests?",
    ])
    def test_explanation_is_answered_without_starting_work(self, user_message, patch_litellm):
        scripted = patch_litellm([_response(content="Here is the explanation.")])

        result = run_chat_agent(user_message)

        assert result.kind == "respond"
        assert result.reply == "Here is the explanation."
        assert len(scripted.calls) == 1

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
    @pytest.mark.parametrize("user_message", [
        "Could you create app.py and run tests?",
        "¿Podés crear app.py y ejecutar tests?",
        "Fix `app.py` and run tests.",
        'Set the name to "create app.py and run tests" in config.json and run tests.',
        "Run tests.py and report the results.",
        "Fix create.py and run tests.",
    ])
    def test_polite_or_quoted_target_change_still_starts_work(
        self, user_message, patch_litellm,
    ):
        scripted = patch_litellm([
            _response([_tc("escalate", {"understanding": "Do requested work"})]),
        ])

        result = run_chat_agent(user_message)

        assert result.kind == "escalate"
        assert len(scripted.calls) == 1

    def test_explicit_repo_change_uses_primary_llm_router(self, patch_litellm):
        scripted = patch_litellm([
            _response([_tc("escalate", {"understanding": "Do requested work"})]),
        ])

        result = run_chat_agent(
            "In this repository, add src/demo.py and run focused tests."
        )

        assert result.kind == "escalate"
        assert result.escalation is not None
        assert len(scripted.calls) == 1

    def test_detailed_bugfix_uses_primary_llm_router(self, patch_litellm):
        scripted = patch_litellm([
            _response([_tc("escalate", {"understanding": "Do requested work"})]),
        ])

        result = run_chat_agent(
            "Fix pages_needed so a partial final page is included, exact multiples stay "
            "unchanged, zero items need zero pages, and invalid page sizes keep raising "
            "ValueError. Change only the implementation and run the relevant tests."
        )

        assert result.kind == "escalate"
        assert result.escalation is not None
        assert len(scripted.calls) == 1

    def test_literal_bugfix_uses_primary_llm_router(self, patch_litellm):
        scripted = patch_litellm([
            _response([_tc("escalate", {"understanding": "Do requested work"})]),
        ])

        result = run_chat_agent(
            "Fix retry scheduling so Retry-After accepts both delta-seconds and HTTP "
            "dates relative to the injected clock, past dates clamp to zero, and "
            "exhausted requests never sleep after the final attempt. Preserve the "
            "public API and exponential fallback, change only implementation files, "
            "do not edit tests, and run the full suite."
        )

        assert result.kind == "escalate"
        assert result.escalation is not None
        assert len(scripted.calls) == 1

    def test_referenced_continuation_uses_primary_llm_router(self, patch_litellm):
        scripted = patch_litellm([
            _response([_tc("escalate", {"understanding": "Do requested work"})]),
        ])

        result = run_chat_agent(
            "Lee infinigpu/CONTINUE.md y continua el trabajo"
        )

        assert result.kind == "escalate"
        assert result.escalation is not None
        assert len(scripted.calls) == 1

    def test_escalate_populates_packet(self, patch_litellm, monkeypatch):
        monkeypatch.setattr(settings, "TASK_POLICIES_ENABLED", False)
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
        # If recovery is unavailable, describe the failed handoff rather
        # than claiming the user's request was ambiguous.
        assert result.kind == "respond"
        assert "handoff" in result.reply.lower()
        assert "isn't clear" not in result.reply

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
    @pytest.mark.parametrize("answer", [
        "I'm escalating to the developer to apply the fix.",
        "Okay, I'm escalating to the developer to apply the fix.",
        "Sure, I'll hand off to the developer to apply the fix.",
        "I’m escalating to the developer to apply the fix.",
        "I'll hand off to the developer to apply the fix.",
        "Voy a escalar al desarrollador para aplicar el cambio.",
    ])
    def test_actual_first_person_handoff_is_preserved(self, answer, patch_litellm, monkeypatch):
        from infinidev.engine.orchestration.semantic_routing import RoutingDecision
        monkeypatch.setattr(
            chat_agent_mod, "classify_chat_response",
            lambda *a, **kw: RoutingDecision(kind="escalate", understanding="Apply the fix"),
        )
        patch_litellm([_response(content=answer)])

        result = run_chat_agent("Necesito ayuda con esto.")

        assert result.kind == "escalate"
        assert "semantic recovery" in result.escalation.user_signal

    @pytest.mark.parametrize("answer", [
        'Para explicarlo: "I am escalating to the developer" significa delegar.',
        "No voy a escalar al desarrollador. Solo voy a explicar el diseño.",
        "The documentation says handing off to the developer is an optional step.",
        "Handing off to the developer does not mean I will change files.",
        "I am not escalating to the developer; this is only an explanation.",
        "Example:\n```\nI'm escalating to the developer.\n```\nThat is a handoff.",
    ])
    def test_explained_or_negated_handoff_is_only_an_answer(self, answer, patch_litellm):
        scripted = patch_litellm([_response(content=answer)])

        result = run_chat_agent("Explain how agent handoffs work.")

        assert result.kind == "respond"
        assert result.reply == answer
        assert len(scripted.calls) == 1

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

    def test_structured_text_reply_is_normalized(self, patch_litellm):
        patch_litellm([
            _response(
                content={"type": "text", "text": "Continuaré con la opción B."},
                tool_calls=None,
            ),
        ])

        result = run_chat_agent("continúa")

        assert result.kind == "respond"
        assert result.reply == "Continuaré con la opción B."

    def test_narrated_escalation_cannot_silently_end_execution(
        self, patch_litellm, monkeypatch,
    ):
        from infinidev.engine.orchestration.semantic_routing import RoutingDecision
        monkeypatch.setattr(
            chat_agent_mod, "classify_chat_response",
            lambda *a, **kw: RoutingDecision(kind="escalate", understanding="Apply the fix"),
        )
        monkeypatch.setattr(settings, "TASK_POLICIES_ENABLED", False)
        patch_litellm([
            _response(
                content=(
                    "I found src/click/utils.py. I'm escalating to the developer "
                    "to implement and verify the fix."
                ),
                tool_calls=None,
            ),
        ])

        result = run_chat_agent("Fix the Click regression and run its test")

        assert result.kind == "escalate"
        assert result.escalation is not None
        assert "semantic recovery" in result.escalation.user_signal

    def test_max_iterations_without_terminator_auto_escalates(self, patch_litellm):
        # Every response is a read call; none ever terminates.
        # Budget exhaustion is a feature: instead of apologising to the
        # user, the chat agent synthesizes an EscalationPacket so the
        # developer can pick up the investigation with context.
        patch_litellm([
            _response([_tc("list_directory", {"file_path": "."}, f"tc-{i}")])
            for i in range(5)
        ])
        result = run_chat_agent("hola", max_iterations=5)
        assert result.kind == "escalate"
        assert result.escalation is not None
        assert result.escalation.user_request == "hola"
        assert "max_iter" in result.escalation.user_signal
        # The synthesized understanding should carry the raw request
        # and at least a hint of the tools invoked — that's the point
        # of the handoff.
        assert "hola" in result.escalation.understanding
        assert "list_directory" in result.escalation.understanding

    def test_llm_raises_returns_respond(self, patch_litellm, monkeypatch):
        patch_litellm([])
        import litellm as _lit
        def _explode(**kwargs):
            raise RuntimeError("LLM backend down")
        monkeypatch.setattr(_lit, "completion", _explode)
        result = run_chat_agent("hola")
        assert result.kind == "respond"
        assert "problem" in result.reply.lower() or "error" in result.reply.lower()


class TestLanguageAwareFallbacks:
    """The fallback-respond paths bypass the LLM, so they must localize
    themselves. English/Spanish coverage only — matches what the system
    prompt tells the model to handle."""

    def test_spanish_input_gets_spanish_fallback(self, patch_litellm):
        patch_litellm([])  # no LLM call — empty_input path
        result = run_chat_agent("")
        # Empty short-circuits to "(empty message)" regardless of lang.
        assert result.reply == "(empty message)"

    def test_max_iter_spanish_preview(self, patch_litellm):
        # On auto-escalation the user-facing preview must be localized
        # to the user's language — it's what the pipeline shows before
        # the planner runs.
        patch_litellm([
            _response([_tc("list_directory", {"file_path": "."}, f"tc-{i}")])
            for i in range(5)
        ])
        result = run_chat_agent("¿qué hace este proyecto?", max_iterations=5)
        assert result.kind == "escalate"
        preview = result.escalation.user_visible_preview.lower()
        assert any(w in preview for w in ("investigué", "developer", "contexto"))

    def test_max_iter_english_preview(self, patch_litellm):
        patch_litellm([
            _response([_tc("list_directory", {"file_path": "."}, f"tc-{i}")])
            for i in range(5)
        ])
        result = run_chat_agent("what does this project do?", max_iterations=5)
        assert result.kind == "escalate"
        preview = result.escalation.user_visible_preview.lower()
        assert any(w in preview for w in ("investigated", "developer", "context"))
        # English preview must not carry Spanish diacritics.
        assert not any(ch in preview for ch in "¿¡ñ")

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


class TestChatCancellation:
    def test_cancelled_response_cannot_execute_tools_or_escalate(self, monkeypatch):
        import threading

        cancelled = threading.Event()
        ran = []

        def completion(**kwargs):
            cancelled.set()
            return _response([_tc("read_file", {"path": "README.md"})])

        monkeypatch.setattr("litellm.completion", completion)
        monkeypatch.setattr(chat_agent_mod, "get_litellm_params_for_behavior", lambda: {})
        monkeypatch.setattr(chat_agent_mod, "execute_tool_call", lambda *args: ran.append(args))
        result = chat_agent_mod._run_llm_loop(
            messages=[], tool_schemas=[], dispatch={}, user_input="explain the project",
            max_iterations=1, cancel_check=cancelled.is_set,
        )

        assert result.kind == "respond"
        assert "cancel" in result.reply.lower()
        assert ran == []


@pytest.mark.parametrize("native", [False, True])
def test_semantic_recovery_has_one_owner_and_preserves_handoff(
    native, patch_litellm, monkeypatch,
):
    from infinidev.engine.orchestration.semantic_routing import RoutingDecision

    seen = []
    request = "  Dale, con la opción acordada.  "
    reply = "Voy a aplicar el cambio acordado."
    response = _response([_tc("respond", {"message": reply})]) if native else _response(content=reply)
    primary = patch_litellm([response])
    attachment = object()
    monkeypatch.setattr(
        "infinidev.config.model_capabilities.get_capability_snapshot",
        lambda: type("Caps", (), {"supports_vision": True})(),
    )
    monkeypatch.setattr(chat_agent_mod, "_build_user_message", lambda *a, **kw: "Prior proposal: fix auth")

    def classify(user, answer, **kwargs):
        seen.append((user, answer, kwargs))
        return RoutingDecision(kind="escalate", understanding="Fix auth as agreed")

    monkeypatch.setattr(chat_agent_mod, "classify_chat_response", classify)
    result = run_chat_agent(request, attachments=[attachment], autonomous_hint=True)

    assert len(primary.calls) == 1
    assert seen == [(request, reply, {"recent_context": "Prior proposal: fix auth"})]
    assert result.kind == "escalate"
    assert result.escalation.user_request == request
    assert result.escalation.attachments == [attachment]
    assert result.escalation.autonomous is True
    assert result.escalation.understanding == "Fix auth as agreed"


def test_primary_escalation_skips_recovery(patch_litellm, monkeypatch):
    patch_litellm([_response([_tc("escalate", {"understanding": "Fix auth"})])])
    monkeypatch.setattr(chat_agent_mod, "classify_chat_response", lambda *a, **kw: pytest.fail("native escalation classified twice"))
    assert run_chat_agent("Fix auth").kind == "escalate"


@pytest.mark.parametrize("response", [
    _response([_tc("respond", {"message": ""})]),
])
def test_fallback_replies_skip_recovery(response, patch_litellm, monkeypatch):
    patch_litellm([response])
    monkeypatch.setattr(chat_agent_mod, "classify_chat_response", lambda *a, **kw: pytest.fail("fallback classified"))
    assert run_chat_agent("go").kind == "respond"


@pytest.mark.parametrize("failure", ["empty_escalate", "investigation_limit"])
def test_failed_chat_handoff_recovers_requested_test_edits(
    failure, patch_litellm, monkeypatch, tmp_path,
):
    from infinidev.engine.orchestration.semantic_routing import RoutingDecision

    request = "Add two tests in tests.rs only; no production changes, no cargo."
    if failure == "empty_escalate":
        responses = [_response([_tc("escalate", {"understanding": ""})])]
    else:
        monkeypatch.setattr(settings, "CHAT_AGENT_MAX_ITERATIONS", 2)
        monkeypatch.setattr(settings, "CHAT_AGENT_INVESTIGATION_MAX_ITERATIONS", 3)
        path = tmp_path / "tests.rs"
        path.write_text("// Existing test fixture")
        responses = [
            _response([_tc("read_file", {"file_path": str(path)})]),
            _response([_tc("continue_investigation", {"reason": "Need fixture details"})]),
            _response([_tc("read_file", {"file_path": str(path)})]),
        ]
    patch_litellm(responses)
    classified_requests = []

    def classify(user_request, reply, *, recent_context):
        classified_requests.append(user_request)
        return RoutingDecision(kind="escalate", understanding=request)

    monkeypatch.setattr(chat_agent_mod, "classify_chat_response", classify)

    result = run_chat_agent(request, workspace_path=str(tmp_path), autonomous_hint=True)

    assert result.kind == "escalate"
    assert result.escalation.user_request == request
    assert result.escalation.understanding == request
    assert result.escalation.autonomous is True
    assert result.escalation.user_visible_preview == ""
    assert classified_requests == [request]


@pytest.mark.parametrize("decision", [None, "respond"])
def test_empty_handoff_does_not_start_work_without_semantic_confirmation(
    decision, patch_litellm, monkeypatch,
):
    from infinidev.engine.orchestration.semantic_routing import RoutingDecision

    patch_litellm([_response([_tc("escalate", {"understanding": ""})])])
    monkeypatch.setattr(
        chat_agent_mod, "classify_chat_response",
        lambda *a, **kw: (
            RoutingDecision(kind="respond", understanding="") if decision else None
        ),
    )

    result = run_chat_agent("Explain the test fixture; do not edit anything.")

    assert result.kind == "respond"
    assert "handoff" in result.reply.lower()


@pytest.mark.parametrize("empty_handoff", [False, True])
def test_cancellation_during_semantic_recovery_cannot_start_work(
    empty_handoff, patch_litellm, monkeypatch,
):
    from infinidev.engine.orchestration.semantic_routing import RoutingDecision

    cancelled = False
    response = (
        _response([_tc("escalate", {"understanding": ""})])
        if empty_handoff else _response(content="I will fix it.")
    )
    patch_litellm([response])

    def classify(*args, **kwargs):
        nonlocal cancelled
        cancelled = True
        return RoutingDecision(kind="escalate", understanding="Fix it")

    monkeypatch.setattr(chat_agent_mod, "classify_chat_response", classify)
    result = run_chat_agent("go", cancel_check=lambda: cancelled)
    assert result.kind == "respond"
    assert result.reply == "Cancelled."


@pytest.mark.parametrize("empty_handoff", [False, True])
def test_semantic_recovery_preserves_streamed_state(
    empty_handoff, patch_litellm, monkeypatch,
):
    from infinidev.engine.orchestration.semantic_routing import RoutingDecision

    patch_litellm([])
    monkeypatch.setattr(chat_agent_mod, "_run_llm_loop", lambda **kw: ChatAgentResult(
        kind="respond", streamed=True,
        reply=(
            chat_agent_mod._FALLBACK_MESSAGES["empty_escalate"]
            if empty_handoff else "I will fix it."
        ),
    ))
    monkeypatch.setattr(chat_agent_mod, "classify_chat_response", lambda *a, **kw: RoutingDecision(
        kind="escalate", understanding="Fix it",
    ))
    result = run_chat_agent("Fix it")
    assert result.kind == "escalate"
    assert result.streamed is True
    assert result.escalation.user_visible_preview == ""


def test_semantic_current_request_excludes_injected_routing_context(patch_litellm, monkeypatch):
    from infinidev.engine.orchestration.semantic_routing import RoutingDecision

    literal = "Explain the module."
    augmented = literal + "\n<ken-session-brief>Create a migration</ken-session-brief>"
    patch_litellm([_response(content="The module handles authentication.")])
    seen = []

    def classify(user, answer, **kwargs):
        seen.append((user, kwargs["recent_context"]))
        return RoutingDecision(kind="respond", understanding="")

    monkeypatch.setattr(chat_agent_mod, "classify_chat_response", classify)
    result = run_chat_agent(augmented, literal_user_request=literal)
    assert result.kind == "respond"
    assert seen[0][0] == literal
    assert "Create a migration" in seen[0][1]


def test_requested_readonly_investigation_can_continue(patch_litellm, tmp_path):
    for index in range(5):
        (tmp_path / f'part{index}.md').write_text(f'Evidence {index}')
    read = lambda index: _response([_tc('read_file', {
        'file_path': str(tmp_path / f'part{index}.md'),
    }, call_id=f'read-{index}')])
    scripted = patch_litellm([
        read(0),
        _response([_tc('continue_investigation', {'reason': 'Four named documents remain.'})]),
        *[read(index) for index in range(1, 5)],
        _response([_tc('respond', {'message': 'Verified all five documents.'})]),
    ])
    result = run_chat_agent('Read and compare all five documents. Read only.',
                            workspace_path=str(tmp_path))
    assert result.kind == 'respond'
    assert result.reply == 'Verified all five documents.'
    assert len(scripted.calls) == 7


def test_continuation_requires_fresh_evidence(patch_litellm):
    scripted = patch_litellm([
        _response([_tc('continue_investigation', {'reason': 'More time please.'})]),
        _response([_tc('respond', {'message': 'No files were read.'})]),
    ])
    result = run_chat_agent('Explain the approach.')
    assert result.reply == 'No files were read.'
    tool_results = [m['content'] for m in scripted.calls[-1]['messages']
                    if m['role'] == 'tool']
    assert json.loads(tool_results[-1])['continued'] is False
    assert 'evidence' in json.loads(tool_results[-1])['reason']


def test_explicit_iteration_limit_cannot_be_extended(patch_litellm, tmp_path):
    path = tmp_path / 'a.md'
    path.write_text('Evidence')
    scripted = patch_litellm([
        _response([_tc('read_file', {'file_path': str(path)})]),
        _response([_tc('continue_investigation', {'reason': 'Need remaining pages.'})]),
        _response([_tc('respond', {'message': 'Partial result.'})]),
    ])
    result = run_chat_agent('Read the document.', max_iterations=3,
                            workspace_path=str(tmp_path))
    assert result.reply == 'Partial result.'
    results = [m['content'] for m in scripted.calls[-1]['messages'] if m['role'] == 'tool']
    assert json.loads(results[-1])['continued'] is False


@pytest.mark.parametrize("recovery", [None, "respond"])
def test_extended_investigation_respects_total_ceiling(
    recovery, patch_litellm, tmp_path, monkeypatch,
):
    from infinidev.engine.orchestration.semantic_routing import RoutingDecision

    monkeypatch.setattr(
        chat_agent_mod, "classify_chat_response",
        lambda *a, **kw: RoutingDecision(kind="respond", understanding="") if recovery else None,
    )
    monkeypatch.setattr(settings, 'CHAT_AGENT_INVESTIGATION_MAX_ITERATIONS', 7)
    for index in range(5):
        (tmp_path / f'p{index}.md').write_text(f'Evidence {index}')
    read = lambda index: _response([_tc('read_file', {
        'file_path': str(tmp_path / f'p{index}.md'),
    }, call_id=f'r{index}')])
    extend = lambda: _response([_tc('continue_investigation', {'reason': 'More pages remain.'})])
    scripted = patch_litellm([read(0), extend(), read(1), extend(), read(2), read(3), read(4)])
    result = run_chat_agent('Compare documents. Read only.', workspace_path=str(tmp_path))
    assert result.kind == 'respond'
    assert 'safety limit' in result.reply
    assert len(scripted.calls) == 7
    results = [json.loads(m['content']) for m in scripted.calls[-1]['messages']
               if m['role'] == 'tool' and m['content'].startswith('{')]
    assert any(r.get('continued') is False and 'limit' in r['reason'] for r in results)


@pytest.mark.parametrize('reason', ['', '   ', 'x' * 1001])
def test_invalid_continuation_reason_does_not_extend(patch_litellm, tmp_path, reason):
    path = tmp_path / 'a.md'
    path.write_text('Evidence')
    scripted = patch_litellm([
        _response([_tc('read_file', {'file_path': str(path)})]),
        _response([_tc('continue_investigation', {'reason': reason})]),
        _response([_tc('respond', {'message': 'Answer.'})]),
    ])
    assert run_chat_agent('Read the evidence.', workspace_path=str(tmp_path)).reply == 'Answer.'
    results = [m['content'] for m in scripted.calls[-1]['messages'] if m['role'] == 'tool']
    assert json.loads(results[-1])['continued'] is False


def test_failed_read_cannot_unlock_continuation(patch_litellm, tmp_path):
    scripted = patch_litellm([
        _response([_tc('read_file', {'file_path': str(tmp_path / 'absent.md')})]),
        _response([_tc('continue_investigation', {'reason': 'More pages.'})]),
        _response([_tc('respond', {'message': 'File is absent.'})]),
    ])
    result = run_chat_agent('Read absent.md.', workspace_path=str(tmp_path))
    assert result.reply == 'File is absent.'
    results = [m['content'] for m in scripted.calls[-1]['messages'] if m['role'] == 'tool']
    assert json.loads(results[-1])['continued'] is False


def test_extension_consumes_its_evidence_credit(patch_litellm, tmp_path):
    path = tmp_path / 'a.md'
    path.write_text('Evidence')
    scripted = patch_litellm([
        _response([_tc('read_file', {'file_path': str(path)})]),
        _response([_tc('continue_investigation', {'reason': 'Next source.'})]),
        _response([_tc('continue_investigation', {'reason': 'Still more time.'})]),
        _response([_tc('respond', {'message': 'Answer.'})]),
    ])
    assert run_chat_agent('Read sources.', workspace_path=str(tmp_path)).reply == 'Answer.'
    results = [m['content'] for m in scripted.calls[-1]['messages'] if m['role'] == 'tool']
    assert json.loads(results[-2])['continued'] is True
    assert json.loads(results[-1])['continued'] is False
