"""Final correction uses the same provider transcript contract as the main loop."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.loop import guardrail_runner as runner
from infinidev.engine.loop.loop_state import LoopState


@pytest.fixture
def ctx():
    return SimpleNamespace(
        state=LoopState(), guardrail=lambda text: (text == "CORRECTED", "fix it"),
        guardrail_max_retries=1, llm_params={}, system_prompt="developer", desc="Task",
        expected="Verified result", tool_schemas=[{"type": "function"}], tool_dispatch={},
        max_per_action=0, max_total_calls=None, max_prompt_tokens=None, manual_tc=False,
        file_tracker=FileChangeTracker(), project_id=1, agent_id="developer",
    )


def _call(name, arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or name, function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(content="", calls=None, **extra):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=content, tool_calls=calls, **extra,
        ))],
        usage=None,
    )


def test_required_reasoning_echo_is_preserved_between_correction_rounds(ctx, monkeypatch):
    requests = []

    def model(params, messages, schemas):
        requests.append(messages)
        if len(requests) == 1:
            return _response(calls=[_call("inspect")], reasoning_content="required reasoning")
        assistant = next(
            message for message in reversed(messages) if message["role"] == "assistant"
        )
        assert assistant["reasoning_content"] == "required reasoning"
        return _response("CORRECTED")

    monkeypatch.setattr(runner, "_call_llm", model)
    monkeypatch.setattr(
        "infinidev.config.model_capabilities.get_model_capabilities",
        lambda: SimpleNamespace(requires_reasoning_echo=True),
    )

    outcome = runner.evaluate_guardrail(ctx, "ORIGINAL", run_tool=lambda *args: "observed")

    assert outcome.status == "done"
    assert len(requests) == 2


def test_structured_content_is_normalized_without_mutating_provider_response(ctx, monkeypatch):
    response = _response({"text": "CORRECTED"})
    monkeypatch.setattr(runner, "_call_llm", lambda *args: response)

    outcome = runner.evaluate_guardrail(ctx, "ORIGINAL")

    assert outcome.status == "done"
    assert outcome.text == "CORRECTED"
    assert response.choices[0].message.content == {"text": "CORRECTED"}


def test_structured_arguments_reach_the_executor_as_canonical_json(ctx, monkeypatch):
    responses = iter([_response(calls=[_call("inspect", {"value": 7})]), _response("CORRECTED")])
    seen = []

    def run_tool(name, arguments):
        seen.append((name, json.loads(arguments)))
        return "observed"

    monkeypatch.setattr(runner, "_call_llm", lambda *args: next(responses))
    outcome = runner.evaluate_guardrail(ctx, "ORIGINAL", run_tool=run_tool)

    assert outcome.status == "done"
    assert seen == [("inspect", {"value": 7})]


def test_manual_correction_parses_text_tools_without_native_tool_messages(ctx, monkeypatch):
    ctx.manual_tc = True
    requests = []
    actions = []

    def model(params, messages, schemas):
        assert schemas is None
        requests.append(messages)
        if len(requests) == 1:
            return _response('{"name":"inspect","arguments":{"value":7}}')
        assert all(message["role"] != "tool" for message in messages)
        assert any("observed" in str(message.get("content", "")) for message in messages)
        return _response("CORRECTED")

    def run_tool(name, arguments):
        actions.append((name, json.loads(arguments)))
        return "observed"

    monkeypatch.setattr(runner, "_call_llm", model)
    outcome = runner.evaluate_guardrail(ctx, "ORIGINAL", run_tool=run_tool)

    assert outcome.status == "done"
    assert actions == [("inspect", {"value": 7})]


@pytest.mark.parametrize("terminal", [False, True])
def test_native_batches_answer_skipped_calls_without_executing_them(ctx, monkeypatch, terminal):
    messages_seen = []
    actions = []
    ctx.max_per_action = 1
    calls = (
        [_call("step_complete", '{"summary":"CORRECTED","status":"done"}'), _call("write")]
        if terminal else [_call("inspect"), _call("write")]
    )

    def model(params, messages, schemas):
        messages_seen.append(messages)
        return _response(calls=calls)

    def run_tool(name, arguments):
        actions.append(name)
        return "observed"

    monkeypatch.setattr(runner, "_call_llm", model)
    outcome = runner.evaluate_guardrail(ctx, "ORIGINAL", run_tool=run_tool)

    assert "write" not in actions
    assert outcome.status == ("done" if terminal else "exhausted")
    results = [message for message in messages_seen[0] if message["role"] == "tool"]
    assert sorted(message["tool_call_id"] for message in results) == sorted(
        call.id for call in calls
    )


def test_empty_completion_ends_the_attempt_instead_of_repeating_model_calls(ctx, monkeypatch):
    model_calls = []

    def model(*args):
        model_calls.append(1)
        return _response(calls=[_call("step_complete", '{"summary":"","status":"done"}')])

    monkeypatch.setattr(runner, "_call_llm", model)
    outcome = runner.evaluate_guardrail(ctx, "ORIGINAL")

    assert outcome.status == "blocked"
    assert len(model_calls) == 1


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("status", ["blocked", "explore"])
def test_incomplete_correction_cannot_be_overridden_by_validation(ctx, monkeypatch, manual, status):
    ctx.manual_tc = manual
    ctx.guardrail = lambda text: (text.startswith("{"), "Return JSON")
    arguments = {"status": status, "summary": '{"tests":"credentials missing"}'}
    call = _call("step_complete", json.dumps(arguments))
    response = (
        _response(json.dumps({"name": "step_complete", "arguments": arguments}))
        if manual else _response(calls=[call])
    )
    monkeypatch.setattr(runner, "_call_llm", lambda *args: response)

    outcome = runner.evaluate_guardrail(ctx, "ORIGINAL")

    assert outcome.status == "blocked"
    assert "credentials missing" in outcome.text
