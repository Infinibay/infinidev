"""Final validation outcomes must distinguish acceptance from stopped correction."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.loop import guardrail_runner as runner
from infinidev.engine.loop.loop_state import LoopState


def _ctx(guardrail=None, **overrides):
    values = dict(
        state=LoopState(), guardrail=guardrail, guardrail_max_retries=0,
        llm_params={}, system_prompt="developer", desc="Implement rollback",
        expected="Verified rollback", tool_schemas=[], tool_dispatch={},
        max_per_action=0, max_total_calls=None, max_prompt_tokens=None,
        file_tracker=FileChangeTracker(), project_id=1, agent_id="developer", manual_tc=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _response(text="CORRECTED", tool_calls=None, usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=tool_calls))],
        usage=usage,
    )


@pytest.mark.parametrize(("validation", "status"), [
    (True, "done"), (False, "blocked"),
    ((True, "unused"), "done"), ((False, "rollback did not pass"), "blocked"),
])
def test_acceptance_and_rejection_are_distinct(validation, status):
    outcome = runner.evaluate_guardrail(_ctx(lambda text: validation), "ORIGINAL")

    assert outcome.status == status
    assert outcome.text == "ORIGINAL"
    if status != "done":
        assert outcome.reason


def test_no_guardrail_needs_no_model_call(monkeypatch):
    monkeypatch.setattr(runner, "_call_llm", lambda *args: pytest.fail("unexpected model call"))

    outcome = runner.evaluate_guardrail(_ctx(), "ORIGINAL")

    assert outcome.status == "done"
    assert outcome.text == "ORIGINAL"


def test_validator_exception_is_not_success(caplog):
    def broken(text):
        raise RuntimeError("validator unavailable")

    with caplog.at_level("ERROR"):
        outcome = runner.evaluate_guardrail(_ctx(broken), "ORIGINAL")

    assert outcome.status == "failed"
    assert "validator unavailable" in outcome.reason
    assert outcome.text == "ORIGINAL"


@pytest.mark.parametrize("validation", [
    "false", {"passed": False}, (False,), ("false", "rejected"), None,
])
def test_invalid_validation_shapes_cannot_be_accepted(validation):
    outcome = runner.evaluate_guardrail(_ctx(lambda text: validation), "ORIGINAL")

    assert outcome.status == "failed"
    assert outcome.reason


def test_last_corrected_candidate_is_revalidated(monkeypatch):
    validated = []

    def validate(text):
        validated.append(text)
        return False, "still invalid"

    monkeypatch.setattr(runner, "_call_llm", lambda *args: _response())
    outcome = runner.evaluate_guardrail(_ctx(validate, guardrail_max_retries=1), "ORIGINAL")

    assert validated == ["ORIGINAL", "CORRECTED"]
    assert outcome.status == "blocked"
    assert outcome.text == "CORRECTED"
    assert "still invalid" in outcome.reason


def test_successful_last_candidate_can_use_the_remaining_token_budget(monkeypatch):
    monkeypatch.setattr(runner, "_call_llm", lambda *args: _response(usage=SimpleNamespace(
        prompt_tokens=10, completion_tokens=2, total_tokens=12,
    )))
    ctx = _ctx(lambda text: (text == "CORRECTED", "fix it"),
               guardrail_max_retries=1, max_prompt_tokens=10)

    outcome = runner.evaluate_guardrail(ctx, "ORIGINAL")

    assert outcome.status == "done"
    assert ctx.state.total_prompt_tokens == 10


def test_prompt_fuse_is_exhaustion_not_acceptance(monkeypatch):
    monkeypatch.setattr(runner, "_call_llm", lambda *args: pytest.fail("budget was exhausted"))
    ctx = _ctx(lambda text: (False, "invalid"), guardrail_max_retries=2, max_prompt_tokens=10)
    ctx.state.total_prompt_tokens = 10

    outcome = runner.evaluate_guardrail(ctx, "ORIGINAL")

    assert outcome.status == "exhausted"
    assert "prompt" in outcome.reason.lower()


@pytest.mark.parametrize("cancel_at", ["before", "validator", "model", "tool"])
def test_cancellation_stops_correction_and_later_batch_tools(monkeypatch, cancel_at):
    cancelled = threading.Event()
    actions = []
    if cancel_at == "before":
        cancelled.set()

    def validate(text):
        actions.append("validate")
        if cancel_at == "validator":
            cancelled.set()
            return True
        return False, "fix it"

    def model(*args):
        actions.append("model")
        if cancel_at == "model":
            cancelled.set()
        calls = [SimpleNamespace(id=name, function=SimpleNamespace(name=name, arguments="{}"))
                 for name in ("first", "second")]
        return _response(tool_calls=calls)

    def tool(name, arguments):
        actions.append(name)
        cancelled.set()
        return "partial result"

    monkeypatch.setattr(runner, "_call_llm", model)
    outcome = runner.evaluate_guardrail(
        _ctx(validate, guardrail_max_retries=2), "ORIGINAL",
        is_cancelled=cancelled.is_set, run_tool=tool,
    )

    assert outcome.status == "cancelled"
    assert "second" not in actions
    if cancel_at == "before":
        assert actions == []
    if cancel_at in {"before", "validator", "model"}:
        assert "first" not in actions


def test_legacy_string_adapter_keeps_fallback_and_does_not_mutate_context():
    ctx = _ctx(lambda text: True)
    original_guardrail = ctx.guardrail

    text = runner.apply_guardrail(
        ctx, "PARTIAL", lambda text: False, 0, {}, "sys", "task", "expected",
        ctx.state, [], {},
    )

    assert text == "PARTIAL"
    assert ctx.guardrail is original_guardrail


@pytest.mark.parametrize("phase", ["validator", "model"])
def test_cancellation_wins_when_the_interrupted_phase_raises(monkeypatch, phase):
    cancelled = threading.Event()

    def broken(*args):
        cancelled.set()
        raise RuntimeError("interrupted while stopping")

    ctx = _ctx(broken if phase == "validator" else lambda text: (False, "fix it"),
               guardrail_max_retries=1)
    monkeypatch.setattr(runner, "_call_llm", broken)

    outcome = runner.evaluate_guardrail(ctx, "ORIGINAL", is_cancelled=cancelled.is_set)

    assert outcome.status == "cancelled"


def _tool_response(names, *, manual=False):
    if manual:
        return _response(json.dumps({"tool_calls": [
            {"name": name, "arguments": {}} for name in names
        ]}))
    return _response("", tool_calls=[
        SimpleNamespace(id=f"call-{index}", function=SimpleNamespace(name=name, arguments="{}"))
        for index, name in enumerate(names)
    ])


@pytest.mark.parametrize("manual,evaluations", [(False, 1), (True, 1), (False, 2)])
def test_unbounded_fuse_spans_attempts_but_renews_for_another_evaluation(
    monkeypatch, manual, evaluations,
):
    ctx = _ctx(lambda text: (False, "still invalid"), guardrail_max_retries=5, manual_tc=manual)
    actions = []

    for evaluation in range(evaluations):
        model_calls = []
        contexts = []

        def build_context():
            contexts.append(1)
            return [{"role": "user", "content": f"Fresh scope {len(contexts)}"}]

        def model(*args):
            model_calls.append(1)
            if len(model_calls) % 2:
                return _tool_response(["inspect"] * 7, manual=manual)
            return _response(f"DRAFT-{len(model_calls) // 2}")

        monkeypatch.setattr(runner, "_call_llm", model)
        outcome = runner.evaluate_guardrail(
            ctx, "ORIGINAL", build_context_messages=build_context,
            run_tool=lambda name, args: actions.append(name) or "observed",
        )

        assert outcome.status == "exhausted"
        assert outcome.text == "DRAFT-2"
        assert len(model_calls) == 5
        assert len(contexts) == 3
        assert actions == ["inspect"] * (16 * (evaluation + 1))
        assert ctx.state.total_tool_calls == 16 * (evaluation + 1)


def test_per_action_allowance_renews_without_resetting_run_accounting(monkeypatch):
    responses = iter([
        _tool_response(["inspect"]), _response("DRAFT"),
        _tool_response(["inspect"]), _response("CORRECTED"),
    ])
    ctx = _ctx(lambda text: (text == "CORRECTED", "fix it"),
               guardrail_max_retries=2, max_per_action=1)
    ctx.state.total_tool_calls = 4
    actions = []
    monkeypatch.setattr(runner, "_call_llm", lambda *args: next(responses))

    outcome = runner.evaluate_guardrail(
        ctx, "ORIGINAL", run_tool=lambda name, args: actions.append(name) or "observed",
    )

    assert outcome.status == "done"
    assert outcome.text == "CORRECTED"
    assert actions == ["inspect", "inspect"]
    assert ctx.state.total_tool_calls == 6


def test_accepted_terminal_candidate_wins_over_same_batch_tool_exhaustion(monkeypatch):
    response = _tool_response(["inspect", "skip", "step_complete"])
    response.choices[0].message.tool_calls[-1].function.arguments = json.dumps({
        "summary": "CORRECTED", "status": "done",
    })
    ctx = _ctx(lambda text: (text == "CORRECTED", "fix it"),
               guardrail_max_retries=1, max_per_action=1)
    actions = []
    monkeypatch.setattr(runner, "_call_llm", lambda *args: response)

    outcome = runner.evaluate_guardrail(
        ctx, "ORIGINAL", run_tool=lambda name, args: actions.append(name) or "observed",
    )

    assert outcome.status == "done"
    assert outcome.text == "CORRECTED"
    assert actions == ["inspect"]
    assert ctx.state.total_tool_calls == 1


def test_model_round_fuse_stops_a_nonterminal_correction(monkeypatch):
    requests = []
    actions = []

    def model(*args):
        requests.append(1)
        return _tool_response(["inspect"])

    monkeypatch.setattr(runner, "_call_llm", model)
    ctx = _ctx(lambda text: (False, "fix it"), guardrail_max_retries=3)

    outcome = runner.evaluate_guardrail(
        ctx, "ORIGINAL", run_tool=lambda name, args: actions.append(name) or "observed",
    )

    assert outcome.status == "exhausted"
    assert "round" in outcome.reason
    assert len(requests) == 8
    assert actions == ["inspect"] * 8
    assert ctx.state.total_tool_calls == 8


def test_tool_exception_keeps_the_latest_candidate_and_attempted_call_count(monkeypatch):
    responses = iter([_response("DRAFT"), _tool_response(["inspect"])])
    monkeypatch.setattr(runner, "_call_llm", lambda *args: next(responses))
    ctx = _ctx(lambda text: (False, "fix it"), guardrail_max_retries=2)

    def broken_tool(*args):
        raise RuntimeError("tool unavailable")

    outcome = runner.evaluate_guardrail(ctx, "ORIGINAL", run_tool=broken_tool)

    assert outcome.status == "failed"
    assert outcome.text == "DRAFT"
    assert "tool unavailable" in outcome.reason
    assert ctx.state.total_tool_calls == 1


def test_denied_correction_budget_does_not_revalidate_an_unchanged_candidate(monkeypatch):
    validated = []

    def validate(text):
        validated.append(text)
        return len(validated) > 1, "fix it"

    def unexpected(*args):
        pytest.fail("no correction can start without prompt budget")

    ctx = _ctx(validate, guardrail_max_retries=1, max_prompt_tokens=10)
    ctx.state.total_prompt_tokens = 10
    monkeypatch.setattr(runner, "_call_llm", unexpected)

    outcome = runner.evaluate_guardrail(
        ctx, "ORIGINAL", build_context_messages=unexpected,
    )

    assert outcome.status == "exhausted"
    assert validated == ["ORIGINAL"]
