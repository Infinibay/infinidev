"""Runtime contract tests for the evidence-driven Stage Planner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from infinidev.engine.analysis.stage_planner import (
    _MAX_STALL_NUDGES,
    run_stage_planner,
)
from infinidev.engine.analysis.staged_planning import (
    BlockGoalDecision,
    CompleteGoalDecision,
    EmitStageDecision,
    EvidenceEntry,
    GoalSpec,
    StageSpec,
    StageTaskSpec,
    StagedPlanningState,
)


@dataclass
class _Function:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    function: _Function
    type: str = "function"


@dataclass
class _Message:
    content: str = ""
    tool_calls: list[_ToolCall] | None = None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]


def _call(name: str, args: dict[str, Any], call_id: str = "call-1") -> _ToolCall:
    return _ToolCall(call_id, _Function(name, json.dumps(args)))


def _response(
    calls: list[_ToolCall] | None = None,
    content: str = "",
) -> _Response:
    return _Response([_Choice(_Message(content=content, tool_calls=calls))])


class _Scripted:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Stage Planner requested an unscripted response")
        return self.responses.pop(0)


@pytest.fixture
def scripted(monkeypatch):
    def install(responses: list[_Response]) -> _Scripted:
        runner = _Scripted(responses)
        monkeypatch.setattr("litellm.completion", runner)
        return runner

    monkeypatch.setattr(
        "infinidev.engine.analysis.stage_planner.get_litellm_params_for_behavior",
        lambda: {"model": "test/mock"},
    )
    return install


def _state(*, evidence: bool = False) -> StagedPlanningState:
    state = StagedPlanningState(goal=GoalSpec(
        title="Repair the staged runtime",
        user_request="Repair the staged runtime and verify its behavior.",
    ))
    if evidence:
        state.add_evidence(EvidenceEntry(kind="test", summary="Focused tests pass."))
    return state


def _stage_args(title: str = "Repair one slice") -> dict[str, Any]:
    return {
        "title": title,
        "outcome": "One end-to-end behavior works",
        "exit_criteria": ["The focused behavior has a passing regression test"],
        "tasks": [{
            "id": "slice",
            "title": "Implement the slice",
            "outcome": "The slice works through the runtime",
            "acceptance_criteria": ["The focused regression test passes"],
            "depends_on": [],
        }],
    }


def test_emit_stage_returns_typed_validated_decision(scripted):
    scripted([_response([_call("emit_stage", _stage_args())])])

    decision = run_stage_planner(_state())

    assert isinstance(decision, EmitStageDecision)
    assert decision.stage.tasks[0].id == "slice"


def test_complete_goal_requires_observed_evidence(scripted):
    scripted([
        _response([_call("complete_goal", {"evidence": ["Everything is done"]})]),
        _response([_call("emit_stage", _stage_args())]),
    ])

    decision = run_stage_planner(_state())

    assert isinstance(decision, EmitStageDecision)


def test_complete_goal_with_ledger_evidence_is_accepted(scripted):
    state = _state(evidence=True)
    scripted([_response([_call(
        "complete_goal",
        {"evidence": [f"{state.evidence[0].id}: Focused tests pass"]},
    )])])

    decision = run_stage_planner(state)

    assert isinstance(decision, CompleteGoalDecision)


def test_complete_goal_rejects_uncited_evidence_claim(scripted):
    scripted([
        _response([_call("complete_goal", {"evidence": ["Tests pass"]})]),
        _response([_call("emit_stage", _stage_args())]),
    ])

    decision = run_stage_planner(_state(evidence=True))

    assert isinstance(decision, EmitStageDecision)


def test_text_json_recovery_supports_models_without_function_calling(scripted):
    scripted([_response(content=json.dumps({
        "kind": "goal_blocked",
        "reason": "A user-owned target is missing",
        "missing": "The exact target",
        "evidence": ["Two candidates were observed"],
    }))])

    decision = run_stage_planner(_state())

    assert isinstance(decision, BlockGoalDecision)
    assert decision.missing == "The exact target"


def test_multiple_terminal_calls_are_rejected(scripted):
    scripted([
        _response([
            _call("complete_goal", {"evidence": ["claim"]}, "complete"),
            _call("block_goal", {
                "reason": "blocked", "missing": "input", "evidence": [],
            }, "block"),
        ]),
        _response([_call("block_goal", {
            "reason": "Need a choice", "missing": "User answer", "evidence": [],
        })]),
    ])

    decision = run_stage_planner(_state())

    assert isinstance(decision, BlockGoalDecision)
    assert decision.reason == "Need a choice"


def test_no_terminal_decision_nudges_and_recovers(scripted):
    runner = scripted([
        _response(content="I think this is complete."),
        _response([_call("emit_stage", _stage_args())]),
    ])

    decision = run_stage_planner(_state())

    assert isinstance(decision, EmitStageDecision)
    assert len(runner.calls) == 2
    second_call_messages = runner.calls[-1]["messages"]
    user_messages = [m for m in second_call_messages if m.get("role") == "user"]
    assert user_messages, "expected a user-role nudge appended after the stall"
    nudge = user_messages[-1]["content"]
    for tool_name in ("emit_stage", "complete_goal", "block_goal"):
        assert tool_name in nudge
    assert "exactly one" in nudge.lower() or "one terminal" in nudge.lower()


def test_max_iterations_guard_still_returns_block_goal_decision(scripted):
    runner = scripted([
        _response(content="I am still thinking about the stage."),
        _response(content="Still no terminal decision."),
    ])

    decision = run_stage_planner(_state(), max_iterations=2)

    assert isinstance(decision, BlockGoalDecision)
    assert decision.reason == (
        "Stage Planner exhausted its iteration budget without a valid decision."
    )
    assert decision.missing == "A valid Stage Planner decision on a later retry."
    assert decision.evidence == []
    assert len(runner.calls) == 2, (
        "max_iterations=2 must drive exactly two litellm.completion calls before "
        "the guard returns _failure_decision."
    )


def test_stage_planner_role_has_only_its_terminal_tools(scripted):
    runner = scripted([_response([_call("emit_stage", _stage_args())])])

    run_stage_planner(_state())

    names = {
        tool["function"]["name"] for tool in runner.calls[0]["tools"]
    }
    assert {"emit_stage", "complete_goal", "block_goal"} <= names
    assert "emit_task_plan" not in names
    assert "emit_plan" not in names
    assert "create_file" not in names


def test_qwen_subscription_requires_a_tool_call(scripted, monkeypatch):
    monkeypatch.setattr(
        "infinidev.engine.llm_client.settings.LLM_PROVIDER",
        "qwen_subscription",
    )
    runner = scripted([_response([_call("emit_stage", _stage_args())])])

    run_stage_planner(_state())

    assert runner.calls[0]["tool_choice"] == "required"


def test_bounded_stall_returns_failure_decision_after_max_stall_nudges(scripted):
    """Repeated stalls (no tool_calls, no embedded decision) must early-return
    via _failure_decision once _MAX_STALL_NUDGES nudges have been emitted,
    without draining the iteration budget."""
    stall_responses = [_response(content=f"Thinking... {i}") for i in range(_MAX_STALL_NUDGES)]
    runner = scripted(stall_responses)

    decision = run_stage_planner(_state(), max_iterations=20)

    assert isinstance(decision, BlockGoalDecision)
    assert decision.reason == (
        f"Stage Planner stalled without a terminal decision after "
        f"{_MAX_STALL_NUDGES} nudges."
    )
    assert decision.missing == "A valid Stage Planner decision on a later retry."
    assert decision.evidence == []
    assert len(runner.calls) == _MAX_STALL_NUDGES, (
        f"bounded-stall must drive exactly {_MAX_STALL_NUDGES} litellm.completion "
        "calls (initial + N-1 nudges) before returning _failure_decision on the "
        "Nth increment, regardless of the iteration budget."
    )


def test_stall_counter_resets_on_resumed_tool_calls(scripted):
    """The stall counter must reset on the first response that carries
    tool_calls, so an early stall does not poison a later valid turn."""
    runner = scripted([
        _response(content="I am still thinking."),
        _response([_call("emit_stage", _stage_args())]),
    ])

    decision = run_stage_planner(_state())

    assert isinstance(decision, EmitStageDecision)
    assert decision.stage.tasks[0].id == "slice"
    assert len(runner.calls) == 2, (
        "one stall plus one valid emit_stage must drive exactly two litellm "
        "calls; the second response carries tool_calls so the stall counter "
        "resets before any bounded-stall early-return can fire."
    )


def test_exploration_budget_refuses_extra_calls_in_a_single_batch(scripted, monkeypatch):
    calls: list[str] = []

    def execute(_dispatch, name, _arguments):
        calls.append(name)
        return "observed"

    monkeypatch.setattr(
        "infinidev.engine.analysis.stage_planner.execute_tool_call", execute
    )
    scripted([
        _response([
            _call("read_file", {"file_path": "a.py"}, "one"),
            _call("read_file", {"file_path": "b.py"}, "two"),
        ]),
        _response([_call("emit_stage", _stage_args())]),
    ])

    decision = run_stage_planner(_state(), max_exploration_calls=1)

    assert isinstance(decision, EmitStageDecision)
    assert calls == ["read_file"]


def test_implementation_goal_rejects_consecutive_discovery_stages(scripted):
    state = _state()
    state.goal = state.goal.model_copy(update={"intent": "implementation"})
    first = EmitStageDecision(stage=StageSpec(
        title="Resolve one uncertainty",
        purpose="discovery",
        outcome="The integration point is known",
        exit_criteria=["The integration point is observed"],
        tasks=[StageTaskSpec(
            id="discover",
            title="Inspect the integration point",
            outcome="One deciding fact is observed",
            acceptance_criteria=["The deciding fact is recorded"],
        )],
    ))
    state.add_stage(first.stage).status = "evaluating"
    discovery_args = _stage_args("Try another discovery")
    discovery_args["purpose"] = "discovery"
    scripted([
        _response([_call("emit_stage", discovery_args)]),
        _response([_call("block_goal", {
            "reason": "Delivery needs user authority",
            "missing": "The user's approval",
            "evidence": [],
        })]),
    ])

    decision = run_stage_planner(state)

    assert isinstance(decision, BlockGoalDecision)
    assert decision.reason == "Delivery needs user authority"
