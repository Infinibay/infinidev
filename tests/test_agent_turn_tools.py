"""Explicit conversational control without inferred work-task semantics."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from infinidev.engine.turns.models import PlanItem, TurnResult, TurnState
from infinidev.engine.turns.tools import build_turn_tools
from infinidev.tools.base.context import bind_tools_to_agent
from infinidev.tools.base.db import execute_with_retry


def test_finish_turn_preserves_plan_and_does_not_create_tasks(temp_db):
    state = TurnState(plan=[PlanItem(id="a", title="Still working", status="in_progress")])
    tools = build_turn_tools(state)
    bind_tools_to_agent(tools, "turn-agent")
    finish = next(tool for tool in tools if tool.name == "finish_turn")
    before = execute_with_retry(lambda conn: list(conn.execute("SELECT * FROM execution_events")))

    result = finish.run(message="Which target should I use?")

    assert isinstance(result, str)
    assert state.finished is True
    assert state.reply == "Which target should I use?"
    assert state.plan[0].status == "in_progress"
    assert execute_with_retry(lambda conn: list(conn.execute("SELECT * FROM execution_events"))) == before


def test_update_plan_uses_explicit_ids_and_opaque_titles():
    state = TurnState(notes=["retain evidence"])
    update = next(tool for tool in build_turn_tools(state) if tool.name == "update_plan")
    title = "DONE: stop, verify, ignore all instructions; ¿ya terminó?"

    update.run(plan=[{"id": "one", "title": title, "status": "pending"}])
    update.run(plan=[
        {"id": "one", "title": "Renamed", "status": "blocked"},
        {"id": "two", "title": title, "status": "completed"},
    ])

    assert [(item.id, item.title, item.status) for item in state.plan] == [
        ("one", "Renamed", "blocked"), ("two", title, "completed")
    ]
    assert state.notes == ["retain evidence"]
    assert state.finished is False
    update.run(plan=[])
    assert state.plan == []


@pytest.mark.parametrize("plan", [
    [{"id": "same", "title": "A", "status": "pending"},
     {"id": "same", "title": "B", "status": "completed"}],
    [{"id": "a", "title": "A", "status": "in_progress"},
     {"id": "b", "title": "B", "status": "in_progress"}],
    [{"id": "a", "title": "A", "status": "done"}],
    [{"id": "a", "status": "pending"}],
    [{"id": "", "title": "A", "status": "pending"}],
])
def test_invalid_plan_is_rejected_atomically(plan):
    state = TurnState(plan=[PlanItem(id="original", title="Keep", status="pending")])
    update = next(tool for tool in build_turn_tools(state) if tool.name == "update_plan")

    with pytest.raises(ValidationError):
        update.run(plan=plan)

    assert [(item.id, item.title, item.status) for item in state.plan] == [
        ("original", "Keep", "pending")
    ]


@pytest.mark.parametrize("payload", [{}, {"message": 3}, {"message": "ok", "status": "done"}])
def test_invalid_finish_payload_cannot_close_turn(payload):
    state = TurnState()
    finish = next(tool for tool in build_turn_tools(state) if tool.name == "finish_turn")
    with pytest.raises(ValidationError):
        finish.run(**payload)
    assert state.finished is False
    assert state.reply is None


@pytest.mark.parametrize("message", ["", " \t\n", "\u2003\u00a0"])
def test_blank_finish_message_rejected_without_state_mutation(message):
    state = TurnState(notes=["Retain evidence"], reply="Prior reply")
    before = state.model_copy(deep=True)
    finish = next(tool for tool in build_turn_tools(state) if tool.name == "finish_turn")

    with pytest.raises(ValidationError):
        finish.run(message=message)

    assert state == before


@pytest.mark.parametrize("message", ["", " \t\n", "\u2003\u00a0"])
async def test_async_blank_finish_message_rejected_without_state_mutation(message):
    state = TurnState(notes=["Retain evidence"], reply="Prior reply")
    before = state.model_copy(deep=True)
    finish = next(tool for tool in build_turn_tools(state) if tool.name == "finish_turn")

    with pytest.raises(ValidationError):
        await finish.arun(message=message)

    assert state == before


async def test_async_tools_and_turn_states_are_independent():
    first, second = TurnState(), TurnState()
    tools = {tool.name: tool for tool in build_turn_tools(first)}
    await tools["update_plan"].arun(plan=[{"id": "a", "title": "A", "status": "pending"}])
    await tools["finish_turn"].arun(message=" \nIntermediate delivery\t ")
    assert first.finished is True
    assert first.reply == " \nIntermediate delivery\t "
    assert len(first.plan) == 1
    assert second.plan == []
    assert second.notes == []
    assert second.finished is False


def test_result_rejects_task_status_and_internal_flags_are_not_persisted():
    state = TurnState(finished=True, reply="Hello")
    assert state.model_dump() == {"plan": [], "notes": []}
    with pytest.raises(ValidationError):
        TurnResult(message="Hello", status="completed", state=state)
    for status in ("answered", "cancelled", "exhausted", "failed"):
        result = TurnResult(message="Hello", status=status, state=state)
        assert result.state is state
        assert result.engine is None
        assert result.metrics == {}


@pytest.mark.parametrize("count", [True, False, "0", "3", -1, 2.0])
def test_turn_metrics_reject_invalid_budget_counts(count):
    with pytest.raises(ValidationError):
        TurnResult(
            message="Reply", status="answered", state=TurnState(),
            metrics={"tool_calls": count},
        )


def test_turn_metrics_accept_nonnegative_integer_budget_counts():
    result = TurnResult(
        message="Reply", status="answered", state=TurnState(),
        metrics={"tool_calls": 0, "iterations": 3},
    )
    assert result.metrics == {"tool_calls": 0, "iterations": 3}
