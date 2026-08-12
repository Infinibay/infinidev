"""Observable runtime behavior detection and bounded intervention tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

from infinidev.engine.behavior.runtime_policy import (
    detect_runtime_behavior,
    drain_runtime_intervention,
    observe_reasoning_behavior,
    observe_runtime_behavior,
)
from infinidev.engine.loop.guidance_handler import GuidanceHandler
from infinidev.engine.loop.loop_state import LoopState
from infinidev.engine.loop.plan_step import PlanStep


def _messages(calls: list[tuple[str, dict[str, object], str]]) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for index, (name, arguments, result) in enumerate(calls):
        call_id = f"call-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "content": result},
            ]
        )
    return messages


def _discovery_window() -> list[dict[str, object]]:
    return _messages(
        [
            ("list_directory", {"file_path": "."}, "ok"),
            ("code_search", {"query": "target"}, "match"),
            ("read_file", {"file_path": "src/a.py"}, "source"),
            ("read_file", {"file_path": "tests/test_a.py"}, "tests"),
            ("describe_tool", {"tool_name": "edit_file"}, "schema"),
            ("add_step", {"title": "Edit src/a.py"}, "added"),
            ("modify_step", {"index": 1}, "modified"),
            ("add_note", {"content": "target found"}, "saved"),
        ]
    )


def test_excessive_discovery_requires_modifying_task_and_no_progress() -> None:
    state = LoopState()

    modifying = detect_runtime_behavior(
        _discovery_window(), state, modifying_task=True
    )
    research = detect_runtime_behavior(
        _discovery_window(), state, modifying_task=False
    )

    assert [signal.label for signal in modifying] == ["excessive_discovery"]
    assert research == ()


def test_shadow_mode_records_signal_without_mutating_prompt_budget() -> None:
    state = LoopState()

    queued = observe_runtime_behavior(
        state,
        _discovery_window(),
        task=SimpleNamespace(kind="bugfix", task_profile=None),
        shadow_mode=True,
        max_interventions=2,
        opened_files_budget_chars=16_000,
    )

    assert queued is None
    assert state.pending_runtime_intervention == ""
    assert state.opened_files_prompt_max_chars == 0
    assert state.runtime_behavior_events[0]["label"] == "excessive_discovery"
    assert not state.runtime_behavior_events[0]["intervention_queued"]


def test_candidate_queues_once_and_reduces_opened_file_budget() -> None:
    state = LoopState()
    task = SimpleNamespace(kind="bugfix", task_profile=None)

    queued = observe_runtime_behavior(
        state,
        _discovery_window(),
        task=task,
        shadow_mode=False,
        max_interventions=2,
        opened_files_budget_chars=16_000,
    )
    duplicate = observe_runtime_behavior(
        state,
        _discovery_window(),
        task=task,
        shadow_mode=False,
        max_interventions=2,
        opened_files_budget_chars=16_000,
    )

    assert queued == "excessive_discovery"
    assert duplicate is None
    assert state.opened_files_prompt_max_chars == 16_000
    assert "smallest scoped edit" in drain_runtime_intervention(state)
    assert drain_runtime_intervention(state) == ""

    progress_messages = _discovery_window() + _messages(
        [("edit_file", {"file_path": "src/a.py"}, "updated")]
    )
    observe_runtime_behavior(
        state,
        progress_messages,
        task=task,
        shadow_mode=False,
        max_interventions=2,
        opened_files_budget_chars=16_000,
    )
    assert state.opened_files_prompt_max_chars == 0


def test_timeout_intervention_wins_before_schema_telemetry() -> None:
    state = LoopState()
    messages = _messages(
        [
            (
                "execute_command",
                {"command": "node probe.js"},
                '{"error":"Command timed out after 15s"}',
            ),
            (
                "execute_command",
                {"command": "node probe.js", "summary": "retry"},
                '{"error":"wrong parameter name(s): summary"}',
            ),
        ]
    )

    queued = observe_runtime_behavior(
        state,
        messages,
        task=SimpleNamespace(kind="bugfix", task_profile=None),
        shadow_mode=False,
        max_interventions=1,
        opened_files_budget_chars=16_000,
    )

    assert queued == "command_timeout"
    assert [event["label"] for event in state.runtime_behavior_events] == [
        "command_timeout",
        "tool_schema_mismatch",
    ]
    assert "finite input" in state.pending_runtime_intervention


def test_prior_task_test_does_not_hide_new_step_excessive_discovery() -> None:
    state = LoopState(last_test_output="old passing test output")

    signals = detect_runtime_behavior(
        _discovery_window(), state, modifying_task=True
    )

    assert [signal.label for signal in signals] == ["excessive_discovery"]


def test_current_step_test_suppresses_excessive_discovery() -> None:
    messages = _discovery_window() + _messages(
        [("execute_command", {"command": "uv run pytest tests/test_a.py -q"}, "1 passed")]
    )

    signals = detect_runtime_behavior(messages, LoopState(), modifying_task=True)

    assert "excessive_discovery" not in {signal.label for signal in signals}


def test_runner_name_in_diagnostic_command_is_not_test_progress() -> None:
    messages = _discovery_window() + _messages(
        [
            (
                "execute_command",
                {"command": "ls node_modules/ava/package.json && echo ava installed"},
                '{"exit_code":0,"stdout":"node_modules/ava/package.json\\nava installed"}',
            )
        ]
    )

    signals = detect_runtime_behavior(messages, LoopState(), modifying_task=True)

    assert "excessive_discovery" in {signal.label for signal in signals}


def test_mid_step_runtime_intervention_reaches_immediate_next_model_call(
    monkeypatch,
) -> None:
    from infinidev.engine.loop import step_manager

    settings = SimpleNamespace(
        ADAPTIVE_RUNTIME_BEHAVIOR_ENABLED=True,
        ADAPTIVE_RUNTIME_BEHAVIOR_SHADOW_MODE=False,
        ADAPTIVE_RUNTIME_MAX_INTERVENTIONS=2,
        ADAPTIVE_RUNTIME_OPENED_FILES_MAX_CHARS=16_000,
        ADAPTIVE_RUNTIME_SEMANTIC_SHADOW_ENABLED=False,
        LOOP_GUIDANCE_ENABLED=False,
    )
    monkeypatch.setattr(step_manager, "_get_settings", lambda: settings)
    state = LoopState()
    state.plan.steps = [PlanStep(index=3, title="Implement fix", status="active")]
    ctx = SimpleNamespace(
        state=state,
        task=SimpleNamespace(kind="bugfix", task_profile=None),
        is_small=False,
        verbose=False,
    )
    messages = _discovery_window()

    GuidanceHandler().try_queue(ctx, messages, 0, mid_step=True)

    assert messages[-1]["role"] == "user"
    assert "make the smallest scoped edit now" in messages[-1]["content"]
    assert state.pending_runtime_intervention == ""
    assert state.runtime_behavior_events[0]["step_index"] == 3


def test_reasoning_mini_model_queues_an_evidence_gated_retry_prompt() -> None:
    state = LoopState()
    state.plan.steps = [PlanStep(index=2, title="Implement fix", status="active")]
    messages = _messages(
        [
            (
                "execute_command",
                {"command": "uv run pytest tests/test_a.py"},
                '{"exit_code":1,"stderr":"failed"}',
            ),
            (
                "execute_command",
                {"command": "uv run pytest tests/test_a.py"},
                '{"exit_code":1,"stderr":"failed"}',
            ),
        ]
    )

    event = observe_reasoning_behavior(
        state,
        "The command failed again, so I will repeat the exact same command unchanged.",
        messages,
        task=SimpleNamespace(kind="bugfix", task_profile=None),
        current_tool_calls=None,
        sources=("reasoning_content",),
        shadow_mode=False,
        max_interventions=2,
    )

    assert event is not None
    assert event["label"] == "reasoning:retry_loop"
    assert event["source"] == "static-qwen3-reasoning-mini-head"
    assert event["intervention_queued"] is True
    assert "alter cwd, input" in state.pending_runtime_intervention


def test_visible_reasoning_reaches_the_next_prompt_through_the_mini_model(
    monkeypatch,
) -> None:
    from infinidev.engine.loop import step_manager

    monkeypatch.setattr(
        step_manager,
        "_get_settings",
        lambda: SimpleNamespace(
            ADAPTIVE_RUNTIME_REASONING_ENABLED=True,
            ADAPTIVE_RUNTIME_REASONING_SHADOW_MODE=False,
            ADAPTIVE_RUNTIME_MAX_INTERVENTIONS=2,
        ),
    )
    state = LoopState()
    state.plan.steps = [PlanStep(index=4, title="Fix retry", status="active")]
    messages = _messages(
        [
            (
                "execute_command",
                {"command": "cargo test parser"},
                '{"exit_code":1,"stderr":"failed"}',
            ),
            (
                "execute_command",
                {"command": "cargo test parser"},
                '{"exit_code":1,"stderr":"failed"}',
            ),
        ]
    )
    ctx = SimpleNamespace(
        state=state,
        task=SimpleNamespace(kind="bugfix", task_profile=None),
    )
    result = SimpleNamespace(
        message=SimpleNamespace(
            reasoning_content=(
                "The same command failed, so I will run it unchanged once more."
            )
        ),
        reasoning_content=(
            "The same command failed, so I will run it unchanged once more."
        ),
        tool_calls=[],
    )
    handler = GuidanceHandler()

    handler.observe_reasoning(ctx, messages, result)
    delivered = handler.inject_pending(ctx, messages)

    assert delivered is True
    assert messages[-1]["role"] == "user"
    assert 'source="mini-model"' in messages[-1]["content"]
    assert "Do not repeat it unchanged" in messages[-1]["content"]
    assert state.runtime_behavior_events[-1]["intervention_delivered"] is True
    assert state.runtime_behavior_events[-1]["delivery_channel"] == "next-user-turn"
