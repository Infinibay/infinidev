"""Regression tests for the legacy phase executor's LoopEngine boundary."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from infinidev.engine.loop import LoopEngine
from infinidev.engine.phases.plan_executor import _execute_minimal, _execute_plan
from infinidev.gather.models import DEPTH_CONFIGS, DepthLevel
from infinidev.prompts.phases import STRATEGIES


def _assert_execute_kwargs_are_supported(execute_mock: MagicMock) -> dict:
    kwargs = execute_mock.call_args.kwargs
    supported = set(inspect.signature(LoopEngine.execute).parameters) - {"self"}
    assert set(kwargs) <= supported
    return kwargs


def test_minimal_executor_passes_supported_kwargs_and_phase_identity() -> None:
    engine = MagicMock(spec=LoopEngine)
    engine.execute.return_value = "done"
    strategy = STRATEGIES["bug"]

    with patch(
        "infinidev.engine.phases.plan_executor.LoopEngine",
        return_value=engine,
    ):
        result, returned_engine = _execute_minimal(
            agent=object(),
            description="Fix the bug",
            expected_output="Tests pass",
            strategy=strategy,
            task_tools=[],
            depth_config=DEPTH_CONFIGS[DepthLevel.minimal],
            verbose=False,
        )

    kwargs = _assert_execute_kwargs_are_supported(engine.execute)
    assert kwargs["identity_override"] == strategy.execute_identity
    assert kwargs["max_total_tool_calls"] == strategy.execute_max_tool_calls_per_step
    assert kwargs["max_tool_calls_per_action"] == strategy.execute_max_tool_calls_per_step
    assert result == "done"
    assert returned_engine is engine


def test_plan_executor_passes_supported_kwargs_and_phase_identity() -> None:
    engine = MagicMock(spec=LoopEngine)
    engine.execute.return_value = "done"
    strategy = STRATEGIES["bug"]

    with (
        patch(
            "infinidev.engine.phases.plan_executor.LoopEngine",
            return_value=engine,
        ),
        patch("infinidev.config.llm._is_small_model", return_value=False),
    ):
        result, returned_engine = _execute_plan(
            agent=object(),
            description="Fix the bug",
            expected_output="Tests pass",
            answers=[],
            all_notes=[],
            plan_steps=[{"step": 1, "title": "Fix src/example.py", "files": []}],
            strategy=strategy,
            all_tools=[],
            depth_config=DEPTH_CONFIGS[DepthLevel.deep],
            verbose=False,
        )

    kwargs = _assert_execute_kwargs_are_supported(engine.execute)
    assert kwargs["identity_override"] == strategy.execute_identity
    assert kwargs["max_total_tool_calls"] == strategy.execute_max_tool_calls_per_step
    assert kwargs["max_tool_calls_per_action"] == strategy.execute_max_tool_calls_per_step
    assert result == "done"
    assert returned_engine is engine


def test_plan_executor_reuses_provided_engine_and_stops_after_cancellation() -> None:
    engine = MagicMock(spec=LoopEngine)
    outcomes = iter(
        (("done", "step one done", 2), ("cancelled", "cancelled", 3))
    )

    def execute(**kwargs):
        status, result, tool_calls = next(outcomes)
        engine._last_status = status
        engine._last_total_tool_calls = tool_calls
        return result

    engine.execute.side_effect = execute
    strategy = STRATEGIES["bug"]
    steps = [
        {"step": 1, "title": "Inspect src/example.py", "files": []},
        {"step": 2, "title": "Fix src/example.py", "files": ["src/example.py"]},
        {"step": 3, "title": "Run tests", "files": []},
    ]

    with (
        patch("infinidev.engine.phases.plan_executor.LoopEngine") as loop_cls,
        patch("infinidev.config.llm._is_small_model", return_value=False),
    ):
        result, returned_engine = _execute_plan(
            agent=object(),
            description="Fix the bug",
            expected_output="Tests pass",
            answers=[],
            all_notes=[],
            plan_steps=steps,
            strategy=strategy,
            all_tools=[],
            depth_config=DEPTH_CONFIGS[DepthLevel.deep],
            verbose=False,
            loop_engine=engine,
        )

    assert result == "cancelled"
    assert returned_engine is engine
    assert engine.execute.call_count == 2
    assert engine._last_total_tool_calls == 5
    assert engine.execute.call_args_list[0].kwargs["preserve_file_tracker"] is False
    assert engine.execute.call_args_list[1].kwargs["preserve_file_tracker"] is True
    loop_cls.assert_not_called()
