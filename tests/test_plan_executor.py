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
    assert result == "done"
    assert returned_engine is engine
