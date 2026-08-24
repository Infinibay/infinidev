"""Regression tests for phase-engine execution and planning budgets."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from infinidev.engine.loop import LoopEngine
from infinidev.engine.phases.phase_engine import PhaseEngine
from infinidev.engine.phases.plan_generator import _generate_plan
from infinidev.gather.models import DEPTH_CONFIGS, DepthLevel
from infinidev.prompts.phases import STRATEGIES


def _add_step_response(call_id: str, title: str) -> SimpleNamespace:
    tool_call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="add_step",
            arguments=f'{{"title": "{title}", "explanation": ""}}',
        ),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))]
    )


def test_plan_generator_honors_explicit_round_budget() -> None:
    strategy = replace(STRATEGIES["bug"], plan_min_steps=2)
    responses = [
        _add_step_response("step-1", "Inspect the failing implementation"),
        _add_step_response("step-2", "Fix the verified root cause"),
        _add_step_response("step-3", "This response must not be requested"),
    ]

    with (
        patch("infinidev.config.llm._is_small_model", return_value=False),
        patch(
            "infinidev.config.llm.get_litellm_params",
            return_value={"model": "test/model"},
        ),
        patch(
            "infinidev.engine.loop.context.build_system_prompt",
            return_value="system",
        ),
        patch(
            "infinidev.engine.llm_client.call_llm",
            side_effect=responses,
        ) as call_mock,
    ):
        plan = _generate_plan(
            agent=SimpleNamespace(project_id=1, agent_id="developer"),
            description="Fix the bug",
            answers=[],
            all_notes=[],
            strategy=strategy,
            all_tools=[],
            verbose=False,
            max_rounds=2,
        )

    assert call_mock.call_count == 2
    assert [step["title"] for step in plan] == [
        "Inspect the failing implementation",
        "Fix the verified root cause",
    ]


def test_phase_engine_forwards_depth_round_budget_to_initial_plan_and_replan() -> None:
    loop = MagicMock(spec=LoopEngine)
    loop.is_cancelled = False
    loop._last_status = "done"
    checkpoint = MagicMock()
    checkpoint.run.side_effect = [(1, 2), (2, 2)]
    strategy = replace(STRATEGIES["bug"], auto_test=True)
    plans = [
        [{"step": 1, "title": "Apply the first fix", "files": ["src/example.py"]}],
        [{"step": 1, "title": "Correct the remaining failure", "files": ["src/example.py"]}],
    ]

    def execute_plan(*args, **kwargs):
        loop._last_status = "done"
        return "done", loop

    with (
        patch(
            "infinidev.engine.phases.phase_engine.TestCheckpoint",
            return_value=checkpoint,
        ),
        patch(
            "infinidev.engine.phases.phase_engine.get_strategy",
            return_value=strategy,
        ),
        patch(
            "infinidev.engine.phases.phase_engine._generate_plan",
            side_effect=plans,
        ) as generate_mock,
        patch(
            "infinidev.engine.phases.phase_engine._execute_plan",
            side_effect=execute_plan,
        ),
    ):
        result = PhaseEngine(loop_engine=loop).execute(
            agent=SimpleNamespace(),
            task_prompt=("Fix the bug", "Tests pass"),
            task_type="bug",
            depth_config=DEPTH_CONFIGS[DepthLevel.light],
            prompt_configuration=object(),
            verbose=False,
        )

    assert result == "done"
    assert generate_mock.call_count == 2
    assert [
        call.kwargs["max_rounds"] for call in generate_mock.call_args_list
    ] == [DEPTH_CONFIGS[DepthLevel.light].plan_max_rounds] * 2



def test_plan_review_forwards_depth_round_budget() -> None:
    loop = MagicMock(spec=LoopEngine)
    loop.is_cancelled = False
    loop._last_status = "done"
    checkpoint = MagicMock()
    strategy = replace(STRATEGIES["bug"], auto_test=False)
    plan = [{"step": 1, "title": "Apply the approved fix", "files": ["src/example.py"]}]
    classification = SimpleNamespace(
        depth=DepthLevel.light,
        ticket_type=SimpleNamespace(value="bug"),
    )

    with (
        patch(
            "infinidev.engine.phases.phase_engine.TestCheckpoint",
            return_value=checkpoint,
        ),
        patch(
            "infinidev.engine.phases.phase_engine.get_strategy",
            return_value=strategy,
        ),
        patch(
            "infinidev.engine.phases.phase_engine._investigate_iteratively",
            return_value=([], []),
        ),
        patch(
            "infinidev.engine.phases.phase_engine._generate_plan",
            return_value=plan,
        ) as generate_mock,
        patch(
            "infinidev.engine.phases.phase_engine._execute_plan",
            return_value=("done", loop),
        ),
        patch.object(PhaseEngine, "_classify", return_value=classification),
    ):
        result = PhaseEngine(loop_engine=loop).execute_with_plan_review(
            agent=SimpleNamespace(),
            task_description="Fix the bug",
            on_plan_ready=lambda steps: ("approve", ""),
            prompt_configuration=object(),
            verbose=False,
        )

    assert result == "done"
    assert generate_mock.call_args.kwargs["max_rounds"] == (
        DEPTH_CONFIGS[DepthLevel.light].plan_max_rounds
    )
