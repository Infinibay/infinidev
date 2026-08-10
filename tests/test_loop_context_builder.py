"""Execution-context budget normalization at the public LoopEngine boundary."""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.engine.loop import context_builder


def test_zero_total_tool_budget_reaches_the_loop_as_unlimited(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(context_builder.settings, "LLM_PROVIDER", "minimax")
    monkeypatch.setattr(context_builder.settings, "LLM_MODEL", "minimax/MiniMax-M3")
    monkeypatch.setattr(
        context_builder,
        "get_litellm_params",
        lambda: {"model": "minimax/MiniMax-M3"},
    )
    monkeypatch.setattr(context_builder, "_is_small_model", lambda: False)
    monkeypatch.setattr(
        context_builder,
        "get_model_capabilities",
        lambda: SimpleNamespace(supports_function_calling=True),
    )
    engine = SimpleNamespace(_last_file_tracker=None)
    agent = SimpleNamespace(
        agent_id="developer-1",
        project_id=1,
        name="developer",
        role="developer",
        backstory="",
        workspace_path=str(tmp_path),
    )

    ctx = context_builder.build_execution_context(
        engine,
        agent,
        ("Complete a long task", "Verified result"),
        task_tools=[],
        max_total_tool_calls=0,
        max_tool_calls_per_action=12,
        verbose=False,
    )

    assert ctx.max_total_calls is None
    assert ctx.max_per_action == 12
    assert ctx.model_policy_name == "minimax-m3-v5"
    assert ctx.renew_step_budget_on_progress is True
    assert ctx.semantic_stagnation_control is True
    assert ctx.step_tool_limit == 12


def test_positive_total_tool_budget_remains_available_for_bounded_phases(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        context_builder,
        "get_litellm_params",
        lambda: {"model": "openai/gpt-5.6-terra"},
    )
    monkeypatch.setattr(context_builder, "_is_small_model", lambda: False)
    monkeypatch.setattr(
        context_builder,
        "get_model_capabilities",
        lambda: SimpleNamespace(supports_function_calling=True),
    )
    engine = SimpleNamespace(_last_file_tracker=None)
    agent = SimpleNamespace(
        agent_id="developer-1",
        project_id=1,
        name="developer",
        role="developer",
        backstory="",
        workspace_path=str(tmp_path),
    )

    ctx = context_builder.build_execution_context(
        engine,
        agent,
        ("Run one bounded evaluation", "Report the result"),
        task_tools=[],
        max_total_tool_calls=40,
        max_tool_calls_per_action=12,
        verbose=False,
    )

    assert ctx.max_total_calls == 40
    assert ctx.max_per_action == 12
    assert ctx.renew_step_budget_on_progress is False
    assert ctx.semantic_stagnation_control is False
