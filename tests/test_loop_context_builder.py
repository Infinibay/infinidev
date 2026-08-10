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
        max_iterations=0,
        max_total_tool_calls=0,
        max_tool_calls_per_action=0,
        verbose=False,
    )

    assert ctx.max_iterations is None
    assert ctx.max_total_calls is None
    assert ctx.max_per_action == 0
    assert ctx.model_policy_name == "minimax-m3-v11"
    assert ctx.renew_step_budget_on_progress is True
    assert ctx.semantic_stagnation_control is True
    assert ctx.phase_boundary_control is True
    assert ctx.recovery_direct_reads_only is True
    assert ctx.unlimited_recovery_reads is True
    assert ctx.reuse_unchanged_test_results is True
    assert ctx.freeze_plan_growth_in_recovery is True
    assert ctx.recovery_requires_workspace_change is True
    assert "MiniMax M3 execution calibration" in ctx.system_prompt
    assert "Preserve the literal requested outcome" in ctx.system_prompt
    assert ctx.step_tool_limit is None


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
    assert ctx.semantic_stagnation_control is True
    assert ctx.phase_boundary_control is False
    assert ctx.recovery_direct_reads_only is True
    assert ctx.unlimited_recovery_reads is True
    assert ctx.reuse_unchanged_test_results is True
    assert ctx.freeze_plan_growth_in_recovery is True
    assert ctx.recovery_requires_workspace_change is True
    assert "MiniMax M3 execution calibration" not in ctx.system_prompt
