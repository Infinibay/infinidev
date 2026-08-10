"""Operational model policies stay narrow, deterministic, and route-specific."""

from infinidev.engine.model_execution_policy import resolve_model_execution_policy


def test_minimax_m3_uses_compact_operational_policy() -> None:
    policy = resolve_model_execution_policy("minimax", "minimax/MiniMax-M3")

    assert policy.name == "minimax-m3-v9"
    assert policy.compact_tool_schemas is True
    assert policy.require_step_orientation is False
    assert policy.renew_step_budget_on_progress is True
    assert policy.semantic_stagnation_control is True
    assert policy.recovery_direct_reads_only is True
    assert policy.reuse_unchanged_test_results is True
    assert policy.freeze_plan_growth_in_recovery is True
    assert policy.recovery_requires_workspace_change is True
    assert policy.skip_referenced_continuation_elaboration is True
    assert "call escalate immediately" in policy.chat_prompt_addendum
    assert "Make no read calls first" in policy.chat_prompt_addendum
    assert "Preserve the literal requested outcome" in policy.prompt_addendum
    assert "normalized test target" in policy.prompt_addendum
    assert "sufficient evidence" in policy.prompt_addendum
    assert "Low confidence is not a blocker" in policy.prompt_addendum
    assert policy.step_nudge_threshold(
        max_tool_calls=12,
        configured_threshold=6,
    ) == 10


def test_chat_prompt_addendum_is_route_conditional(monkeypatch) -> None:
    from infinidev.config import llm
    from infinidev.config.settings import settings
    from infinidev.prompts.chat_agent import build_chat_agent_system_prompt

    monkeypatch.setattr(settings, "LLM_PROVIDER", "minimax")
    monkeypatch.setattr(
        llm,
        "get_litellm_params_for_behavior",
        lambda: {"model": "minimax/MiniMax-M3"},
    )
    prompt = build_chat_agent_system_prompt()
    assert "MiniMax M3 routing calibration" in prompt
    assert "Make no read calls first" in prompt

    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(
        llm,
        "get_litellm_params_for_behavior",
        lambda: {"model": "openai/gpt-5.6-terra"},
    )
    assert "MiniMax M3 routing calibration" not in build_chat_agent_system_prompt()


def test_other_routes_keep_the_neutral_baseline() -> None:
    for provider, model in (
        ("openai", "openai/gpt-5.6-terra"),
        ("minimax", "minimax/MiniMax-M2.7"),
        ("openai_compatible", "custom/MiniMax-M3"),
    ):
        policy = resolve_model_execution_policy(provider, model)

        assert policy.name == "baseline"
        assert policy.compact_tool_schemas is False
        assert policy.require_step_orientation is True
        assert policy.renew_step_budget_on_progress is False
        assert policy.semantic_stagnation_control is True
        assert policy.recovery_direct_reads_only is True
        assert policy.reuse_unchanged_test_results is True
        assert policy.prompt_addendum == ""
        assert policy.freeze_plan_growth_in_recovery is True
        assert policy.recovery_requires_workspace_change is True
        assert policy.skip_referenced_continuation_elaboration is False
        assert policy.chat_prompt_addendum == ""
        assert policy.step_nudge_threshold(
            max_tool_calls=12,
            configured_threshold=6,
        ) == 6
