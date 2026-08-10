"""Operational model policies stay narrow, deterministic, and route-specific."""

from infinidev.engine.model_execution_policy import resolve_model_execution_policy


def test_minimax_m3_uses_compact_operational_policy() -> None:
    policy = resolve_model_execution_policy("minimax", "minimax/MiniMax-M3")

    assert policy.name == "minimax-m3-v5"
    assert policy.compact_tool_schemas is True
    assert policy.require_step_orientation is False
    assert policy.renew_step_budget_on_progress is True
    assert policy.semantic_stagnation_control is True
    assert policy.step_nudge_threshold(
        max_tool_calls=12,
        configured_threshold=6,
    ) == 10


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
        assert policy.semantic_stagnation_control is False
        assert policy.step_nudge_threshold(
            max_tool_calls=12,
            configured_threshold=6,
        ) == 6
