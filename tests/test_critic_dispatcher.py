"""Tests for the critic dispatcher.

The dispatcher is a pure function over :class:`DispatchSignal` — these
tests construct signals directly and assert which checks fire. No LLM,
no I/O, no monkeypatching. If a test here fails, the failure mode it
covers will silently disappear from the critic's prompt in production.
"""

from __future__ import annotations

import pytest

from infinidev.engine.loop.critic_dispatcher import (
    CHECK_PROMPTS,
    CheckCode,
    DispatchSignal,
    render_active_checks_block,
    select_checks,
)


def _signal(**overrides) -> DispatchSignal:
    """Build a neutral signal and apply overrides."""
    sig = DispatchSignal()
    for k, v in overrides.items():
        setattr(sig, k, v)
    return sig


# --- Per-mode trigger tests --------------------------------------------------


def test_premature_closure_fires_on_step_complete():
    checks = select_checks(_signal(is_step_complete=True))
    assert CheckCode.PREMATURE_CLOSURE in checks


def test_false_confidence_fires_on_step_complete():
    checks = select_checks(_signal(is_step_complete=True))
    assert CheckCode.FALSE_CONFIDENCE in checks


def test_tunnel_vision_fires_when_grinding_without_step_complete():
    checks = select_checks(_signal(iteration=8, recent_step_complete_iters=[]))
    assert CheckCode.TUNNEL_VISION in checks


def test_tunnel_vision_silent_when_recent_step_complete():
    checks = select_checks(
        _signal(iteration=8, recent_step_complete_iters=[7]),
    )
    assert CheckCode.TUNNEL_VISION not in checks


def test_context_drift_fires_after_iteration_10():
    checks = select_checks(_signal(iteration=11))
    assert CheckCode.CONTEXT_DRIFT in checks


def test_context_drift_silent_at_iteration_10():
    checks = select_checks(_signal(iteration=10))
    assert CheckCode.CONTEXT_DRIFT not in checks


def test_overcorrection_fires_after_recommendation():
    checks = select_checks(_signal(prior_verdict_action="recommendation"))
    assert CheckCode.OVERCORRECTION in checks


def test_overcorrection_fires_after_reject():
    checks = select_checks(_signal(prior_verdict_action="reject"))
    assert CheckCode.OVERCORRECTION in checks


def test_overcorrection_silent_after_continue():
    checks = select_checks(_signal(prior_verdict_action="continue"))
    assert CheckCode.OVERCORRECTION not in checks


def test_overcorrection_silent_when_no_prior_verdict():
    checks = select_checks(_signal(prior_verdict_action=None))
    assert CheckCode.OVERCORRECTION not in checks


def test_confab_api_fires_on_execute_command():
    checks = select_checks(_signal(tool_call_names=["execute_command"]))
    assert CheckCode.CONFAB_API in checks


def test_confab_api_fires_on_replace_lines():
    checks = select_checks(_signal(tool_call_names=["replace_lines"]))
    assert CheckCode.CONFAB_API in checks


def test_confab_path_fires_when_path_not_seen():
    checks = select_checks(_signal(
        tool_call_names=["replace_lines"],
        tool_call_paths=["src/foo.py"],
        paths_seen_recently={"src/bar.py"},
    ))
    assert CheckCode.CONFAB_PATH in checks


def test_confab_path_silent_when_path_seen():
    checks = select_checks(_signal(
        tool_call_names=["replace_lines"],
        tool_call_paths=["src/foo.py"],
        paths_seen_recently={"src/foo.py"},
    ))
    assert CheckCode.CONFAB_PATH not in checks


def test_halluc_output_fires_when_tool_result_present():
    checks = select_checks(_signal(last_tool_result_present=True))
    assert CheckCode.HALLUC_OUTPUT in checks


def test_sycophancy_user_fires_on_first_turn():
    checks = select_checks(_signal(is_first_turn=True, iteration=1))
    assert CheckCode.SYCOPHANCY_USER in checks


def test_sycophancy_user_silent_on_later_turns():
    checks = select_checks(_signal(is_first_turn=False, iteration=3))
    assert CheckCode.SYCOPHANCY_USER not in checks


def test_sycophancy_repo_fires_on_inspect_then_edit():
    checks = select_checks(_signal(
        tool_call_names=["code_search", "replace_lines"],
        tool_call_paths=["src/x.py"],
        paths_seen_recently={"src/x.py"},
    ))
    assert CheckCode.SYCOPHANCY_REPO in checks


def test_sycophancy_repo_silent_on_inspect_only():
    checks = select_checks(_signal(tool_call_names=["code_search"]))
    assert CheckCode.SYCOPHANCY_REPO not in checks


# --- Pattern-driven check additions -----------------------------------------


def test_victory_lap_pattern_adds_premature_closure():
    checks = select_checks(_signal(reasoning_pattern_matches=["victory_lap"]))
    assert CheckCode.PREMATURE_CLOSURE in checks


def test_anchoring_loop_pattern_adds_tunnel_vision():
    # iteration=2 wouldn't trigger TUNNEL_VISION via signals alone.
    checks = select_checks(_signal(
        iteration=2,
        reasoning_pattern_matches=["anchoring_loop"],
    ))
    assert CheckCode.TUNNEL_VISION in checks


# --- Determinism and ordering ------------------------------------------------


def test_determinism_same_signal_same_checks():
    sig = _signal(
        iteration=12,
        is_step_complete=True,
        prior_verdict_action="reject",
        last_tool_result_present=True,
        tool_call_names=["replace_lines"],
        tool_call_paths=["src/foo.py"],
    )
    a = select_checks(sig)
    b = select_checks(sig)
    assert a == b


def test_check_order_matches_enum_declaration():
    sig = _signal(
        iteration=12,
        is_step_complete=True,
        prior_verdict_action="reject",
        last_tool_result_present=True,
        is_first_turn=True,
    )
    checks = select_checks(sig)
    enum_order = list(CheckCode)
    assert checks == [c for c in enum_order if c in checks]


def test_neutral_signal_yields_no_checks():
    assert select_checks(DispatchSignal()) == []


# --- render_active_checks_block ---------------------------------------------


def test_render_block_empty_when_no_checks():
    assert render_active_checks_block([]) == ""


def test_render_block_includes_each_check():
    block = render_active_checks_block([
        CheckCode.PREMATURE_CLOSURE,
        CheckCode.HALLUC_OUTPUT,
    ])
    assert "<active-checks>" in block
    assert "</active-checks>" in block
    assert "[PREMATURE_CLOSURE]" in block
    assert "[HALLUC_OUTPUT]" in block


def test_every_check_has_a_prompt():
    """Guard rail: adding a CheckCode without a prompt would silently
    crash the critic at runtime. Catch it here instead."""
    for code in CheckCode:
        assert code in CHECK_PROMPTS, f"Missing CHECK_PROMPTS entry for {code}"
        assert CHECK_PROMPTS[code].strip(), f"Empty prompt for {code}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
