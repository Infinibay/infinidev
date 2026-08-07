"""How a run ends, and what it says about how it ended.

Three ways the loop used to reach a wrong ending: it reported a user
cancellation as an exhausted iteration budget, it aborted a healthy run
for "failing to produce function calls" when the model had been calling
them all along, and it had one gate with no bound at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine.loop.critic_liaison import CriticLiaison, _MAX_REJECTS_PER_STEP
from infinidev.engine.loop.loop_guard import LoopGuard, _MAX_PSEUDO_ONLY_ROUNDS
from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.step_result import StepResult
from infinidev.engine.loop.step_summarizer import _synthesize_final
from infinidev.engine.loop.action_record import ActionRecord


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        state=LoopState(), project_id=1, agent_id="a1",
        manual_tc=False, is_small=False,
    )


# ── cancellation is a status, not a flag ─────────────────────────────


def test_cancelled_run_does_not_claim_the_iteration_limit():
    state = LoopState()
    state.history = [ActionRecord(step_index=1, summary="edited auth.py")]

    text = _synthesize_final(state, "cancelled")

    assert "iteration limit" not in text.lower(), (
        "a cancelled run reported itself as exhausted to the reviewer, the "
        "task_end hooks and the next turn's work summary"
    )
    assert "stopped by the user" in text.lower()
    assert "edited auth.py" in text, "what did happen is still reported"


def test_exhausted_run_still_says_iteration_limit():
    state = LoopState()
    state.history = [ActionRecord(step_index=1, summary="edited auth.py")]
    assert "iteration limit" in _synthesize_final(state, "exhausted").lower()


def test_synthesize_final_defaults_to_exhausted():
    state = LoopState()
    state.history = [ActionRecord(step_index=1, summary="x")]
    assert "iteration limit" in _synthesize_final(state).lower()


def test_cancelled_with_no_history_says_so():
    assert "stopped by the user" in _synthesize_final(LoopState(), "cancelled").lower()


def test_cancel_survives_until_the_turn_ends():
    """execute() used to clear the flag, so the next phase resurrected the run."""
    from infinidev.engine.loop.engine import LoopEngine

    engine = LoopEngine()
    engine.cancel()
    assert engine.is_cancelled is True

    engine.begin_turn()
    assert engine.is_cancelled is False


# ── liveness is not the budget counter ───────────────────────────────


def test_a_step_closed_with_only_pseudo_tools_is_not_a_stall():
    """think + step_complete executes no regular tool and is not a stall."""
    result = StepResult(summary="done thinking", status="continue")
    result.action_tool_calls = 0
    result.saw_tool_calls = True

    assert result.saw_tool_calls, (
        "the abort reads this; action_tool_calls is a budget counter and "
        "reports 0 for a perfectly well-behaved step"
    )


def test_a_step_with_no_function_calls_at_all_is_a_stall():
    result = StepResult(summary="", status="continue")
    assert result.saw_tool_calls is False


# ── the pseudo-only spin ─────────────────────────────────────────────


def test_pseudo_only_turns_are_bounded():
    guard = LoopGuard(is_small=False)
    guard.reset()
    ctx, messages = _ctx(), []

    for _ in range(_MAX_PSEUDO_ONLY_ROUNDS):
        assert guard.handle_pseudo_only(ctx, messages) is None

    forced = guard.handle_pseudo_only(ctx, messages)
    assert forced is not None, (
        "nothing else bounds this: the inner while advances on "
        "action_tool_calls, which a pseudo-only turn never moves"
    )
    assert forced.status == "continue"


def test_a_real_tool_call_clears_the_pseudo_only_streak():
    guard = LoopGuard(is_small=False)
    guard.reset()
    ctx, messages = _ctx(), []
    for _ in range(_MAX_PSEUDO_ONLY_ROUNDS):
        guard.handle_pseudo_only(ctx, messages)

    guard.pseudo_only_rounds = 0  # what the engine does on a regular call

    assert guard.handle_pseudo_only(ctx, messages) is None


def test_error_circuit_requires_a_diagnosed_alternative_before_blocking():
    guard = LoopGuard(is_small=False)
    ctx, messages = _ctx(), []
    guard.consecutive_tool_errors = 4

    guard.check_error_circuit_breaker(ctx, messages)

    assert len(messages) == 1
    guidance = messages[0]["content"]
    assert "working directory" in guidance
    assert "known local correction must be tried" in guidance
    assert "only when the changed approach also cannot proceed" in guidance


# ── the critic's veto is bounded too ─────────────────────────────────


class _Verdict:
    def __init__(self, action: str) -> None:
        self.action = action
        self.message = "you did not run the tests"
        self.is_silent = False


class _Critic:
    model_short_name = "critic"

    def review(self, *a, **k):
        return _Verdict("reject")


def test_critic_veto_is_demoted_after_the_cap(monkeypatch):
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "ASSISTANT_LLM_INCLUDE_STEP_COMPLETE", True, raising=False)

    liaison = CriticLiaison()
    liaison.reset_run()
    monkeypatch.setattr(liaison, "get", lambda ctx: _Critic())

    ctx = _ctx()
    call = SimpleNamespace(id="sc1")
    overwrites: list[str] = []

    def _overwrite(messages, call_id, body):
        overwrites.append(body)

    blocked = [
        liaison.review_step_complete(ctx, [], call, None, _overwrite).blocked
        for _ in range(_MAX_REJECTS_PER_STEP + 1)
    ]

    assert blocked[:_MAX_REJECTS_PER_STEP] == [True] * _MAX_REJECTS_PER_STEP
    assert blocked[-1] is False, (
        "an unbounded veto holds the step open forever at two LLM calls a "
        "turn, and the pseudo-only turn it produces spends no budget"
    )


def test_critic_objection_still_reaches_the_model_after_demotion(monkeypatch):
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "ASSISTANT_LLM_INCLUDE_STEP_COMPLETE", True, raising=False)

    liaison = CriticLiaison()
    liaison.reset_run()
    monkeypatch.setattr(liaison, "get", lambda ctx: _Critic())

    ctx = _ctx()
    call = SimpleNamespace(id="sc1")
    review = None
    for _ in range(_MAX_REJECTS_PER_STEP + 1):
        review = liaison.review_step_complete(ctx, [], call, None, lambda *a: None)

    assert review.blocked is False
    assert review.followup is not None, "demoted to advice, not to silence"
    assert "tests" in review.followup["content"]
