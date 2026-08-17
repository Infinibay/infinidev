"""Unit tests for the autonomous chain budget.

The chain itself is wired in a later step; these tests pin down the
budget's pure-function behaviour so the downstream pipeline can rely on
it without coupling. The cases cover each of the four topes
(max_plans, token_budget, wall_seconds, idle_passes) plus the terminal
outcome short-circuit and the ``from_settings`` defensive defaults.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from infinidev.engine.orchestration.autonomous import (
    DEFAULT_IDLE_PASSES,
    DEFAULT_MAX_PLANS,
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_WALL_SECONDS,
    AutonomousBudget,
    budget_status_text,
    from_settings,
    should_continue,
    stop_reason,
)


# ── Defaults ──────────────────────────────────────────────────────────


def test_default_budget_allows_continue_on_first_iteration():
    """A fresh chain with sane defaults should be allowed to run a plan.

    Mirrors the expectation from the step description
    (``should_continue(AutonomousBudget(), 'continue')`` is True) and is the
    gate every other test assumes.
    """
    assert should_continue(AutonomousBudget(), "continue") is True


def test_default_topes_match_spec():
    """The defaults baked into the dataclass must match the documented
    conservative values. If a future refactor changes them, the chain
    becomes greedier (or shorter) by surprise — surface the change here.
    """
    budget = AutonomousBudget()
    assert budget.max_plans == DEFAULT_MAX_PLANS == 3
    assert budget.token_budget == DEFAULT_TOKEN_BUDGET == 50_000
    assert budget.wall_seconds == DEFAULT_WALL_SECONDS == 900
    assert budget.idle_passes == DEFAULT_IDLE_PASSES == 2


# ── Per-tope behaviour ────────────────────────────────────────────────


def test_max_plans_tope_stops_after_the_configured_count():
    """max_plans is a hard cap. After N plans the chain must stop even
    if every other fuse has budget left. Counter is incremented BEFORE
    the check, so plans_executed == max_plans is the terminal state.
    """
    budget = AutonomousBudget(max_plans=3)
    budget.record_outcome("continue")
    budget.record_outcome("continue")
    budget.record_outcome("continue")
    assert budget.plans_executed == 3
    assert should_continue(budget, "continue") is False


def test_token_budget_tope_stops_when_consumed_matches():
    """token_budget stops at equality, not ``>``. That keeps the accounting
    intuitive: a budget of 50k means "let it consume up to 50k tokens".
    Counts tokens contributed via ``record_outcome(tokens_used=...)``.
    """
    budget = AutonomousBudget(token_budget=100, max_plans=10)
    budget.record_outcome("continue", tokens_used=40)
    budget.record_outcome("continue", tokens_used=40)
    # 80 consumed, well under 100
    assert should_continue(budget, "continue") is True
    budget.record_outcome("continue", tokens_used=20)
    # exactly 100 — boundary case, must stop
    assert budget.tokens_consumed == 100
    assert should_continue(budget, "continue") is False


def test_wall_seconds_tope_stops_when_elapsed_matches():
    """wall_seconds is evaluated against ``time.monotonic`` anchored at
    ``start``. We override the anchor to a known past value so the test
    is deterministic and does not rely on real wall-clock waits.
    """
    budget = AutonomousBudget(wall_seconds=60, max_plans=10)
    budget.start()
    # Pretend start happened 60s in the past.
    budget.wall_started_at = time.monotonic() - 60.0
    assert should_continue(budget, "continue") is False


def test_wall_seconds_tope_does_not_fire_before_anchor():
    """``start`` is idempotent and must run before the wall fuse checks
    elapsed. A fresh budget with ``wall_started_at=0`` reports 0 elapsed
    and never trips the wall clock — only the start() call arms it.
    """
    budget = AutonomousBudget(wall_seconds=1)
    # never called start(); wall_started_at is 0.0
    assert budget.wall_elapsed == 0.0
    assert should_continue(budget, "continue") is True


def test_idle_passes_tope_stops_after_repeated_idle_reports():
    """``idle_passes=2`` means the chain tolerates exactly two consecutive
    "nothing to do" reports. The third consecutive idle is the stop
    signal — checked against ``idle_runs``, which ``record_outcome``
    increments only on idle outcomes.
    """
    budget = AutonomousBudget(idle_passes=2, max_plans=10)
    budget.record_outcome("idle")
    assert budget.idle_runs == 1
    assert should_continue(budget, "idle") is True  # one idle, still under
    budget.record_outcome("idle")
    assert budget.idle_runs == 2
    assert should_continue(budget, "idle") is False  # at the threshold


def test_non_idle_outcomes_reset_idle_runs_semantically():
    """A real plan after an idle must not extend the idle streak. The
    counter is not auto-reset (the chain does not need to forget the
    signal), but a non-idle outcome neutralises the next idle check
    because ``should_continue`` evaluates the *consecutive* sequence by
    reading ``last_outcome`` against the current ``idle_runs``.
    """
    budget = AutonomousBudget(idle_passes=2, max_plans=10)
    budget.record_outcome("idle")
    budget.record_outcome("continue")  # real plan in between
    # idle_runs stays at 1 (we only increment on idle), but last_outcome
    # is "continue" so the idle check no longer fires.
    assert should_continue(budget, "continue") is True


# ── Terminal outcomes ─────────────────────────────────────────────────


def test_terminal_outcomes_stop_immediately():
    """``done`` / ``blocked`` / ``error`` are short-circuit terminal
    states. They take precedence over every fuse — even a brand-new
    budget with no plans executed yet must stop if the model says
    "done" (e.g. a one-shot user request that the chat agent decided
    needed escalation and the engine then had nothing to do).
    """
    budget = AutonomousBudget()
    for outcome in ("done", "blocked", "error"):
        assert should_continue(budget, outcome) is False, outcome


def test_terminal_outcomes_with_capitalisation_still_stop():
    """The pipeline emits ``'Completed'`` in places; the budget must
    accept casing variance rather than crashing on a value that is
    clearly the same intent. Same for surrounding whitespace.
    """
    budget = AutonomousBudget()
    assert should_continue(budget, "DONE") is False
    assert should_continue(budget, " Blocked ") is False
    assert should_continue(budget, "") is True  # empty == "continue"


def test_unknown_outcome_treated_as_continue():
    """A new outcome value added in the pipeline must not silently stop
    the chain. Anything we do not recognise falls back to "continue",
    so a typo in the engine status is at worst a noisy run, not a
    stuck one.
    """
    budget = AutonomousBudget(max_plans=10)
    assert should_continue(budget, "transcended") is True


# ── ``record_outcome`` accounting ─────────────────────────────────────


def test_record_outcome_counts_plans_and_tokens_when_provided():
    """``record_outcome`` is the single mutation entry point. Counters
    must reflect the inputs exactly: plans_executed always increments,
    tokens_consumed only when tokens_used>0, idle_runs only when
    outcome=="idle".
    """
    budget = AutonomousBudget()
    budget.record_outcome("continue", tokens_used=1_000)
    assert budget.plans_executed == 1
    assert budget.tokens_consumed == 1_000
    assert budget.idle_runs == 0
    assert budget.last_outcome == "continue"

    budget.record_outcome("idle")
    assert budget.idle_runs == 1
    assert budget.last_outcome == "idle"

    # tokens_used=0 / negative is ignored — the caller did not pay attention.
    budget.record_outcome("continue", tokens_used=0)
    assert budget.tokens_consumed == 1_000


def test_reset_counters_restarts_the_chain():
    """``reset_counters`` zeros the per-plan counters and re-anchors the
    wall clock. Used when the chain is cycled without a process restart.
    """
    budget = AutonomousBudget(max_plans=10)
    budget.record_outcome("continue", tokens_used=500)
    budget.record_outcome("idle")
    budget.start()
    budget.reset_counters()
    assert budget.plans_executed == 0
    assert budget.tokens_consumed == 0
    assert budget.idle_runs == 0
    assert budget.last_outcome is None
    assert budget.wall_started_at > 0.0
    assert should_continue(budget, "continue") is True


# ── Status text + stop reason ─────────────────────────────────────────


def test_budget_status_text_includes_all_four_counters():
    """The status string is rendered into the chat agent's banner and
    into the UI; it must include every counter so the user can see
    what is consuming the budget at a glance.
    """
    budget = AutonomousBudget(max_plans=5, token_budget=10_000, wall_seconds=30, idle_passes=1)
    budget.start()
    text = budget_status_text(budget)
    assert "plan 0/5" in text
    assert "tokens 0/10000" in text
    assert "wall 0s/30s" in text
    assert "idle 0/1" in text


def test_stop_reason_is_concrete_when_chain_terminates():
    """After the chain stops, the chat agent should be able to tell the
    user *why* (vs. just "stopped"). The mapping must cover every
    termination path.
    """
    budget = AutonomousBudget(max_plans=2, idle_passes=1, wall_seconds=60, token_budget=10_000)
    budget.record_outcome("continue")
    budget.record_outcome("continue")
    assert stop_reason(budget) == "reached max_plans=2"

    budget2 = AutonomousBudget(max_plans=10, token_budget=100)
    budget2.record_outcome("continue", tokens_used=100)
    assert stop_reason(budget2) == "reached token_budget=100"

    budget3 = AutonomousBudget(max_plans=10, wall_seconds=60)
    budget3.start()
    budget3.wall_started_at = time.monotonic() - 60.0
    assert stop_reason(budget3) == "reached wall_seconds=60"

    budget4 = AutonomousBudget(max_plans=10, idle_passes=1)
    budget4.record_outcome("idle")
    assert stop_reason(budget4) == "no new work in 1 consecutive plans"

    budget5 = AutonomousBudget()
    budget5.record_outcome("done")
    assert stop_reason(budget5) == "engine reported done"


def test_stop_reason_is_none_when_chain_is_still_active():
    """While the chain is allowed to keep going, ``stop_reason`` returns
    None. The chat agent uses that signal to decide whether to print a
    closing explanation.
    """
    budget = AutonomousBudget()
    assert stop_reason(budget) is None


# ── Settings loading ──────────────────────────────────────────────────


def test_from_settings_reads_each_tope_with_defaults_fallback():
    """``from_settings`` accepts a permissive SimpleNamespace so the
    test does not depend on the global settings singleton. Each field
    falls back to the module default when missing, None, or non-positive
    — a misconfiguration must not silently turn the chain into an
    infinite loop (max_plans=0) or into a no-op (wall_seconds=0).
    """
    fake = SimpleNamespace(
        AUTONOMOUS_MAX_PLANS=7,
        AUTONOMOUS_TOKEN_BUDGET=12_345,
        AUTONOMOUS_WALL_SECONDS=300,
        AUTONOMOUS_IDLE_PASSES=4,
    )
    budget = AutonomousBudget.from_settings(fake)
    assert budget.max_plans == 7
    assert budget.token_budget == 12_345
    assert budget.wall_seconds == 300
    assert budget.idle_passes == 4


def test_from_settings_replaces_zero_or_negative_with_default():
    """0 / negative / None values are treated as "user did not configure"
    and replaced with the conservative default. The chain must never
    be permanently halted by a misconfigured zero.
    """
    fake = SimpleNamespace(
        AUTONOMOUS_MAX_PLANS=0,
        AUTONOMOUS_TOKEN_BUDGET=-1,
        AUTONOMOUS_WALL_SECONDS=None,
        AUTONOMOUS_IDLE_PASSES=0,
    )
    budget = AutonomousBudget.from_settings(fake)
    assert budget.max_plans == DEFAULT_MAX_PLANS
    assert budget.token_budget == DEFAULT_TOKEN_BUDGET
    assert budget.wall_seconds == DEFAULT_WALL_SECONDS
    assert budget.idle_passes == DEFAULT_IDLE_PASSES


def test_module_level_from_settings_matches_classmethod():
    """The module re-exports ``from_settings`` as a top-level helper so
    the pipeline can call ``autonomous.from_settings(settings)`` without
    importing the dataclass. The two entry points must produce the same
    budget for the same settings — covers the late-binding re-export at
    module bottom. (Identity comparison is unreliable on classmethod
    descriptors: ``is`` returns False across the module/class boundary
    even when both refer to the same underlying function.)
    """
    fake = SimpleNamespace(
        AUTONOMOUS_MAX_PLANS=4,
        AUTONOMOUS_TOKEN_BUDGET=8_000,
        AUTONOMOUS_WALL_SECONDS=120,
        AUTONOMOUS_IDLE_PASSES=3,
    )
    via_module = from_settings(fake)
    via_class = AutonomousBudget.from_settings(fake)
    assert via_module == via_class
    assert via_module.max_plans == 4
    assert via_class.max_plans == 4


def test_global_settings_exposes_the_autonomous_fields():
    """The settings instance reads from the env / file lazily; this test
    merely confirms the four fields are declared on the class so the
    test for ``from_settings`` can read them via ``getattr(settings, ...)``
    without raising. Without these attributes, ``from_settings`` would
    fall through to defaults even when the user configured a value.
    """
    from infinidev.config.settings import Settings

    assert hasattr(Settings(), "AUTONOMOUS_MAX_PLANS")
    assert hasattr(Settings(), "AUTONOMOUS_TOKEN_BUDGET")
    assert hasattr(Settings(), "AUTONOMOUS_WALL_SECONDS")
    assert hasattr(Settings(), "AUTONOMOUS_IDLE_PASSES")
    # The defaults exposed on the global settings also match the spec.
    s = Settings()
    assert s.AUTONOMOUS_MAX_PLANS == 3
    assert s.AUTONOMOUS_TOKEN_BUDGET == 50_000
    assert s.AUTONOMOUS_WALL_SECONDS == 900
    assert s.AUTONOMOUS_IDLE_PASSES == 2
