"""Tests for the difficulty-adaptive Task Planner system prompt.

Three invariants matter:

1. The hard variant MUST be byte-identical to the legacy module-level
   ``TASK_PLANNER_SYSTEM_PROMPT`` so every existing caller (planner.py,
   prompts/analyst/__init__.py, planner_prompt.py,
   tests/test_specialized_prompt_contracts.py,
   tests/test_staged_planning_prompts.py, tests/test_task_policies.py)
   keeps working without churn.
2. The three variants MUST differ in size — easy < medium < hard — and
   must include / exclude the sections a difficulty should.
3. The builder MUST reject unknown levels instead of silently falling back
   to hard, so a future widening of the union surfaces at the build site.
"""

from __future__ import annotations

import pytest

from infinidev.engine.orchestration.difficulty import DIFFICULTY_LEVELS
from infinidev.prompts.analyst.task_planner_prompt import (
    TASK_PLANNER_SYSTEM_PROMPT,
    build_task_planner_system_prompt,
)


# --- Back-compat invariant ----------------------------------------------------


def test_hard_variant_matches_legacy_constant():
    """The legacy module-level constant must equal the hard variant.

    Anything that imports ``TASK_PLANNER_SYSTEM_PROMPT`` directly relies on
    this byte-identity (planner.py, the prompts/analyst re-exports, three
    test files). A drift here would silently change behaviour for callers
    that never opted into difficulty-aware planning.
    """
    assert (
        build_task_planner_system_prompt("hard") == TASK_PLANNER_SYSTEM_PROMPT
    )


# --- Distinct variants -------------------------------------------------------


@pytest.mark.parametrize("level", DIFFICULTY_LEVELS)
def test_builder_returns_non_empty_string(level):
    out = build_task_planner_system_prompt(level)
    assert isinstance(out, str)
    assert out.strip()


def test_levels_differ_in_size():
    """Easy must be shorter than medium, medium shorter than hard.

    The whole point of difficulty-adaptive planning is that we stop
    spending hard-level tokens on typo fixes; a size invariant is the
    most direct guard against a regression that silently equalises the
    three prompts.
    """
    easy = build_task_planner_system_prompt("easy")
    medium = build_task_planner_system_prompt("medium")
    hard = build_task_planner_system_prompt("hard")
    assert len(easy) < len(medium) < len(hard)


def test_easy_drops_critique_and_review_language():
    """The easy variant must not invite critique/review sub-steps.

    The hard prompt tells the planner to surface the observation that
    would disprove each Step; the easy variant deliberately omits that
    guidance because the developer loop already verifies each Step
    against its own check. A negative mention ("do not add a separate
    critique sub-step") is fine — it tells the model what to avoid —
    but the easy prompt must not contain positive instructions for
    critique or review.
    """
    easy = build_task_planner_system_prompt("easy")
    lower = easy.lower()
    # No positive "## Critique" / "Critique each" / "Critique the plan"
    # header-style or imperative-style instruction.
    assert "## critique" not in lower
    assert "critique each" not in lower
    assert "critique the plan" not in lower
    assert "critique this" not in lower
    assert "review each step" not in lower
    assert "review the plan" not in lower
    # The only allowed mention is the negative guidance.
    assert "do not add a separate critique" in lower


def test_easy_announces_its_level():
    """The easy variant must self-identify so the model reads it as such.

    A prompt that omits the level would still be valid, but the model
    wouldn't know it should produce 1–2 Steps instead of the usual
    full-depth decomposition. The framing line is the cheapest way to
    keep the model on-piste.
    """
    easy = build_task_planner_system_prompt("easy")
    assert "EASY" in easy


def test_medium_announces_its_level():
    medium = build_task_planner_system_prompt("medium")
    assert "MEDIUM" in medium


def test_hard_announces_its_level():
    hard = build_task_planner_system_prompt("hard")
    assert "HARD" in hard


def test_medium_keeps_planning_vocabulary():
    """The medium variant must include the planning vocabulary.

    Medium-depth requests are still grounded multi-step tasks; the
    vocabulary is the shared semantics (Goal / Stage / Task / Step /
    Evidence) the planner relies on. Removing it would break the
    contract ``tests/test_staged_planning_prompts.py`` exercises on the
    hard prompt.
    """
    medium = build_task_planner_system_prompt("medium")
    assert "**Goal**" in medium
    assert "**Step**" in medium
    assert "**Evidence**" in medium


def test_easy_excludes_planning_vocabulary():
    """The easy variant can drop the deep vocabulary.

    A 1–2 step typo fix doesn't need the full Goal / Stage / Task / Step
    / Evidence exposition; the model can emit a focused plan without
    it. Keeping it would inflate the easy prompt for no benefit.
    """
    easy = build_task_planner_system_prompt("easy")
    assert "**Goal**" not in easy
    assert "**Evidence**" not in easy


def test_hard_keeps_planning_vocabulary():
    hard = build_task_planner_system_prompt("hard")
    assert "**Goal**" in hard
    assert "**Stage**" in hard
    assert "**Step**" in hard
    assert "**Evidence**" in hard


def test_easy_explains_few_steps_target():
    """The easy variant must tell the planner to emit 1–2 Steps."""
    easy = build_task_planner_system_prompt("easy")
    assert "1–2" in easy or "1-2" in easy


def test_easy_keeps_emit_task_plan_call_site():
    """All three variants must still emit ``emit_task_plan`` — the
    planner's terminator. If a future edit accidentally replaces the
    call site, every variant would fail at the dispatch layer.
    """
    for level in DIFFICULTY_LEVELS:
        out = build_task_planner_system_prompt(level)
        assert "emit_task_plan" in out, level


# --- Failure modes ----------------------------------------------------------


def test_unknown_level_raises():
    with pytest.raises(ValueError, match="unknown difficulty level"):
        build_task_planner_system_prompt("trivial")  # type: ignore[arg-type]


# --- Per-level budget map (planner.py wiring) --------------------------------


def test_difficulty_budgets_shrink_with_easier_levels():
    """The per-difficulty budget map must shrink calls/iterations for
    easier levels so a single typo fix doesn't run the full hard-depth
    loop. Read from planner.py's private map because that's where the
    wiring actually happens.
    """
    from infinidev.engine.analysis.planner import _DIFFICULTY_BUDGETS

    easy_calls, easy_iters = _DIFFICULTY_BUDGETS["easy"]
    med_calls, med_iters = _DIFFICULTY_BUDGETS["medium"]
    hard_calls, hard_iters = _DIFFICULTY_BUDGETS["hard"]

    assert easy_calls <= med_calls <= hard_calls
    assert easy_iters <= med_iters <= hard_iters
    # Hard row keeps the previous defaults so behaviour is unchanged for
    # callers that never opted into difficulty-aware planning.
    assert hard_calls == 4
    assert hard_iters == 6


# --- run_planner wiring -----------------------------------------------------


def test_run_planner_resolves_difficulty_and_scales_caps(monkeypatch):
    """run_planner must call resolve_difficulty and use the level to pick
    caps. We don't drive the LLM; we patch resolve_difficulty so it
    returns a known level, and patch ``_run_llm_loop`` to capture the
    prompt the planner sends. Asserting on the prompt contents is the
    most direct proof that the difficulty level threaded through.
    """
    from infinidev.engine.analysis import planner as planner_mod
    from infinidev.engine.orchestration.escalation_packet import EscalationPacket

    captured: dict[str, object] = {}

    class _StubDecision:
        def __init__(self) -> None:
            self.level = "easy"
            self.confidence = 0.9
            self.reason = "stubbed for test"

    def _fake_resolve(req, *, opened_files=()):
        captured["request"] = req
        captured["opened_files"] = opened_files
        return _StubDecision()

    # ``run_planner`` imports ``resolve_difficulty`` lazily inside the
    # function body, so we patch the source module not the binding.
    monkeypatch.setattr(
        "infinidev.engine.orchestration.difficulty.resolve_difficulty",
        _fake_resolve,
    )

    def _capture_run_llm_loop(*args, **kwargs):
        captured["prompt"] = kwargs.get("messages", [{}])[0].get("content", "")
        raise RuntimeError("stop after prompt build")

    monkeypatch.setattr(planner_mod, "_run_llm_loop", _capture_run_llm_loop)
    # run_planner catches broad exceptions and converts them to a
    # fallback plan; we patch _fallback_plan to re-raise so the test
    # still sees the capture.
    def _re_raise(*args, **kwargs):
        raise RuntimeError("stop after prompt build")
    monkeypatch.setattr(planner_mod, "_fallback_plan", _re_raise)

    packet = EscalationPacket(
        user_request="Fix the typo in README.md.",
        understanding="typo fix",
        opened_files=["README.md"],
    )

    with pytest.raises(RuntimeError, match="stop after prompt build"):
        planner_mod.run_planner(packet, session_id="s", project_id=1)

    # resolve_difficulty saw the user_request and opened_files from the
    # escalation packet.
    assert captured["request"] == "Fix the typo in README.md."
    assert captured["opened_files"] == ("README.md",)
    # The easy prompt reached the LLM message.
    prompt = captured["prompt"]
    assert isinstance(prompt, str)
    assert "EASY" in prompt
    # And the easy prompt's "1–2 focused Steps" instruction is present.
    assert "1–2" in prompt or "1-2" in prompt


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
