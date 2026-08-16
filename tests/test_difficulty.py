"""Tests for the deterministic difficulty classifier."""

from __future__ import annotations

import pytest

from infinidev.engine.orchestration.difficulty import (
    DIFFICULTY_LEVELS,
    DifficultyDecision,
    DifficultyLevel,
    resolve_difficulty,
)


# --- Returns and shape ------------------------------------------------------


def test_returns_difficulty_decision():
    decision = resolve_difficulty("Fix the typo in README.md")
    assert isinstance(decision, DifficultyDecision)
    assert decision.level in DIFFICULTY_LEVELS
    assert isinstance(decision.confidence, float)
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.reason  # non-empty human-readable reason
    assert isinstance(decision.signals, dict)
    assert decision.signals  # non-empty


def test_empty_input_defaults_to_hard():
    decision = resolve_difficulty("")
    assert decision.level == "hard"
    assert decision.confidence == 0.0
    assert "empty" in decision.reason.lower()


def test_whitespace_only_input_defaults_to_hard():
    decision = resolve_difficulty("   \n\t  ")
    assert decision.level == "hard"


# --- Easy path --------------------------------------------------------------


def test_typo_fix_is_easy():
    decision = resolve_difficulty("Fix the typo in README.md.")
    assert decision.level == "easy"


def test_short_rename_is_easy():
    decision = resolve_difficulty("Rename foo to bar in src/utils.py")
    assert decision.level == "easy"


def test_quick_fix_is_easy():
    decision = resolve_difficulty("Quick fix for the lint warning")
    assert decision.level == "easy"


def test_short_request_with_single_file_is_easy():
    decision = resolve_difficulty("Add docstring to src/x.py")
    assert decision.level == "easy"


def test_bump_version_is_easy():
    decision = resolve_difficulty("Bump the version in pyproject.toml")
    assert decision.level == "easy"


# --- Hard path --------------------------------------------------------------


def test_refactor_is_hard():
    decision = resolve_difficulty("Refactor the auth module")
    assert decision.level == "hard"


def test_investigate_is_hard():
    decision = resolve_difficulty(
        "Investigate why the planner emits too many steps"
    )
    assert decision.level == "hard"


def test_analyze_in_detail_is_hard():
    decision = resolve_difficulty(
        "Analyze the engine loop in detail and report token usage"
    )
    assert decision.level == "hard"


def test_design_new_api_is_hard():
    decision = resolve_difficulty("Design a new API for streaming responses")
    assert decision.level == "hard"


def test_many_files_is_hard():
    decision = resolve_difficulty(
        "Update these files: a.py b.py c.py d.py",
        opened_files=["a.py", "b.py", "c.py", "d.py"],
    )
    assert decision.level == "hard"


def test_long_session_is_hard():
    decision = resolve_difficulty(
        "Continue with the refactor",
        prior_turn_count=20,
    )
    assert decision.level == "hard"


def test_migration_keyword_is_hard():
    decision = resolve_difficulty("Migrate the schema to v2")
    assert decision.level == "hard"


# --- Medium path ------------------------------------------------------------


def test_grounded_execution_request_is_medium():
    decision = resolve_difficulty(
        "Add a JWT auth on /login and verify with pytest tests/test_auth.py"
    )
    assert decision.level in ("medium", "hard")


def test_ambiguous_request_is_medium():
    decision = resolve_difficulty("Look at the code")
    assert decision.level == "medium"


# --- Default-to-hard preservation -------------------------------------------


def test_unknown_request_without_grounding_is_not_easy():
    """A request with no clear signal must not collapse to easy."""
    decision = resolve_difficulty(
        "Help me with some thing in the project"  # vague, no path, no verb
    )
    assert decision.level != "easy"


def test_explicit_difficulty_in_task_construction_preserves_value():
    """The schema must honour an explicit difficulty over the resolver."""
    from infinidev.engine.orchestration.task_schema import (
        Task,
        task_from_free_text,
    )

    # Direct construction
    t = Task(
        title="Fix typo in README",
        description="Fix the typo in README.md.",
        kind="bugfix",
        acceptance_criteria=["README is updated"],
        difficulty="easy",
    )
    assert t.difficulty == "easy"

    # task_from_free_text explicit override
    t2 = task_from_free_text(
        "Fix the typo in README.md",
        difficulty="easy",
    )
    assert t2.difficulty == "easy"


# --- Signals exposed --------------------------------------------------------


def test_signals_include_required_keys():
    decision = resolve_difficulty("Refactor the planner", prior_turn_count=3)
    for key in (
        "char_count",
        "path_count",
        "easy_keyword_hits",
        "hard_keyword_hits",
        "execution_score",
        "prior_turn_count",
    ):
        assert key in decision.signals, f"missing signal: {key}"
    assert decision.signals["prior_turn_count"] == 3


def test_confidence_is_monotonic_in_hits():
    easy_one = resolve_difficulty("typo")
    # Force the easy path by adding corroborating signals.
    easy_many = resolve_difficulty(
        "Quick fix for a small typo in README.md",
        opened_files=["README.md"],
    )
    assert easy_many.confidence >= easy_one.confidence


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])