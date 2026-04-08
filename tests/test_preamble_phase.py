"""Pipeline-level tests for the mandatory pre-analysis preamble.

These tests use a fake ``OrchestrationHooks`` and monkeypatch the
internal ``try_conversational_fastpath`` so no real LLM call is
made. They verify the contract the user asked for:

  1. The preamble runs even when ``ANALYSIS_ENABLED=False``. The
     gate must NOT silence the preamble — that gate is reserved
     for the heavy analyst phase.
  2. When the preamble decides ``status="done"`` the pipeline
     short-circuits: gather/execute/review are skipped and the
     reply text comes back as the function result.
  3. When the preamble decides ``status="continue"`` the reply is
     shown to the user and the pipeline falls through to the
     analyst.
  4. The ``preamble`` phase is published via ``hooks.on_phase``
     so UIs can render an indicator.

These tests intentionally avoid pexpect / TUI: they call
``run_task`` directly with a fake hooks object, which is exactly
what the TUI worker does. If these pass, the TUI receives the
same notifications.
"""

from __future__ import annotations

from typing import Any

import pytest

from infinidev.engine.analysis.analysis_result import AnalysisResult


# ─────────────────────────────────────────────────────────────────────
# Test doubles
# ─────────────────────────────────────────────────────────────────────


class FakeHooks:
    """Records every hook call so tests can assert on the trace."""

    def __init__(self) -> None:
        self.phases: list[str] = []
        self.statuses: list[tuple[str, str]] = []
        self.notifications: list[tuple[str, str, str]] = []
        self.questions: list[tuple[str, str]] = []
        self.steps: list[tuple[int, int]] = []
        self.file_changes: list[str] = []

    def on_phase(self, phase: str) -> None:
        self.phases.append(phase)

    def on_status(self, level: str, msg: str) -> None:
        self.statuses.append((level, msg))

    def notify(self, speaker: str, msg: str, kind: str = "agent") -> None:
        self.notifications.append((speaker, msg, kind))

    def ask_user(self, prompt: str, kind: str = "text") -> str | None:
        self.questions.append((prompt, kind))
        return None  # non-interactive

    def on_step_start(
        self,
        step_num: int,
        total: int,
        all_steps: list[dict],
        completed: list[int],
    ) -> None:
        self.steps.append((step_num, total))

    def on_file_change(self, path: str) -> None:
        self.file_changes.append(path)


class _StubAgent:
    """Minimal agent — only the attributes the pipeline touches."""

    def __init__(self) -> None:
        self._session_summaries: list[str] = []
        self._system_prompt_identity = ""
        self.backstory = ""

    def activate_context(self, **kwargs: Any) -> None:
        pass

    def deactivate(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def patch_db(monkeypatch):
    """Stub out the conversation history fetch so no SQLite is needed."""
    from infinidev.db import service as db_service

    monkeypatch.setattr(
        db_service, "get_recent_summaries", lambda session_id, limit=10: []
    )
    return monkeypatch


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────


def _patch_preamble(monkeypatch, status: str, message: str):
    """Force the preamble to return a deterministic decision."""
    from infinidev.engine.orchestration import conversational_fastpath

    def fake(user_input, session_summaries=None, session_id=None):
        result = AnalysisResult(
            action="passthrough",
            original_input=user_input,
            reason=message,
            flow="develop" if status == "continue" else "done",
        )
        return result, message, status == "continue"

    monkeypatch.setattr(
        conversational_fastpath, "try_conversational_fastpath", fake
    )
    # The pipeline imports the symbol locally inside _run_preamble_phase,
    # so the module-level patch above is enough — the local import
    # resolves to the patched module.


def test_preamble_runs_when_analysis_disabled(patch_db, monkeypatch):
    """The preamble must NOT be gated by ANALYSIS_ENABLED.

    With ANALYSIS_ENABLED=False the analyst is bypassed, but the
    preamble must still run, still notify the user, and still
    short-circuit on ``done``.
    """
    from infinidev.config.settings import settings
    from infinidev.engine.orchestration.pipeline import run_task

    monkeypatch.setattr(settings, "ANALYSIS_ENABLED", False)
    _patch_preamble(monkeypatch, "done", "Hola! Estoy aquí.")

    hooks = FakeHooks()
    result = run_task(
        agent=_StubAgent(),
        user_input="hola",
        session_id="test-session",
        engine=object(),  # never reached
        analyst=object(),  # never reached
        reviewer=object(),  # never reached
        hooks=hooks,
    )

    # The preamble's reply came back as the function result.
    # The caller (TUI worker / classic loop) is responsible for
    # rendering it through its normal final-output path.
    assert result == "Hola! Estoy aquí."

    # CRITICAL: the preamble must NOT have notified the reply on
    # the ``done`` path. Doing so causes a duplicate message in
    # the TUI because the worker also calls add_message(result).
    # See the regression at conversational_fastpath / preamble.
    assert all(
        msg != "Hola! Estoy aquí." for _speaker, msg, _kind in hooks.notifications
    ), f"preamble notified on done path: {hooks.notifications}"

    # The preamble phase was published, then idle. Analysis/gather/
    # execute/review must NOT have been published.
    assert "preamble" in hooks.phases
    assert "analysis" not in hooks.phases
    assert "gather" not in hooks.phases
    assert "execute" not in hooks.phases
    assert "review" not in hooks.phases
    assert hooks.phases[-1] == "idle"


def test_preamble_done_short_circuits_even_when_analysis_enabled(
    patch_db, monkeypatch
):
    """Even with the analyst available, ``done`` must skip it."""
    from infinidev.config.settings import settings
    from infinidev.engine.orchestration.pipeline import run_task

    monkeypatch.setattr(settings, "ANALYSIS_ENABLED", True)
    _patch_preamble(monkeypatch, "done", "Hi there.")

    # Tripwire: if the analyst is touched the test fails loudly.
    class _ExplodingAnalyst:
        def reset(self):
            raise AssertionError("analyst.reset() must not be called on done")

        def analyze(self, *a, **kw):
            raise AssertionError("analyst.analyze() must not be called on done")

    hooks = FakeHooks()
    result = run_task(
        agent=_StubAgent(),
        user_input="thanks",
        session_id="test-session",
        engine=object(),
        analyst=_ExplodingAnalyst(),
        reviewer=object(),
        hooks=hooks,
    )

    assert result == "Hi there."
    assert "analysis" not in hooks.phases


def test_preamble_continue_publishes_phase_and_falls_through(
    patch_db, monkeypatch
):
    """``continue`` must show the preview AND let the analyst run."""
    from infinidev.config.settings import settings
    from infinidev.engine.orchestration.pipeline import run_task

    monkeypatch.setattr(settings, "ANALYSIS_ENABLED", True)
    _patch_preamble(
        monkeypatch, "continue", "I'll read src/auth.py to find the bug."
    )

    # Stub the analyst so it raises a sentinel exception immediately
    # after being entered. This proves the pipeline reached the
    # analysis phase without us having to drag the whole engine in.
    class _SentinelAnalyst:
        def reset(self):
            raise RuntimeError("analyst-reached")

    hooks = FakeHooks()
    with pytest.raises(RuntimeError, match="analyst-reached"):
        run_task(
            agent=_StubAgent(),
            user_input="fix the auth bug",
            session_id="test-session",
            engine=object(),
            analyst=_SentinelAnalyst(),
            reviewer=object(),
            hooks=hooks,
        )

    # Reply was shown.
    assert (
        "Infinidev",
        "I'll read src/auth.py to find the bug.",
        "agent",
    ) in hooks.notifications

    # Both phases were published, in order.
    assert hooks.phases.index("preamble") < hooks.phases.index("analysis")
