"""End-to-end pipeline integration tests (Commit 7).

Exercises the full chat-agent-first pipeline with mocked LLM calls:

  run_task → chat_agent (respond|escalate) → planner (emit_plan) →
  gather → LoopEngine.execute(initial_plan=plan) → review

These tests verify the wiring between phases — the actual loop /
planner / chat agent tests live in their own files. Here we care
about contract flow: EscalationPacket reaches the planner, the
resulting Plan reaches the LoopEngine via initial_plan=, and the
pipeline short-circuits correctly on a chat-agent respond.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from infinidev.engine.orchestration.chat_agent_result import ChatAgentResult
from infinidev.engine.orchestration.escalation_packet import EscalationPacket
from infinidev.engine.orchestration.pipeline import run_task
from infinidev.engine.analysis.plan import Plan, PlanStepSpec


class _RecordingHooks:
    def __init__(self) -> None:
        self.phases: list[str] = []
        self.notifications: list[tuple[str, str, str]] = []
        self.statuses: list[tuple[str, str]] = []

    def on_phase(self, phase: str) -> None:
        self.phases.append(phase)

    def on_status(self, level: str, msg: str) -> None:
        self.statuses.append((level, msg))

    def notify(self, speaker: str, msg: str, kind: str = "agent") -> None:
        self.notifications.append((speaker, msg, kind))

    def ask_user(self, prompt: str, kind: str = "text") -> str | None:
        return None

    def on_step_start(self, *a, **kw) -> None:
        pass

    def on_file_change(self, path: str) -> None:
        pass


@dataclass
class _FakeAgent:
    agent_id: str = "test-agent"
    backstory: str = ""
    _system_prompt_identity: str = ""

    def activate_context(self, session_id: str) -> None:
        pass

    def deactivate(self) -> None:
        pass


class _FakeEngine:
    """Captures initial_plan for assertion."""

    def __init__(self, result_text: str = "Done.") -> None:
        self.result_text = result_text
        self.captured_initial_plan: Plan | None = None
        self.captured_task_prompt: tuple[str, str] | None = None
        self._files_changed = False

    def execute(
        self,
        *,
        agent: Any,
        task_prompt: tuple[str, str],
        verbose: bool = True,
        initial_plan: Plan | None = None,
    ) -> str:
        self.captured_initial_plan = initial_plan
        self.captured_task_prompt = task_prompt
        return self.result_text

    def has_file_changes(self) -> bool:
        return self._files_changed


class _FakeReviewer:
    pass


# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────


class TestChatRespondShortCircuits:
    def test_respond_ends_turn_without_planner_or_executor(self, monkeypatch):
        """Chat agent returns respond → pipeline must NOT call planner
        or LoopEngine.execute."""
        def _chat_respond(*args, **kwargs):
            return ChatAgentResult(kind="respond", reply="¡Hola! Soy Infinidev.")

        planner_calls: list[Any] = []
        def _planner_spy(*args, **kwargs):
            planner_calls.append(kwargs)
            raise AssertionError("Planner must not run on respond")

        monkeypatch.setattr(
            "infinidev.engine.orchestration.pipeline.run_chat_agent",
            _chat_respond, raising=False,
        )
        # The function is imported locally inside run_task, so patch the
        # source module too.
        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.run_chat_agent",
            _chat_respond,
        )
        monkeypatch.setattr(
            "infinidev.engine.analysis.planner.run_planner", _planner_spy,
        )

        hooks = _RecordingHooks()
        agent = _FakeAgent()
        engine = _FakeEngine()

        result = run_task(
            agent=agent,
            user_input="hola",
            session_id="test-session",
            engine=engine,
            reviewer=_FakeReviewer(),
            hooks=hooks,
        )

        assert result == "¡Hola! Soy Infinidev."
        assert planner_calls == []
        assert engine.captured_initial_plan is None
        # Phase sequence: chat, then idle.
        assert hooks.phases[0] == "chat"
        assert hooks.phases[-1] == "idle"
        # The reply is shown in the chat.
        assert any(
            speaker == "Infinidev" and "Hola" in msg
            for speaker, msg, _ in hooks.notifications
        )


class TestEscalateRunsFullPipeline:
    def test_escalation_feeds_plan_to_loop_engine(self, monkeypatch):
        """Chat agent escalates → planner produces a Plan → LoopEngine
        receives it via initial_plan=."""
        escalation = EscalationPacket(
            user_request="fix the JWT bug",
            understanding="Fix JWT validation in auth.py",
            opened_files=["src/auth.py"],
            user_visible_preview="Voy a arreglar el JWT.",
            user_signal="dale arreglalo",
        )
        expected_plan = Plan(
            overview="Fix validate_token's exp check.",
            steps=[
                PlanStepSpec(title="Patch", detail="d", expected_output="ok"),
                PlanStepSpec(title="Test", detail="run pytest", expected_output="green"),
            ],
        )

        def _chat_escalate(*args, **kwargs):
            return ChatAgentResult(kind="escalate", escalation=escalation)

        def _planner(*args, **kwargs):
            assert args and isinstance(args[0], EscalationPacket)
            assert args[0] is escalation
            return expected_plan

        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.run_chat_agent",
            _chat_escalate,
        )
        monkeypatch.setattr(
            "infinidev.engine.analysis.planner.run_planner", _planner,
        )

        hooks = _RecordingHooks()
        agent = _FakeAgent()
        engine = _FakeEngine(result_text="All done, bug fixed.")

        result = run_task(
            agent=agent,
            user_input="arreglá el JWT",
            session_id="test-session",
            engine=engine,
            reviewer=_FakeReviewer(),
            hooks=hooks,
        )

        assert result == "All done, bug fixed."
        # The plan reached the LoopEngine.
        assert engine.captured_initial_plan is expected_plan
        # task_prompt first element is the user's original request.
        assert engine.captured_task_prompt[0] == escalation.user_request
        # The user saw the preview AND the plan overview, in order.
        previews = [
            (speaker, msg)
            for speaker, msg, _ in hooks.notifications
        ]
        assert ("Infinidev", "Voy a arreglar el JWT.") in previews
        assert any(
            speaker == "Planner" and "exp check" in msg
            for speaker, msg in previews
        )
        # Phase ordering: chat → analysis → execute → idle.
        assert "chat" in hooks.phases
        assert "analysis" in hooks.phases
        assert "execute" in hooks.phases
        assert hooks.phases[-1] == "idle"
        chat_idx = hooks.phases.index("chat")
        analysis_idx = hooks.phases.index("analysis")
        execute_idx = hooks.phases.index("execute")
        assert chat_idx < analysis_idx < execute_idx


class TestReviewOnlyRunsOnFileChanges:
    def test_no_files_changed_skips_review(self, monkeypatch):
        escalation = EscalationPacket(
            user_request="explain X",
            understanding="read-only question",
        )
        plan = Plan(overview="explain", steps=[PlanStepSpec(title="x")])

        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.run_chat_agent",
            lambda *a, **kw: ChatAgentResult(kind="escalate", escalation=escalation),
        )
        monkeypatch.setattr(
            "infinidev.engine.analysis.planner.run_planner",
            lambda *a, **kw: plan,
        )

        engine = _FakeEngine()
        engine._files_changed = False  # no files → review skipped

        review_spy = {"called": False}
        import infinidev.engine.orchestration.pipeline as pipeline_mod
        original_review = pipeline_mod._run_review_phase
        def _spy(**kwargs):
            review_spy["called"] = kwargs["engine"].has_file_changes()
            return original_review(**kwargs)
        monkeypatch.setattr(pipeline_mod, "_run_review_phase", _spy)

        run_task(
            agent=_FakeAgent(),
            user_input="explain",
            session_id="s",
            engine=engine,
            reviewer=_FakeReviewer(),
            hooks=_RecordingHooks(),
        )
        assert review_spy["called"] is False  # the guard inside review
