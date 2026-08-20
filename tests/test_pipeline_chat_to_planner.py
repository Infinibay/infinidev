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
from infinidev.engine.analysis.staged_planning import (
    CompleteGoalDecision,
    EmitStageDecision,
    StageSpec,
    StageTaskSpec,
)
from infinidev.tools.base.context import (
    clear_agent_context,
    get_context_for_agent,
    set_context,
)


@pytest.fixture(autouse=True)
def _single_stage_planner(temp_db, monkeypatch):
    """Keep pipeline tests focused while exercising the real staged wiring."""
    def decide(state, **_kwargs):
        if state.stages:
            return CompleteGoalDecision(
                evidence=[
                    f"{state.evidence[-1].id}: The Stage Task result is present "
                    "in the evidence ledger"
                ]
            )
        return EmitStageDecision(stage=StageSpec(
            title="Execute the requested change",
            outcome="The requested change is implemented and reviewed",
            exit_criteria=["The Task execution and review produce evidence"],
            tasks=[StageTaskSpec(
                id="request",
                title="Implement the request",
                outcome="The requested behavior is implemented",
                acceptance_criteria=["The requested behavior is checked"],
            )],
        ))

    monkeypatch.setattr(
        "infinidev.engine.analysis.stage_planner.run_stage_planner", decide
    )


@pytest.fixture(autouse=True)
def _pin_staged_engine(monkeypatch):
    """These tests verify the staged plan→loop handoff, so pin the engine.

    The normal ``TASK_ENGINE_MODE`` is ``task`` and deliberately bypasses the
    planner. These tests assert
    the staged contract (EscalationPacket → planner → ``initial_plan=``), so
    they select the staged engine explicitly.
    """
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "TASK_ENGINE_MODE", "staged")


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
        initial_attachments: list[Any] | None = None,
        task: Any | None = None,
        max_total_tool_calls: int | None = None,
        max_prompt_tokens: int | None = None,
        allow_plan_mutation: bool = True,
    ) -> str:
        self.captured_initial_plan = initial_plan
        self.captured_task_prompt = task_prompt
        self.captured_initial_attachments = initial_attachments
        self.captured_task = task
        self.captured_max_total_tool_calls = max_total_tool_calls
        self.captured_max_prompt_tokens = max_prompt_tokens
        self.captured_allow_plan_mutation = allow_plan_mutation
        return self.result_text

    def has_file_changes(self) -> bool:
        return self._files_changed


class _FakeReviewer:
    pass


class _SnapshotReviewer:
    _prompt_configuration = None


# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────


class TestPromptSnapshotLifecycle:
    def test_escalated_turn_loads_profiles_once_and_shares_the_snapshot(
        self, tmp_path, monkeypatch,
    ):
        from infinidev.engine.engines.base import EngineResult
        from infinidev.prompts import profiles

        profile_path = tmp_path / "prompts.json"
        profile_path.write_text(
            json.dumps({"develop": {"loop.identity": False}}),
            encoding="utf-8",
        )
        real_loader = profiles.load_prompt_profiles
        reads = 0
        observed = {}

        def tracked_loader(path=None):
            nonlocal reads
            reads += 1
            return real_loader(path)

        escalation = EscalationPacket(
            user_request="implement the requested change",
            understanding="Implement the requested change",
        )

        def chat_agent(*_args, **kwargs):
            observed["chat"] = kwargs["prompt_configuration"]
            profile_path.write_text(
                json.dumps({"develop": {"loop.identity": True}}),
                encoding="utf-8",
            )
            return ChatAgentResult(kind="escalate", escalation=escalation)

        engine = _FakeEngine(result_text="Done with one prompt snapshot.")

        def selected_engine(**kwargs):
            observed["engine"] = kwargs["prompt_configuration"]
            return EngineResult(
                engine_name="task",
                status="completed",
                user_message=engine.result_text,
                engine=engine,
            )

        monkeypatch.setattr(profiles, "get_prompt_profile_path", lambda: profile_path)
        monkeypatch.setattr(profiles, "load_prompt_profiles", tracked_loader)
        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.run_chat_agent",
            chat_agent,
        )
        monkeypatch.setattr(
            "infinidev.engine.engines.run_selected_engine",
            selected_engine,
        )
        monkeypatch.setattr(
            "infinidev.engine.orchestration.pipeline._run_elaboration_phase",
            lambda **kwargs: kwargs["escalation"],
        )
        monkeypatch.setattr(
            "infinidev.engine.orchestration.pipeline._run_council_phase",
            lambda **kwargs: kwargs["escalation"],
        )

        reviewer = _SnapshotReviewer()
        result = run_task(
            agent=_FakeAgent(),
            user_input=escalation.user_request,
            session_id="prompt-snapshot",
            engine=engine,
            reviewer=reviewer,
            hooks=_RecordingHooks(),
        )

        assert result == engine.result_text
        assert reads == 1
        assert observed["chat"] is observed["engine"]
        assert reviewer._prompt_configuration is observed["chat"]
        assert observed["chat"].resolve("develop", "loop.identity").enabled is False


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

        # Patch the SOURCE modules: run_task imports both functions
        # lazily (local `from ... import` inside the body) to break a
        # circular import. Each call re-reads the current module state,
        # so monkeypatching the source module is what takes effect.
        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.run_chat_agent",
            _chat_respond,
        )
        monkeypatch.setattr(
            "infinidev.engine.analysis.planner.run_planner",
            _planner_spy,
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


    def test_continuation_attaches_nested_repository_before_chat(
        self, monkeypatch, tmp_path
    ):
        repository = tmp_path / "infinigpu"
        (repository / ".git").mkdir(parents=True)
        (repository / "CONTINUE.md").write_text("Continue the implementation.\n")
        captured: dict[str, Any] = {}

        def _chat_respond(user_input, **kwargs):
            captured["input"] = user_input
            captured["workspace_path"] = kwargs["workspace_path"]
            return ChatAgentResult(kind="respond", reply="captured")

        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.run_chat_agent",
            _chat_respond,
        )
        set_context(
            project_id=1,
            agent_id="test-agent",
            workspace_path=str(tmp_path),
        )
        hooks = _RecordingHooks()
        engine = _FakeEngine()
        try:
            run_task(
                agent=_FakeAgent(),
                user_input="Lee infinigpu/CONTINUE.md y continua el trabajo",
                session_id="nested-repo-session",
                engine=engine,
                reviewer=_FakeReviewer(),
                hooks=hooks,
            )

            context = get_context_for_agent("test-agent")
            assert context.workspace_path == str(tmp_path)
            assert context.repository_path == str(repository)
            assert engine._repository_path == str(repository)
            assert captured["workspace_path"] == str(tmp_path)
            assert f"Target Git repository: {repository}" in captured["input"]
            assert ("info", "Target repository: infinigpu") in hooks.statuses
        finally:
            clear_agent_context("test-agent")

    def test_continuation_resolves_nested_repository_from_cwd_without_context(
        self, monkeypatch, tmp_path
    ):
        repository = tmp_path / "infinigpu"
        (repository / ".git").mkdir(parents=True)
        (repository / "CONTINUE.md").write_text("Continue the implementation.\n")
        captured: dict[str, Any] = {}

        def _chat_respond(user_input, **kwargs):
            captured["input"] = user_input
            captured["workspace_path"] = kwargs["workspace_path"]
            return ChatAgentResult(kind="respond", reply="captured")

        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.run_chat_agent",
            _chat_respond,
        )
        monkeypatch.chdir(tmp_path)
        clear_agent_context("test-agent")
        hooks = _RecordingHooks()
        engine = _FakeEngine()
        try:
            run_task(
                agent=_FakeAgent(),
                user_input="Lee infinigpu/CONTINUE.md y continua el trabajo",
                session_id="nested-repo-cwd-session",
                engine=engine,
                reviewer=_FakeReviewer(),
                hooks=hooks,
            )

            assert engine._repository_path == str(repository)
            assert captured["workspace_path"] == str(tmp_path)
            assert f"Target Git repository: {repository}" in captured["input"]
            assert ("info", "Target repository: infinigpu") in hooks.statuses
        finally:
            clear_agent_context("test-agent")


class TestEscalateRunsFullPipeline:
    def test_escalation_feeds_plan_to_loop_engine(self, monkeypatch):
        """Chat agent escalates → planner produces a Plan → LoopEngine
        receives it via initial_plan=."""
        escalation = EscalationPacket(
            user_request=(
                "arreglá el JWT\n\n"
                "<retrieval-context>weigh several alternatives</retrieval-context>"
            ),
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
            assert args[0] is not escalation
            assert args[0].user_request == "arreglá el JWT"
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
        engine._files_changed = True

        result = run_task(
            agent=agent,
            user_input="arreglá el JWT",
            session_id="test-session",
            engine=engine,
            reviewer=_FakeReviewer(),
            hooks=hooks,
        )

        assert result == "All done, bug fixed."
        # The planner's tactics reach LoopEngine unchanged, followed by the
        # scheduler-owned acceptance-coverage repair when needed.
        assert engine.captured_initial_plan.steps[:2] == expected_plan.steps
        assert len(engine.captured_initial_plan.steps) == 3
        assert "verify remaining" in engine.captured_initial_plan.steps[2].title
        assert expected_plan.overview in engine.captured_initial_plan.overview
        # task_prompt first element is the user's original request.
        assert "arreglá el JWT" in engine.captured_task_prompt[0]
        assert "weigh several alternatives" not in engine.captured_task_prompt[0]
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


# ─────────────────────────────────────────────────────────────────────────
# Ken sees the turn
# ─────────────────────────────────────────────────────────────────────────
#
# Ken's ranker is fed by the session, not by the query. These cover the
# wiring: which of the pipeline's exits report the turn, and where the two
# blocks Ken answers with actually land.


class _FakeKen:
    def __init__(self) -> None:
        self.brief: str | None = "<ken-session-brief>last time: auth.py</ken-session-brief>"
        self.ranked: str | None = "<context-rank>src/auth.py</context-rank>"
        self.prompts: list[str] = []
        self.turn_ends: list[str] = []
        self.starts = 0

    def start(self, workspace=None):
        self.starts += 1
        return self.brief if self.starts == 1 else None

    def prompt(self, text):
        self.prompts.append(text)
        return self.ranked

    def turn_end(self, assistant_text=""):
        self.turn_ends.append(assistant_text)


@pytest.fixture
def ken(monkeypatch):
    fake = _FakeKen()
    monkeypatch.setattr(
        "infinidev.engine.ken_session.get_ken_session",
        lambda workspace=None, session_id=None: fake,
    )
    return fake


class TestKenSeesTheTurn:
    def test_a_chat_only_turn_is_still_a_turn(self, monkeypatch, ken):
        """The reply the chat agent produced is scanned for cited paths just
        like the developer's, and the turn advances Ken's decay clock either
        way. Reporting only the develop path left every conversational
        exchange invisible to the ranker."""
        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.run_chat_agent",
            lambda *a, **kw: ChatAgentResult(
                kind="respond", reply="It lives in src/auth.py."
            ),
        )

        run_task(
            agent=_FakeAgent(), user_input="where is the JWT check?",
            session_id="s", engine=_FakeEngine(), reviewer=_FakeReviewer(),
            hooks=_RecordingHooks(),
        )

        assert ken.prompts == ["where is the JWT check?"]
        assert ken.turn_ends == ["It lives in src/auth.py."]

    def test_kens_blocks_reach_the_chat_agent(self, monkeypatch, ken):
        """The chat agent decides whether to escalate. Handing it the ranked
        context after that decision would be too late to matter."""
        seen: dict[str, str] = {}

        def _chat(message, *a, **kw):
            seen["input"] = message
            return ChatAgentResult(kind="respond", reply="ok")

        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.run_chat_agent", _chat,
        )

        run_task(
            agent=_FakeAgent(), user_input="fix the JWT bug", session_id="s",
            engine=_FakeEngine(), reviewer=_FakeReviewer(),
            hooks=_RecordingHooks(),
        )

        assert "fix the JWT bug" in seen["input"]
        assert "<ken-session-brief>" in seen["input"]
        assert "<context-rank>" in seen["input"]

    def test_kens_blocks_reach_the_developer(self, monkeypatch, ken):
        """They ride the task description, never the expected_output — that
        one is the flow's contract and the reviewer judges against it."""
        escalation = EscalationPacket(
            user_request="fix the JWT bug",
            understanding="Fix JWT validation",
            opened_files=["src/auth.py"],
            user_visible_preview="Voy.",
            user_signal="dale",
        )
        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.run_chat_agent",
            lambda *a, **kw: ChatAgentResult(kind="escalate", escalation=escalation),
        )
        monkeypatch.setattr(
            "infinidev.engine.analysis.planner.run_planner",
            lambda *a, **kw: Plan(
                overview="Patch it.",
                steps=[PlanStepSpec(title="Patch", detail="d", expected_output="ok")],
            ),
        )

        engine = _FakeEngine(result_text="Fixed src/auth.py.")
        engine._files_changed = True
        run_task(
            agent=_FakeAgent(), user_input="fix the JWT bug", session_id="s",
            engine=engine, reviewer=_FakeReviewer(), hooks=_RecordingHooks(),
        )

        description, expected_output = engine.captured_task_prompt
        assert "<context-rank>" in description
        assert "<context-rank>" not in expected_output
        assert ken.turn_ends == ["Fixed src/auth.py."]

    def test_the_session_is_opened_once_per_conversation(self, monkeypatch, ken):
        """Not once per task: /sessions/start INSERTs a fresh cr_sessions row,
        so a row per turn shreds one conversation into many and restarts the
        per-turn decay counter each time."""
        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.run_chat_agent",
            lambda *a, **kw: ChatAgentResult(kind="respond", reply="ok"),
        )

        for text in ("first", "second", "third"):
            run_task(
                agent=_FakeAgent(), user_input=text, session_id="s",
                engine=_FakeEngine(), reviewer=_FakeReviewer(),
                hooks=_RecordingHooks(),
            )

        assert ken.starts == 3, "every turn asks; the client decides"
        assert ken.prompts == ["first", "second", "third"]
        assert len(ken.turn_ends) == 3
