"""End-to-end tests for the autonomous ("manejate vos") chain.

Wires together three concerns that the budget-only tests in
``test_autonomous_budget.py`` leave on the shelf:

  * the chat-agent-side intent detector (``detect_autonomous_intent`` /
    ``apply_autonomous_to_packet``);
  * the pipeline-side ``autonomous`` kwarg that drives chained plans;
  * the per-tope stopping conditions evaluated against a stub engine.

The integration check exercises the real
:func:`infinidev.engine.orchestration.pipeline.run_task` end to end with
its chat agent and selected engine stubbed — same strategy used by
``test_pipeline_chat_to_planner.py``. We do not need to spin up the LLM
loop; the chain lives between ``run_task`` and the engine adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from infinidev.engine.orchestration.autonomous import (
    AutonomousBudget,
    DEFAULT_IDLE_PASSES,
    DEFAULT_MAX_PLANS,
    apply_autonomous_to_packet,
    detect_autonomous_intent,
    should_continue,
)
from infinidev.engine.orchestration.chat_agent_result import ChatAgentResult
from infinidev.engine.orchestration.escalation_packet import EscalationPacket
from infinidev.engine.orchestration.pipeline import run_task
from infinidev.tools.base.context import (
    clear_agent_context,
    set_context,
)


# ─────────────────────────────────────────────────────────────────────────
# Intent detector — pure-function unit tests
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "phrase",
    [
        "manejate vos",
        "manejáte vos",
        "siga investigando",
        "siga vos",
        "vos solo",
        "tomá el control",
        "seguí adelante",
        "sin preguntarme",
        "you handle it",
        "keep going",
        "do your thing",
    ],
)
def test_detect_autonomous_intent_matches_known_phrases(phrase: str) -> None:
    """Spanish and English variants of the "keep going" intent all hit."""
    assert detect_autonomous_intent(phrase) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "please implement the bug fix in auth.py:412",
        "explain the difference between list and tuple",
        "look at the layout",  # contains "vos" but not the autonomous phrase
        "goodbye",
    ],
)
def test_detect_autonomous_intent_rejects_unrelated_text(text: str) -> None:
    """The detector must NOT match normal conversational requests."""
    assert detect_autonomous_intent(text) is False


def test_apply_autonomous_to_packet_stamps_packet_when_user_signal_matches() -> None:
    """The chat agent stamps the ``user_signal``; the helper must react.

    Mirrors the accepted contract from the Step 3 description: a packet
    whose ``user_signal`` contains "manejate vos" is treated the same
    as if the user had typed it in their original message — the chain
    fires. This protects against the model paraphrasing rather than
    echoing the literal phrase.
    """
    packet = EscalationPacket(
        user_request="fix bug",
        understanding="fix bug",
        user_signal="user said 'maneja vos solo con esto'",
    )
    stamped = apply_autonomous_to_packet(packet)
    assert stamped.autonomous is True
    assert stamped.user_signal == "user said 'maneja vos solo con esto'"


def test_apply_autonomous_to_packet_respects_explicit_hint() -> None:
    """``explicit_hint=True`` overrides any text matching — the caller
    (pipeline autonomous kwarg) drives the chain regardless of wording.
    """
    packet = EscalationPacket(
        user_request="fix bug",
        understanding="fix bug",
    )
    stamped = apply_autonomous_to_packet(packet, explicit_hint=True)
    assert stamped.autonomous is True


def test_apply_autonomous_to_packet_is_idempotent() -> None:
    """Re-applying on an already-stamped packet returns the same object."""
    packet = EscalationPacket(
        user_request="manejate vos",
        understanding="fix bug",
    )
    once = apply_autonomous_to_packet(packet)
    twice = apply_autonomous_to_packet(once)
    assert twice.autonomous is True
    # Once replaces, but the value is unchanged — id is per-build, not stable.
    assert twice.user_request == "manejate vos"


def test_apply_autonomous_to_packet_leaves_unrelated_alone() -> None:
    """Plain requests do not accidentally trigger the chain."""
    packet = EscalationPacket(
        user_request="fix the typo in README",
        understanding="fix typo",
    )
    stamped = apply_autonomous_to_packet(packet)
    assert stamped.autonomous is False


# ─────────────────────────────────────────────────────────────────────────
# Pipeline integration — chain wiring
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeAgent:
    agent_id: str = "auto-test-agent"
    backstory: str = ""
    _system_prompt_identity: str = ""
    project_id: int = 1

    def activate_context(self, session_id: str) -> None:  # pragma: no cover - stub
        pass

    def deactivate(self) -> None:  # pragma: no cover - stub
        pass


class _CountingEngine:
    """Selected-engine stub that records every execute() call.

    Each call appends a tuple ``(plan_index, plan_text)`` so the test can
    assert both that the chain fired the expected number of times AND
    that each re-entry's continuation text reached the developer.
    """

    def __init__(self, plan_count: int, status: str = "completed") -> None:
        self._plan_count = plan_count
        self._status = status
        self.calls: list[tuple[int, str]] = []
        self._last_status = status
        self.is_cancelled = False

    def execute(self, *args: Any, **kwargs: Any) -> str:
        task_prompt = kwargs.get("task_prompt") or args[1] if len(args) > 1 else None
        prompt_user = task_prompt[0] if isinstance(task_prompt, tuple) else ""
        idx = len(self.calls) + 1
        self.calls.append((idx, prompt_user))
        return f"result-{idx}"

    def has_file_changes(self) -> bool:
        return False

    def build_work_summary(self, result: str, status: str) -> str:
        return ""


class _CountingHooks:
    def __init__(self) -> None:
        self.phases: list[str] = []
        self.statuses: list[tuple[str, str]] = []

    def on_phase(self, phase: str) -> None:
        self.phases.append(phase)

    def on_status(self, level: str, msg: str) -> None:
        self.statuses.append((level, msg))

    def notify(self, speaker: str, msg: str, kind: str = "agent") -> None:
        pass

    def notify_stream_chunk(self, *args: Any, **kwargs: Any) -> None:
        pass

    def notify_stream_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def ask_user(self, prompt: str, kind: str = "text") -> str | None:
        return None

    def on_step_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_file_change(self, path: str) -> None:
        pass

    def on_stage_update(self, snapshot: dict[str, Any]) -> None:
        pass


class _FakeReviewer:
    pass


@pytest.fixture(autouse=True)
def _reset_context():
    """Clear any global agent context between tests."""
    try:
        yield
    finally:
        try:
            clear_agent_context("auto-test-agent")
        except Exception:
            pass


def _make_packet(
    *,
    autonomous: bool = False,
    user_signal: str = "",
    user_request: str = "manejate vos, finish the auth migration",
) -> EscalationPacket:
    return EscalationPacket(
        user_request=user_request,
        understanding="continue working",
        user_signal=user_signal,
        suggested_flow="develop",
        autonomous=autonomous,
    )


def _patch_pipeline(monkeypatch, *, chat_packets, engine):
    """Stub the chat agent + selected engine + planner plumbing.

    ``chat_packets`` is a list of packets returned one per chain turn
    (popping from the front each time ``run_chat_agent`` is called).
    ``engine`` is the counting engine stub.
    """
    state = {"chat_calls": 0}

    def _chat(*args: Any, **kwargs: Any) -> ChatAgentResult:
        idx = state["chat_calls"]
        state["chat_calls"] += 1
        try:
            pkt = chat_packets[idx]
            return ChatAgentResult(kind="escalate", escalation=pkt)
        except IndexError:
            # Out of packets → respond "done" so the chain ends gracefully.
            return ChatAgentResult(kind="respond", reply="done")

    # Capture engine via default arg (Python class body has no closure access).
    def _selected_engine(_engine=engine, **kwargs: Any) -> Any:
        class _Result:
            user_message = _engine.execute(
                agent=kwargs.get("agent"),
                task_prompt=("user", "task"),
            )
            engine = _engine
            status = _engine._last_status

        return _Result()

    monkeypatch.setattr(
        "infinidev.engine.orchestration.chat_agent.run_chat_agent",
        _chat,
    )
    monkeypatch.setattr(
        "infinidev.engine.engines.run_selected_engine",
        _selected_engine,
    )
    # Disable gather/elaboration/council paths so the test stays focused
    # on the chain wiring.
    monkeypatch.setattr(
        "infinidev.engine.orchestration.pipeline._run_elaboration_phase",
        lambda **kw: kw["escalation"],
    )
    monkeypatch.setattr(
        "infinidev.engine.orchestration.pipeline._run_council_phase",
        lambda **kw: kw["escalation"],
    )
    # Staged planner shortcut: emit a one-task plan so gather/execute
    # proceed without pulling in the full planner.
    def _task_policies(*args: Any, **kwargs: Any):
        return None

    monkeypatch.setattr(
        "infinidev.engine.orchestration.pipeline.resolve_task_profile",
        _task_policies,
        raising=False,
    )
    # Pin settings to skip the policy resolver path entirely.
    from infinidev.config.settings import settings as _settings

    monkeypatch.setattr(_settings, "TASK_POLICIES_ENABLED", False)
    # Bound the chain so the integration test mirrors the budget-default
    # contract without depending on env vars.
    monkeypatch.setattr(
        _settings, "AUTONOMOUS_IDLE_PASSES", DEFAULT_IDLE_PASSES,
    )
    # test that cares about it (``test_pipeline_stops_after_exactly_max_plans``)
    # sets its own value before calling this fixture. Resetting it would
    # silently overwrite the test's intent and break the assertion.
    monkeypatch.setattr(_settings, "AUTONOMOUS_WALL_SECONDS", 900)
    monkeypatch.setattr(_settings, "AUTONOMOUS_TOKEN_BUDGET", 50_000)
    return state


def test_pipeline_chains_plans_under_budget(monkeypatch) -> None:
    """The chain runs every queued plan while the budget remains.

    Configured with the default ``max_plans=3`` and three packets
    stamped ``autonomous=True``. The pipeline must call the engine
    three times and return after the third plan exhausts the fuse.
    """
    packets = [_make_packet(autonomous=True) for _ in range(3)]
    engine = _CountingEngine(plan_count=10)
    state = _patch_pipeline(monkeypatch, chat_packets=packets, engine=engine)

    set_context(
        agent_id="auto-test-agent",
        project_id=1,
        session_id="autonomous-1",
        workspace_path="/tmp/auto",
    )
    hooks = _CountingHooks()

    result = run_task(
        agent=_FakeAgent(),
        user_input="manejate vos",
        session_id="autonomous-1",
        engine=engine,
        reviewer=_FakeReviewer(),
        hooks=hooks,
        autonomous=True,
    )

    assert len(engine.calls) == 3, (
        f"expected exactly 3 chained plans, got {len(engine.calls)}: {engine.calls}"
    )
    assert state["chat_calls"] == 3, (
        "chat agent must re-run once per chained plan"
    )
    # Final result comes from the last executed engine call.
    assert result == "result-3"
    # Status banner: at least one status line about the autonomous chain.
    status_msgs = [m for _, m in hooks.statuses if "Autonomous chain" in m]
    assert len(status_msgs) >= 1


def test_pipeline_stops_after_exactly_max_plans(monkeypatch) -> None:
    """Once ``max_plans`` is reached, the chain refuses the third call.

    Configured for ``max_plans=2`` with two packets stamped autonomous.
    After the second plan completes, the budget fuse trips and the
    chain returns the second plan's result without producing a third
    chat call (the budget check runs *before* ``run_chat_agent`` on a
    would-be third turn).
    """
    from infinidev.config.settings import settings as _settings

    monkeypatch.setattr(_settings, "AUTONOMOUS_MAX_PLANS", 2)
    monkeypatch.setattr(_settings, "AUTONOMOUS_WALL_SECONDS", 900)
    monkeypatch.setattr(_settings, "AUTONOMOUS_TOKEN_BUDGET", 50_000)

    packets = [_make_packet(autonomous=True) for _ in range(2)]
    engine = _CountingEngine(plan_count=10)
    state = _patch_pipeline(monkeypatch, chat_packets=packets, engine=engine)

    set_context(
        agent_id="auto-test-agent",
        project_id=1,
        session_id="autonomous-2",
        workspace_path="/tmp/auto",
    )
    hooks = _CountingHooks()

    result = run_task(
        agent=_FakeAgent(),
        user_input="manejate vos",
        session_id="autonomous-2",
        engine=engine,
        reviewer=_FakeReviewer(),
        hooks=hooks,
        autonomous=True,
    )

    assert len(engine.calls) == 2, (
        f"chain must stop at max_plans=2, got {len(engine.calls)}: {engine.calls}"
    )
    # Two chat-agent runs (one per chained plan), no third — the chain
    # is gated *after* the developer path, not before the chat agent.
    assert state["chat_calls"] == 2
    assert result == "result-2"
    stop_msgs = [m for _, m in hooks.statuses if "stopped" in m]
    assert any("max_plans=2" in m for m in stop_msgs), (
        f"expected a max_plans stop line, got {stop_msgs}"
    )


def test_chat_agent_user_signal_manejate_vos_enables_autonomous(monkeypatch) -> None:
    """Detection runs on the packet's ``user_signal`` even when the literal
    request did not include the autonomous phrase — the chain still fires.

    This mirrors the Step 3 acceptance criterion that the chat agent
    signals "manejate vos" *in user_signal* turns on autonomous mode
    for that turn: the helper stamps the packet on its way out of the
    chat agent, the pipeline picks it up via ``escalation.autonomous``.
    """
    packet_in = EscalationPacket(
        user_request="please fix the auth migration",
        understanding="fix auth",
        user_signal="(user said: manejate vos con esto)",
    )
    stamped = apply_autonomous_to_packet(packet_in)
    assert stamped.autonomous is True

    # Now exercise the pipeline with the stamped packet and confirm the
    # engine is invoked twice (the user's literal text contained no
    # autonomous phrase; the chain flag came entirely from user_signal).
    packets = [stamped, stamped]
    engine = _CountingEngine(plan_count=10)
    monkeypatch.setattr(_settings_module(), "AUTONOMOUS_MAX_PLANS", 2)
    monkeypatch.setattr(_settings_module(), "AUTONOMOUS_WALL_SECONDS", 900)
    monkeypatch.setattr(_settings_module(), "AUTONOMOUS_TOKEN_BUDGET", 50_000)
    _patch_pipeline(monkeypatch, chat_packets=packets, engine=engine)

    set_context(
        agent_id="auto-test-agent",
        project_id=1,
        session_id="autonomous-3",
        workspace_path="/tmp/auto",
    )

    run_task(
        agent=_FakeAgent(),
        user_input="please fix the auth migration",
        session_id="autonomous-3",
        engine=engine,
        reviewer=_FakeReviewer(),
        hooks=_CountingHooks(),
        # NOTE: no autonomous kwarg — must still chain because the
        # chat-agent stamp set the flag.
    )

    assert len(engine.calls) == 2, (
        f"chain should fire from stamped packet; got {len(engine.calls)}"
    )


def _settings_module():
    from infinidev.config.settings import settings

    return settings


# ─────────────────────────────────────────────────────────────────────────
# Budget-property guard — should_continue obeys the chain contract
# ─────────────────────────────────────────────────────────────────────────


def test_should_continue_refuses_a_fourth_plan_under_default_budget() -> None:
    """Sanity check that the budget helper the pipeline uses respects
    plans_executed == max_plans as the terminal state.
    """
    budget = AutonomousBudget(max_plans=DEFAULT_MAX_PLANS, wall_seconds=900, token_budget=50_000)
    for outcome in ("completed", "completed", "completed"):
        budget.record_outcome(outcome)
    # Three plans done; the next iteration must be rejected.
    assert should_continue(budget, "completed") is False
