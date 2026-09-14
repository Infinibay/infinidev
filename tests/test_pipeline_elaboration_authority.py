"""High-impact elaborator choices cannot silently reach the planner."""

from __future__ import annotations

from infinidev.engine.analysis.grounded_spec import Clarification, GroundedSpec
from infinidev.engine.orchestration.escalation_packet import EscalationPacket
from infinidev.engine.orchestration.pipeline import _run_elaboration_phase


class _Hooks:
    def __init__(self, answer: str | None = None) -> None:
        self.answer = answer
        self.asked: list[str] = []
        self.notified: list[str] = []

    def on_phase(self, phase: str) -> None:
        pass

    def on_status(self, level: str, message: str) -> None:
        pass

    def ask_user(self, prompt: str, kind: str = "text") -> str | None:
        self.asked.append(prompt)
        return self.answer

    def notify(self, speaker: str, message: str, kind: str = "agent") -> None:
        self.notified.append(message)


def _packet() -> EscalationPacket:
    return EscalationPacket(
        user_request="Choose and implement the public persistence contract",
        understanding="A product-level storage contract must be chosen",
    )


def _spec(risk: str) -> GroundedSpec:
    return GroundedSpec(
        deliverable="Implement persistence",
        clarifications_needed=[
            Clarification(
                question="Use SQLite or PostgreSQL?",
                options=["SQLite", "PostgreSQL"],
                default="SQLite",
                impact="Changes the public deployment and storage contract",
                risk=risk,  # type: ignore[arg-type]
            )
        ],
    )


def test_a_caller_that_cannot_ask_proceeds_on_the_declared_default(monkeypatch) -> None:
    """``None`` means "cannot ask", not "the user declined".

    The hook interface (``OrchestrationHooks.ask_user``) states that a
    branch receiving ``None`` must proceed with sensible defaults instead of
    failing. Halting here made every one-shot run do zero work on any
    request whose elaboration found a consequential fork.
    """
    monkeypatch.setattr(
        "infinidev.engine.analysis.spec_elaborator.should_elaborate", lambda _: True
    )
    monkeypatch.setattr(
        "infinidev.engine.analysis.spec_elaborator.elaborate",
        lambda *args, **kwargs: _spec("costly_to_reverse"),
    )
    hooks = _Hooks(answer=None)

    result = _run_elaboration_phase(
        escalation=_packet(), session_id="s", project_id=None,
        workspace_path=None, hooks=hooks,
    )

    spec = result.grounded_spec
    assert result.execution_blocked_reason == ""
    assert spec.blocking_clarifications == []
    assert [c.default for c in spec.unconfirmed_decisions] == ["SQLite"]
    # The decision has to reach the user, not be silently taken.
    assert any("No user was available" in message for message in hooks.notified)
    rendered = spec.render_for_planner()
    assert "UNCONFIRMED PRODUCT DECISIONS" in rendered
    assert "proceeding with: SQLite" in rendered


def test_an_asked_user_who_gives_nothing_still_blocks(monkeypatch) -> None:
    """An empty answer is a real answer: the user was asked and said nothing."""
    monkeypatch.setattr(
        "infinidev.engine.analysis.spec_elaborator.should_elaborate", lambda _: True
    )
    monkeypatch.setattr(
        "infinidev.engine.analysis.spec_elaborator.elaborate",
        lambda *args, **kwargs: _spec("costly_to_reverse"),
    )

    result = _run_elaboration_phase(
        escalation=_packet(), session_id="s", project_id=None,
        workspace_path=None, hooks=_Hooks(answer=""),
    )

    assert "waiting for confirmation" in result.execution_blocked_reason
    assert result.grounded_spec.blocking_clarifications
    assert result.grounded_spec.unconfirmed_decisions == []


def test_user_answer_replaces_high_impact_default_with_authority(monkeypatch) -> None:
    monkeypatch.setattr(
        "infinidev.engine.analysis.spec_elaborator.should_elaborate", lambda _: True
    )
    monkeypatch.setattr(
        "infinidev.engine.analysis.spec_elaborator.elaborate",
        lambda *args, **kwargs: _spec("costly_to_reverse"),
    )

    result = _run_elaboration_phase(
        escalation=_packet(), session_id="s", project_id=None,
        workspace_path=None, hooks=_Hooks(answer="PostgreSQL"),
    )

    assert result.execution_blocked_reason == ""
    assert result.grounded_spec.blocking_clarifications == []
    assert "PostgreSQL" in result.grounded_spec.confirmed_decisions[0]


def test_local_reversible_default_is_nonblocking(monkeypatch) -> None:
    monkeypatch.setattr(
        "infinidev.engine.analysis.spec_elaborator.should_elaborate", lambda _: True
    )
    monkeypatch.setattr(
        "infinidev.engine.analysis.spec_elaborator.elaborate",
        lambda *args, **kwargs: _spec("local_reversible"),
    )
    hooks = _Hooks()

    result = _run_elaboration_phase(
        escalation=_packet(), session_id="s", project_id=None,
        workspace_path=None, hooks=hooks,
    )

    assert result.execution_blocked_reason == ""
    assert hooks.asked == []
    assert any("Local reversible defaults" in message for message in hooks.notified)
