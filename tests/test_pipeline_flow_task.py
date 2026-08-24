"""Tests for direct flow execution through the shared pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from infinidev.engine.orchestration.pipeline import run_flow_task


@dataclass
class _Agent:
    backstory: str = ""
    _system_prompt_identity: str = ""
    activated: list[str] = field(default_factory=list)
    deactivations: int = 0

    def activate_context(self, session_id: str) -> None:
        self.activated.append(session_id)

    def deactivate(self) -> None:
        self.deactivations += 1


class _Engine:
    def __init__(
        self,
        *,
        status: str,
        result: str = "",
        cancelled: bool = False,
        error: Exception | None = None,
    ) -> None:
        self._last_status = status
        self._result = result
        self.is_cancelled = cancelled
        self._error = error

    def execute(self, **_kwargs) -> str:
        if self._error is not None:
            raise self._error
        return self._result


@dataclass
class _Hooks:
    phases: list[str] = field(default_factory=list)
    statuses: list[tuple[str, str]] = field(default_factory=list)

    def on_phase(self, phase: str) -> None:
        self.phases.append(phase)

    def on_status(self, level: str, message: str) -> None:
        self.statuses.append((level, message))


@pytest.mark.parametrize(
    ("status", "cancelled", "expected", "level"),
    [
        ("failed", False, "Flow failed. (no additional output)", "error"),
        ("", False, "Flow failed. (no additional output)", "error"),
        ("unknown", False, "Flow failed. (no additional output)", "error"),
        (
            "cancelled",
            False,
            "Execution cancelled; the flow remains incomplete.",
            "warn",
        ),
        (
            "done",
            True,
            "Execution cancelled; the flow remains incomplete.",
            "warn",
        ),
        ("blocked", False, "Flow blocked before completion.", "warn"),
        ("exhausted", False, "Flow blocked before completion.", "warn"),
    ],
)
def test_direct_flow_empty_non_success_never_reports_done(
    status, cancelled, expected, level
):
    agent = _Agent()
    hooks = _Hooks()

    result = run_flow_task(
        agent=agent,
        flow="sysadmin",
        task_prompt=("Inspect the service", "Report findings"),
        session_id="flow-terminal",
        engine=_Engine(status=status, cancelled=cancelled),
        hooks=hooks,
    )

    assert result == expected
    assert agent.activated == ["flow-terminal"]
    assert agent.deactivations == 1
    assert hooks.phases == ["execute", "idle"]
    assert hooks.statuses[-1][0] == level


def test_direct_flow_preserves_detailed_failure_output() -> None:
    hooks = _Hooks()

    result = run_flow_task(
        agent=_Agent(),
        flow="sysadmin",
        task_prompt=("Inspect the service", "Report findings"),
        session_id="flow-failed-output",
        engine=_Engine(status="failed", result="Connection refused by service."),
        hooks=hooks,
    )

    assert result == "Connection refused by service."
    assert hooks.statuses[-1][0] == "error"


def test_direct_flow_returns_to_idle_when_execution_raises() -> None:
    agent = _Agent()
    hooks = _Hooks()

    with pytest.raises(RuntimeError, match="provider unavailable"):
        run_flow_task(
            agent=agent,
            flow="sysadmin",
            task_prompt=("Inspect the service", "Report findings"),
            session_id="flow-exception",
            engine=_Engine(
                status="",
                error=RuntimeError("provider unavailable"),
            ),
            hooks=hooks,
        )

    assert agent.deactivations == 1
    assert hooks.phases == ["execute", "idle"]
