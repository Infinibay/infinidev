"""Contract tests for PhaseEngine's LoopEngine compatibility surface."""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.engine.loop import LoopEngine
from infinidev.engine.phases.phase_engine import PhaseEngine


class _LoopResult:
    def __init__(self) -> None:
        self._last_status = "completed"
        self._last_state = SimpleNamespace(iteration_count=2)
        self._last_total_tool_calls = 7
        self.is_cancelled = False
        self.summary_calls = []

    def get_objective_checks(self):
        return [{"criterion": "works", "status": "passed"}]

    def build_work_summary(self, result, status):
        self.summary_calls.append((result, status))
        return "summary"


def test_phase_engine_proxies_terminal_state_metrics_and_summary() -> None:
    phase = PhaseEngine()
    loop = _LoopResult()
    phase._last_engine = loop

    assert phase._last_status == "completed"
    assert phase._last_state is loop._last_state
    assert phase._last_total_tool_calls == 7
    assert phase.is_cancelled is False
    assert phase.get_objective_checks() == [
        {"criterion": "works", "status": "passed"}
    ]
    assert phase.build_work_summary("done", "completed") == "summary"
    assert loop.summary_calls == [("done", "completed")]


def test_phase_engine_status_assignment_updates_wrapped_loop() -> None:
    phase = PhaseEngine()
    loop = _LoopResult()
    phase._last_engine = loop

    phase._last_status = "failed"

    assert phase._last_status == "failed"
    assert loop._last_status == "failed"


def test_phase_engine_contract_is_safe_before_loop_execution() -> None:
    phase = PhaseEngine()

    assert phase._last_status == ""
    assert phase._last_state is None
    assert phase._last_total_tool_calls == 0
    assert phase.is_cancelled is False
    assert phase.get_objective_checks() == []
    assert phase.build_work_summary("nothing", "failed") is None


def test_phase_engine_shares_task_and_tool_cancellation_with_wrapped_loop() -> None:
    loop = LoopEngine()
    phase = PhaseEngine(loop_engine=loop)

    loop._begin_tool_batch()
    try:
        assert phase.has_active_tool is True
        assert phase.cancel_active_tool() is True
        assert loop._tool_cancel_event.is_set() is True
        assert loop.is_cancelled is False
    finally:
        loop._finish_tool_batch()

    phase.cancel()

    assert phase.is_cancelled is True
    assert loop.is_cancelled is True
