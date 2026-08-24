"""Cancellation regressions for PhaseEngine investigation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from infinidev.engine.phases.investigator import _investigate_iteratively
from infinidev.prompts.phases import STRATEGIES


class _CancellingLoop:
    def __init__(self) -> None:
        self.calls = 0
        self.is_cancelled = False
        self._last_state = None
        self._last_status = ""

    def execute(self, **kwargs) -> str:
        self.calls += 1
        self.is_cancelled = True
        self._last_status = "cancelled"
        return "Cancelled by user."


def test_iterative_investigation_stops_after_shared_loop_is_cancelled() -> None:
    loop = _CancellingLoop()
    questions = [
        {"question": "Where is the first implementation?", "intent": "location"},
        {"question": "Which tests cover the behavior?", "intent": "tests"},
    ]

    with (
        patch(
            "infinidev.engine.phases.investigator._generate_questions",
            return_value=questions,
        ),
        patch("infinidev.engine.phases.investigator._generate_followups") as followups,
        patch("infinidev.config.llm._is_small_model", return_value=False),
    ):
        answers, notes = _investigate_iteratively(
            agent=SimpleNamespace(tools=[]),
            description="Fix cancellation",
            strategy=STRATEGIES["bug"],
            all_tools=[],
            verbose=False,
            max_questions=2,
            loop_engine=loop,
            cancel_check=lambda: loop.is_cancelled,
        )

    assert answers == []
    assert notes == []
    assert loop.calls == 1
    followups.assert_not_called()
