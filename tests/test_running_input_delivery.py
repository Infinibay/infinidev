"""Regression tests for user input while an engine task is running."""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.ui.app import InfinidevApp


class _InjectingEngine:
    def __init__(self) -> None:
        self.messages: list[tuple[str, list[str]]] = []

    def inject_message(self, text: str, attachments: list[str]) -> None:
        self.messages.append((text, attachments))


def _running_app(engine: object) -> InfinidevApp:
    app = object.__new__(InfinidevApp)
    app._autocomplete = SimpleNamespace(dismiss=lambda: None)
    app._permission_waiting = False
    app._permission_event = None
    app._analysis_waiting = False
    app._analysis_event = None
    app._plan_review_waiting = False
    app._plan_review_event = None
    app._resolve_attachments = lambda text: (text, [])
    app._engine_running = True
    app.engine = engine
    app._pending_inputs = []
    app.add_message = lambda *args: None
    return app


def test_running_input_is_delivered_without_cancelling() -> None:
    engine = _InjectingEngine()
    app = _running_app(engine)

    app._handle_submit("Please address this before finishing.")

    assert engine.messages == [("Please address this before finishing.", [])]
    assert app._pending_inputs == []


def test_running_input_queues_when_engine_has_no_injection_hook() -> None:
    app = _running_app(object())

    app._handle_submit("Please address this before finishing.")

    assert app._pending_inputs == ["Please address this before finishing."]
