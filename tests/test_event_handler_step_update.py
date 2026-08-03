"""Step transitions keep plan titles out of the TUI chat transcript."""

from __future__ import annotations

from infinidev.ui.event_handler import process_event


class _FakeApp:
    def __init__(self) -> None:
        self._thinking_text = "stale"
        self._thinking_full = "stale"
        self._streaming_tool_name = "read_file"
        self._streaming_token_count = 10
        self._actions_text = "old action"
        self._session_plan_steps: list[dict] = []
        self._steps_text = ""
        self._plan_text = ""
        self.messages: list[tuple[str, str, str]] = []
        self.logs: list[str] = []

    def add_message(
        self,
        sender: str,
        text: str,
        msg_type: str = "agent",
        **_kw,
    ) -> None:
        self.messages.append((sender, text, msg_type))

    def add_log(self, text: str) -> None:
        self.logs.append(text)


def test_active_step_title_stays_in_side_panels_not_chat() -> None:
    app = _FakeApp()

    process_event(app, "loop_step_update", {
        "iteration": 2,
        "step_title": "Trace the event path",
        "status": "active",
        "plan_steps": [
            {"index": 1, "title": "Read prompts", "status": "done"},
            {"index": 2, "title": "Trace the event path", "status": "active"},
        ],
    })

    assert app.messages == []
    assert "> Trace the event path" in app._steps_text
    assert app._plan_text == "Step 2: Trace the event path"


def test_agent_orientation_still_reaches_chat() -> None:
    app = _FakeApp()

    process_event(app, "loop_user_message", {
        "message": "Voy a seguir el evento hasta el chat para quitar la línea duplicada.",
    })

    assert app.messages == [(
        "Infinidev",
        "Voy a seguir el evento hasta el chat para quitar la línea duplicada.",
        "agent",
    )]
