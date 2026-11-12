"""Task cancellation releases workers waiting for modal input."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import Mock

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from infinidev.engine.loop.engine import LoopEngine
from infinidev.ui.app import InfinidevApp


@pytest.mark.parametrize("modal, expected", [("permission_request", False), ("stdin_prompt", None)])
def test_cancel_hold_releases_modal_that_opens_after_escape(monkeypatch, modal, expected) -> None:
    async def run() -> None:
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            app = InfinidevApp()
            app.engine = LoopEngine()
            app._engine_running = True
            app._cancel_watcher_active = True
            monkeypatch.setattr(app, "_persist_session_message", Mock())
            monkeypatch.setattr(app, "_persist_runtime_state", Mock())
            shown = threading.Event()
            monkeypatch.setattr(
                app, "invalidate", lambda: shown.set() if app.active_dialog == modal else None,
            )

            # Escape starts the timer before the tool requests human input.
            app.handle_escape()
            results = []

            def request() -> None:
                if modal == "permission_request":
                    results.append(app._handle_permission_request("execute_command", "Run", "pwd"))
                else:
                    results.append(app._handle_stdin_input("read reply", "Reply:", "", ""))

            worker = threading.Thread(target=request, daemon=True)
            worker.start()
            try:
                assert shown.wait(timeout=1)
                app._cancel_hold_start -= 4
                app._cancel_hold_watcher()
                worker.join(timeout=1)
                assert not worker.is_alive(), "Cancellation left the modal worker blocked"
                assert results == [expected]
                assert app.engine.is_cancelled
                assert app.active_dialog is None
            finally:
                # A failing regression must not leave a worker behind.
                pending = getattr(app, "_permission_event", None)
                pending = pending or getattr(app, "_stdin_prompt_event", None)
                if pending is not None:
                    pending.set()
                worker.join(timeout=1)

    asyncio.run(run())
