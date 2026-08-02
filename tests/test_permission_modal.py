"""Permission-gated tool calls must demand an explicit, visible decision."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.output import DummyOutput

from infinidev.ui.keybindings import create_global_keybindings


class _SizedOutput(DummyOutput):
    def get_size(self) -> Size:
        return Size(rows=28, columns=100)


def _draw(app_state) -> str:
    screen = Screen(default_char=None, initial_width=100, initial_height=28)
    app_state._float_container.write_to_screen(
        screen,
        MouseHandlers(),
        WritePosition(0, 0, 100, 28),
        "",
        False,
        None,
    )
    screen.draw_all_floats()
    return "\n".join(
        "".join(screen.data_buffer[y][x].char for x in range(100)).rstrip()
        for y in range(28)
    )


def test_permission_request_renders_as_a_centered_modal() -> None:
    async def _run() -> str:
        with create_pipe_input() as pipe, create_app_session(
            input=pipe, output=_SizedOutput()
        ):
            from prompt_toolkit.document import Document

            from infinidev.ui.app import InfinidevApp
            from infinidev.ui.dialogs.permission_detail import DIALOG_NAME

            app = InfinidevApp()
            app._permission_waiting = True
            app._permission_event = threading.Event()
            app._permission_state = {
                "tool_name": "execute_command",
                "description": "Run a command outside the sandbox",
                "details": "sudo apt-get purge tailscale",
            }
            app._permission_details_buffer.set_document(
                Document(app._permission_state["details"]),
                bypass_readonly=True,
            )
            app.active_dialog = DIALOG_NAME
            app.app.layout.focus(app._permission_deny_button)
            return _draw(app)

    rendered = asyncio.run(_run())
    assert "Permission required" in rendered
    assert "execute_command" in rendered
    assert "sudo apt-get purge tailscale" in rendered
    assert "[    Allow     ]" in rendered
    assert "[     Deny     ]" in rendered
    title_row = next(row for row in rendered.splitlines() if "Permission required" in row)
    assert 0 < title_row.index("Permission required") < 50


def test_permission_handler_blocks_until_the_modal_is_resolved() -> None:
    async def _run() -> tuple[list[bool], bool, str, str | None]:
        with create_pipe_input() as pipe, create_app_session(
            input=pipe, output=_SizedOutput()
        ):
            from infinidev.ui.app import InfinidevApp
            from infinidev.ui.dialogs.permission_detail import DIALOG_NAME

            app = InfinidevApp()
            result: list[bool] = []
            worker = threading.Thread(
                target=lambda: result.append(
                    app._handle_permission_request(
                        "execute_command",
                        "Run a command outside the sandbox",
                        "sudo apt-get purge tailscale",
                    )
                )
            )
            worker.start()
            deadline = time.monotonic() + 1
            while not app._permission_waiting and time.monotonic() < deadline:
                time.sleep(0.01)

            waiting = app._permission_waiting
            active_dialog = app.active_dialog
            details = app._permission_details_buffer.text
            app._resolve_permission(True)
            worker.join(timeout=1)
            return result, waiting, details, active_dialog

    result, waiting, details, active_dialog = asyncio.run(_run())
    assert waiting is True
    assert active_dialog == "permission_request"
    assert "sudo apt-get purge tailscale" in details
    assert result == [True]


def test_escape_denies_and_releases_the_waiting_worker() -> None:
    async def _run() -> tuple[bool, bool, str | None]:
        with create_pipe_input() as pipe, create_app_session(
            input=pipe, output=_SizedOutput()
        ):
            from infinidev.ui.app import InfinidevApp
            from infinidev.ui.dialogs.permission_detail import DIALOG_NAME

            app = InfinidevApp()
            app._permission_waiting = True
            app._permission_event = threading.Event()
            app.active_dialog = DIALOG_NAME
            app.handle_escape()
            return (
                app._permission_approved,
                app._permission_event.is_set(),
                app.active_dialog,
            )

    approved, released, active_dialog = asyncio.run(_run())
    assert approved is False
    assert released is True
    assert active_dialog is None


def test_permission_shortcuts_resolve_without_using_chat_input() -> None:
    decisions: list[bool] = []
    app_state = SimpleNamespace(
        _chat_buffer=SimpleNamespace(on_text_changed=[]),
        active_dialog="permission_request",
        _permission_event=threading.Event(),
        _resolve_permission=decisions.append,
    )
    bindings = create_global_keybindings(app_state)

    def fire(key: str) -> None:
        binding = next(
            binding
            for binding in bindings.bindings
            if tuple(getattr(k, "value", k) for k in binding.keys) == (key,)
            and binding.filter()
        )
        binding.handler(SimpleNamespace())

    fire("y")
    fire("n")
    assert decisions == [True, False]
