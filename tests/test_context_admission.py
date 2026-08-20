"""Regression coverage for small-context task admission."""

from __future__ import annotations

import asyncio

from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.output import DummyOutput

from infinidev.ui.context_admission import (
    LARGE_CONTEXT_CANDIDATE_MIN,
    SMALL_CONTEXT_ADMISSION_MAX,
    find_context_admission,
)


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


def test_small_window_offers_only_a_verified_larger_model(monkeypatch) -> None:
    windows = {
        "small": SMALL_CONTEXT_ADMISSION_MAX - 1,
        "unknown": None,
        "barely-large": LARGE_CONTEXT_CANDIDATE_MIN,
        "large": LARGE_CONTEXT_CANDIDATE_MIN + 1,
    }
    monkeypatch.setattr(
        "infinidev.ui.context_admission.get_model_context_window",
        lambda params, provider: windows[params["model"]],
    )

    admission = find_context_admission(
        model="small",
        provider_id="provider",
        llm_params={"model": "small"},
        candidates=["unknown", "barely-large", "large"],
    )

    assert admission is not None
    assert admission.active_window == SMALL_CONTEXT_ADMISSION_MAX - 1
    assert admission.replacement_model == "large"


def test_small_window_without_verified_candidate_falls_back_to_compaction(monkeypatch) -> None:
    monkeypatch.setattr(
        "infinidev.ui.context_admission.get_model_context_window",
        lambda params, provider: {
            "small": 80_000,
            "unknown": None,
            "two-hundred-k": LARGE_CONTEXT_CANDIDATE_MIN,
        }[params["model"]],
    )

    admission = find_context_admission(
        model="small",
        provider_id="provider",
        llm_params={"model": "small"},
        candidates=["unknown", "two-hundred-k"],
    )

    assert admission is not None
    assert admission.replacement_model is None


def test_context_admission_modal_hides_switch_when_no_candidate() -> None:
    async def _run() -> str:
        with create_pipe_input() as pipe, create_app_session(
            input=pipe, output=_SizedOutput()
        ):
            from infinidev.ui.app import InfinidevApp

            app = InfinidevApp()
            app._context_admission_state = {
                "active_window": 80_000,
                "replacement_model": None,
            }
            app.active_dialog = "context_admission"
            return _draw(app)

    rendered = asyncio.run(_run())

    assert "Context window decision" in rendered
    assert "80,000-token context window" in rendered
    assert "Compact and continue" in rendered
    assert "Use larger model" not in rendered


def test_context_admission_modal_offers_verified_larger_model() -> None:
    async def _run() -> str:
        with create_pipe_input() as pipe, create_app_session(
            input=pipe, output=_SizedOutput()
        ):
            from infinidev.ui.app import InfinidevApp

            app = InfinidevApp()
            app._context_admission_state = {
                "active_window": 80_000,
                "replacement_model": "openai/gpt-5.6",
            }
            app.active_dialog = "context_admission"
            return _draw(app)

    rendered = asyncio.run(_run())

    assert "compatible larger model is available: openai/gpt-5.6" in rendered
    assert "Use larger model" in rendered


def test_context_admission_resolution_keeps_the_queued_session_task() -> None:
    from infinidev.ui.app import InfinidevApp

    app = object.__new__(InfinidevApp)
    app._context_admission_pending = ("continue the active task", ["attachment"])
    app._context_admission_state = {"active_window": 80_000, "replacement_model": None}
    app.active_dialog = "context_admission"
    app.add_message = lambda *args: None
    launched: list[tuple[str, list[str]]] = []
    app._start_engine_task = lambda text, attachments: launched.append((text, attachments))

    app._resolve_context_admission(False)

    assert launched == [("continue the active task", ["attachment"])]
    assert app._context_admission_pending is None
    assert app.active_dialog is None
