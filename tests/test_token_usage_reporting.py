"""The context meter reads what the model received, not what the user typed.

A chat turn once displayed 108 tokens while carrying tens of thousands: the
TUI estimated from the user's message plus the session summaries, which is the
one part of the prompt that never matters. The system prompt alone is ~4 400
tokens before a single tool schema.
"""

from __future__ import annotations

import pytest

from infinidev.engine.token_usage import report_prompt_tokens
from infinidev.ui.context_calculator import ContextWindowCalculator


class _Recorder:
    """Stands in for TUIHooks."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def notify_token_usage(self, prompt_tokens: int, lane: str = "chat") -> None:
        self.calls.append((prompt_tokens, lane))


class _Response:
    def __init__(self, prompt_tokens: int | None) -> None:
        if prompt_tokens is not None:
            self.usage = type("U", (), {"prompt_tokens": prompt_tokens})()


class TestReadingUsage:
    def test_usage_is_forwarded_verbatim(self):
        hooks = _Recorder()
        assert report_prompt_tokens(hooks, _Response(24_680), lane="task") == 24_680
        assert hooks.calls == [(24_680, "task")]

    def test_the_lane_defaults_to_chat(self):
        hooks = _Recorder()
        report_prompt_tokens(hooks, _Response(900))
        assert hooks.calls == [(900, "chat")]

    def test_a_streamed_response_falls_back_to_counting_the_messages(self):
        """litellm omits usage on a stream unless it was asked for."""
        hooks = _Recorder()
        messages = [
            {"role": "system", "content": "x" * 40_000},
            {"role": "user", "content": "hola"},
        ]
        n = report_prompt_tokens(hooks, _Response(None), messages=messages, model="gpt-4")
        assert n > 1_000
        assert hooks.calls[0][0] == n

    def test_nothing_measurable_reports_nothing(self):
        hooks = _Recorder()
        assert report_prompt_tokens(hooks, _Response(None)) == 0
        assert hooks.calls == []

    def test_a_missing_hook_is_not_an_error(self):
        assert report_prompt_tokens(None, _Response(500)) == 500
        assert report_prompt_tokens(object(), _Response(500)) == 500

    def test_a_hook_that_raises_never_breaks_the_turn(self):
        class Broken:
            def notify_token_usage(self, prompt_tokens, lane="chat"):
                raise RuntimeError("boom")

        assert report_prompt_tokens(Broken(), _Response(500)) == 500


class TestTheMeterEndToEnd:
    def test_the_bar_shows_what_the_model_received(self):
        calc = ContextWindowCalculator(max_context=1_000_000)

        class Hooks:
            def notify_token_usage(self, prompt_tokens, lane="chat"):
                calc.update_chat(prompt_tokens) if lane == "chat" else calc.update_task(prompt_tokens)

        report_prompt_tokens(Hooks(), _Response(24_680))
        assert calc.get_context_status()["chat"]["current_tokens"] == 24_680

    def test_the_user_message_alone_is_never_the_answer(self):
        """The shape of the original bug: 108 for a prompt of tens of thousands."""
        calc = ContextWindowCalculator(max_context=1_000_000)
        calc.update_chat(108)          # what the old estimate produced
        calc.update_chat(24_680)       # what usage actually reports
        assert calc.get_context_status()["chat"]["current_tokens"] == 24_680


class TestTheLoopsThatReport:
    @pytest.mark.parametrize(
        "module",
        [
            "infinidev.engine.orchestration.chat_agent",
            "infinidev.engine.analysis.planner",
        ],
    )
    def test_the_loop_reports_its_prompt_size(self, module):
        """Guards the wiring: an import that drifts away fails here."""
        import importlib
        import inspect

        src = inspect.getsource(importlib.import_module(module))
        assert "report_prompt_tokens(" in src, f"{module} stopped metering"

    def test_the_council_deliberately_does_not_report(self):
        """Members run concurrently; one meter cannot describe several."""
        import inspect

        import infinidev.engine.council.agent_loop as council

        src = inspect.getsource(council)
        assert "report_prompt_tokens" not in src
        assert "does not feed the context meter" in src
