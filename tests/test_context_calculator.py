"""Tests for context window calculator."""

import pytest
from infinidev.ui.context_calculator import ContextWindowCalculator


class TestContextWindowCalculator:
    """Tests for ContextWindowCalculator class."""

    def test_initial_state(self):
        """Test initial calculator state — max_context is None until detected."""
        calc = ContextWindowCalculator()
        assert calc.model_name == ""
        # Unknown until update_model_context() is called
        assert calc.max_context is None
        # Remaining properties return 0 when max is unknown
        assert calc.chat_remaining == 0
        assert calc.task_remaining == 0

    def test_initial_state_with_explicit_max(self):
        """Test initial state when max_context is explicitly provided."""
        calc = ContextWindowCalculator(max_context=8192)
        assert calc.max_context == 8192
        assert calc.chat_remaining == 8192
        assert calc.task_remaining == 8192

    def test_update_chat_takes_the_real_prompt_size(self):
        """The chat meter records usage.prompt_tokens, never an estimate."""
        calc = ContextWindowCalculator(max_context=8192)
        calc.update_chat(4400)

        assert calc.chat_window["current_tokens"] == 4400
        assert calc.chat_remaining == 3792

    def test_update_chat_ignores_a_missing_measurement(self):
        """A provider that sent no usage must not zero a real reading."""
        calc = ContextWindowCalculator(max_context=8192)
        calc.update_chat(4400)
        calc.update_chat(0)
        assert calc.chat_window["current_tokens"] == 4400

    def test_update_task(self):
        """Test updating task context with prompt tokens."""
        calc = ContextWindowCalculator(max_context=8192)
        calc.update_task(task_prompt_tokens=3000)

        status = calc.get_context_status()
        assert status["tasks"]["current_tokens"] == 3000
        assert status["tasks"]["remaining_tokens"] == 5192

    def test_chat_and_task_independent(self):
        """The two lanes hold different prompts and move independently."""
        calc = ContextWindowCalculator(max_context=4096)
        calc.update_chat(900)
        calc.update_task(task_prompt_tokens=2000)

        status = calc.get_context_status()
        assert status["chat"]["current_tokens"] == 900
        assert status["tasks"]["current_tokens"] == 2000

    def test_start_task_resets_only_the_task_lane(self):
        """A new task rebuilds the developer prompt; the chat lane persists."""
        calc = ContextWindowCalculator(max_context=4096)
        calc.update_chat(900)
        calc.update_task(task_prompt_tokens=2000)
        calc.start_task()

        status = calc.get_context_status()
        assert status["tasks"]["current_tokens"] == 0
        assert status["chat"]["current_tokens"] == 900

    def test_full_context_window(self):
        """Test when prompt fills entire context window."""
        calc = ContextWindowCalculator(max_context=100)
        calc.update_task(task_prompt_tokens=100)

        assert calc.task_remaining == 0
        assert calc.task_window["usage_percentage"] == 1.0

    def test_max_context_updated(self):
        """Test that max_context affects calculations."""
        calc = ContextWindowCalculator(max_context=1024)
        calc.update_task(task_prompt_tokens=256)

        assert calc.task_remaining == 768

    def test_model_name(self):
        """Test setting model name."""
        calc = ContextWindowCalculator(model_name="test-model")
        status = calc.get_context_status()
        assert status["model"] == "test-model"

    def test_chat_usage_percentage(self):
        """Test chat usage percentage calculation."""
        calc = ContextWindowCalculator(max_context=4096)
        # Directly set for deterministic test
        calc._last_prompt_tokens = 2048
        assert calc.chat_usage_percentage == 0.5

    def test_task_usage_percentage(self):
        """Test task prompt tokens usage percentage calculation."""
        calc = ContextWindowCalculator(max_context=4096)
        calc.update_task(task_prompt_tokens=2048)
        assert calc.task_usage_percentage == 0.5

    def test_usage_cannot_exceed_max(self):
        """Test that remaining tokens never go negative."""
        calc = ContextWindowCalculator(max_context=1000)
        calc.update_task(task_prompt_tokens=1500)
        assert calc.task_remaining == 0
        assert calc.task_usage_percentage == 1.0

    def test_context_status_format(self):
        """Test context status dictionary format."""
        calc = ContextWindowCalculator(
            max_context=4096,
            model_max_context=8192,
        )
        calc.update_chat(500)
        calc.update_task(task_prompt_tokens=1000)
        status = calc.get_context_status()

        assert "model" in status
        assert "max_context" in status
        assert status["model_max_context"] == 8192
        assert "chat" in status
        assert "tasks" in status
        assert status["max_context"] == 4096
        assert status["chat"]["current_tokens"] > 0
        assert status["tasks"]["current_tokens"] == 1000

    def test_update_chat_replaces_previous(self):
        """Each call rebuilds the prompt, so the newest reading wins."""
        calc = ContextWindowCalculator(max_context=4096)
        calc.update_chat(300)
        calc.update_chat(2800)
        assert calc.chat_window["current_tokens"] == 2800

    def test_resolve_model_context_works_inside_a_running_loop(self):
        """Regression: asyncio.run here raised, and the raise was swallowed.

        Startup runs inside prompt_toolkit's loop, so every model displayed
        ``?`` for its context window no matter what the catalog said.
        """
        import asyncio

        async def main():
            calc = ContextWindowCalculator()
            calc.resolve_model_context()
            return calc.max_context

        assert asyncio.run(main()) is not None
