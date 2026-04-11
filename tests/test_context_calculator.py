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

    def test_update_chat(self):
        """Test updating chat context from user input + summaries."""
        calc = ContextWindowCalculator(max_context=8192)
        calc.update_chat("Hello, can you help me?", ["Previous summary 1", "Previous summary 2"])

        # ~4 chars per token estimate
        assert calc.chat_window["current_tokens"] > 0
        assert calc.chat_remaining < 8192

    def test_update_task(self):
        """Test updating task context with prompt tokens."""
        calc = ContextWindowCalculator(max_context=8192)
        calc.update_task(task_prompt_tokens=3000)

        status = calc.get_context_status()
        assert status["tasks"]["current_tokens"] == 3000
        assert status["tasks"]["remaining_tokens"] == 5192

    def test_chat_and_task_independent(self):
        """Test that chat and task contexts are tracked independently."""
        calc = ContextWindowCalculator(max_context=4096)
        calc.update_chat("Hello world")
        calc.update_task(task_prompt_tokens=2000)

        status = calc.get_context_status()
        # Chat is estimated from text, task is exact
        assert status["chat"]["current_tokens"] > 0
        assert status["tasks"]["current_tokens"] == 2000

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
        calc = ContextWindowCalculator(max_context=4096)
        calc.update_chat("test input")
        calc.update_task(task_prompt_tokens=1000)
        status = calc.get_context_status()

        assert "model" in status
        assert "max_context" in status
        assert "chat" in status
        assert "tasks" in status
        assert status["max_context"] == 4096
        assert status["chat"]["current_tokens"] > 0
        assert status["tasks"]["current_tokens"] == 1000

    def test_update_chat_replaces_previous(self):
        """Test that each update_chat replaces the previous count."""
        calc = ContextWindowCalculator(max_context=4096)
        calc.update_chat("short")
        first = calc.chat_window["current_tokens"]
        calc.update_chat("a much longer message with more content")
        second = calc.chat_window["current_tokens"]
        assert second > first

    def test_update_chat_empty(self):
        """Test update_chat with empty input."""
        calc = ContextWindowCalculator(max_context=4096)
        calc.update_chat("")
        assert calc.chat_window["current_tokens"] >= 0
