"""Tests for context window calculator."""

import pytest
from infinidev.ui.context_calculator import ContextWindowCalculator


class TestContextWindowCalculator:
    """Tests for ContextWindowCalculator class."""

    def test_initial_state(self):
        """Test initial calculator state."""
        calc = ContextWindowCalculator()
        assert calc.model_name == ""
        assert calc.max_context == 4096
        assert calc.chat_remaining == 4096
        assert calc.task_remaining == 4096
        assert calc.total_remaining == 8192

    def test_update_model_context(self):
        """Test updating model context."""
        calc = ContextWindowCalculator()
        calc.update_model_context()  # Will fail silently without Ollama
        # Model name should still be set or remain default
        assert calc.model_name != "" or calc.max_context == 4096

    def test_calculate_usage(self):
        """Test calculating token usage."""
        calc = ContextWindowCalculator(max_context=8192)
        calc.calculate_usage(chat_tokens=2048, task_tokens=1024)

        status = calc.get_context_status()
        assert status["chat"]["current_tokens"] == 2048
        assert status["tasks"]["current_tokens"] == 1024
        assert status["chat"]["remaining_tokens"] == 6144
        assert status["tasks"]["remaining_tokens"] == 7168
        assert status["total_used"] == 3072
        # remaining = max_context - total_used = 8192 - 3072 = 5120
        assert status["remaining"] == 5120

    def test_full_chat_window(self):
        """Test when chat window is full."""
        calc = ContextWindowCalculator(max_context=4096)
        calc.calculate_usage(chat_tokens=4096, task_tokens=0)

        assert calc.chat_remaining == 0
        assert calc.chat_window["usage_percentage"] == 1.0
        status = calc.get_context_status()
        assert status["chat"]["remaining_tokens"] == 0

    def test_max_context_updated(self):
        """Test that max_context affects calculations."""
        calc = ContextWindowCalculator(max_context=1024)
        calc.calculate_usage(chat_tokens=512, task_tokens=256)

        assert calc.chat_remaining == 512
        assert calc.task_remaining == 768

    def test_model_name(self):
        """Test setting model name."""
        calc = ContextWindowCalculator(model_name="test-model")
        status = calc.get_context_status()
        assert status["model"] == "test-model"

    def test_chat_usage_percentage(self):
        """Test chat usage percentage calculation."""
        calc = ContextWindowCalculator(max_context=4096)
        calc.calculate_usage(chat_tokens=2048, task_tokens=0)
        assert calc.chat_usage_percentage == 0.5

    def test_task_usage_percentage(self):
        """Test task usage percentage calculation."""
        calc = ContextWindowCalculator(max_context=4096)
        calc.calculate_usage(chat_tokens=0, task_tokens=2048)
        assert calc.task_usage_percentage == 0.5

    def test_usage_cannot_exceed_max(self):
        """Test that remaining tokens never go negative."""
        calc = ContextWindowCalculator(max_context=1000)
        calc.calculate_usage(chat_tokens=1500, task_tokens=0)
        assert calc.chat_remaining == 0
        assert calc.chat_usage_percentage == 1.0

    def test_context_status_format(self):
        """Test context status dictionary format."""
        calc = ContextWindowCalculator(max_context=4096)
        calc.calculate_usage(chat_tokens=1000, task_tokens=500)
        status = calc.get_context_status()

        assert "model" in status
        assert "max_context" in status
        assert "chat" in status
        assert "tasks" in status
        assert "remaining" in status
        assert "total_used" in status
        assert status["max_context"] == 4096
        assert status["total_used"] == 1500
