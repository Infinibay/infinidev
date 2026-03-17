"""Tests for ContextPanel and context calculator integration."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static, Label

from infinidev.cli.tui import ContextPanel
from infinidev.ui.context_calculator import ContextWindowCalculator


# Minimal app to mount ContextPanel so compose() runs
class ContextPanelTestApp(App):
    def compose(self) -> ComposeResult:
        yield ContextPanel(id="context-panel")


class TestContextPanel:
    """Test ContextPanel widget."""

    async def test_context_panel_creation(self):
        """Test ContextPanel initializes with expected child widgets."""
        app = ContextPanelTestApp()
        async with app.run_test():
            panel = app.query_one(ContextPanel)
            assert panel.query_one("#context-model-name", Static)
            assert panel.query_one("#chat-details", Static)
            assert panel.query_one("#chat-bar", Static)
            assert panel.query_one("#task-details", Static)
            assert panel.query_one("#task-bar", Static)

    async def test_context_panel_update_status(self):
        """Test updating context panel with status data."""
        app = ContextPanelTestApp()
        async with app.run_test():
            panel = app.query_one(ContextPanel)
            status = {
                "model": "llama2",
                "max_context": 4096,
                "chat": {
                    "current_tokens": 1000,
                    "remaining_tokens": 3096,
                    "usage_percentage": 0.244,
                },
                "tasks": {
                    "current_tokens": 500,
                    "remaining_tokens": 3596,
                    "usage_percentage": 0.122,
                },
                "remaining": 2596,
                "total_used": 1500,
            }
            panel.update_status(status)

            # Model name visible
            model_w = panel.query_one("#context-model-name", Static)
            assert "llama2" in str(model_w._Static__content)

            # Chat details show used and available
            chat_w = panel.query_one("#chat-details", Static)
            content = str(chat_w._Static__content)
            assert "1000" in content
            assert "3096" in content

    async def test_context_panel_no_markup_errors(self):
        """Test that progress bars render without markup errors."""
        app = ContextPanelTestApp()
        async with app.run_test():
            panel = app.query_one(ContextPanel)
            status = {
                "model": "test-model",
                "max_context": 1000,
                "chat": {
                    "current_tokens": 500,
                    "remaining_tokens": 500,
                    "usage_percentage": 0.5,
                },
                "tasks": {
                    "current_tokens": 250,
                    "remaining_tokens": 750,
                    "usage_percentage": 0.25,
                },
                "remaining": 250,
                "total_used": 750,
            }
            # Should not raise MarkupError
            panel.update_status(status)
            chat_bar = panel.query_one("#chat-bar", Static)
            assert "50.0%" in str(chat_bar._Static__content)


class TestContextWindowCalculatorIntegration:
    """Test ContextWindowCalculator integration with ContextPanel."""

    async def test_calculator_with_panel(self):
        """Test that calculator data works with panel display."""
        calculator = ContextWindowCalculator(model_name="test-model", max_context=8192)
        calculator.calculate_usage(chat_tokens=2048, task_tokens=1024)

        status = calculator.get_context_status()

        app = ContextPanelTestApp()
        async with app.run_test():
            panel = app.query_one(ContextPanel)
            panel.update_status(status)

            model_w = panel.query_one("#context-model-name", Static)
            assert "test-model" in str(model_w._Static__content)
            assert status["remaining"] > 0

    def test_calculator_full_context(self):
        """Test calculator at full context capacity."""
        calculator = ContextWindowCalculator(max_context=4096)
        calculator.calculate_usage(chat_tokens=2048, task_tokens=2048)

        status = calculator.get_context_status()
        assert status["remaining"] == 0
        assert status["chat"]["usage_percentage"] == 0.5
        assert status["tasks"]["usage_percentage"] == 0.5

    def test_calculator_zero_context(self):
        """Test calculator with zero tokens."""
        calculator = ContextWindowCalculator()
        calculator.calculate_usage(chat_tokens=0, task_tokens=0)

        status = calculator.get_context_status()
        assert status["chat"]["remaining_tokens"] == 4096
        assert status["chat"]["usage_percentage"] == 0.0
        assert status["remaining"] == 4096
