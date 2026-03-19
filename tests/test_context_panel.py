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
            assert panel.query_one("#ctx-model", Static)
            assert panel.query_one("#ctx-chat", Static)
            assert panel.query_one("#ctx-task", Static)

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
            }
            panel.update_status(status)

            model_w = panel.query_one("#ctx-model", Static)
            assert "llama2" in str(model_w._Static__content)

            chat_w = panel.query_one("#ctx-chat", Static)
            content = str(chat_w._Static__content)
            assert "Chat" in content

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
            }
            panel.update_status(status)
            chat_w = panel.query_one("#ctx-chat", Static)
            assert "50%" in str(chat_w._Static__content)

    async def test_context_panel_flow(self):
        """Test flow indicator in model line."""
        app = ContextPanelTestApp()
        async with app.run_test():
            panel = app.query_one(ContextPanel)
            panel.update_status({"model": "test", "max_context": 4096,
                                 "chat": {"usage_percentage": 0},
                                 "tasks": {"usage_percentage": 0}})
            panel.set_flow("research")
            model_w = panel.query_one("#ctx-model", Static)
            assert "research" in str(model_w._Static__content)

            panel.set_flow("")
            model_w = panel.query_one("#ctx-model", Static)
            assert "research" not in str(model_w._Static__content)


class TestContextWindowCalculatorIntegration:
    """Test ContextWindowCalculator integration with ContextPanel."""

    async def test_calculator_with_panel(self):
        """Test that calculator data works with panel display."""
        calculator = ContextWindowCalculator(model_name="test-model", max_context=8192)
        calculator.update_chat("Hello", ["summary1", "summary2"])
        calculator.update_task(task_prompt_tokens=2048)

        status = calculator.get_context_status()

        app = ContextPanelTestApp()
        async with app.run_test():
            panel = app.query_one(ContextPanel)
            panel.update_status(status)

            model_w = panel.query_one("#ctx-model", Static)
            assert "test-model" in str(model_w._Static__content)

    def test_calculator_full_context(self):
        """Test calculator at full context capacity."""
        calculator = ContextWindowCalculator(max_context=4096)
        calculator.update_task(task_prompt_tokens=4096)

        status = calculator.get_context_status()
        assert status["tasks"]["remaining_tokens"] == 0
        assert status["tasks"]["usage_percentage"] == 1.0

    def test_calculator_zero_context(self):
        """Test calculator with zero tokens."""
        calculator = ContextWindowCalculator()
        status = calculator.get_context_status()
        assert status["chat"]["remaining_tokens"] == 4096
        assert status["chat"]["usage_percentage"] == 0.0
