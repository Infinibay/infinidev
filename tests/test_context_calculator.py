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

    def _over_budget_warning(self, monkeypatch, caplog, provider_id, model):
        """Drive the one-shot overflow warning for a given backend."""
        from infinidev.config.settings import settings

        monkeypatch.setattr(settings, "LLM_PROVIDER", provider_id)
        monkeypatch.setattr(settings, "LLM_MODEL", model)
        monkeypatch.setattr(settings, "OLLAMA_NUM_CTX", 0)
        monkeypatch.setattr(settings, "LLM_CONTEXT_WINDOW", 0)

        caplog.set_level("WARNING", logger="infinidev.ui.context_calculator")
        calc = ContextWindowCalculator(model_name=model, max_context=1000)
        calc.update_task(task_prompt_tokens=1500)
        return "\n".join(r.message for r in caplog.records)

    def test_over_budget_warning_names_ollama_knob_for_ollama(self, monkeypatch, caplog):
        """Ollama is the one backend that takes num_ctx from a setting."""
        message = self._over_budget_warning(
            monkeypatch, caplog, "ollama", "ollama_chat/qwen3"
        )
        assert "INFINIDEV_OLLAMA_NUM_CTX" in message
        assert "INFINIDEV_LLM_CONTEXT_WINDOW" not in message

    @pytest.mark.parametrize("provider_id", ["llama_cpp", "vllm", "openai_compatible"])
    def test_over_budget_warning_names_server_flag_when_self_hosted(
        self, monkeypatch, caplog, provider_id
    ):
        """A self-hosted server allocates its own window; num_ctx is Ollama-only.

        Telling a llama.cpp/vLLM user to raise INFINIDEV_OLLAMA_NUM_CTX points
        at a variable config/llm.py never sends them, and raising
        INFINIDEV_LLM_CONTEXT_WINDOW alone would only make Infinidev assume a
        window the server still truncates to.
        """
        message = self._over_budget_warning(
            monkeypatch, caplog, provider_id, "custom_openai/qwen"
        )
        assert "INFINIDEV_OLLAMA_NUM_CTX" not in message
        assert "--ctx-size" in message
        assert "--max-model-len" in message
        assert "INFINIDEV_LLM_CONTEXT_WINDOW" in message

    def test_over_budget_warning_names_server_flag_for_unregistered_provider(
        self, monkeypatch, caplog
    ):
        """An id outside the registry still reaches the generic OpenAI transport.

        ``_is_self_hosted_openai`` falls back to the ``custom_openai/`` prefix
        in that case, which only works because the advice reads the configured
        model id with its prefix rather than the bare display name.
        """
        message = self._over_budget_warning(
            monkeypatch, caplog, "strata", "custom_openai/qwen"
        )
        assert "INFINIDEV_OLLAMA_NUM_CTX" not in message
        assert "--ctx-size" in message

    def test_over_budget_warning_names_no_setting_for_cloud(self, monkeypatch, caplog):
        """A cloud limit isn't the user's to raise, so only the prompt is."""
        message = self._over_budget_warning(
            monkeypatch, caplog, "anthropic", "anthropic/claude-sonnet-5"
        )
        assert "INFINIDEV_OLLAMA_NUM_CTX" not in message
        assert "INFINIDEV_LLM_CONTEXT_WINDOW" not in message
        assert "shorten the task" in message

    def test_over_budget_warning_fires_once_per_task(self, monkeypatch, caplog):
        """The one-shot fuse still holds after the advice became a method call."""
        from infinidev.config.settings import settings

        monkeypatch.setattr(settings, "LLM_PROVIDER", "llama_cpp")
        monkeypatch.setattr(settings, "LLM_MODEL", "custom_openai/qwen")

        caplog.set_level("WARNING", logger="infinidev.ui.context_calculator")
        calc = ContextWindowCalculator(model_name="qwen", max_context=1000)
        calc.update_task(task_prompt_tokens=1500)
        calc.update_task(task_prompt_tokens=1800)
        warnings = [r for r in caplog.records if "truncating context" in r.message]
        assert len(warnings) == 1

        calc.start_task()
        calc.update_task(task_prompt_tokens=1500)
        warnings = [r for r in caplog.records if "truncating context" in r.message]
        assert len(warnings) == 2

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

    def test_resolve_model_context_works_inside_a_running_loop(self, monkeypatch):
        """Regression: asyncio.run here raised, and the raise was swallowed.

        Startup runs inside prompt_toolkit's loop, so every model displayed
        ``?`` for its context window no matter what the catalog said.
        """
        import asyncio

        from infinidev.config.settings import settings

        monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
        monkeypatch.setattr(settings, "LLM_MODEL", "ollama_chat/test-model")
        monkeypatch.setattr(
            "infinidev.engine.loop.model_context._fetch_ollama_trained_context",
            lambda *_args: 262_144,
        )

        async def main():
            calc = ContextWindowCalculator()
            calc.resolve_model_context()
            return calc.max_context

        assert asyncio.run(main()) == 262_144
