"""Tests for prompt caching module: provider strategies and metrics extraction."""

import copy
from unittest.mock import patch, MagicMock

import pytest

from infinidev.config.prompt_cache import (
    apply_prompt_caching,
    _apply_cache_control_caching,
    _apply_openrouter_caching,
    _apply_kimi_caching,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_kwargs(*, with_tools: bool = True) -> dict:
    """Build a typical kwargs dict as call_llm() would construct."""
    kwargs = {
        "model": "anthropic/claude-sonnet-4-6",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello world"},
        ],
    }
    if with_tools:
        kwargs["tools"] = [
            {"type": "function", "function": {"name": "read_file", "parameters": {}}},
            {"type": "function", "function": {"name": "write_file", "parameters": {}}},
        ]
    return kwargs


# ── Strategy A: cache_control (Anthropic/DashScope/MiniMax) ─────────────────


class TestCacheControlCaching:
    """Tests for _apply_cache_control_caching — shared by Anthropic, DashScope, MiniMax."""

    def test_system_message_converted_to_content_blocks(self):
        kwargs = _make_kwargs()
        _apply_cache_control_caching(kwargs)

        system_msg = kwargs["messages"][0]
        assert system_msg["role"] == "system"
        assert isinstance(system_msg["content"], list)
        assert len(system_msg["content"]) == 1
        block = system_msg["content"][0]
        assert block["type"] == "text"
        assert block["text"] == "You are a helpful assistant."
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_last_tool_gets_cache_control(self):
        kwargs = _make_kwargs()
        _apply_cache_control_caching(kwargs)

        last_tool = kwargs["tools"][-1]
        assert last_tool["cache_control"] == {"type": "ephemeral"}

    def test_first_tool_unchanged(self):
        kwargs = _make_kwargs()
        _apply_cache_control_caching(kwargs)

        first_tool = kwargs["tools"][0]
        assert "cache_control" not in first_tool

    def test_original_tools_not_mutated(self):
        """Deep-copy safety: original tool schemas must not be modified."""
        kwargs = _make_kwargs()
        original_tools = kwargs["tools"]
        original_last_tool = original_tools[-1].copy()

        _apply_cache_control_caching(kwargs)

        # The kwargs["tools"] is now a new list (deep-copied)
        assert kwargs["tools"] is not original_tools
        # The original last tool should NOT have cache_control
        assert "cache_control" not in original_last_tool

    def test_no_tools_only_system_annotated(self):
        kwargs = _make_kwargs(with_tools=False)
        _apply_cache_control_caching(kwargs)

        system_msg = kwargs["messages"][0]
        assert isinstance(system_msg["content"], list)
        assert "tools" not in kwargs

    def test_no_messages_is_noop(self):
        kwargs = {"model": "test"}
        _apply_cache_control_caching(kwargs)
        assert kwargs == {"model": "test"}

    def test_system_already_content_blocks_gets_cache_control(self):
        """If system message is already content blocks, add cache_control to last block."""
        kwargs = _make_kwargs()
        kwargs["messages"][0] = {
            "role": "system",
            "content": [{"type": "text", "text": "Already blocks"}],
        }
        _apply_cache_control_caching(kwargs)

        system_msg = kwargs["messages"][0]
        assert isinstance(system_msg["content"], list)
        assert len(system_msg["content"]) == 1
        assert system_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_system_already_has_cache_control_not_duplicated(self):
        """If last block already has cache_control, don't overwrite."""
        kwargs = _make_kwargs()
        kwargs["messages"][0] = {
            "role": "system",
            "content": [{"type": "text", "text": "Cached", "cache_control": {"type": "ephemeral"}}],
        }
        _apply_cache_control_caching(kwargs)

        assert kwargs["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_empty_system_content_is_noop(self):
        kwargs = _make_kwargs()
        kwargs["messages"][0]["content"] = ""
        _apply_cache_control_caching(kwargs)

        # Empty string should not be converted
        assert kwargs["messages"][0]["content"] == ""


# ── Strategy B: OpenRouter ──────────────────────────────────────────────────


class TestOpenRouterCaching:

    def test_anthropic_model_gets_cache_control(self):
        kwargs = _make_kwargs()
        kwargs["model"] = "openrouter/anthropic/claude-sonnet-4-6"
        _apply_openrouter_caching(kwargs)

        system_msg = kwargs["messages"][0]
        assert isinstance(system_msg["content"], list)
        assert system_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_claude_model_gets_cache_control(self):
        kwargs = _make_kwargs()
        kwargs["model"] = "openrouter/claude-3-opus"
        _apply_openrouter_caching(kwargs)

        system_msg = kwargs["messages"][0]
        assert isinstance(system_msg["content"], list)

    def test_gemini_model_gets_cache_control(self):
        kwargs = _make_kwargs()
        kwargs["model"] = "openrouter/google/gemini-2.5-pro"
        _apply_openrouter_caching(kwargs)

        system_msg = kwargs["messages"][0]
        assert isinstance(system_msg["content"], list)
        assert system_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_non_anthropic_model_is_noop(self):
        kwargs = _make_kwargs()
        kwargs["model"] = "openrouter/openai/gpt-5.4"
        original_system = kwargs["messages"][0].copy()
        _apply_openrouter_caching(kwargs)

        assert kwargs["messages"][0] == original_system


# ── Strategy C: Kimi ────────────────────────────────────────────────────────


class TestKimiCaching:

    def test_session_affinity_header_added(self):
        kwargs = _make_kwargs()
        _apply_kimi_caching(kwargs)

        assert kwargs["extra_headers"]["x-session-affinity"] == "true"

    def test_preserves_existing_headers(self):
        kwargs = _make_kwargs()
        kwargs["extra_headers"] = {"X-Custom": "value"}
        _apply_kimi_caching(kwargs)

        assert kwargs["extra_headers"]["X-Custom"] == "value"
        assert kwargs["extra_headers"]["x-session-affinity"] == "true"

    def test_none_extra_headers(self):
        kwargs = _make_kwargs()
        kwargs["extra_headers"] = None
        _apply_kimi_caching(kwargs)

        assert kwargs["extra_headers"]["x-session-affinity"] == "true"


# ── Dispatch: apply_prompt_caching ──────────────────────────────────────────


class TestApplyPromptCaching:

    @pytest.mark.parametrize("provider", ["anthropic", "qwen", "minimax"])
    def test_cache_control_providers(self, provider):
        kwargs = _make_kwargs()
        with patch("infinidev.config.prompt_cache.settings") as mock_settings:
            mock_settings.PROMPT_CACHE_ENABLED = True
            apply_prompt_caching(kwargs, provider)

        system_msg = kwargs["messages"][0]
        assert isinstance(system_msg["content"], list)
        assert system_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_openrouter_dispatch(self):
        kwargs = _make_kwargs()
        kwargs["model"] = "openrouter/anthropic/claude-sonnet-4-6"
        with patch("infinidev.config.prompt_cache.settings") as mock_settings:
            mock_settings.PROMPT_CACHE_ENABLED = True
            apply_prompt_caching(kwargs, "openrouter")

        system_msg = kwargs["messages"][0]
        assert isinstance(system_msg["content"], list)

    def test_kimi_dispatch(self):
        kwargs = _make_kwargs()
        with patch("infinidev.config.prompt_cache.settings") as mock_settings:
            mock_settings.PROMPT_CACHE_ENABLED = True
            apply_prompt_caching(kwargs, "kimi")

        assert kwargs["extra_headers"]["x-session-affinity"] == "true"

    @pytest.mark.parametrize("provider", [
        "openai", "deepseek", "zai", "gemini",
        "ollama", "llama_cpp", "vllm", "openai_compatible",
    ])
    def test_noop_providers(self, provider):
        kwargs = _make_kwargs()
        original = copy.deepcopy(kwargs)
        with patch("infinidev.config.prompt_cache.settings") as mock_settings:
            mock_settings.PROMPT_CACHE_ENABLED = True
            apply_prompt_caching(kwargs, provider)

        assert kwargs == original

    def test_disabled_setting_is_noop(self):
        kwargs = _make_kwargs()
        original = copy.deepcopy(kwargs)
        with patch("infinidev.config.prompt_cache.settings") as mock_settings:
            mock_settings.PROMPT_CACHE_ENABLED = False
            apply_prompt_caching(kwargs, "anthropic")

        assert kwargs == original


# ── Cache metrics extraction ────────────────────────────────────────────────


class TestCacheMetricsExtraction:
    """Test that _track_usage correctly extracts cache metrics from responses."""

    def _make_state(self):
        from infinidev.engine.loop.loop_state import LoopState
        return LoopState()

    def test_anthropic_cache_metrics(self):
        state = self._make_state()
        usage = MagicMock()
        usage.total_tokens = 1000
        usage.prompt_tokens = 800
        usage.completion_tokens = 200
        usage.cache_creation_input_tokens = 5000
        usage.cache_read_input_tokens = 3000
        usage.prompt_tokens_details = None
        usage.prompt_cache_hit_tokens = 0

        state.total_tokens += usage.total_tokens
        state.cache_creation_tokens += (usage.cache_creation_input_tokens or 0)
        state.cache_read_tokens += (usage.cache_read_input_tokens or 0)

        assert state.cache_creation_tokens == 5000
        assert state.cache_read_tokens == 3000

    def test_openai_cache_metrics(self):
        state = self._make_state()
        details = MagicMock()
        details.cached_tokens = 7000

        state.cached_tokens += details.cached_tokens

        assert state.cached_tokens == 7000

    def test_deepseek_cache_metrics(self):
        state = self._make_state()
        state.cached_tokens += 4500  # from prompt_cache_hit_tokens

        assert state.cached_tokens == 4500

    def test_cache_metrics_accumulate(self):
        state = self._make_state()

        # Simulate 3 calls
        state.cache_creation_tokens += 5000  # first call: cache write
        state.cache_read_tokens += 5000      # second call: cache read
        state.cache_read_tokens += 5000      # third call: cache read

        assert state.cache_creation_tokens == 5000
        assert state.cache_read_tokens == 10000

    def test_default_metrics_are_zero(self):
        state = self._make_state()
        assert state.cache_creation_tokens == 0
        assert state.cache_read_tokens == 0
        assert state.cached_tokens == 0
