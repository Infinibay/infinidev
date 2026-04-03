"""Thinking budget — provider-aware reasoning token limits.

Translates the user-facing THINKING_BUDGET preset into provider-specific
LLM parameters. Each provider has its own mechanism:

    Provider        | Mechanism
    ────────────────┼──────────────────────────────────────────
    Anthropic       | thinking.budget_tokens  (dedicated field)
    OpenAI (o-series)| reasoning_effort "low"/"medium"/"high"
    Gemini          | thinking_config.thinking_budget (tokens)
    DeepSeek        | max_tokens (total, includes reasoning)
    Ollama/Qwen     | /think vs /no_think prompt tag + max_tokens
    llama.cpp       | max_tokens
    vLLM            | max_tokens
    OpenRouter      | provider-dependent, uses max_tokens
    Others          | max_tokens (universal fallback)
"""

from __future__ import annotations

import logging
from typing import Any

from infinidev.config.settings import settings

logger = logging.getLogger(__name__)

# ── Preset definitions ──────────────────────────────────────────────

# Token budgets per preset.  These are *thinking* tokens, not total
# output tokens.  The actual parameter sent depends on the provider.
_PRESET_TOKENS: dict[str, int] = {
    "low": 1024,
    "medium": 4096,
    "high": 16384,
    "ultra": 0,  # 0 = no limit
}

# OpenAI o-series models use a keyword instead of token count
_OPENAI_EFFORT: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "ultra": "high",  # o-series has no "unlimited"
}


def _resolve_tokens() -> int:
    """Return the thinking budget in tokens from settings."""
    preset = settings.THINKING_BUDGET.lower().strip()
    if preset == "custom":
        return max(0, settings.THINKING_BUDGET_TOKENS)
    return _PRESET_TOKENS.get(preset, _PRESET_TOKENS["medium"])


def _is_openai_reasoning_model(model: str) -> bool:
    """Return True if the model is an OpenAI o-series reasoning model."""
    # o1, o1-mini, o1-pro, o3, o3-mini, o3-pro, o4-mini, etc.
    model_lower = model.lower()
    for prefix in ("o1", "o3", "o4"):
        if prefix in model_lower:
            return True
    return False


# ── Public API ──────────────────────────────────────────────────────

def apply_thinking_budget(
    kwargs: dict[str, Any],
    provider_id: str,
    model: str,
) -> None:
    """Mutate *kwargs* in-place to apply the thinking budget.

    Called from ``call_llm()`` after basic kwargs are assembled but
    before the LLM call is made.
    """
    # ── Master toggle ────────────────────────────────────────────
    if not settings.THINKING_ENABLED:
        return _disable_thinking(kwargs, provider_id, model)

    preset = settings.THINKING_BUDGET.lower().strip()
    tokens = _resolve_tokens()

    # ── Anthropic ────────────────────────────────────────────────
    if provider_id == "anthropic":
        if preset == "ultra" or tokens == 0:
            # Let the model think freely — Anthropic's max is 128k
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 128_000}
        elif preset == "low" and tokens <= 1024:
            # Disable extended thinking entirely for low budget
            kwargs.pop("thinking", None)
        else:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": max(1024, tokens)}
        return

    # ── OpenAI (o-series reasoning models) ───────────────────────
    if provider_id == "openai" and _is_openai_reasoning_model(model):
        effort = _OPENAI_EFFORT.get(preset, "medium")
        if preset == "custom":
            # Map custom token count to effort keyword
            if tokens <= 1024:
                effort = "low"
            elif tokens <= 8192:
                effort = "medium"
            else:
                effort = "high"
        kwargs["reasoning_effort"] = effort
        return

    # ── Gemini ───────────────────────────────────────────────────
    if provider_id == "gemini":
        if preset == "ultra" or tokens == 0:
            kwargs["thinking"] = {"thinking_budget": -1}  # -1 = no limit
        elif preset == "low" and tokens <= 1024:
            kwargs["thinking"] = {"thinking_budget": 0}  # disable
        else:
            kwargs["thinking"] = {"thinking_budget": tokens}
        return

    # ── Ollama (Qwen3, QwQ, DeepSeek, etc.) ─────────────────────
    if provider_id == "ollama":
        # Ollama models with thinking use /think and /no_think tags
        # The tag injection is already handled in call_llm for FC mode;
        # here we handle max_tokens budget.
        if preset == "low":
            # Inject /no_think to suppress thinking entirely
            _inject_prompt_tag(kwargs, "/no_think")
        else:
            # Make sure /no_think is NOT present, allow thinking
            _remove_prompt_tag(kwargs, "/no_think")
            if preset != "ultra" and tokens > 0:
                # Set a generous max_tokens that includes thinking + response
                # Thinking tokens are typically 2-5x the response tokens,
                # so we give extra headroom
                kwargs["max_tokens"] = tokens * 3
        return

    # ── DeepSeek (native provider) ───────────────────────────────
    if provider_id in ("deepseek",):
        # DeepSeek reasoning appears in reasoning_content, controlled by max_tokens
        if preset != "ultra" and tokens > 0:
            kwargs["max_tokens"] = tokens * 3  # thinking + response headroom
        return

    # ── llama.cpp / vLLM / OpenAI-compatible / OpenRouter ────────
    if provider_id in ("llama_cpp", "vllm", "openai_compatible", "openrouter"):
        if preset == "low":
            _inject_prompt_tag(kwargs, "/no_think")
            if tokens > 0:
                kwargs["max_tokens"] = tokens * 2
        elif preset != "ultra" and tokens > 0:
            _remove_prompt_tag(kwargs, "/no_think")
            kwargs["max_tokens"] = tokens * 3
        else:
            _remove_prompt_tag(kwargs, "/no_think")
        return

    # ── Fallback for unknown providers ───────────────────────────
    if preset != "ultra" and tokens > 0:
        kwargs["max_tokens"] = tokens * 3


# ── Prompt tag helpers ──────────────────────────────────────────────

def _inject_prompt_tag(kwargs: dict[str, Any], tag: str) -> None:
    """Inject a tag (e.g. /no_think) into the last user message."""
    msgs = kwargs.get("messages")
    if not msgs:
        return
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            content = msgs[i].get("content", "")
            if tag not in content:
                msgs[i] = {**msgs[i], "content": tag + "\n" + content}
            return


def _remove_prompt_tag(kwargs: dict[str, Any], tag: str) -> None:
    """Remove a prompt tag from user messages if present."""
    msgs = kwargs.get("messages")
    if not msgs:
        return
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            content = msgs[i].get("content", "")
            if tag in content:
                msgs[i] = {**msgs[i], "content": content.replace(tag + "\n", "").replace(tag, "")}
            return


def _disable_thinking(
    kwargs: dict[str, Any], provider_id: str, model: str,
) -> None:
    """Disable thinking/reasoning entirely for all providers.

    Called when ``THINKING_ENABLED=False``.
    """
    # Anthropic: don't send thinking param at all
    if provider_id == "anthropic":
        kwargs.pop("thinking", None)
        return

    # OpenAI o-series: set reasoning_effort to low (closest to "off")
    if provider_id == "openai" and _is_openai_reasoning_model(model):
        kwargs["reasoning_effort"] = "low"
        return

    # Gemini: set budget to 0 (disabled)
    if provider_id == "gemini":
        kwargs["thinking"] = {"thinking_budget": 0}
        return

    # Ollama / llama.cpp / vLLM / OpenRouter / compatible / others:
    # inject /no_think prompt tag
    _inject_prompt_tag(kwargs, "/no_think")
