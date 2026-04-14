"""Prompt caching — provider-aware cache annotation for LLM calls.

Annotates messages and tool schemas with provider-specific caching
markers to reduce cost on repeated prompts. Each provider has its own
mechanism:

    Provider        | Mechanism
    ────────────────┼──────────────────────────────────────────
    Anthropic       | cache_control on content blocks + tools
    Qwen/DashScope  | cache_control (same format as Anthropic)
    MiniMax         | cache_control (same format as Anthropic)
    OpenRouter      | Pass-through to underlying provider
    Kimi            | x-session-affinity header for sticky routing
    OpenAI          | Automatic prefix caching (no-op)
    DeepSeek        | Automatic prefix caching (no-op)
    Gemini          | Implicit caching on 2.5+ (no-op)
    ZAI/Zhipu       | Automatic caching (no-op)
    Ollama/local    | N/A (no-op)
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from infinidev.config.settings import settings

logger = logging.getLogger(__name__)

# Providers that support Anthropic-style cache_control on content blocks.
_CACHE_CONTROL_PROVIDERS = frozenset({"anthropic", "qwen", "minimax"})


# ── Public API ──────────────────────────────────────────────────────

def apply_prompt_caching(
    kwargs: dict[str, Any],
    provider_id: str,
) -> None:
    """Mutate *kwargs* in-place to enable provider-specific prompt caching.

    Called from ``call_llm()`` after thinking budget is applied but
    before the actual LLM call.

    Also strips the ``CACHE_BREAKPOINT_MARKER`` sentinel from any system
    message that happens to reach a provider without explicit cache
    breakpoints — otherwise a benign-looking HTML comment would leak to
    the model.
    """
    if not settings.PROMPT_CACHE_ENABLED:
        _strip_cache_breakpoint(kwargs)
        return

    if provider_id in _CACHE_CONTROL_PROVIDERS:
        _apply_cache_control_caching(kwargs)
    elif provider_id == "openrouter":
        _apply_openrouter_caching(kwargs)
    elif provider_id == "kimi":
        _apply_kimi_caching(kwargs)
    # openai, deepseek, zai, gemini: automatic prefix caching, no-op
    # ollama, llama_cpp, vllm, openai_compatible: local/no cache API, no-op

    # Anything not handled above — strip the marker so it never reaches
    # a provider that can't use it.
    _strip_cache_breakpoint(kwargs)


def _strip_cache_breakpoint(kwargs: dict[str, Any]) -> None:
    """Remove the cache-breakpoint sentinel from string system messages."""
    from infinidev.engine.loop.context import CACHE_BREAKPOINT_MARKER

    messages = kwargs.get("messages")
    if not messages:
        return
    for i, msg in enumerate(messages):
        if msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and CACHE_BREAKPOINT_MARKER in content:
            messages[i] = {
                "role": "system",
                "content": content.replace(
                    f"\n\n{CACHE_BREAKPOINT_MARKER}\n\n", "\n\n"
                ).replace(CACHE_BREAKPOINT_MARKER, ""),
            }


# ── Strategy A: cache_control (Anthropic / DashScope / MiniMax) ─────

def _apply_cache_control_caching(kwargs: dict[str, Any]) -> None:
    """Annotate system message and tools with cache_control breakpoints.

    Uses 2 of the 4 available breakpoints:
      1. Stable system prefix (identity + tech + protocol)
      2. Last tool schema (static per session)

    When the system prompt contains ``CACHE_BREAKPOINT_MARKER``, the
    message is split at that marker into two content blocks — only the
    first (stable prefix) gets ``cache_control``. The second (session
    context, changes per iteration) is sent plain so growing it doesn't
    invalidate the cache. Without the marker, the whole system message
    is cached (legacy behavior).
    """
    from infinidev.engine.loop.context import CACHE_BREAKPOINT_MARKER

    messages = kwargs.get("messages")
    if not messages:
        return

    for i, msg in enumerate(messages):
        if msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            if CACHE_BREAKPOINT_MARKER in content:
                stable, _, dynamic = content.partition(CACHE_BREAKPOINT_MARKER)
                blocks: list[dict[str, Any]] = [{
                    "type": "text",
                    "text": stable.rstrip(),
                    "cache_control": {"type": "ephemeral"},
                }]
                dynamic_text = dynamic.lstrip()
                if dynamic_text:
                    blocks.append({"type": "text", "text": dynamic_text})
                messages[i] = {"role": "system", "content": blocks}
            else:
                messages[i] = {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }],
                }
        elif isinstance(content, list) and content:
            # Already content blocks (e.g. from thinking_budget) —
            # add cache_control to the last block.
            last_block = content[-1]
            if "cache_control" not in last_block:
                last_block["cache_control"] = {"type": "ephemeral"}
        break

    # Last tool schema → cache_control (deep-copy to avoid mutating
    # the shared schema list).
    tools = kwargs.get("tools")
    if tools:
        kwargs["tools"] = copy.deepcopy(tools)
        kwargs["tools"][-1]["cache_control"] = {"type": "ephemeral"}


# ── Strategy B: OpenRouter pass-through ─────────────────────────────

def _apply_openrouter_caching(kwargs: dict[str, Any]) -> None:
    """Apply caching for OpenRouter based on the underlying model.

    OpenRouter passes cache_control through to providers that support
    explicit breakpoints:
      - Anthropic/Claude: cache_control with ephemeral type
      - Google/Gemini: cache_control on content blocks

    Other providers (OpenAI, DeepSeek, Grok, Groq) use automatic
    prefix caching — no annotation needed.
    """
    model = kwargs.get("model", "").lower()
    if any(k in model for k in ("anthropic", "claude", "gemini", "google")):
        _apply_cache_control_caching(kwargs)


# ── Strategy C: Kimi session affinity ───────────────────────────────

def _apply_kimi_caching(kwargs: dict[str, Any]) -> None:
    """Add session affinity header for Kimi K2 models.

    Routes requests to the same model instance, improving cache hit
    rates for automatic prefix caching.
    """
    extra = kwargs.get("extra_headers") or {}
    extra["x-session-affinity"] = "true"
    kwargs["extra_headers"] = extra
