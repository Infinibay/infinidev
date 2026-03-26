"""LLM client with retry logic and model capability adaptation.

Handles:
- Transient error retries with exponential backoff
- Malformed tool call detection (delegated to loop-level handler)
- Model capability adaptation (tool_choice downgrade, JSON mode, /no_think)
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Retry configuration ──────────────────────────────────────────────────

LLM_RETRIES = 5
LLM_RETRY_DELAY = 3.0  # seconds (base; exponential backoff applied)

# ── Error classification ─────────────────────────────────────────────────

# Transient LLM errors that should be retried
TRANSIENT_ERRORS = (
    "connection error",
    "connectionerror",
    "disconnected",
    "rate limit",
    "timeout",
    "503",
    "502",
    "429",
    "overloaded",
    "internal server error",
)

# Permanent errors that look transient but aren't (substrings that
# override a TRANSIENT_ERRORS match when present)
PERMANENT_ERRORS = (
    "does not support tools",
    "does not support function",
    "tool_choice is not supported",
    "tools is not supported",
    "not found",        # Ollama: {"error":"tool 'X' not found"}
)

# Patterns that indicate the LLM produced a malformed tool call
# (e.g. Ollama mixing natural language text with JSON arguments)
MALFORMED_TOOL_PATTERNS = (
    "error parsing tool call",
    "invalid character",
    "looking for beginning of value",
    "unexpected end of json",
    "failed to parse json",
    "unexpected token",
    "unterminated string",
    "after top-level value",
    "after object key:value pair",
)


def is_transient(exc: Exception) -> bool:
    """Check if an LLM exception is transient (worth retrying)."""
    msg = str(exc).lower()
    # Check permanent exclusions first — these are capability errors
    # wrapped in APIConnectionError, not actual network issues
    if any(p in msg for p in PERMANENT_ERRORS):
        return False
    # Malformed tool call errors (Ollama returns 500 for these) should NOT
    # be retried at the LLM call level — the same context will produce the
    # same malformed output.  Let the loop-level malformed handler deal with
    # these instead (it can force a step completion or switch to manual mode).
    if is_malformed_tool_call(exc):
        return False
    return any(p in msg for p in TRANSIENT_ERRORS)


def is_malformed_tool_call(exc: Exception) -> bool:
    """Check if an LLM error is due to a malformed tool call response."""
    msg = str(exc).lower()
    return any(p in msg for p in MALFORMED_TOOL_PATTERNS)


def call_llm(
    params: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] = "auto",
) -> Any:
    """Call litellm.completion with retry for transient errors.

    Adapts request parameters based on probed model capabilities:
    - Downgrades tool_choice="required" → "auto" if unsupported
    - Skips response_format=json_object if unsupported
    - Injects /no_think for models with thinking sections in FC mode
    """
    import litellm
    from infinidev.config.model_capabilities import get_model_capabilities

    caps = get_model_capabilities()

    kwargs: dict[str, Any] = {**params, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        # Downgrade tool_choice if model doesn't support "required"
        if tool_choice == "required" and not caps.supports_tool_choice_required:
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["tool_choice"] = tool_choice
        # Suppress thinking tags for models that emit <think> in FC mode
        # (e.g. Qwen 3.x). Ollama can't parse <think> mixed with tool call
        # JSON, causing "invalid character '<'" errors on every request.
        if caps.has_thinking_sections:
            msgs = kwargs["messages"]
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == "user":
                    content = msgs[i].get("content", "")
                    if "/no_think" not in content:
                        msgs[i] = {**msgs[i], "content": "/no_think\n" + content}
                    break
    # Force JSON output — prevents models (especially Ollama) from mixing
    # natural language text into tool call arguments.
    # NOTE: Do NOT set response_format when tools are present — for llama-server
    # and similar backends, the JSON grammar constraint conflicts with the
    # function calling grammar, causing the model to intermittently return
    # JSON text instead of tool calls.
    if caps.supports_json_mode and not tools:
        kwargs["response_format"] = {"type": "json_object"}

    last_exc: Exception | None = None
    for attempt in range(1, LLM_RETRIES + 1):
        try:
            return litellm.completion(**kwargs)
        except Exception as exc:
            last_exc = exc
            if not is_transient(exc) or attempt == LLM_RETRIES:
                raise
            delay = LLM_RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Transient LLM error (attempt %d/%d), retrying in %.1fs: %s",
                attempt, LLM_RETRIES, delay, str(exc)[:200],
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]
