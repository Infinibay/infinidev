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

from infinidev.config.settings import settings

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
    "failed to parse input",
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


def _stream_and_assemble(
    litellm_mod: Any,
    kwargs: dict,
    on_chunk: Any,
    on_stream_status: "Callable[[str, int, str | None], None] | None" = None,
) -> Any:
    """Stream a completion and assemble into a normal response.

    Calls *on_chunk(text)* for each content or reasoning_content delta,
    giving the UI real-time thinking visibility.

    *on_stream_status(phase, token_count, tool_name)* is called periodically
    to report streaming progress:
      - phase="thinking" — reasoning tokens arriving
      - phase="content"  — regular content tokens arriving
      - phase="tool_detected" — a tool call name was detected in content

    Returns the fully assembled response.
    """
    kwargs_stream = {**kwargs, "stream": True}
    stream = litellm_mod.completion(**kwargs_stream)

    chunks = []
    content_buffer = ""
    token_count = 0
    detected_tool: str | None = None
    is_reasoning = False

    for chunk in stream:
        chunks.append(chunk)
        try:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            reasoning = getattr(delta, "reasoning_content", None) or ""
            content = getattr(delta, "content", None) or ""

            if reasoning:
                is_reasoning = True
                token_count += 1
                on_chunk(reasoning)
                if on_stream_status and token_count % 5 == 0:
                    on_stream_status("thinking", token_count, None)
            elif content:
                is_reasoning = False
                token_count += 1
                content_buffer += content
                on_chunk(content)

                # Early tool call detection from content stream
                if not detected_tool:
                    detected_tool = _detect_tool_name(content_buffer)
                    if detected_tool and on_stream_status:
                        on_stream_status("tool_detected", token_count, detected_tool)

                if on_stream_status and token_count % 5 == 0:
                    on_stream_status("content", token_count, detected_tool)
        except (IndexError, AttributeError):
            pass

    # Final status update
    if on_stream_status:
        phase = "tool_detected" if detected_tool else ("thinking" if is_reasoning else "content")
        on_stream_status(phase, token_count, detected_tool)

    # Assemble chunks into a complete response object
    try:
        assembled = litellm_mod.stream_chunk_builder(chunks)
    except (AttributeError, Exception):
        kwargs_no_stream = {k: v for k, v in kwargs.items() if k != "stream"}
        assembled = litellm_mod.completion(**kwargs_no_stream)
    return assembled


# Regex patterns for early tool name detection from streamed JSON/text
import re

_TOOL_NAME_PATTERNS = [
    # {"name": "tool_name", ...}
    re.compile(r'"name"\s*:\s*"([a-z_][a-z0-9_]*)"', re.IGNORECASE),
    # <tool_call> or <|tool_call|> formats with name field
    re.compile(r'<\|?tool_call\|?>\s*\{[^}]*"name"\s*:\s*"([a-z_][a-z0-9_]*)"', re.IGNORECASE),
    # function_call style
    re.compile(r'"function"\s*:\s*"([a-z_][a-z0-9_]*)"', re.IGNORECASE),
]

# Known tool names to validate detected names against
_KNOWN_TOOLS = frozenset({
    "read_file", "partial_read", "create_file", "replace_lines",
    "list_directory", "code_search", "glob",
    "get_symbol_code", "list_symbols", "search_symbols",
    "find_references", "edit_symbol", "add_symbol", "remove_symbol",
    "project_structure",
    "git_branch", "git_commit", "git_diff", "git_status",
    "execute_command", "code_interpreter",
    "record_finding", "read_findings", "search_findings",
    "step_complete", "help",
})


def _detect_tool_name(text: str) -> str | None:
    """Try to detect a tool name from partially streamed text."""
    for pattern in _TOOL_NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            name = match.group(1)
            # Validate against known tools (models sometimes hallucinate)
            if name in _KNOWN_TOOLS:
                return name
    return None


def call_llm(
    params: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] = "auto",
    on_thinking_chunk: Any | None = None,
    on_stream_status: "Callable[[str, int, str | None], None] | None" = None,
) -> Any:
    """Call litellm.completion with retry for transient errors.

    Adapts request parameters based on probed model capabilities:
    - Downgrades tool_choice="required" → "auto" if unsupported
    - Skips response_format=json_object if unsupported
    - Injects /no_think for models with thinking sections in FC mode

    If *on_thinking_chunk* is provided, enables streaming mode and calls
    ``on_thinking_chunk(text)`` for each content/reasoning chunk as it
    arrives. *on_stream_status(phase, tokens, tool_name)* reports streaming
    progress. The final assembled response is still returned normally.
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
        # Suppress when thinking is disabled or budget is low — higher budgets
        # mean the user explicitly wants reasoning, so we let it through.
        if caps.has_thinking_sections and (
            not settings.THINKING_ENABLED or settings.THINKING_BUDGET.lower() == "low"
        ):
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

    # --- Apply thinking budget ---
    from infinidev.config.thinking_budget import apply_thinking_budget
    apply_thinking_budget(kwargs, settings.LLM_PROVIDER, kwargs["model"])

    # --- Pre-LLM hook ---
    from infinidev.engine.hooks.hooks import hook_manager, HookContext, HookEvent

    llm_ctx = HookContext(
        event=HookEvent.PRE_LLM_CALL,
        metadata={"messages": messages, "tools": tools, "kwargs": kwargs},
    )
    hook_manager.dispatch(llm_ctx)
    if llm_ctx.skip:
        return llm_ctx.metadata.get("response")

    # Enable streaming only for non-tool calls (streaming + FC is unreliable
    # on some providers like Ollama). Thinking is most useful in manual mode anyway.
    use_streaming = on_thinking_chunk is not None and not tools

    last_exc: Exception | None = None
    for attempt in range(1, LLM_RETRIES + 1):
        try:
            if use_streaming:
                response = _stream_and_assemble(
                    litellm, kwargs, on_thinking_chunk,
                    on_stream_status=on_stream_status,
                )
            else:
                response = litellm.completion(**kwargs)

            # --- Post-LLM hook ---
            llm_ctx.event = HookEvent.POST_LLM_CALL
            llm_ctx.metadata["response"] = response
            hook_manager.dispatch(llm_ctx)

            return llm_ctx.metadata["response"]
        except Exception as exc:
            last_exc = exc
            # Auto-drop tool_choice if the endpoint doesn't support it at all
            err_msg = str(exc).lower()
            if "tool_choice" in err_msg or "tool_choise" in err_msg:
                if "tool_choice" in kwargs:
                    logger.info("Dropping tool_choice — endpoint does not support it")
                    del kwargs["tool_choice"]
                    caps.supports_tool_choice_required = False
                    continue  # retry without tool_choice
            if not is_transient(exc) or attempt == LLM_RETRIES:
                raise
            delay = LLM_RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Transient LLM error (attempt %d/%d), retrying in %.1fs: %s",
                attempt, LLM_RETRIES, delay, str(exc)[:200],
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]
