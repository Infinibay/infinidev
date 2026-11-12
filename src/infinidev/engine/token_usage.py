"""Report what a model actually received, for the context meter.

The LoopEngine reads ``response.usage`` itself and threads the number through
``LoopState.last_prompt_tokens``. The four loops that are not the LoopEngine —
chat agent, planner, council, spec elaborator — never did, and the TUI filled
the gap by estimating from the user's message plus the session summaries.

That estimate measured the one part of the prompt that never matters. The
system prompt alone is ~4 400 tokens before a single tool schema or tool
result, so a turn carrying tens of thousands displayed 108.

``usage`` is the truth when the provider sends it. A streamed response carries
none unless the caller asked for it, so the fallback counts the messages and tool schemas
that were just sent — still a measurement of the real prompt, not a guess at
one fragment of it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _usage_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    try:
        return int(getattr(usage, "prompt_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _count_messages(
    messages: list[dict] | None,
    model: str,
    *,
    tools: list[dict] | None = None,
) -> int:
    """Estimate the complete prompt, including any tool schemas sent with it."""
    if not messages and not tools:
        return 0
    try:
        import litellm

        kwargs: dict[str, Any] = {"model": model or "gpt-4", "messages": messages or []}
        if tools:
            kwargs["tools"] = tools
        return int(litellm.token_counter(**kwargs))
    except Exception:
        try:
            payload: dict[str, Any] = {"messages": messages or []}
            if tools:
                payload["tools"] = tools
            # Include framing and schema keys; three chars/token errs above the
            # usual four-char heuristic when the model tokenizer is unavailable.
            chars = len(json.dumps(payload, ensure_ascii=False, default=str))
            return (chars + 2) // 3
        except Exception:
            logger.debug("prompt size estimation failed", exc_info=True)
            return 0


def report_prompt_tokens(
    hooks: Any,
    response: Any = None,
    *,
    lane: str = "chat",
    messages: list[dict] | None = None,
    model: str = "",
    tools: list[dict] | None = None,
) -> int:
    """Forward one call's real prompt size to whoever is metering.

    Returns the count so a caller can log or assert on it. Never raises: a
    meter is not worth failing a turn over.
    """
    tokens = _usage_tokens(response)
    if tokens <= 0:
        tokens = _count_messages(messages, model, tools=tools)
    if tokens <= 0 or hooks is None:
        return tokens
    notify = getattr(hooks, "notify_token_usage", None)
    if notify is None:
        return tokens
    try:
        notify(tokens, lane)
    except Exception:  # pragma: no cover - a meter never breaks a run
        logger.debug("token usage notify failed", exc_info=True)
    return tokens
