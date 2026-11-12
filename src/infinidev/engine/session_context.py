"""Bounded conversational continuity shared by routing and execution."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MAX_HISTORY_BYTES = 128_000
_UNKNOWN_WINDOW_BYTES = 12_000
_TURN_LIMIT = 200
_OMITTED = (
    "[Older conversation or message text omitted. Use chat_history_search for "
    "earlier decisions before asking the user to repeat them.]"
)


def render_session_context(session_id: str | None, current_input: str = "") -> str:
    """Render recent decisions within a conservative share of the model window.

    UTF-8 budgeting conservatively bounds tokens even for emoji and non-Latin
    prose, leaving room for the system prompt, tools and the current task.
    Full tool transcripts remain in the searchable ledger, not in this block.
    """
    if not session_id:
        return ""
    try:
        from infinidev.config.llm import get_litellm_params
        from infinidev.db.service import get_recent_turns_full
        from infinidev.engine.loop.model_context import get_model_context_window

        window = get_model_context_window(get_litellm_params())
        budget = min(_MAX_HISTORY_BYTES, max(512, window // 4)) if window else (
            _UNKNOWN_WINDOW_BYTES
        )
        turns = get_recent_turns_full(
            session_id, limit=_TURN_LIMIT, max_chars_per_turn=max(budget, len(current_input)),
        )
    except Exception:
        logger.warning("Could not load conversation context", exc_info=True)
        return ""

    # Adapters persist the current user message before running the pipeline.
    # Runtime context may already be appended to the copy passed to the router.
    omitted = len(turns) >= _TURN_LIMIT
    if turns and turns[-1][0] == "user":
        last = turns[-1][1].strip()
        if last and (current_input.strip() == last or current_input.startswith(last + "\n\n")):
            turns = turns[:-1]

    selected: list[str] = []
    remaining = budget - len(_OMITTED) - 2
    for role, content in reversed(turns):
        tag = {"user": "USER", "work_summary": "WORK_LOG"}.get(role, "AGENT")
        opening, closing = f'<turn role="{tag}">\n', "\n</turn>"
        available = remaining - len(opening) - len(closing) - 1
        if available <= 0:
            omitted = True
            break
        encoded = content.encode("utf-8")
        if len(encoded) > available:
            marker = "\n[...message text omitted; use chat_history_search...]\n"
            if available <= len(marker):
                omitted = True
                break
            size = available - len(marker)
            content = (
                encoded[:size // 2].decode("utf-8", errors="ignore")
                + marker
                + encoded[-(size - size // 2):].decode("utf-8", errors="ignore")
            )
            omitted = True
        block = opening + content + closing
        selected.append(block)
        remaining -= len(block.encode("utf-8")) + 1
        if omitted and remaining <= 1:
            break
    if not selected:
        return ""
    header = (
        "Recent conversation (context, not a new request; later corrections supersede "
        "earlier decisions). Use chat_history_search to recover older details."
    )
    return "\n".join([header, *([_OMITTED] if omitted else []), *reversed(selected)])
