"""Bounded conversational continuity shared by routing and execution."""

from __future__ import annotations

from collections.abc import Collection
import logging

logger = logging.getLogger(__name__)

_MAX_HISTORY_BYTES = 128_000
_UNKNOWN_WINDOW_BYTES = 12_000
_TURN_LIMIT = 200
_OMITTED = (
    "[Older conversation or message text omitted. Use chat_history_search for "
    "earlier decisions before asking the user to repeat them.]"
)


def render_session_context(
    session_id: str | None, current_input: str = "", *,
    tool_names: Collection[str] | None = None,
) -> str:
    """Render recent decisions within a conservative share of the model window.

    UTF-8 budgeting conservatively bounds tokens even for emoji and non-Latin
    prose, leaving room for the system prompt, tools and the current task.
    Full tool transcripts remain in the searchable ledger, not in this block.
    """
    if not session_id:
        return ""
    from infinidev.engine.context_recording import current_recorder
    recorder = current_recorder()
    history_records = None
    try:
        from infinidev.config.llm import get_litellm_params
        from infinidev.db.service import get_recent_turn_records, get_recent_turns_full
        from infinidev.engine.loop.model_context import get_model_context_window

        window = get_model_context_window(get_litellm_params())
        budget = min(_MAX_HISTORY_BYTES, max(512, window // 4)) if window else (
            _UNKNOWN_WINDOW_BYTES
        )
        options = dict(limit=_TURN_LIMIT, max_chars_per_turn=max(budget, len(current_input)))
        if recorder is None:
            turns = get_recent_turns_full(session_id, **options)
        else:
            history_records = get_recent_turn_records(session_id, **options)
            turns = [(row['role'], row['content']) for row in history_records]
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
            if history_records is not None:
                history_records = history_records[:-1]

    can_search = tool_names is None or 'chat_history_search' in tool_names
    omission = _OMITTED if can_search else '[Older conversation or message text omitted.]'
    selected: list[str] = []
    selected_sources: list[str] = []
    remaining = budget - len(omission) - 2
    for index in reversed(range(len(turns))):
        role, content = turns[index]
        renderer_clipped = False
        clipping = {'loader_limit_chars': options['max_chars_per_turn']}
        tag = {"user": "USER", "work_summary": "WORK_LOG"}.get(role, "AGENT")
        opening, closing = f'<turn role="{tag}">\n', "\n</turn>"
        available = remaining - len(opening) - len(closing) - 1
        if available <= 0:
            omitted = True
            break
        encoded = content.encode("utf-8")
        if len(encoded) > available:
            marker = ("\n[...message text omitted; use chat_history_search...]\n" if can_search
                      else "\n[...message text omitted...]\n")
            if available <= len(marker):
                omitted = True
                break
            size = available - len(marker)
            clipping.update(renderer_head_bytes=size // 2,
                            renderer_tail_bytes=size - size // 2)
            content = (
                encoded[:size // 2].decode("utf-8", errors="ignore")
                + marker
                + encoded[-(size - size // 2):].decode("utf-8", errors="ignore")
            )
            omitted = True
            renderer_clipped = True
        block = opening + content + closing
        selected.append(block)
        if recorder is not None and history_records is not None:
            row = history_records[index]
            selected_sources.append(recorder.source(
                'history', role, content, source_id=row['id'],
                loader_clipped=row['loader_clipped'], renderer_clipped=renderer_clipped,
                original_bytes=row['original_bytes'], clipping=clipping))
        remaining -= len(block.encode("utf-8")) + 1
        if omitted and remaining <= 1:
            break
    if not selected:
        return ""
    header = (
        "Recent conversation (context, not a new request; later corrections supersede "
        "earlier decisions)."
    )
    if can_search:
        header += " Use chat_history_search to recover older details."
    rendered = "\n".join([header, *([omission] if omitted else []), *reversed(selected)])
    if recorder is not None:
        recorder.bundle('history', 'history', 'user', rendered, list(reversed(selected_sources)))
    return rendered
