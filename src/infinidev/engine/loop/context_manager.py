"""Message window management for the loop engine.

Extracted from ``engine.py`` so context-window policies (expiring old
thinking, compacting tool outputs for small models) live in one place
and can be unit-tested without spinning up a full ``LoopEngine``.

The policies here are pure: ``list[dict]`` in, ``list[dict]`` mutated
in place. No LLM calls, no state, no I/O.
"""

from __future__ import annotations

from typing import Any

CONTEXT_COMPACTION_USED_FRACTION = 0.70
CONTEXT_COMPACTION_MIN_REMAINING = 100_000
CONTEXT_PRESSURE_TOOL_RESULT_CHARS = 400


class ContextManager:
    """Context-window policies applied between loop iterations."""

    # How many tool call rounds before assistant thinking is truncated.
    THINKING_TTL = 3
    # A tool result is fully visible to the model on the request immediately
    # after it is produced. Once a newer assistant turn exists, resending the
    # complete body charges for evidence the model already consumed. Keep the
    # current result intact and compact prior rounds deterministically.
    TOOL_RESULT_TTL = 1
    TOOL_RESULT_COMPACT_CHARS = 1_200

    @staticmethod
    def expire_thinking(messages: list[dict[str, Any]], ttl: int = THINKING_TTL) -> None:
        """Truncate old assistant thinking to save context window.

        Assistant messages carry a ``_thinking_age`` counter that increments
        each time this method is called.  Once a message is older than
        ``ttl`` rounds, its ``content`` (reasoning text) is replaced with
        a short placeholder — the ``tool_calls`` structure stays intact
        so the API conversation remains valid.

        For manual-TC mode (no ``tool_calls``), the entire assistant
        content is the reasoning, so we truncate it to the first line.
        """
        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if not content or len(content) < 80:
                continue  # Already short, skip

            age = msg.get("_thinking_age", 0) + 1
            msg["_thinking_age"] = age

            if age <= ttl:
                continue

            first_line = content.split("\n", 1)[0][:120]
            # Same placeholder for FC and manual modes; distinction kept
            # for future divergence but both paths collapse to this today.
            msg["content"] = f"[thinking truncated] {first_line}"

    @staticmethod
    def compact_for_small(messages: list[dict[str, Any]]) -> None:
        """Compact old messages in the inner loop for small models.

        Small models have limited context. This truncates tool result
        messages older than the last 2 assistant rounds to their first
        200 chars, preventing context bloat from large tool outputs.
        The system and first user message are always preserved.
        """
        # Count assistant messages from the end to find the cutoff.
        # The default is "nothing is old yet": with fewer than two assistant
        # rounds there is no history to compact, and defaulting to
        # len(messages) instead truncated the result the model was about to
        # read — every step's first tool call came back as a 224-char stub.
        assistant_count = 0
        cutoff_idx = 2
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                assistant_count += 1
                if assistant_count >= 2:
                    cutoff_idx = i
                    break

        # Truncate tool results before the cutoff (skip system + first user)
        for i in range(2, cutoff_idx):
            msg = messages[i]
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if len(content) > 200:
                    msg["content"] = content[:200] + "\n[truncated for context]"
            elif msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content and len(content) > 100:
                    first_line = content.split("\n", 1)[0][:100]
                    msg["content"] = f"[compacted] {first_line}"

    @staticmethod
    def compact_old_tool_results(
        messages: list[dict[str, Any]],
        *,
        keep_assistant_rounds: int = TOOL_RESULT_TTL,
        max_chars: int = TOOL_RESULT_COMPACT_CHARS,
    ) -> None:
        """Bound old tool-result bodies while preserving recent evidence.

        A long-context model still pays for every old file body and command
        output on every subsequent request in the same Step. The model has
        already consumed results older than ``keep_assistant_rounds``; retain
        a deterministic head/tail excerpt for error identity and protocol
        continuity instead of repeatedly resending the full body.
        """
        if keep_assistant_rounds < 1 or max_chars < 80:
            return

        assistant_count = 0
        cutoff_idx = 0
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") != "assistant":
                continue
            assistant_count += 1
            if assistant_count >= keep_assistant_rounds:
                cutoff_idx = i
                break
        else:
            return

        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars
        for message in messages[:cutoff_idx]:
            if message.get("role") != "tool":
                continue
            content = message.get("content", "")
            if not isinstance(content, str) or len(content) <= max_chars:
                continue
            omitted = len(content) - max_chars
            message["content"] = (
                content[:head_chars]
                + f"\n[... {omitted} chars compacted after prior delivery ...]\n"
                + content[-tail_chars:]
            )

    @staticmethod
    def under_context_pressure(prompt_tokens: int, max_context_tokens: int) -> bool:
        """Whether the real prompt crossed either automatic compaction trigger."""
        if prompt_tokens <= 0 or max_context_tokens <= 0:
            return False
        remaining = max(0, max_context_tokens - prompt_tokens)
        return (
            prompt_tokens / max_context_tokens >= CONTEXT_COMPACTION_USED_FRACTION
            or remaining < CONTEXT_COMPACTION_MIN_REMAINING
        )

    @staticmethod
    def compact_for_pressure(messages: list[dict[str, Any]]) -> None:
        """Aggressively compact evidence already consumed by a newer model turn."""
        ContextManager.expire_thinking(messages, ttl=0)
        ContextManager.compact_for_small(messages)
        ContextManager.compact_old_tool_results(
            messages,
            keep_assistant_rounds=1,
            max_chars=CONTEXT_PRESSURE_TOOL_RESULT_CHARS,
        )
