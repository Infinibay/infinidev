"""Message window management for the loop engine.

Extracted from ``engine.py`` so context-window policies (expiring old
thinking, compacting tool outputs for small models) live in one place
and can be unit-tested without spinning up a full ``LoopEngine``.

The policies here are pure: ``list[dict]`` in, ``list[dict]`` mutated
in place. No LLM calls, no state, no I/O.
"""

from __future__ import annotations

from typing import Any


class ContextManager:
    """Context-window policies applied between loop iterations."""

    # How many tool call rounds before assistant thinking is truncated.
    THINKING_TTL = 3

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
        # Count assistant messages from the end to find the cutoff
        assistant_count = 0
        cutoff_idx = len(messages)
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
