"""Message grouping — identifies runs of consecutive same-type messages.

Used by ChatHistoryControl to collapse consecutive system/think/diff
messages into a single group header + the last message visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MessageGroup:
    """A run of consecutive messages sharing the same type."""
    msg_type: str
    messages: list[dict[str, Any]]
    start_index: int  # index of first msg in the full messages list

    @property
    def is_group(self) -> bool:
        return len(self.messages) > 1


# Types that must NEVER be folded into multi-message groups. The user
# wants every tool call, every diff, every exec block, every think and
# every error visible permanently — no accordion. Each message of these
# types is rendered as its own singleton group.
NEVER_GROUP_TYPES: frozenset[str] = frozenset({
    "tool_call", "exec", "diff", "error", "think",
})


def identify_groups(messages: list[dict[str, Any]]) -> list[MessageGroup]:
    """Scan messages into groups of consecutive same-type visible messages.

    Hidden messages (visible=False) are skipped entirely.
    Types in :data:`NEVER_GROUP_TYPES` are forced to singleton groups so
    the chat never collapses tool calls / diffs / thinking under a header.
    """
    groups: list[MessageGroup] = []
    i = 0
    n = len(messages)

    while i < n:
        msg = messages[i]
        if not msg.get("visible", True):
            i += 1
            continue

        msg_type = msg.get("type", "agent")

        # Singleton-only types: emit one group per message and move on.
        if msg_type in NEVER_GROUP_TYPES:
            groups.append(MessageGroup(msg_type=msg_type, messages=[msg], start_index=i))
            i += 1
            continue

        run = [msg]
        start = i
        j = i + 1

        while j < n:
            next_msg = messages[j]
            if not next_msg.get("visible", True):
                j += 1
                continue
            if next_msg.get("type", "agent") != msg_type:
                break
            run.append(next_msg)
            j += 1

        groups.append(MessageGroup(msg_type=msg_type, messages=run, start_index=start))
        i = j

    return groups
