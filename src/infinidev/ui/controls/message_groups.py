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


def identify_groups(messages: list[dict[str, Any]]) -> list[MessageGroup]:
    """Scan messages into groups of consecutive same-type visible messages.

    Hidden messages (visible=False) are skipped entirely.
    Each group holds references to the original message dicts.
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
