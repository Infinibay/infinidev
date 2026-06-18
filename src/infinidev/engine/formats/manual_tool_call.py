"""ManualToolCall — a lightweight stand-in for native tool call objects.

Used in manual tool-call mode (models without native function calling) so the
rest of the loop can treat parsed text tool calls uniformly with provider
``ChatCompletionMessageToolCall`` objects.
"""

from __future__ import annotations


class ManualToolCall:
    """Lightweight stand-in for native tool call objects in manual TC mode."""

    __slots__ = ("id", "function")

    class _Function:
        __slots__ = ("name", "arguments")

        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.function = self._Function(name, arguments)
