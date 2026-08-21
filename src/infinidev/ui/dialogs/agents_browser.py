"""Council and subagent picker for the prompt-toolkit TUI."""

from __future__ import annotations

from typing import Any, Callable

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import FormattedTextControl

from infinidev.engine.council import observer
from infinidev.ui.theme import ACCENT, PRIMARY, SUCCESS, TEXT, TEXT_DIM, TEXT_MUTED


class AgentsBrowserControl(FormattedTextControl):
    """Live, keyboard-navigable list of council debates and members."""

    def __init__(self, on_open: Callable[[str, str | None], None]) -> None:
        self.cursor = 0
        self._on_open = on_open
        super().__init__(self._fragments, focusable=True)

    def entries(self) -> list[tuple[str, str | None, dict[str, Any]]]:
        entries: list[tuple[str, str | None, dict[str, Any]]] = []
        for council in observer.list_councils(include_messages=False):
            entries.append((council["id"], None, council))
            for member in council.get("members", {}).values():
                entries.append((council["id"], member.get("member_id"), member))
        return entries

    def move(self, delta: int) -> None:
        entries = self.entries()
        if entries:
            self.cursor = (self.cursor + delta) % len(entries)

    def open_selected(self) -> None:
        entries = self.entries()
        if not entries:
            return
        council_id, member_id, _ = entries[min(self.cursor, len(entries) - 1)]
        self._on_open(council_id, member_id)

    def _fragments(self) -> FormattedText:
        entries = self.entries()
        evicted = observer.council_eviction_count()
        if not entries:
            if evicted:
                return FormattedText([
                    (TEXT_MUTED, " No recent councils are retained.\n"),
                    (TEXT_DIM, f" {evicted} older council transcript(s) were evicted."),
                ])
            return FormattedText([
                (TEXT_MUTED, " No councils have run in this process.\n"),
                (TEXT_DIM, " Start a council and its agents will appear here."),
            ])
        self.cursor = min(self.cursor, len(entries) - 1)
        fragments: list[tuple[str, str]] = []
        for index, (council_id, member_id, item) in enumerate(entries):
            selected = index == self.cursor
            marker = "› " if selected else "  "
            base = f"bg:{PRIMARY} #ffffff bold" if selected else TEXT
            if member_id is None:
                status = item.get("status", "")
                running = sum(
                    member.get("status") == "running"
                    for member in item.get("members", {}).values()
                )
                label = f"{council_id}  [{status}]"
                if running:
                    label += f"  Running {running} agents"
                fragments.append((base, f"{marker}{label}\n"))
                fragments.append((TEXT_DIM, f"    {item.get('question', '')[:100]}\n"))
            else:
                status = item.get("status", "waiting")
                colour = SUCCESS if status == "completed" else ACCENT
                persona = item.get("persona", "")
                objective = item.get("objective", "")
                fragments.append((base, f"{marker}  {member_id} · {persona}"))
                fragments.append((colour, f"  {status}\n"))
                fragments.append((TEXT_MUTED, f"      {objective[:96]}\n"))
        if evicted:
            fragments.append((
                TEXT_DIM,
                f"\n {evicted} older council transcript(s) were evicted.\n",
            ))
        fragments.append((TEXT_DIM, "\n ↑/↓ select   Enter open tab   Esc close"))
        return FormattedText(fragments)


__all__ = ["AgentsBrowserControl"]
