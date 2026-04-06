"""Debug panel — tabbed modal for inspecting agent internals.

Left panel: section tabs (Notes, History, Plan, State).
Right panel: scrollable content for the selected section.
"""

from __future__ import annotations

from typing import Any

from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import UIControl, UIContent
from prompt_toolkit.mouse_events import MouseEventType

from infinidev.ui.theme import (
    PRIMARY, ACCENT, TEXT, TEXT_MUTED, SURFACE_LIGHT, SUCCESS, ERROR,
)


# ── Sections ──────────────────────────────────────────────────────────

SECTIONS = ["Notes", "History", "Plan", "State", "Behavior"]


# ── State ─────────────────────────────────────────────────────────────

class DebugPanelState:
    """Holds all data for the debug panel, refreshed on each open."""

    def __init__(self) -> None:
        self.section_cursor: int = 0
        self.scroll: int = 0
        self.focus: str = "sections"  # "sections" or "content"

        # Data — populated by open_debug()
        self.session_notes: list[str] = []
        self.task_notes: list[str] = []
        self.history: list[Any] = []       # list of ActionRecord
        self.plan_text: str = ""
        self.state_info: dict[str, Any] = {}
        # Behavior subsystem
        self.behavior_scores: list[tuple[str, int]] = []   # [(agent_id, score)]
        self.behavior_events: list[Any] = []                # list[BehaviorEvent], oldest first

    @property
    def current_section(self) -> str:
        return SECTIONS[self.section_cursor] if 0 <= self.section_cursor < len(SECTIONS) else ""

    def move_section(self, delta: int) -> None:
        self.section_cursor = max(0, min(len(SECTIONS) - 1, self.section_cursor + delta))
        self.scroll = 0

    def scroll_up(self) -> None:
        self.scroll = max(0, self.scroll - 1)

    def scroll_down(self, max_lines: int) -> None:
        self.scroll = min(max(0, max_lines - 1), self.scroll + 1)

    def content_lines(self, width: int) -> list[list[tuple[str, str]]]:
        """Build styled lines for the current section."""
        section = self.current_section
        if section == "Notes":
            return self._render_notes(width)
        elif section == "History":
            return self._render_history(width)
        elif section == "Plan":
            return self._render_plan(width)
        elif section == "State":
            return self._render_state(width)
        elif section == "Behavior":
            return self._render_behavior(width)
        return [[(f"{TEXT_MUTED}", " (empty)")]]

    # ── Renderers ──

    def _render_notes(self, w: int) -> list[list[tuple[str, str]]]:
        lines: list[list[tuple[str, str]]] = []
        lines.append([(f"bg:{PRIMARY} #ffffff bold", _pad("Session Notes", w))])
        if self.session_notes:
            for i, n in enumerate(self.session_notes, 1):
                for wl in _wrap(f"  {i}. {n}", w):
                    lines.append([(f"{TEXT}", wl)])
        else:
            lines.append([(f"{TEXT_MUTED}", "  (none)")])

        lines.append([("", "")])
        lines.append([(f"bg:{ACCENT} #ffffff bold", _pad("Task Notes", w))])
        if self.task_notes:
            for i, n in enumerate(self.task_notes, 1):
                for wl in _wrap(f"  {i}. {n}", w):
                    lines.append([(f"{TEXT}", wl)])
        else:
            lines.append([(f"{TEXT_MUTED}", "  (none)")])
        return lines

    def _render_history(self, w: int) -> list[list[tuple[str, str]]]:
        lines: list[list[tuple[str, str]]] = []
        if not self.history:
            lines.append([(f"{TEXT_MUTED}", "  No steps completed yet.")])
            return lines

        for record in self.history:
            # Header
            lines.append([(f"{TEXT} bold",
                           f" Step {record.step_index}: {record.summary[:w - 12]}")])
            if record.changes_made:
                for wl in _wrap(f"   Changes: {record.changes_made}", w):
                    lines.append([(f"{TEXT}", wl)])
            if record.discovered_context:
                for wl in _wrap(f"   Context: {record.discovered_context}", w):
                    lines.append([(f"{TEXT}", wl)])
            if record.pending_items:
                for wl in _wrap(f"   Pending: {record.pending_items}", w):
                    lines.append([(f"{TEXT}", wl)])
            if record.anti_patterns:
                for wl in _wrap(f"   Avoid: {record.anti_patterns}", w):
                    lines.append([(f"{TEXT_MUTED}", wl)])
            tc = record.tool_calls_count
            lines.append([(f"{TEXT_MUTED}", f"   ({tc} tool call{'s' if tc != 1 else ''})")])
            lines.append([("", "")])
        return lines

    def _render_plan(self, w: int) -> list[list[tuple[str, str]]]:
        lines: list[list[tuple[str, str]]] = []
        if not self.plan_text:
            lines.append([(f"{TEXT_MUTED}", "  No plan yet.")])
            return lines
        for raw_line in self.plan_text.split("\n"):
            for wl in _wrap(f" {raw_line}", w):
                lines.append([(f"{TEXT}", wl)])
        return lines

    def _render_behavior(self, w: int) -> list[list[tuple[str, str]]]:
        import time as _time

        lines: list[list[tuple[str, str]]] = []
        lines.append([(f"bg:{PRIMARY} #ffffff bold", _pad("Behavior Scores", w))])
        if not self.behavior_scores:
            lines.append([(f"{TEXT_MUTED}", "  (no checkers have run yet)")])
        else:
            for agent_id, score in self.behavior_scores:
                color = SUCCESS if score > 0 else (ERROR if score < 0 else TEXT_MUTED)
                sign = "+" if score > 0 else ""
                lines.append([
                    (f"{TEXT}", f"  {agent_id}: "),
                    (f"{color} bold", f"{sign}{score}"),
                ])

        lines.append([("", "")])
        lines.append([(f"bg:{ACCENT} #ffffff bold", _pad("Verdict History (newest last)", w))])

        if not self.behavior_events:
            lines.append([(f"{TEXT_MUTED}", "  (no verdicts yet)")])
            lines.append([(f"{TEXT_MUTED}", "  Enable in /settings → Behavior Checkers.")])
            return lines

        now = _time.time()
        for ev in self.behavior_events:
            ago = max(0, int(now - ev.timestamp))
            if ago < 60:
                ago_str = f"{ago}s ago"
            elif ago < 3600:
                ago_str = f"{ago // 60}m ago"
            else:
                ago_str = f"{ago // 3600}h ago"

            sign = "+" if ev.delta > 0 else ""
            color = SUCCESS if ev.delta > 0 else ERROR
            score_color = SUCCESS if ev.score_after > 0 else (
                ERROR if ev.score_after < 0 else TEXT_MUTED
            )
            score_sign = "+" if ev.score_after > 0 else ""

            # Header line: [delta] checker — agent (ago) → score
            header = [
                (f"{color} bold", f"  {sign}{ev.delta:>2}"),
                (f"{TEXT} bold", f"  {ev.checker}"),
                (f"{TEXT_MUTED}", f"  ({ev.agent_id}, {ago_str})  →  total "),
                (f"{score_color} bold", f"{score_sign}{ev.score_after}"),
            ]
            lines.append(header)
            if ev.reason:
                for wl in _wrap(f"      {ev.reason}", w):
                    lines.append([(f"{TEXT_MUTED}", wl)])
            lines.append([("", "")])
        return lines

    def _render_state(self, w: int) -> list[list[tuple[str, str]]]:
        lines: list[list[tuple[str, str]]] = []
        info = self.state_info
        if not info:
            lines.append([(f"{TEXT_MUTED}", "  No engine state available.")])
            return lines

        lines.append([(f"bg:{PRIMARY} #ffffff bold", _pad("Loop State", w))])
        for key, val in info.items():
            label = key.replace("_", " ").title()
            for wl in _wrap(f"  {label}: {val}", w):
                lines.append([(f"{TEXT}", wl)])
        return lines


# ── Sections list control (left panel) ────────────────────────────────

class DebugSectionsControl(UIControl):
    """Left panel: clickable section tabs."""

    def __init__(self, state: DebugPanelState) -> None:
        self._state = state

    def is_focusable(self) -> bool:
        return True

    def mouse_handler(self, mouse_event) -> None:
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            row = mouse_event.position.y
            if 0 <= row < len(SECTIONS):
                self._state.section_cursor = row
                self._state.scroll = 0

    def get_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        s = self._state

        @kb.add("up")
        @kb.add("k")
        def _up(event):
            s.move_section(-1)

        @kb.add("down")
        @kb.add("j")
        def _down(event):
            s.move_section(1)

        @kb.add("enter")
        @kb.add("right")
        @kb.add("tab")
        def _enter(event):
            s.focus = "content"

        return kb

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        lines = []
        for i, section in enumerate(SECTIONS):
            active = self._state.focus == "sections"
            if i == self._state.section_cursor:
                style = f"bg:{PRIMARY} #ffffff bold" if active else f"bg:{SURFACE_LIGHT} {TEXT} bold"
            else:
                style = f"{TEXT}"
            pad = " " * max(0, width - len(section) - 2)
            lines.append([(style, f" {section}{pad} ")])

        def get_line(i):
            return lines[i] if 0 <= i < len(lines) else []
        return UIContent(get_line=get_line, line_count=len(lines))


# ── Content control (right panel) ─────────────────────────────────────

class DebugContentControl(UIControl):
    """Right panel: scrollable content for the selected section."""

    def __init__(self, state: DebugPanelState) -> None:
        self._state = state
        self._last_line_count: int = 0

    def is_focusable(self) -> bool:
        return True

    def get_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        s = self._state

        @kb.add("up")
        @kb.add("k")
        def _up(event):
            s.scroll_up()

        @kb.add("down")
        @kb.add("j")
        def _down(event):
            s.scroll_down(self._last_line_count)

        @kb.add("left")
        @kb.add("s-tab")
        def _back(event):
            s.focus = "sections"

        return kb

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        lines = self._state.content_lines(max(width - 1, 20))
        self._last_line_count = len(lines)

        scroll = min(self._state.scroll, max(0, len(lines) - 1))

        def get_line(i):
            return lines[i] if 0 <= i < len(lines) else []

        return UIContent(
            get_line=get_line,
            line_count=len(lines),
            cursor_position=Point(x=0, y=scroll),
        )


# ── Helpers ───────────────────────────────────────────────────────────

def _pad(text: str, width: int) -> str:
    """Pad text to fill width with leading space."""
    return f" {text}" + " " * max(0, width - len(text) - 1)


def _wrap(text: str, width: int) -> list[str]:
    """Word-wrap a string into lines that fit *width*."""
    if width <= 0:
        width = 80
    if len(text) <= width:
        return [text]
    result: list[str] = []
    while text:
        if len(text) <= width:
            result.append(text)
            break
        cut = text.rfind(" ", 0, width)
        if cut <= 0:
            cut = width
        result.append(text[:cut])
        text = text[cut:].lstrip(" ")
        if text:
            text = "    " + text
    return result
