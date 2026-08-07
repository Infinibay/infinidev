"""Chat history control — renders all messages as FormattedText.

This replaces the Textual ChatHistory(VerticalScroll) that mounted Static
widgets. Here, messages are plain dicts in a list. The UIControl only
generates FormattedText for the visible viewport, giving us natural
viewport culling without the ±200px visibility hack.
"""

from __future__ import annotations

import time
from typing import Any

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import UIControl, UIContent
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from infinidev.config.settings import settings
from infinidev.ui.theme import TEXT_DIM, TEXT_MUTED, PRIMARY


# Minimum interval between full line rebuilds (seconds)
_REBUILD_MIN_INTERVAL = 0.18  # ~5.5 rebuilds/sec max

# Working indicator. Braille frames read as continuous rotation at low
# frame rates, which is what the ~3 fps animation timer can deliver
# without burning CPU on repaints.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_FPS = 8


# Exact number of transcript lines moved by each mouse-wheel event.
_MOUSE_WHEEL_LINES = 5


class ChatHistoryControl(UIControl):
    """Custom UIControl that renders chat messages as formatted text lines.

    Messages are stored as a list of dicts:
        {"sender": str, "text": str, "type": str, "is_diff": bool, ...}
    """

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages
        self._line_cache: list[list[tuple[str, str]]] | None = None
        self._cache_len = 0
        self._cache_width = 0
        self._cache_show_thinking_messages = False
        self._show_thinking = False
        # Working-indicator state: when the current piece of work started
        # (for the elapsed counter) and a short label for what it is.
        self._work_started_at: float = 0.0
        self._work_label: str = ""
        self._follow_tail: bool = True  # stick to bottom
        self._scroll_offset: int = 0    # lines from bottom (when not following)
        self._line_count: int = 0
        # Generalized click targets: line index → callback
        self._clickable_lines: dict[int, Any] = {}
        # Blank rows create_content prepends to bottom-anchor a short
        # transcript. Clicks arrive in content coordinates, so this is what
        # separates them from the _clickable_lines keys.
        self._top_pad: int = 0
        # Group collapse state: start_index of group → collapsed bool
        self._group_states: dict[int, bool] = {}
        # Rebuild throttle
        self._last_rebuild: float = 0.0
        # Last *rendered* line list (message lines only — excludes the
        # thinking indicator). Survives invalidate_cache() so the throttle
        # can reuse it during streaming bursts and so the scroll-anchor
        # delta has a stable, thinking-independent reference.
        self._last_lines: list[list[tuple[str, str]]] | None = None
        # One-shot guard: a trailing rebuild has been scheduled for after
        # the current throttle window (see _schedule_trailing_rebuild).
        self._trailing_scheduled: bool = False
        # ── Compact tool-group state ──────────────────────────────────
        # A group reads "Running" only while one of its tool messages is
        # active; otherwise it reads "Ran". Collapse/expand state is keyed
        # by the group's start_index (default: collapsed).
        self._busy: bool = False
        self._tool_group_states: dict[int, bool] = {}   # start_index → collapsed
        self._tool_expanded: dict[int, set[int]] = {}   # start_index → expanded tool idxs

    @property
    def busy(self) -> bool:
        return self._busy

    @busy.setter
    def busy(self, value: bool) -> None:
        """Track whether the agent is working and invalidate its cached view."""
        if bool(value) != self._busy:
            self._busy = bool(value)
            self._line_cache = None
            # Bypass the throttle so lifecycle changes appear next frame.
            self._last_rebuild = 0.0

    def invalidate_cache(self) -> None:
        """Mark cache as stale.

        Preserves the user's scroll position. The actual rebuild is
        deferred to the next ``create_content()`` call and throttled
        to avoid excessive rebuilds during rapid event bursts.

        Previously this method also forced ``_follow_tail = True`` and
        ``_scroll_offset = 0`` on every new message, which yanked the
        viewport to the bottom each time a tool call arrived — making
        it impossible to read older content while the agent worked.
        Now scroll position is fully under user control: only the
        ``end`` / ``pagedown`` keys, or scrolling all the way down with
        the mouse wheel, re-engage tail-following.
        """
        self._line_cache = None

    def is_focusable(self) -> bool:
        return True

    def mouse_handler(self, mouse_event: MouseEvent):
        """Handle fixed-distance wheel scroll and clicks on registered lines.

        We consume SCROLL_UP/DOWN here rather than letting the hosting
        Window handle them: Window._scroll_up/_scroll_down only invoke the
        control's scroll hooks when the phantom cursor sits at a viewport
        edge, creating dead zones where notches do nothing. Handling the
        wheel directly applies one fixed step regardless of cursor position.
        """
        et = mouse_event.event_type
        if et == MouseEventType.SCROLL_UP:
            self.move_cursor_up()
            return None
        if et == MouseEventType.SCROLL_DOWN:
            self.move_cursor_down()
            return None
        if et == MouseEventType.MOUSE_UP:
            # prompt_toolkit hands us a row in the *content*, which includes
            # the bottom-anchor padding create_content prepends. The callback
            # table is keyed without it.
            line_idx = mouse_event.position.y - self._top_pad
            callback = self._clickable_lines.get(line_idx)
            if callback is not None:
                callback()
                # Rebuild on next frame — and bypass the streaming
                # throttle so a collapse/expand toggle clicked during a
                # burst renders immediately instead of reusing stale
                # _last_lines for up to _REBUILD_MIN_INTERVAL (C3 regression).
                self._line_cache = None
                self._last_rebuild = 0.0
                return None
        return NotImplemented

    def get_vertical_scroll(self, window) -> int:
        """Authoritative viewport scroll, derived from the line offset.

        Passed to the chat ``Window`` as ``get_vertical_scroll``. Without
        it, prompt_toolkit derives ``vertical_scroll`` from the phantom
        cursor via ``do_scroll``, which only moves once the cursor reaches a
        viewport edge, leaving the viewport static across several notches.

        By pinning ``vertical_scroll`` to ``cursor_y - height + 1`` (cursor
        at the bottom edge), every line of ``_scroll_offset`` maps 1:1 to a
        viewport line, and prompt_toolkit's subsequent ``do_scroll`` leaves
        the value untouched because the cursor is already exactly on the
        edge. On the very first render (no ``render_info`` yet) we return 0
        and let ``do_scroll`` position the tail — it self-corrects.
        """
        line_count = self._line_count
        info = getattr(window, "render_info", None)
        height = info.window_height if info is not None else 0
        if height <= 0 or line_count <= height:
            return 0
        max_scroll = line_count - height
        if self._follow_tail:
            return max_scroll
        cursor_y = max(0, line_count - 1 - self._scroll_offset)
        return max(0, min(max_scroll, cursor_y - height + 1))

    # Mouse wheel uses fixed-distance steps in both directions.
    # The wheel is consumed in mouse_handler() above to avoid Window cursor
    # dead zones. These hooks remain for paths that delegate to Window.

    def move_cursor_down(self) -> None:
        """Move toward the newest messages by one fixed wheel step."""
        self.page_down(_MOUSE_WHEEL_LINES)

    def move_cursor_up(self) -> None:
        """Move into older messages by one fixed wheel step."""
        self.page_up(_MOUSE_WHEEL_LINES)

    # ── Public scroll API (callable from global keybindings) ──────────
    # These mirror the per-control bindings below but can be invoked
    # regardless of which widget has focus — see
    # ``ui/keybindings.py:create_global_keybindings``.

    def page_up(self, lines: int = 15) -> None:
        self._follow_tail = False
        self._scroll_offset = min(
            self._scroll_offset + lines,
            max(0, self._line_count - 1),
        )

    def page_down(self, lines: int = 15) -> None:
        self._scroll_offset = max(0, self._scroll_offset - lines)
        if self._scroll_offset == 0:
            self._follow_tail = True

    def scroll_home(self) -> None:
        self._follow_tail = False
        self._scroll_offset = max(0, self._line_count - 1)

    def scroll_end(self) -> None:
        self._follow_tail = True
        self._scroll_offset = 0

    def get_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("pageup")
        def _pgup(event):
            self.page_up()

        @kb.add("pagedown")
        def _pgdn(event):
            self.page_down()

        @kb.add("home")
        def _home(event):
            self.scroll_home()

        @kb.add("end")
        def _end(event):
            self.scroll_end()

        return kb

    @property
    def show_thinking(self) -> bool:
        return self._show_thinking

    @show_thinking.setter
    def show_thinking(self, value: bool) -> None:
        if value != self._show_thinking:
            self._show_thinking = value
            # Reset the clock on every transition so the elapsed counter
            # measures this piece of work, not the session.
            self._work_started_at = time.monotonic() if value else 0.0
            if not value:
                self._work_label = ""

    @property
    def work_label(self) -> str:
        return self._work_label

    @work_label.setter
    def work_label(self, value: str) -> None:
        """What the agent is doing right now, shown beside the spinner."""
        normalized = (value or "").strip()
        if normalized != self._work_label:
            self._work_label = normalized
            if self._show_thinking:
                self._work_started_at = time.monotonic()

    def preferred_width(self, max_available_width: int) -> int | None:
        return None  # fill available

    def preferred_height(self, width: int, max_available_height: int,
                         wrap_lines: bool, get_line_prefix) -> int | None:
        return None  # fill available

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        """Build the displayable content from messages."""
        lines, line_count, custom_get_line = self._build_lines(width)
        if line_count == 0:
            lines = [[(f"{TEXT_MUTED}", " Type your instruction, or / for commands.")]]
            line_count = 1
            custom_get_line = None

        # Bottom-anchor a short conversation: pad above it so the newest
        # message always sits just over the composer. Without this the
        # first exchanges cling to the top of the terminal with a growing
        # void beneath them, which reads as a rendering glitch rather than
        # as a young conversation.
        # ``_clickable_lines`` is keyed by index into ``lines``; this padding
        # shifts every one of them down in the content the mouse actually
        # hits. Record the offset so ``mouse_handler`` can undo it — without
        # this, a click lands `pad` rows away from its callback and toggles
        # nothing, which reads as "the group opens sometimes": a transcript
        # taller than the terminal pads by zero and works, a short one does
        # not.
        self._top_pad = 0
        if height and line_count < height:
            pad = height - line_count
            self._top_pad = pad
            inner_get = custom_get_line
            inner_lines = lines

            def custom_get_line(i: int, _pad=pad, _get=inner_get, _lines=inner_lines):
                if i < _pad:
                    return []
                index = i - _pad
                if _get is not None:
                    return _get(index)
                return _lines[index] if 0 <= index < len(_lines) else []

            line_count += pad

        self._line_count = line_count

        if self._follow_tail:
            cursor_y = max(0, line_count - 1)
        else:
            cursor_y = max(0, line_count - 1 - self._scroll_offset)

        if custom_get_line:
            getter = custom_get_line
        else:
            def getter(i: int) -> list[tuple[str, str]]:
                if 0 <= i < len(lines):
                    return lines[i]
                return []

        return UIContent(
            get_line=getter,
            line_count=line_count,
            cursor_position=Point(x=0, y=cursor_y),
            show_cursor=False,
        )

    def _build_lines(self, width: int) -> list[list[tuple[str, str]]]:
        """Build flat line list with message grouping.

        Consecutive same-type messages are grouped. Groups of 2+ show a
        clickable header; when collapsed only the last message is visible.

        Optimizations:
        - Cache is keyed on (msg_count, width) — repaints without new
          messages reuse the cache.
        - Rebuilds are throttled to ~5/sec to keep the UI responsive
          when the engine emits events rapidly.
        - Message indices are computed from group.start_index instead of
          list.index() to avoid O(n²) lookups.
        """
        from infinidev.ui.controls.message_groups import identify_groups
        from infinidev.ui.controls.message_widgets import get_widget

        msg_count = len(self._messages)
        show_thinking_messages = bool(settings.UI_SHOW_THINKING_IN_CHAT)

        # ── Cache check ──────────────────────────────────────────────
        cache_valid = (
            self._line_cache is not None
            and self._cache_len == msg_count
            and self._cache_width == width
            and self._cache_show_thinking_messages == show_thinking_messages
        )

        if cache_valid:
            lines = self._line_cache
        else:
            # Throttle: skip the full rebuild if we did one very recently
            # and the message geometry is unchanged (same count + width) —
            # this is the streaming-burst case, where the last message's
            # text grows every frame. Always rebuild when a *new* message
            # arrives so it appears immediately.
            #
            # The reuse source is _last_lines, NOT _line_cache: the normal
            # streaming path calls invalidate_cache() each frame, which
            # nulls _line_cache, so gating on it left this throttle dead
            # and ran a full rebuild every frame. _last_lines survives
            # invalidation, so the throttle now actually engages.
            now = time.monotonic()
            throttled = (
                self._last_lines is not None
                and self._cache_len == msg_count
                and self._cache_width == width
                and self._cache_show_thinking_messages == show_thinking_messages
                and now - self._last_rebuild < _REBUILD_MIN_INTERVAL
            )
            if throttled:
                lines = self._last_lines
                # Reused lines are stale (mid-stream text). Guarantee a
                # real rebuild once the throttle window closes so the
                # final state always renders — text never stays frozen.
                self._schedule_trailing_rebuild()
            else:
                lines = self._do_rebuild(msg_count, width)
                self._last_rebuild = now
                # A real rebuild just produced fresh lines, so any pending
                # trailing-rebuild guard is moot. Clear it defensively so a
                # scheduled timer that never fired (app torn down mid-window)
                # can't wedge the flag True and permanently disable future
                # trailing rebuilds.
                self._trailing_scheduled = False
        # Append the working indicator if active. One animated line \u2014
        # spinner, what it is doing, and how long it has been at it \u2014 beats
        # two static lines of "thinking...", which give no signal about
        # whether anything is actually happening.
        if self._show_thinking:
            total = len(lines) + 1
            spinner = _SPINNER_FRAMES[
                int(time.monotonic() * _SPINNER_FPS) % len(_SPINNER_FRAMES)
            ]
            elapsed = ""
            if self._work_started_at:
                seconds = int(time.monotonic() - self._work_started_at)
                if seconds >= 1:
                    elapsed = f"  ({seconds}s \u00b7 esc to interrupt)"
            label = self._work_label or "Working"
            indicator = [
                (f"{PRIMARY}", f"  {spinner} "),
                (f"{TEXT_MUTED}", label),
                (f"{TEXT_DIM}", elapsed),
            ]

            def get_line_with_thinking(i: int) -> list[tuple[str, str]]:
                if i < len(lines):
                    return lines[i]
                if i == len(lines):
                    return indicator
                return []

            return lines, total, get_line_with_thinking

        return lines, len(lines), None

    def _do_rebuild(self, msg_count: int, width: int) -> list[list[tuple[str, str]]]:
        """Full line rebuild — separated from _build_lines for clarity."""
        from infinidev.ui.controls.message_groups import identify_groups
        from infinidev.ui.controls.message_widgets import get_widget

        # Snapshot the previous line count BEFORE we rebuild. After the
        # rebuild we'll bump _scroll_offset by the delta when the user
        # is scrolled up, so the visible content stays anchored at the
        # same position instead of drifting down with new entries.
        #
        # Use the previous *rebuild's* line count (message lines only) —
        # NOT self._line_count, which includes the 2 thinking-indicator
        # lines _build_lines appends. new_line_count below is message-only
        # too, so both sides of the delta stay consistent; otherwise the
        # anchor drifts ~2 lines per rebuild while "thinking" is active.
        prev_line_count = len(self._last_lines) if self._last_lines is not None else 0

        # After a /clear (or any shrink) the old collapse/expand state is
        # keyed by now-invalid indices — drop it so a fresh tool group at
        # index 0 doesn't inherit stale expanded/collapsed state.
        if msg_count < self._cache_len:
            self._tool_group_states.clear()
            self._tool_expanded.clear()
            self._group_states.clear()

        lines: list[list[tuple[str, str]]] = []
        self._clickable_lines = {}
        groups = identify_groups(self._messages)
        show_thinking_messages = bool(settings.UI_SHOW_THINKING_IN_CHAT)

        for group in groups:
            if group.msg_type == "think" and not show_thinking_messages:
                continue
            # Compact, collapsible tool groups (claude-code style) take a
            # dedicated render path instead of the generic header+messages.
            if group.msg_type == "tool_call":
                self._render_tool_group(group, width, lines)
                continue
            if group.msg_type == "critic":
                self._render_critic_group(group, width, lines)
                continue

            widget = get_widget(group.msg_type)
            if widget is None:
                for msg in group.messages:
                    lines.extend(self._render_fallback(msg, width))
                continue

            if group.is_group:
                # Default EXPANDED (False): consecutive agent/system/user
                # replies must all stay visible — collapsing by default hid
                # earlier messages behind a "Responses (N)" header and read
                # as "content vanished". The manual collapse toggle still
                # works; clicking the header flips this state.
                collapsed = self._group_states.get(group.start_index, False)

                header_result = widget.render_group_header(
                    len(group.messages), collapsed, width,
                )
                header_start = len(lines)
                lines.extend(header_result.lines)

                def _toggle_group(idx=group.start_index):
                    self._group_states[idx] = not self._group_states.get(idx, False)
                self._clickable_lines[header_start] = _toggle_group

                if collapsed:
                    visible_msgs = [group.messages[-1]]
                    # Index of last msg = start_index + len - 1
                    visible_indices = [group.start_index + len(group.messages) - 1]
                else:
                    visible_msgs = group.messages
                    visible_indices = [group.start_index + i for i in range(len(group.messages))]

                for msg, msg_idx in zip(visible_msgs, visible_indices):
                    result = widget.render(msg, width)
                    start = len(lines)
                    lines.extend(result.lines)
                    for offset, cb in result.clickable_offsets.items():
                        self._clickable_lines[start + offset] = cb
            else:
                msg = group.messages[0]
                msg_idx = group.start_index
                result = widget.render(msg, width)
                start = len(lines)
                lines.extend(result.lines)
                for offset, cb in result.clickable_offsets.items():
                    self._clickable_lines[start + offset] = cb
        # Anchor compensation: if the user is scrolled up (tail-follow
        # disabled), bump _scroll_offset by the number of newly-rendered
        # lines so cursor_y (computed as line_count-1-offset) stays
        # the same — they keep seeing the same content, while new
        # entries pile up below the viewport.
        new_line_count = len(lines)
        if not self._follow_tail and prev_line_count > 0:
            delta = new_line_count - prev_line_count
            if delta > 0:
                self._scroll_offset = min(
                    self._scroll_offset + delta,
                    max(0, new_line_count - 1),
                )

        self._line_cache = lines
        self._last_lines = lines
        self._cache_len = msg_count
        self._cache_width = width
        self._cache_show_thinking_messages = show_thinking_messages
        return lines

    def _render_tool_group(self, group, width: int,
                           lines: list[list[tuple[str, str]]]) -> None:
        """Render a run of tool calls as one compact, collapsible group.

        Appends the rendered lines to ``lines`` and registers this group's
        clickable offsets (collapse toggle + per-tool detail toggles).
        """
        from infinidev.ui.controls.tool_call_widget import build_tool_group

        idx = group.start_index
        running = any(bool(message.get("running")) for message in group.messages)
        collapsed = self._tool_group_states.get(idx, not running)
        expanded_set = self._tool_expanded.get(idx, set())
        live = running

        def _toggle_group(_idx=idx, _default=not running):
            self._tool_group_states[_idx] = not self._tool_group_states.get(
                _idx, _default,
            )

        def _toggle_tool(local_i: int, _idx=idx):
            s = self._tool_expanded.setdefault(_idx, set())
            if local_i in s:
                s.discard(local_i)
            else:
                s.add(local_i)

        rr = build_tool_group(
            group.messages,
            collapsed=collapsed,
            expanded_set=expanded_set,
            width=width,
            live=live,
            on_toggle_group=_toggle_group,
            on_toggle_tool=_toggle_tool,
        )
        start = len(lines)
        lines.extend(rr.lines)
        for offset, cb in rr.clickable_offsets.items():
            self._clickable_lines[start + offset] = cb

    def _render_critic_group(self, group, width: int,
                             lines: list[list[tuple[str, str]]]) -> None:
        """Render a run of critic verdicts as one compact, collapsible group.

        Shares ``_tool_group_states`` / ``_tool_expanded`` with the tool
        groups: both are keyed by ``group.start_index`` into the same
        message list, so the keys cannot collide, and one collapse-state
        store means one place to clear when the transcript resets.
        """
        from infinidev.ui.controls.critic_widget import build_critic_group

        idx = group.start_index
        collapsed = self._tool_group_states.get(idx, True)   # default collapsed
        expanded_set = self._tool_expanded.get(idx, set())

        def _toggle_group(_idx=idx):
            self._tool_group_states[_idx] = not self._tool_group_states.get(_idx, True)

        def _toggle_item(local_i: int, _idx=idx):
            s = self._tool_expanded.setdefault(_idx, set())
            if local_i in s:
                s.discard(local_i)
            else:
                s.add(local_i)

        rr = build_critic_group(
            group.messages,
            collapsed=collapsed,
            expanded_set=expanded_set,
            width=width,
            on_toggle_group=_toggle_group,
            on_toggle_item=_toggle_item,
        )
        start = len(lines)
        lines.extend(rr.lines)
        for offset, cb in rr.clickable_offsets.items():
            self._clickable_lines[start + offset] = cb

    def _schedule_trailing_rebuild(self) -> None:
        """Ensure a real rebuild lands once the throttle window closes.

        When ``_build_lines`` reuses stale ``_last_lines`` during a burst,
        the most recent change (e.g. the final streamed chunk, or the
        markdown re-render on stream end) would never appear if no further
        redraw is triggered. Schedule a one-shot invalidate just past the
        throttle interval: ``create_content`` runs again with the interval
        elapsed, so the throttle no longer applies and a fresh rebuild
        renders the final state.

        Idempotent within a window via ``_trailing_scheduled``. Wrapped in
        try/except so it is a no-op when no application is running (tests),
        where ``get_app()`` returns a loop-less DummyApplication.
        """
        if self._trailing_scheduled:
            return
        try:
            from prompt_toolkit.application import get_app
            app = get_app()
            loop = app.loop
        except Exception:
            return
        if loop is None:
            return

        def _fire() -> None:
            self._trailing_scheduled = False
            try:
                app.invalidate()
            except Exception:
                pass

        try:
            loop.call_later(_REBUILD_MIN_INTERVAL, _fire)
            self._trailing_scheduled = True
        except Exception:
            self._trailing_scheduled = False

    def _render_fallback(self, msg: dict, width: int) -> list[list[tuple[str, str]]]:
        """Minimal fallback for unknown message types."""
        if not msg.get("visible", True):
            return []
        text = msg.get("text", "")
        sender = msg.get("sender", "")
        lines = [
            [(f"{TEXT_MUTED}", f"  {sender}: {text[:width - 4]}")],
            [("", "")],
        ]
        return lines


def format_tool_chat_message(tool_name: str, detail: str,
                             error: str, output: str) -> str:
    """Format a tool call as a chat message string. Returns empty to skip."""
    # Note: execute_command is rendered by ExecCommandWidget — see event_handler.
    if tool_name == "create_file":
        path = detail or "?"
        if error:
            return f"create {path}\n  x {error}"
        return f"+ created {path}"

    if tool_name in ("replace_lines", "edit_symbol", "add_symbol", "remove_symbol"):
        if error:
            path = detail or "?"
            label = {"replace_lines": "edit", "edit_symbol": "edit",
                     "add_symbol": "add", "remove_symbol": "remove"}.get(tool_name, "edit")
            return f"{label} {path}\n  x {error}"
        return ""

    if tool_name == "git_commit":
        return f"commit {detail}" if detail else "commit"

    if tool_name == "git_branch":
        return f"branch {detail}" if detail else "branch"

    return ""
