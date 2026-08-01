"""Tests for the compact, collapsible tool-group rendering (claude-code style).

Consecutive tool calls coalesce into ONE group that is COLLAPSED by default
to a single summary line ("Ran N tools ▸"). Clicking the summary expands to
one compact line per tool; clicking a tool line expands its full detail
(the original ToolCallWidget block). While the agent is still working the
last group reads "Running N tools…".
"""

from __future__ import annotations

from infinidev.ui.controls.chat_history import ChatHistoryControl
from infinidev.ui.controls.tool_call_widget import build_tool_group


def _tool(name, text, *, result="", error="", running=False):
    return {"type": "tool_call", "tool_name": name, "text": text,
            "args": {}, "result": result, "error": error, "running": running}


def _flat(lines) -> str:
    return "\n".join("".join(t for _, t in ln) for ln in lines)


def _render(c: ChatHistoryControl, width: int = 80) -> str:
    # Bypass the streaming throttle exactly like the live mouse_handler does.
    c._line_cache = None
    c._last_rebuild = 0.0
    lines, _, _ = c._build_lines(width)
    return _flat(lines)


# ── Default collapsed summary ───────────────────────────────────────────


def test_tool_group_collapsed_by_default_hides_payloads():
    msgs = [_tool("read_file", "read_file src/main.py", result="line1\nline2"),
            _tool("code_search", "code_search TODO", result="match!"),
            _tool("glob", "glob *.py", result="a.py")]
    c = ChatHistoryControl(list(msgs))
    out = _render(c)
    # Single compact summary — count + collapsed arrow.
    assert "3 tools" in out
    assert "▸" in out
    # Individual tool payloads must NOT be visible while collapsed.
    assert "read_file" not in out
    assert "line1" not in out
    assert "match!" not in out


def test_single_tool_uses_singular_unit():
    c = ChatHistoryControl([_tool("read_file", "read_file a.py")])
    out = _render(c)
    assert "1 tool" in out
    assert "1 tools" not in out


# ── Expand group → one-liners ───────────────────────────────────────────


def test_expand_group_reveals_one_liners_not_full_payload():
    msgs = [_tool("read_file", "read_file src/main.py", result="secret body"),
            _tool("code_search", "code_search TODO")]
    c = ChatHistoryControl(list(msgs))
    _render(c)
    # Summary is clickable at offset 0 → toggles collapse.
    c._clickable_lines[0]()
    out = _render(c)
    # Now the one-liner labels are visible…
    assert "read_file src/main.py" in out
    assert "code_search TODO" in out
    # …but still NOT the full result body (that needs a second click).
    assert "secret body" not in out
    # Expanded arrow on the summary.
    assert "▾" in out


def test_expand_individual_tool_reveals_full_detail():
    msgs = [_tool("code_search", "code_search TODO",
                  result='{"pattern":"TODO","match_count":2,"truncated":false,"matches":[]}')]
    c = ChatHistoryControl(list(msgs))
    _render(c)
    c._clickable_lines[0]()          # expand the group
    _render(c)
    # Offset 1 is the single tool one-liner → expand its detail.
    c._clickable_lines[1]()
    out = _render(c)
    # The full ToolCallWidget block (accent bar + args) is now embedded.
    assert "▌" in out
    assert "match_count" in out or "matches" in out


# ── Live vs done summary ────────────────────────────────────────────────


def test_live_group_reads_running_for_active_tool():
    c = ChatHistoryControl([_tool("read_file", "read_file a.py", running=True)])
    c.busy = True
    out = _render(c)
    assert "Running" in out
    assert "…" in out


def test_done_group_reads_ran_when_idle():
    c = ChatHistoryControl([_tool("read_file", "read_file a.py")])
    c.busy = False
    out = _render(c)
    assert "Ran" in out
    assert "Running" not in out


def test_busy_agent_does_not_keep_completed_tool_running():
    c = ChatHistoryControl([_tool("read_file", "read_file a.py", running=False)])
    c.busy = True
    out = _render(c)
    assert "Ran" in out
    assert "Running" not in out


def test_busy_toggle_busts_cache():
    # Turning busy off must drop the cache so the summary flips Running→Ran
    # even though msg_count/width are unchanged.
    c = ChatHistoryControl([_tool("read_file", "read_file a.py")])
    c.busy = True
    _render(c)
    assert c._line_cache is not None
    c.busy = False
    assert c._line_cache is None      # setter dropped it
    assert c._last_rebuild == 0.0     # …and reset the throttle clock


def test_running_flips_to_ran_immediately_within_throttle_window():
    # Regression: nulling _line_cache alone does NOT bypass the streaming
    # throttle (which reuses _last_lines within _REBUILD_MIN_INTERVAL), so
    # the summary would stay on "Running…" for up to ~180ms. The busy setter
    # must also reset _last_rebuild so the flip lands on the very next frame.
    import time
    message = _tool("read_file", "read_file a.py", running=True)
    c = ChatHistoryControl([message])
    c.busy = True
    lines, _, _ = c._build_lines(80)          # establishes _last_lines + a
    assert "Running" in _flat(lines)          # recent _last_rebuild
    # Simulate the turn ending an instant later (still inside the throttle
    # window) — do NOT touch _line_cache/_last_rebuild manually.
    c._last_rebuild = time.monotonic()
    message["running"] = False
    c.busy = False
    lines, _, _ = c._build_lines(80)
    flat = _flat(lines)
    assert "Ran" in flat
    assert "Running" not in flat


# ── Error accounting ────────────────────────────────────────────────────


def test_summary_counts_failures():
    msgs = [_tool("read_file", "read_file a.py"),
            _tool("write_file", "write_file /etc/x", error="PermissionError")]
    c = ChatHistoryControl(list(msgs))
    out = _render(c)
    assert "1 failed" in out
    assert "✗" in out


def test_shell_nonzero_exit_counts_as_failure():
    msgs = [_tool("execute_command", "execute_command false",
                  result='{"exit_code": 1, "stdout": "", "stderr": "boom"}')]
    c = ChatHistoryControl(list(msgs))
    out = _render(c)
    assert "✗" in out
    assert "1 failed" in out


# ── build_tool_group wiring contract ────────────────────────────────────


def test_build_tool_group_wires_callbacks():
    calls = {"group": 0, "tool": []}
    msgs = [_tool("read_file", "read_file a.py"), _tool("glob", "glob *.py")]
    rr = build_tool_group(
        msgs, collapsed=False, expanded_set=set(), width=80, live=False,
        on_toggle_group=lambda: calls.__setitem__("group", calls["group"] + 1),
        on_toggle_tool=lambda i: calls["tool"].append(i),
    )
    # Offset 0 = group toggle; offsets 1,2 = per-tool toggles.
    rr.clickable_offsets[0]()
    assert calls["group"] == 1
    rr.clickable_offsets[1]()
    rr.clickable_offsets[2]()
    assert calls["tool"] == [0, 1]
