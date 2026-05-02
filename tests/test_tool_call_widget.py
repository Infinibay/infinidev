"""Tests for ToolCallWidget — every tool renders its full payload."""

from __future__ import annotations

import json

import pytest

from infinidev.ui.controls.tool_call_widget import ToolCallWidget
from infinidev.ui.controls.message_groups import NEVER_GROUP_TYPES, identify_groups


@pytest.fixture()
def widget():
    return ToolCallWidget()


def _flat(rendered) -> str:
    return "\n".join("".join(t for _, t in line) for line in rendered.lines)


def test_widget_registered_under_tool_call_type():
    from infinidev.ui.controls.message_widgets import get_widget
    assert get_widget("tool_call").__class__.__name__ == "ToolCallWidget"


# ── Per-tool formatter coverage ─────────────────────────────────────────


def test_read_file_shows_only_path_no_content(widget):
    """read_file is args-only — content would clutter the chat."""
    msg = {
        "tool_name": "read_file",
        "args": {"path": "src/main.py"},
        "result": "line1\nline2\nline3",
        "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert "read_file" in out
    assert "src/main.py" in out
    # Body must NOT leak into the rendered block.
    assert "line1" not in out
    assert "line2" not in out
    assert "line3" not in out


def test_list_directory_shows_only_args_not_listing(widget):
    """list_directory is args-only — the listing would clutter the chat."""
    msg = {
        "tool_name": "list_directory",
        "args": {"path": "src/"},
        "result": "src/\n├── main.py\n├── config/\n└── ui/",
        "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert "list_directory" in out
    assert "src/" in out
    # Listing entries must NOT appear in the block.
    assert "main.py" not in out
    assert "├──" not in out


def test_code_search_shows_query_and_match_count(widget):
    msg = {
        "tool_name": "code_search",
        "args": {"query": "TODO", "file_path": "src"},
        "result": '{"pattern": "TODO", "match_count": 2, "truncated": false, "matches": [{"file": "src/foo.py", "line": 12, "content": "# TODO"}, {"file": "src/bar.py", "line": 8, "content": "# TODO"}]}',
        "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert "TODO" in out
    assert "matches" in out and "2" in out
    # The actual matches must NOT appear in the chat — they're noise.
    assert "src/foo.py" not in out
    assert "src/bar.py" not in out


def test_code_search_marks_truncated(widget):
    msg = {
        "tool_name": "code_search",
        "args": {"query": "foo"},
        "result": '{"pattern": "foo", "match_count": 12, "truncated": true, "matches": []}',
        "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert "12" in out
    assert "truncated" in out


def test_glob_shows_pattern_and_paths(widget):
    msg = {
        "tool_name": "glob",
        "args": {"pattern": "*.py"},
        "result": "main.py\nfoo.py\nbar.py",
        "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert "*.py" in out
    assert "paths" in out and "3" in out


def test_replace_lines_shows_path_range_and_new_code(widget):
    msg = {
        "tool_name": "replace_lines",
        "args": {
            "path": "src/foo.py",
            "line_start": 12,
            "line_end": 20,
            "new_lines": "def f():\n    return 42",
        },
        "result": "",
        "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert "src/foo.py:12-20" in out
    assert "def f():" in out
    assert "return 42" in out


def test_code_interpreter_shows_language_code_and_output(widget):
    result = json.dumps({
        "exit_code": 0,
        "stdout": "hello\nworld\n",
        "stderr": "",
    })
    msg = {
        "tool_name": "code_interpreter",
        "args": {"code": "print('hello')\nprint('world')", "language": "python"},
        "result": result,
        "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert "code_interpreter" in out
    assert "language" in out
    assert "python" in out
    assert "print('hello')" in out
    assert "✓ exit 0" in out
    assert "hello" in out
    assert "world" in out


def test_execute_command_shows_command_and_output(widget):
    result = json.dumps({"exit_code": 0, "stdout": "total 4\n", "stderr": ""})
    msg = {
        "tool_name": "execute_command",
        "args": {"command": "ls -la /tmp"},
        "result": result,
        "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert "execute_command" in out
    assert "ls -la /tmp" in out
    assert "✓ exit 0" in out
    assert "total 4" in out


def test_execute_command_failure_shows_exit_code_and_stderr(widget):
    result = json.dumps({"exit_code": 1, "stdout": "", "stderr": "Permission denied"})
    msg = {
        "tool_name": "execute_command",
        "args": {"command": "rm /etc/passwd"},
        "result": result,
        "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert "✗ exit 1" in out
    assert "Permission denied" in out


def test_git_commit_renders_message_and_result(widget):
    msg = {
        "tool_name": "git_commit",
        "args": {"message": "fix: race condition"},
        "result": "[main abc123] fix: race condition",
        "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert "fix: race condition" in out
    assert "abc123" in out


def test_unknown_tool_falls_back_to_default_kv_dump(widget):
    msg = {
        "tool_name": "totally_made_up_tool",
        "args": {"foo": "bar", "baz": 42},
        "result": "ok",
        "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert "totally_made_up_tool" in out
    assert "foo" in out
    assert "bar" in out
    assert "baz" in out
    assert "42" in out
    assert "ok" in out


def test_error_renders_red_marker_and_message(widget):
    msg = {
        "tool_name": "read_file",
        "args": {"path": "/etc/shadow"},
        "result": "",
        "error": "PermissionError: read access denied",
    }
    out = _flat(widget.render(msg, 80))
    assert "read_file" in out
    assert "✗" in out
    assert "PermissionError" in out


def test_args_are_optional_widget_does_not_crash(widget):
    msg = {"tool_name": "weird", "args": None, "result": "", "error": ""}
    rr = widget.render(msg, 80)
    assert any("weird" in "".join(t for _, t in line) for line in rr.lines)


def test_widget_handles_non_dict_args(widget):
    msg = {"tool_name": "weird", "args": "not a dict", "result": "ok", "error": ""}
    rr = widget.render(msg, 80)
    out = _flat(rr)
    assert "weird" in out


# ── Grouping rules ──────────────────────────────────────────────────────


def test_tool_call_messages_are_never_grouped():
    assert "tool_call" in NEVER_GROUP_TYPES
    msgs = [
        {"type": "tool_call", "tool_name": "read_file", "args": {}, "result": "", "error": ""},
        {"type": "tool_call", "tool_name": "code_search", "args": {}, "result": "", "error": ""},
        {"type": "tool_call", "tool_name": "glob", "args": {}, "result": "", "error": ""},
    ]
    groups = identify_groups(msgs)
    # Three tool calls → three singleton groups, NOT one group of 3.
    assert len(groups) == 3
    for g in groups:
        assert len(g.messages) == 1
        assert g.is_group is False


def test_diff_exec_error_think_are_singleton_groups():
    msgs = [
        {"type": "diff", "text": "x"},
        {"type": "diff", "text": "y"},
        {"type": "exec", "cmd": "ls"},
        {"type": "exec", "cmd": "pwd"},
        {"type": "think", "text": "..."},
        {"type": "think", "text": "..."},
        {"type": "error", "text": "boom"},
    ]
    groups = identify_groups(msgs)
    assert len(groups) == len(msgs)


# ── JSON-aware result rendering ────────────────────────────────────────


def test_list_directory_args_only_skips_json_listing(widget):
    """Even with full JSON entries, list_directory shows only path/recursive/pattern."""
    result = json.dumps({
        "file_path": "src/",
        "entries": [
            {"file_path": "main.py", "size": 4123, "type": "file"},
            {"file_path": "config", "type": "dir"},
        ],
        "total": 2, "truncated": False,
    })
    msg = {
        "tool_name": "list_directory",
        "args": {"path": "src/", "recursive": True, "pattern": "*.py"},
        "result": result, "error": "",
    }
    out = _flat(widget.render(msg, 80))
    # Args appear
    assert "src/" in out
    assert "recursive" in out
    assert "*.py" in out
    # Listing data does NOT appear
    assert "main.py" not in out
    assert "config" not in out
    assert "4.0 KB" not in out
    assert "entries" not in out


def test_glob_renders_matches_array_not_raw_json(widget):
    result = json.dumps({
        "pattern": "*.py", "file_path": ".", "content_pattern": None,
        "match_count": 3, "truncated": False,
        "matches": ["src/main.py", "src/foo.py", "tests/test_x.py"],
    })
    msg = {
        "tool_name": "glob",
        "args": {"pattern": "*.py"},
        "result": result, "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert '"matches"' not in out
    assert "src/main.py" in out
    assert "src/foo.py" in out
    assert "tests/test_x.py" in out
    assert "matches" in out and "3" in out


def test_smart_result_pretty_prints_json_dict_for_search_tools(widget):
    """Search tools (which keep their result) pretty-print JSON dicts."""
    result = json.dumps({
        "status": "ok",
        "match_count": 42,
        "title": "Some hit",
    })
    msg = {
        "tool_name": "search_findings",
        "args": {"query": "bug"},
        "result": result, "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert '"match_count"' not in out  # raw JSON should not appear
    assert "ok" in out
    assert "42" in out


def test_smart_result_inlines_list_of_dicts_compactly(widget):
    """A list of dicts should render as `key=val  key=val` rows, not a JSON block."""
    result = json.dumps({
        "findings": [
            {"id": 1, "title": "X"},
            {"id": 2, "title": "Y"},
        ],
        "count": 2,
    })
    msg = {
        "tool_name": "search_findings",
        "args": {"query": "auth"},
        "result": result, "error": "",
    }
    out = _flat(widget.render(msg, 80))
    assert "id=1" in out and "title=X" in out
    assert "id=2" in out and "title=Y" in out
    # And NOT a JSON dump
    assert '"findings": [' not in out


def test_format_size_helper():
    from infinidev.ui.controls.tool_call_widget import _format_size
    assert _format_size(0) == "0 B"
    assert _format_size(842) == "842 B"
    assert _format_size(4123) == "4.0 KB"
    assert _format_size(1234567) == "1.2 MB"
    assert _format_size(None) == ""


# ── Copy button ────────────────────────────────────────────────────────


def test_header_has_copy_button_clickable_at_offset_0(widget):
    msg = {
        "tool_name": "read_file",
        "args": {"path": "src/main.py"},
        "result": "x",
        "error": "",
    }
    rr = widget.render(msg, 80)
    # The widget exposes the copy callback at line 0 (the header).
    assert 0 in rr.clickable_offsets
    assert callable(rr.clickable_offsets[0])
    # Header line includes the copy icon.
    header = "".join(t for _, t in rr.lines[0])
    assert "⧉" in header or "Copied" in header or "Failed" in header


def test_copy_button_invokes_clipboard_with_serialized_text(widget, monkeypatch):
    captured = {}

    def fake_copy(text: str) -> bool:
        captured["text"] = text
        return True

    # Stub the clipboard module so the test is hermetic.
    import infinidev.ui.clipboard as cb
    monkeypatch.setattr(cb, "copy_to_clipboard", fake_copy)

    msg = {
        "tool_name": "code_interpreter",
        "args": {"code": "print('hi')", "language": "python"},
        "result": json.dumps({"exit_code": 0, "stdout": "hi\n", "stderr": ""}),
        "error": "",
    }
    rr = widget.render(msg, 80)
    rr.clickable_offsets[0]()  # simulate click on header
    assert "code_interpreter" in captured["text"]
    assert "language: python" in captured["text"]
    assert "print('hi')" in captured["text"]
    assert "exit 0" in captured["text"]
    assert "hi" in captured["text"]


def test_serialize_for_copy_preserves_args_and_result():
    from infinidev.ui.controls.tool_call_widget import _serialize_for_copy
    msg = {
        "tool_name": "git_commit",
        "args": {"message": "fix: x"},
        "result": "[main abc123] fix: x",
        "error": "",
    }
    text = _serialize_for_copy(msg)
    assert "git_commit" in text
    assert "fix: x" in text
    assert "abc123" in text


def test_user_agent_messages_are_still_grouped():
    msgs = [
        {"type": "agent", "text": "hi"},
        {"type": "agent", "text": "again"},
        {"type": "agent", "text": "third"},
    ]
    groups = identify_groups(msgs)
    assert len(groups) == 1
    assert groups[0].is_group is True
    assert len(groups[0].messages) == 3
