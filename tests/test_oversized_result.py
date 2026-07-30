"""Oversized tool results: refuse a file read, trim what cannot paginate.

Written against one observed livelock. A 22 KB HANDOFF_PROMPT.md was read 21
times by the chat agent because ``...[truncated]`` said content was missing
and never said how to get it. In that project's history, files over the cap
averaged 14.5 reads each and files under it 1.5.
"""

from __future__ import annotations

import json

from infinidev.engine.oversized_result import (
    OVERSIZED_ERROR,
    DuplicateCallGuard,
    handle_oversized_result,
    is_oversized_refusal,
)


def _numbered(lines: list[str]) -> str:
    """Shape text the way read_file does: right-aligned number, tab, content."""
    return "\n".join(f"{i:6d}\t{t}" for i, t in enumerate(lines, 1))


def _markdown(n_sections: int, filler: int = 40) -> str:
    out: list[str] = []
    for s in range(n_sections):
        out.append(f"## Section {s}")
        out.extend(["body text " * 6] * filler)
    return _numbered(out)


class TestUnderTheLimit:
    def test_a_short_result_is_returned_verbatim(self):
        text = _numbered(["a", "b", "c"])
        assert handle_oversized_result(
            text, max_chars=8000, tool_name="read_file",
            tool_args='{"file_path": "a.md"}',
        ) == text

    def test_a_result_exactly_at_the_limit_is_untouched(self):
        text = "a" * 500
        assert handle_oversized_result(text, max_chars=500) == text


class TestFileReadIsRefusedNotTruncated:
    def _refuse(self, path="a.md", **kw):
        return handle_oversized_result(
            _markdown(6), max_chars=1000, tool_name="read_file",
            tool_args=json.dumps({"file_path": path}), **kw,
        )

    def test_no_file_content_reaches_the_model(self):
        """The point of refusing: no misleading partial file in the transcript."""
        assert "body text body text" not in self._refuse()

    def test_the_refusal_says_plainly_that_nothing_was_read(self):
        assert "NOTHING was read" in self._refuse()

    def test_the_refusal_carries_the_size_that_caused_it(self):
        meta = json.loads(self._refuse().splitlines()[0])
        assert meta["error"] == OVERSIZED_ERROR
        assert meta["lines"] > 0
        assert meta["characters"] > meta["limit_characters"]

    def test_the_refusal_names_the_call_that_reads_a_range(self):
        out = self._refuse()
        assert "read_file(file_path='a.md', offset=<first line>, limit=" in out

    def test_partial_read_gets_its_own_argument_name(self):
        out = handle_oversized_result(
            _markdown(6), max_chars=1000, tool_name="partial_read",
            tool_args='{"file_path": "a.md"}',
        )
        assert "start_line=<first line>" in out

    def test_the_refusal_is_far_smaller_than_the_truncation_it_replaced(self):
        assert len(self._refuse()) < 1000


class TestTheOutline:
    def test_markdown_headings_are_listed_with_line_numbers(self):
        out = handle_oversized_result(
            _markdown(5), max_chars=1000, tool_name="read_file",
            tool_args='{"file_path": "notes.md"}',
        )
        assert "## Section 0" in out and "## Section 4" in out
        assert "The file's structure" in out

    def test_python_definitions_are_listed(self):
        src = _numbered(
            ["import os", "", "", "class Parser:", "    def parse(self):",
             *["        pass"] * 300, "", "def helper():", "    return 1"]
        )
        out = handle_oversized_result(
            src, max_chars=500, tool_name="read_file",
            tool_args='{"file_path": "p.py"}',
        )
        assert "class Parser:" in out
        assert "def helper():" in out

    def test_a_deeply_nested_definition_does_not_crowd_the_outline(self):
        src = _numbered(
            ["class A:", *["            def inner(self): pass"] * 400]
        )
        out = handle_oversized_result(
            src, max_chars=500, tool_name="read_file",
            tool_args='{"file_path": "p.py"}',
        )
        assert "class A:" in out
        assert "def inner" not in out

    def test_a_huge_outline_is_capped_and_says_so(self):
        src = _numbered([f"## H{i}" for i in range(400)])
        out = handle_oversized_result(
            src, max_chars=500, tool_name="read_file",
            tool_args='{"file_path": "a.md"}',
        )
        assert "and 340 more, not listed" in out

    def test_an_unknown_extension_gets_block_advice_instead(self):
        src = _numbered(["data,1,2,3"] * 400)
        out = handle_oversized_result(
            src, max_chars=500, tool_name="read_file",
            tool_args='{"file_path": "rows.csv"}',
        )
        assert "The file's structure" not in out
        assert "work forward in blocks of" in out


class TestUnpaginatedResultsStillTrim:
    def test_a_diff_is_trimmed_because_there_is_no_range_to_ask_for(self):
        out = handle_oversized_result(
            "+" * 5000, max_chars=1000, tool_name="git_diff", tool_args="{}",
        )
        assert "TRUNCATED" in out
        assert "4,000 of 5,000 characters" in out

    def test_a_search_result_is_trimmed_not_refused(self):
        out = handle_oversized_result(
            "hit\n" * 2000, max_chars=1000, tool_name="code_search",
            tool_args='{"query": "auth"}',
        )
        assert "NOTHING was read" not in out
        assert "Narrow the request instead." in out

    def test_a_read_without_a_resolvable_path_falls_back_to_trimming(self):
        out = handle_oversized_result(
            _markdown(6), max_chars=1000, tool_name="read_file", tool_args="{}",
        )
        assert "TRUNCATED" in out

    def test_broken_json_arguments_do_not_raise(self):
        out = handle_oversized_result(
            _markdown(6), max_chars=1000, tool_name="read_file",
            tool_args="{not json",
        )
        assert out


class TestRefusalDetection:
    def test_a_refusal_is_recognised(self):
        out = handle_oversized_result(
            _markdown(6), max_chars=1000, tool_name="read_file",
            tool_args='{"file_path": "a.md"}',
        )
        assert is_oversized_refusal(out)

    def test_ordinary_output_is_not(self):
        assert not is_oversized_refusal("     1\thello")
        assert not is_oversized_refusal("")

    def test_a_different_tool_error_is_not(self):
        assert not is_oversized_refusal('{"error": "File not found"}')

    def test_a_file_quoting_the_marker_is_not(self):
        """Parsed, not substring-matched, so content cannot masquerade."""
        assert not is_oversized_refusal(f"     1\t{OVERSIZED_ERROR}")


class TestDuplicateCallGuard:
    def test_the_first_call_is_allowed_through(self):
        assert DuplicateCallGuard().refusal_for("read_file", '{"file_path": "a"}') is None

    def test_the_second_identical_call_is_refused(self):
        g = DuplicateCallGuard()
        g.refusal_for("read_file", '{"file_path": "a"}')
        refusal = g.refusal_for("read_file", '{"file_path": "a"}')
        assert json.loads(refusal.splitlines()[0])["error"] == "duplicate call"

    def test_key_order_does_not_disguise_a_repeat(self):
        g = DuplicateCallGuard()
        g.refusal_for("read_file", '{"file_path": "a", "offset": 1}')
        assert g.refusal_for("read_file", '{"offset": 1, "file_path": "a"}')

    def test_a_different_offset_is_a_new_call(self):
        g = DuplicateCallGuard()
        g.refusal_for("read_file", '{"file_path": "a", "offset": 1}')
        assert g.refusal_for("read_file", '{"file_path": "a", "offset": 140}') is None

    def test_a_different_tool_is_a_new_call(self):
        g = DuplicateCallGuard()
        g.refusal_for("read_file", '{"file_path": "a"}')
        assert g.refusal_for("list_symbols", '{"file_path": "a"}') is None

    def test_two_guards_do_not_share_state(self):
        """Re-reading in a later turn is legitimate, not a livelock."""
        a, b = DuplicateCallGuard(), DuplicateCallGuard()
        a.refusal_for("read_file", '{"file_path": "a"}')
        assert b.refusal_for("read_file", '{"file_path": "a"}') is None


class TestTheObservedLivelock:
    def test_three_identical_reads_become_one_read_and_two_refusals(self):
        g = DuplicateCallGuard()
        args = '{"file_path": "HANDOFF_PROMPT.md"}'
        outcomes = [g.refusal_for("read_file", args) for _ in range(3)]
        assert outcomes[0] is None
        assert all(o is not None for o in outcomes[1:])

    def test_the_first_read_hands_back_a_map_instead_of_a_third_of_the_file(self):
        out = handle_oversized_result(
            _markdown(16), max_chars=8000, tool_name="read_file",
            tool_args='{"file_path": "HANDOFF_PROMPT.md"}',
        )
        assert is_oversized_refusal(out)
        assert out.count("## Section") == 16
        assert len(out) < 8000


class TestTheUiShowsARefusalApartFromAFailure:
    """The user asked to tell a size refusal apart at a glance."""

    def _msg(self, result: str, error: str = "") -> dict:
        return {
            "type": "tool_call", "tool_name": "read_file",
            "args": {"file_path": "a.md"}, "result": result,
            "error": error, "exec_data": None,
        }

    def _refusal(self) -> str:
        return handle_oversized_result(
            _markdown(6), max_chars=1000, tool_name="read_file",
            tool_args='{"file_path": "a.md"}',
        )

    def test_a_refusal_is_neither_ok_nor_err(self):
        from infinidev.ui.controls.tool_call_widget import _tool_status

        assert _tool_status(self._msg(self._refusal())) == "skipped"
        assert _tool_status(self._msg("     1\tfine")) == "ok"
        assert _tool_status(self._msg("", error="boom")) == "err"

    def test_the_row_uses_its_own_icon(self):
        from infinidev.ui.controls.tool_call_widget import build_tool_group

        r = build_tool_group(
            [self._msg(self._refusal())], collapsed=False, expanded_set=set(),
            width=80, live=False, on_toggle_group=lambda: None,
            on_toggle_tool=lambda i: None,
        )
        text = "\n".join("".join(t for _, t in line) for line in r.lines)
        assert "⊘" in text
        assert "✗" not in text

    def test_the_group_header_counts_the_oversized_reads(self):
        from infinidev.ui.controls.tool_call_widget import build_tool_group

        r = build_tool_group(
            [self._msg(self._refusal()), self._msg("     1\tfine")],
            collapsed=True, expanded_set=set(), width=80, live=False,
            on_toggle_group=lambda: None, on_toggle_tool=lambda i: None,
        )
        header = "".join(t for _, t in r.lines[0])
        assert "1 too large" in header

    def test_a_real_failure_still_outranks_a_refusal_in_the_header(self):
        from infinidev.ui.controls.tool_call_widget import build_tool_group

        r = build_tool_group(
            [self._msg(self._refusal()), self._msg("", error="boom")],
            collapsed=True, expanded_set=set(), width=80, live=False,
            on_toggle_group=lambda: None, on_toggle_tool=lambda i: None,
        )
        header = "".join(t for _, t in r.lines[0])
        assert header.strip().startswith("✗")

    def test_the_expanded_detail_explains_the_size_instead_of_dumping_json(self):
        from infinidev.ui.controls.tool_call_widget import build_tool_group

        r = build_tool_group(
            [self._msg(self._refusal())], collapsed=False, expanded_set={0},
            width=88, live=False, on_toggle_group=lambda: None,
            on_toggle_tool=lambda i: None,
        )
        text = "\n".join("".join(t for _, t in line) for line in r.lines)
        assert "not read —" in text
        assert "limit" in text
        assert "outline" in text
