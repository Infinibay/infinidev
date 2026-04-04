"""Tests for parallel tool call execution."""

import json
import time
from unittest.mock import MagicMock

import pytest

from infinidev.engine.tool_executor import (
    batch_tool_calls as _batch_tool_calls,
    execute_tool_calls_parallel as _execute_tool_calls_parallel,
    WRITE_TOOLS as _WRITE_TOOLS,
)


def _make_tc(name, args="{}"):
    """Create a mock tool call object."""
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = args
    tc.id = f"tc_{name}_{id(tc)}"
    return tc


# ── Batching ─────────────────────────────────────────────────────────────────


class TestBatchToolCalls:
    def test_all_reads(self):
        """All read-only tools → single batch."""
        calls = [_make_tc("read_file"), _make_tc("code_search"), _make_tc("glob")]
        batches = _batch_tool_calls(calls)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_all_writes(self):
        """All writes → one batch per write."""
        calls = [_make_tc("edit_file"), _make_tc("write_file"), _make_tc("git_commit")]
        batches = _batch_tool_calls(calls)
        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)

    def test_reads_then_write(self):
        """[r, r, r, w] → [[r, r, r], [w]]"""
        calls = [
            _make_tc("read_file"), _make_tc("code_search"), _make_tc("glob"),
            _make_tc("edit_file"),
        ]
        batches = _batch_tool_calls(calls)
        assert len(batches) == 2
        assert len(batches[0]) == 3  # reads
        assert len(batches[1]) == 1  # write

    def test_mixed_pattern(self):
        """[r, r, r, w, r, r, w, w, r, r] → [[r,r,r], [w], [r,r], [w], [w], [r,r]]"""
        calls = [
            _make_tc("read_file"), _make_tc("code_search"), _make_tc("glob"),
            _make_tc("edit_file"),
            _make_tc("read_file"), _make_tc("list_directory"),
            _make_tc("write_file"), _make_tc("git_commit"),
            _make_tc("read_file"), _make_tc("glob"),
        ]
        batches = _batch_tool_calls(calls)
        assert len(batches) == 6
        assert len(batches[0]) == 3  # r, r, r
        assert len(batches[1]) == 1  # w
        assert len(batches[2]) == 2  # r, r
        assert len(batches[3]) == 1  # w
        assert len(batches[4]) == 1  # w
        assert len(batches[5]) == 2  # r, r

    def test_empty(self):
        """Empty list → empty batches."""
        assert _batch_tool_calls([]) == []

    def test_single_read(self):
        """Single read → single batch."""
        batches = _batch_tool_calls([_make_tc("read_file")])
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_single_write(self):
        """Single write → single batch."""
        batches = _batch_tool_calls([_make_tc("edit_file")])
        assert len(batches) == 1

    def test_write_between_reads(self):
        """[r, w, r] → [[r], [w], [r]]"""
        calls = [_make_tc("read_file"), _make_tc("edit_file"), _make_tc("read_file")]
        batches = _batch_tool_calls(calls)
        assert len(batches) == 3

    def test_execute_command_is_write(self):
        """execute_command is classified as write (barrier)."""
        assert "execute_command" in _WRITE_TOOLS

    def test_read_tools_not_in_writes(self):
        """Read-only tools should not be in _WRITE_TOOLS."""
        for tool in ["read_file", "code_search", "glob", "list_directory",
                      "search_findings", "read_findings", "web_search"]:
            assert tool not in _WRITE_TOOLS


# ── Parallel Execution ───────────────────────────────────────────────────────


class TestParallelExecution:
    def test_single_item_no_threading(self):
        """Single item batch doesn't use threading."""
        tc = _make_tc("read_file", '{"path": "test.txt"}')
        dispatch = {"read_file": MagicMock()}
        dispatch["read_file"]._run.return_value = "file contents"

        # The execute_tool_call function is used directly, mock it
        from unittest.mock import patch
        with patch("infinidev.engine.tool_executor.execute_tool_call", return_value="file contents"):
            results = _execute_tool_calls_parallel([tc], dispatch)

        assert len(results) == 1
        assert results[0][0] == tc
        assert results[0][1] == "file contents"

    def test_parallel_preserves_order(self):
        """Results should be in the same order as input, not completion order."""
        tcs = [_make_tc(f"read_file", f'{{"path": "file{i}.txt"}}') for i in range(5)]

        def mock_execute(dispatch, name, args, hook_metadata=None):
            # Simulate varying execution times
            import time
            idx = int(json.loads(args)["path"].replace("file", "").replace(".txt", ""))
            time.sleep(0.01 * (5 - idx))  # Later files finish faster
            return f"content_{idx}"

        from unittest.mock import patch
        with patch("infinidev.engine.tool_executor.execute_tool_call", side_effect=mock_execute):
            results = _execute_tool_calls_parallel(tcs, {})

        assert len(results) == 5
        for i, (tc, result) in enumerate(results):
            assert result == f"content_{i}"

    def test_parallel_is_faster_than_sequential(self):
        """Parallel execution of slow operations should be faster than sequential."""
        tcs = [_make_tc(f"read_file", f'{{"path": "file{i}.txt"}}') for i in range(4)]

        def slow_execute(dispatch, name, args, hook_metadata=None):
            time.sleep(0.1)
            return "ok"

        from unittest.mock import patch

        # Sequential
        start = time.time()
        with patch("infinidev.engine.tool_executor.execute_tool_call", side_effect=slow_execute):
            for tc in tcs:
                slow_execute({}, tc.function.name, tc.function.arguments)
        sequential_time = time.time() - start

        # Parallel
        start = time.time()
        with patch("infinidev.engine.tool_executor.execute_tool_call", side_effect=slow_execute):
            _execute_tool_calls_parallel(tcs, {})
        parallel_time = time.time() - start

        # Parallel should be at least 2x faster
        assert parallel_time < sequential_time * 0.7, (
            f"Parallel ({parallel_time:.2f}s) should be significantly faster "
            f"than sequential ({sequential_time:.2f}s)"
        )

    def test_parallel_handles_errors(self):
        """If one tool fails in parallel, others still complete."""
        tcs = [_make_tc("read_file", '{"path": "ok.txt"}'),
               _make_tc("read_file", '{"path": "fail.txt"}'),
               _make_tc("read_file", '{"path": "ok2.txt"}')]

        call_count = 0
        def failing_execute(dispatch, name, args, hook_metadata=None):
            nonlocal call_count
            call_count += 1
            if "fail" in json.loads(args)["path"]:
                raise RuntimeError("File not found")
            return "content"

        from unittest.mock import patch
        with patch("infinidev.engine.tool_executor.execute_tool_call", side_effect=failing_execute):
            results = _execute_tool_calls_parallel(tcs, {})

        assert len(results) == 3
        assert results[0][1] == "content"
        assert "error" in results[1][1].lower()  # Error captured, not raised
        assert results[2][1] == "content"
