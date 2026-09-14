"""The corpus measurement behind the tool-argument lever.

The document claims the elision removes most of the argument bytes and never has
to bail out on a real call. Both are counts over stored runs, so they get a test
against a synthetic corpus rather than a paste.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.tool_argument_census import census


def _run(tmp_path: Path, name: str, calls: list[tuple[str, dict]]) -> None:
    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(
        json.dumps(
            {
                "task": {"id": name},
                "tool_trace": [
                    {"tool_name": tool, "arguments": arguments} for tool, arguments in calls
                ],
            }
        ),
        encoding="utf-8",
    )


def test_a_corpus_of_small_calls_reports_no_saving(tmp_path: Path) -> None:
    _run(tmp_path, "small", [("read_file", {"path": "a.py"}), ("glob", {"pattern": "*.py"})])

    report = census(tmp_path)

    assert report["calls"] == 2
    assert report["calls_touched"] == 0
    assert report["elidable_bytes"] == 0
    assert report["elidable_pct"] == 0.0


def test_a_file_body_dominates_the_bytes_it_sits_among(tmp_path: Path) -> None:
    body = "x" * 5_000
    _run(
        tmp_path,
        "writer",
        [
            ("read_file", {"path": "a.py"}),
            ("create_file", {"file_path": "a.py", "content": body}),
            ("glob", {"pattern": "*.py"}),
        ],
    )

    report = census(tmp_path)

    assert report["calls"] == 3
    # One call of three is touched, and it holds nearly all the bytes.
    assert report["calls_touched"] == 1
    assert report["elidable_pct"] > 90.0
    assert report["payload_calls"] == 1
    assert report["payload_elidable_pct"] > 90.0


def test_the_per_tool_breakdown_names_where_the_bytes_are(tmp_path: Path) -> None:
    body = "y" * 4_000
    _run(
        tmp_path,
        "mixed",
        [("create_file", {"file_path": "a.py", "content": body}), ("read_file", {"path": "a.py"})],
    )

    report = census(tmp_path)

    assert report["by_tool"]["create_file"]["elidable"] > 3_000
    assert report["by_tool"]["read_file"]["elidable"] == 0


def test_a_run_without_a_tool_trace_is_skipped_not_counted(tmp_path: Path) -> None:
    _run(tmp_path, "empty", [])
    (tmp_path / "empty" / "run.json").write_text(json.dumps({"task": {"id": "empty"}}))

    report = census(tmp_path)

    assert report["calls"] == 0
    assert report["elidable_pct"] == 0.0
