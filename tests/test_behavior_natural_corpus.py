"""Natural trajectory extraction tests."""

from __future__ import annotations

import json
from pathlib import Path

from bench.behavior_natural_corpus import extract_artifact, sanitize_text


def _artifact(tmp_path: Path, *, category: str, trace: list[dict[str, object]]) -> Path:
    path = tmp_path / "run.json"
    path.write_text(json.dumps({
        "task": {
            "id": "natural-task",
            "family": "natural-family",
            "category": category,
            "repository_fixture": "natural-project",
            "request": "Inspect the repository without leaking token sk-example0123456789.",
            "expected_changed_paths": [] if category == "code_review" else ["src/a.py"],
        },
        "provider": "minimax",
        "model": "MiniMax-M3",
        "model_identity": "minimax:MiniMax-M3@test",
        "condition": "baseline",
        "repetition": 0,
        "verify_exit_code": 0,
        "error": "",
        "action_records": [{
            "step_index": 1,
            "summary": "Read /tmp/infinidev-agent-task-abc/repo/src/a.py for a@b.com",
            "tool_calls_count": len(trace),
            "changes_made": "",
            "discovered_context": "",
            "pending_items": "",
            "anti_patterns": "",
            "test_outcome_fingerprints": [],
            "successful_edit_count": 0,
            "net_workspace_changed": False,
        }],
        "tool_trace": trace,
        "runtime_behavior_events": [],
    }), encoding="utf-8")
    return path


def test_sanitize_text_redacts_secret_path_and_email() -> None:
    value = sanitize_text(
        "sk-example0123456789 /tmp/infinidev-agent-task-abc/repo/a.py a@b.com"
    )

    assert "sk-example" not in value
    assert "/tmp/infinidev" not in value
    assert "a@b.com" not in value


def test_legitimate_read_only_discovery_is_hard_negative(tmp_path: Path) -> None:
    trace = [
        {
            "tool_name": "read_file",
            "arguments": {"file_path": f"/tmp/infinidev-agent-task-abc/repo/{name}.py"},
            "result": "source",
            "failed": False,
        }
        for name in ("a", "b", "c", "d")
    ]
    path = _artifact(tmp_path, category="code_review", trace=trace)

    windows = extract_artifact(path, root=tmp_path)
    window = next(item for item in windows if item.window_kind == "step_summary")

    assert window.label is None
    assert window.review_status == "approved"
    assert window.hard_negative_for == ("excessive_exploration",)
    assert "<workspace>" in window.text


def test_modifying_discovery_window_gets_conservative_label(tmp_path: Path) -> None:
    names = ["list_directory", "read_file", "read_file", "read_file", "read_file"]
    names += ["code_search", "describe_tool", "search_symbols"]
    trace = [
        {
            "tool_name": name,
            "arguments": {"file_path": f"src/{index}.py", "query": "target"},
            "result": "evidence",
            "failed": False,
        }
        for index, name in enumerate(names)
    ]
    path = _artifact(tmp_path, category="bugfix", trace=trace)

    windows = extract_artifact(path, root=tmp_path)
    window = next(item for item in windows if item.window_kind == "step_summary")

    assert window.label == "excessive_exploration"
    assert window.label_source == "deterministic_trace"
    assert window.review_status == "approved"


def test_unaligned_trace_is_sent_to_review(tmp_path: Path) -> None:
    path = _artifact(tmp_path, category="bugfix", trace=[])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["action_records"][0]["tool_calls_count"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    windows = extract_artifact(path, root=tmp_path)
    window = next(item for item in windows if item.window_kind == "step_summary")

    assert not window.trace_aligned
    assert window.review_status == "needs_review"


def test_trace_prefixes_capture_failure_and_verified_progress(tmp_path: Path) -> None:
    trace = [
        {
            "tool_name": "execute_command",
            "arguments": {"command": "uv run pytest tests/test_a.py -q", "bad": True},
            "result": "wrong parameter name(s): bad",
            "failed": True,
        },
        {
            "tool_name": "edit_file",
            "arguments": {"file_path": "src/a.py"},
            "result": "updated",
            "failed": False,
        },
        {
            "tool_name": "execute_command",
            "arguments": {"command": "uv run pytest tests/test_a.py -q"},
            "result": "1 passed",
            "failed": False,
        },
    ]
    path = _artifact(tmp_path, category="bugfix", trace=trace)

    windows = extract_artifact(path, root=tmp_path)
    trace_labels = {
        item.label for item in windows if item.window_kind == "tool_prefix"
    }
    healthy = next(item for item in windows if item.label == "healthy_progress")

    assert trace_labels == {"tool_schema_mismatch", "healthy_progress"}
    assert healthy.hard_negative_for == ("retry_loop",)


def test_nonzero_command_exit_is_a_failed_call(tmp_path: Path) -> None:
    trace = [{
        "tool_name": "execute_command",
        "arguments": {"command": "uv run pytest"},
        "result": json.dumps({"exit_code": 2, "stdout": "", "stderr": "bad args"}),
        "failed": False,
    }]
    path = _artifact(tmp_path, category="bugfix", trace=trace)

    windows = extract_artifact(path, root=tmp_path)
    step = next(item for item in windows if item.window_kind == "step_summary")

    assert step.failed_calls == 1


def test_distinct_read_slices_are_not_a_retry_loop(tmp_path: Path) -> None:
    trace = [
        {
            "tool_name": "read_file",
            "arguments": {
                "file_path": "src/a.py",
                "offset": offset,
                "limit": 40,
            },
            "result": "source",
            "failed": False,
        }
        for offset in (0, 40, 80, 120)
    ]
    path = _artifact(tmp_path, category="bugfix", trace=trace)

    windows = extract_artifact(path, root=tmp_path)

    assert all(item.label != "retry_loop" for item in windows)


def test_read_only_run_emits_progressive_hard_negatives(tmp_path: Path) -> None:
    trace = [
        {
            "tool_name": "read_file",
            "arguments": {"file_path": f"src/{index}.py", "offset": 0, "limit": 20},
            "result": "source",
            "failed": False,
        }
        for index in range(9)
    ]
    path = _artifact(tmp_path, category="code_review", trace=trace)

    windows = extract_artifact(path, root=tmp_path)
    prefixes = [item for item in windows if item.window_kind == "tool_prefix"]

    assert [item.tool_calls for item in prefixes] == [4, 8, 9]
    assert all(item.label is None for item in prefixes)
    assert all(item.hard_negative_for == ("excessive_exploration",) for item in prefixes)
