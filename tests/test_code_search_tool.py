"""Regression tests for bounded code-search subprocess execution."""

from __future__ import annotations

import json

import pytest

from infinidev.engine.subprocess_runner import run_captured
from infinidev.tools.base.context import set_context
from infinidev.tools.file.code_search_tool import CodeSearchTool
from infinidev.tools.file.read_file_tool import ReadFileTool


def test_max_depth_search_uses_safe_shell_quoting(bound_tool, workspace_dir) -> None:
    marker = workspace_dir / "shell-injection-marker"
    (workspace_dir / "source.py").write_text("needle = True\n")
    tool = bound_tool(CodeSearchTool)

    result = json.loads(
        tool._run(
            pattern=f"needle; touch {marker}",
            file_path=str(workspace_dir),
            max_depth=2,
        )
    )

    assert result["match_count"] == 0
    assert not marker.exists()


def test_max_depth_search_returns_matches(bound_tool, workspace_dir) -> None:
    nested = workspace_dir / "nested"
    nested.mkdir()
    (nested / "source.py").write_text("needle = True\n")
    tool = bound_tool(CodeSearchTool)

    result = json.loads(
        tool._run(
            pattern="needle",
            file_path=str(workspace_dir),
            file_extensions=["py"],
            max_depth=2,
        )
    )

    assert result["match_count"] == 1
    assert result["matches"][0]["file"].endswith("nested/source.py")


@pytest.mark.parametrize("backend", ["git", "git-parent", "grep-depth", "grep"])
def test_subdirectory_hit_can_be_passed_directly_to_read_file(
    bound_tool, workspace_dir, backend,
) -> None:
    workspace = workspace_dir
    if backend == "git-parent":
        workspace = workspace_dir / "workspace"
        workspace.mkdir()
        set_context(workspace_path=str(workspace))
    if backend in {"git", "git-parent"}:
        assert run_captured(["git", "init", "-q"], cwd=str(workspace_dir), timeout=5).exit_code == 0
    package = workspace / "pkg"
    package.mkdir()
    (package / "scheduler.py").write_text("def max_pending_jobs():\n    return 37\n")
    (workspace / "scheduler.py").write_text("def max_pending_jobs():\n    return 2\n")

    result = json.loads(bound_tool(CodeSearchTool)._run(
        pattern="def max_pending_jobs", file_path="pkg", context_lines=1,
        max_depth=2 if backend == "grep-depth" else None,
    ))

    assert result.get("match_count") == 1, result
    match = result["matches"][0]
    assert match["context"][-1]["content"] == "    return 37"
    read = bound_tool(ReadFileTool)._run(file_path=match["file"])
    assert read.splitlines()[-1].endswith("    return 37"), (match["file"], read)


def test_search_accepts_a_single_file_path(bound_tool, workspace_dir) -> None:
    source = workspace_dir / "source.py"
    source.write_text("before\nneedle = True\nafter\n")

    result = json.loads(bound_tool(CodeSearchTool)._run(
        pattern="needle",
        file_path=str(source),
        context_lines=1,
    ))

    assert result["match_count"] == 1
    assert result["matches"][0]["line"] == 2
    assert [row["line"] for row in result["matches"][0]["context"]] == [1, 2, 3]


def test_single_file_exact_limit_is_not_truncated(bound_tool, workspace_dir) -> None:
    source = workspace_dir / "source.py"
    source.write_text("needle\nneedle\n")

    result = json.loads(bound_tool(CodeSearchTool)._run(
        pattern="needle",
        file_path=str(source),
        max_results=2,
    ))

    assert result["match_count"] == 2
    assert result["truncated"] is False


def test_single_file_more_than_limit_is_truncated(bound_tool, workspace_dir) -> None:
    source = workspace_dir / "source.py"
    source.write_text("needle\nneedle\nneedle\n")

    result = json.loads(bound_tool(CodeSearchTool)._run(
        pattern="needle",
        file_path=str(source),
        max_results=2,
    ))

    assert result["match_count"] == 2
    assert result["truncated"] is True


@pytest.mark.parametrize("git_backend", [False, True])
@pytest.mark.parametrize("filename", ["source.rs", "a-12-b:c.rs", "á-b:c.rs"])
def test_directory_context_preserves_real_backend_rows(
    bound_tool, workspace_dir, git_backend, filename,
):
    if git_backend:
        assert run_captured(["git", "init", "-q"], cwd=str(workspace_dir), timeout=5).exit_code == 0
    source = workspace_dir / filename
    source.write_text("before:123:text\nfn deliver_chunk() {\nafter-12-context\n")
    result = json.loads(bound_tool(CodeSearchTool)._run(
        pattern="fn deliver_chunk", file_path=str(workspace_dir), context_lines=1,
    ))
    assert result["match_count"] == 1
    hit = result["matches"][0]
    assert hit["file"].endswith(filename) and hit["line"] == 2
    assert hit["content"] == "fn deliver_chunk() {"
    assert hit["context"] == [
        {"line": 1, "content": "before:123:text"},
        {"line": 2, "content": "fn deliver_chunk() {"},
        {"line": 3, "content": "after-12-context"},
    ]
    assert result["search_metadata"]["coverage"] == "completed_scope"
    assert "unparsed_output" not in result["search_metadata"]["reasons"]


def test_directory_context_does_not_consume_match_limit(bound_tool, workspace_dir):
    (workspace_dir / "source.rs").write_text("before\nneedle one\nbetween\nneedle two\nafter\n")
    result = json.loads(bound_tool(CodeSearchTool)._run(
        pattern="needle", file_path=str(workspace_dir), context_lines=1, max_results=1,
    ))
    assert result["match_count"] == 1 and result["truncated"] is True
    assert [r["line"] for r in result["matches"][0]["context"]] == [1, 2, 3]
    assert "result_limit" in result["search_metadata"]["reasons"]


@pytest.mark.parametrize("git_backend", [False, True])
def test_directory_overlapping_context_is_not_a_hit(bound_tool, workspace_dir, git_backend):
    if git_backend:
        assert run_captured(["git", "init", "-q"], cwd=str(workspace_dir), timeout=5).exit_code == 0
    (workspace_dir / "source.rs").write_text("before\nneedle one\nbetween\nneedle two\nafter\n")
    result = json.loads(bound_tool(CodeSearchTool)._run(
        pattern="needle", file_path=str(workspace_dir), context_lines=1,
    ))
    assert result["match_count"] == 2
    assert [hit["line"] for hit in result["matches"]] == [2, 4]
    assert [[r["line"] for r in hit["context"]] for hit in result["matches"]] == [[1, 2, 3], [3, 4, 5]]


@pytest.mark.parametrize("git_backend", [False, True])
@pytest.mark.parametrize("separator", ["\v", "\f", "\u0085", "\u2028"])
def test_directory_non_lf_content_is_preserved(bound_tool, workspace_dir, git_backend, separator):
    if git_backend:
        assert run_captured(["git", "init", "-q"], cwd=str(workspace_dir), timeout=5).exit_code == 0
    content = "prefix" + separator + "needle"
    (workspace_dir / "source.rs").write_text("before\n" + content + "\nafter\n")
    result = json.loads(bound_tool(CodeSearchTool)._run(
        pattern="needle", file_path=str(workspace_dir), context_lines=1,
    ))
    assert result["match_count"] == 1
    assert result["matches"][0]["content"] == content
    assert result["matches"][0]["context"][1]["content"] == content
    assert "unparsed_output" not in result["search_metadata"]["reasons"]


@pytest.mark.parametrize("max_depth", [None, 2])
def test_grep_filename_null_flag_is_portable(bound_tool, monkeypatch, workspace_dir, max_depth):
    from infinidev.tools.file import code_search_tool as module
    from infinidev.engine.subprocess_runner import CapturedProcess
    monkeypatch.setattr(CodeSearchTool, "_has_git", staticmethod(lambda _: False))
    commands = []
    def captured(command, **kwargs):
        commands.append(command)
        return CapturedProcess(0, "source.rs\x001:needle\n", "")
    monkeypatch.setattr(module, "run_captured", captured)
    result = json.loads(bound_tool(CodeSearchTool)._run(
        pattern="needle", file_path=str(workspace_dir), max_depth=max_depth,
    ))
    command = commands[0]
    assert "--null" in command and "-Z" not in command
    assert result["match_count"] == 1
