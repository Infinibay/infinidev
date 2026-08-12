"""Git read tools resolve nested repositories without guessing a broad cwd."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from infinidev.tools.base.context import bind_tools_to_agent, set_context
from infinidev.tools.git import GitDiffTool, GitStatusTool


def _bind(tool, workspace) -> None:
    set_context(agent_id="git-target-agent", workspace_path=str(workspace))
    bind_tools_to_agent([tool], "git-target-agent")


def test_git_status_uses_only_nested_repository(tmp_path):
    repository = tmp_path / "infinigpu"
    (repository / ".git").mkdir(parents=True)
    tool = GitStatusTool()
    _bind(tool, tmp_path)

    status = SimpleNamespace(stdout="", stderr="", returncode=0)
    branch = SimpleNamespace(stdout="main\n", stderr="", returncode=0)
    with patch(
        "infinidev.tools.git.git_status_tool.run_git",
        side_effect=[status, branch],
    ) as run:
        result = json.loads(tool._run())

    assert result["branch"] == "main"
    assert all(call.kwargs["cwd"] == str(repository) for call in run.call_args_list)


def test_git_status_ignores_repository_above_workspace(tmp_path):
    outer = tmp_path / "outer"
    (outer / ".git").mkdir(parents=True)
    workspace = outer / "workspace"
    repository = workspace / "infinigpu"
    (repository / ".git").mkdir(parents=True)
    tool = GitStatusTool()
    _bind(tool, workspace)

    status = SimpleNamespace(stdout="", stderr="", returncode=0)
    branch = SimpleNamespace(stdout="main\n", stderr="", returncode=0)
    with patch(
        "infinidev.tools.git.git_status_tool.run_git",
        side_effect=[status, branch],
    ) as run:
        result = json.loads(tool._run())

    assert result["branch"] == "main"
    assert all(call.kwargs["cwd"] == str(repository) for call in run.call_args_list)


def test_git_status_requires_path_when_nested_target_is_ambiguous(tmp_path):
    for name in ("infinigpu", "other"):
        (tmp_path / name / ".git").mkdir(parents=True)
    tool = GitStatusTool()
    _bind(tool, tmp_path)

    result = json.loads(tool._run())

    assert "Multiple Git repositories" in result["error"]
    assert "path='<repository>'" in result["error"]


def test_explicit_git_path_selects_repository(tmp_path):
    repository = tmp_path / "infinigpu"
    (repository / ".git").mkdir(parents=True)
    (tmp_path / "other" / ".git").mkdir(parents=True)
    tool = GitDiffTool()
    _bind(tool, tmp_path)

    completed = SimpleNamespace(stdout="diff output\n", stderr="", returncode=0)
    with patch(
        "infinidev.tools.git.git_diff_tool.run_git",
        return_value=completed,
    ) as run:
        result = tool._run(path="infinigpu")

    assert result == "diff output\n"
    assert run.call_args.kwargs["cwd"] == str(repository)
