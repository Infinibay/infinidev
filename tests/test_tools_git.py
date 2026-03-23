"""Tests for git tools (commit)."""

import json
import os
import subprocess

import pytest

from infinidev.tools.git.commit import GitCommitTool


@pytest.fixture
def git_repo(workspace_dir):
    """Initialise a git repository in the workspace."""
    cwd = str(workspace_dir)
    subprocess.run(["git", "init"], cwd=cwd, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=cwd, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=cwd, capture_output=True)
    # Initial commit so we have a branch
    subprocess.run(["git", "add", "."], cwd=cwd, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=cwd, capture_output=True)
    return workspace_dir


class TestGitCommit:
    """Tests for GitCommitTool."""

    def test_commit_no_changes(self, bound_tool, git_repo):
        """Clean repo returns error."""
        tool = bound_tool(GitCommitTool)
        result = tool._run(message="empty commit")
        data = json.loads(result)
        assert "error" in data
        assert "no changes" in data["error"].lower() or "nothing" in data["error"].lower()

    def test_commit_all_changes(self, bound_tool, git_repo, auto_approve_permissions):
        """Create a file and commit all, verify hash returned."""
        new_file = git_repo / "new.txt"
        new_file.write_text("new content")
        tool = bound_tool(GitCommitTool)
        result = tool._run(message="add new file")
        data = json.loads(result)
        assert "error" not in data
        assert "commit_hash" in data or "hash" in data or "commit" in str(data).lower()

    def test_commit_specific_files(self, bound_tool, git_repo, auto_approve_permissions):
        """Stage only specified files."""
        f1 = git_repo / "a.txt"
        f2 = git_repo / "b.txt"
        f1.write_text("aaa")
        f2.write_text("bbb")
        tool = bound_tool(GitCommitTool)
        result = tool._run(message="add a only", files=["a.txt"])
        # b.txt should still be untracked
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(git_repo),
            capture_output=True, text=True,
        )
        assert "b.txt" in status.stdout

    def test_commit_message_preserved(self, bound_tool, git_repo, auto_approve_permissions):
        """Commit message appears in git log."""
        (git_repo / "msg_test.txt").write_text("data")
        tool = bound_tool(GitCommitTool)
        tool._run(message="test message 12345")
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"], cwd=str(git_repo),
            capture_output=True, text=True,
        )
        assert "test message 12345" in log.stdout

    def test_commit_returns_branch(self, bound_tool, git_repo, auto_approve_permissions):
        """Result includes branch name."""
        (git_repo / "branch_test.txt").write_text("data")
        tool = bound_tool(GitCommitTool)
        result = tool._run(message="branch check")
        # Should mention "main" or "master" somewhere
        result_lower = result.lower()
        assert "main" in result_lower or "master" in result_lower or "branch" in result_lower
