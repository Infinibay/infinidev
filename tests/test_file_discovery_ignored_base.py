"""Explicit ignored bases stay browsable and filtered discovery is honest."""

from __future__ import annotations

import json
import subprocess

import pytest

from infinidev.tools.file.glob_tool import GlobTool
from infinidev.tools.file.list_directory_tool import ListDirectoryTool


@pytest.fixture
def ignored_docs(workspace_dir):
    subprocess.run(["git", "init", "-q", str(workspace_dir)], check=True)
    (workspace_dir / ".gitignore").write_text("ignored/\n")
    base = workspace_dir / "ignored" / "docs"
    base.mkdir(parents=True)
    (base / "current.md").write_text("Current handoff\n")
    (base / "nested").mkdir()
    (base / "nested" / "design.md").write_text("Design\n")
    (base / "node_modules").mkdir()
    (base / "node_modules" / "junk.md").write_text("Generated\n")
    (base / ".hidden.md").write_text("Hidden\n")
    return base


@pytest.mark.parametrize("recursive", [False, True])
def test_listing_explicit_ignored_base_keeps_docs_and_filters_junk(
    bound_tool, ignored_docs, recursive,
):
    data = json.loads(bound_tool(ListDirectoryTool)._run(
        file_path=str(ignored_docs), recursive=recursive,
    ))
    names = {entry["file_path"] for entry in data["entries"]}
    expected = {"current.md", "nested/design.md"} if recursive else {"current.md", "nested"}
    assert names == expected
    assert data["filtering"]["base_ignored"] is True
    assert data["filtering"]["ignore_rules"] == "fallback"


@pytest.mark.parametrize("tool_class", [ListDirectoryTool, GlobTool])
def test_filtered_discovery_explains_empty_result_and_how_to_include_ignored(
    bound_tool, workspace_dir, ignored_docs, tool_class,
):
    kwargs = {"file_path": str(workspace_dir), "pattern": "*.md"}
    if tool_class is ListDirectoryTool:
        kwargs["recursive"] = True
    else:
        kwargs["pattern"] = "**/*.md"
    tool = bound_tool(tool_class)
    data = json.loads(tool._run(**kwargs))
    assert data.get("entries", data.get("matches")) == []
    assert data["filtering"]["include_ignored"] is False
    assert data["filtering"]["ignore_rules"] == "git"
    assert "include_ignored=true" in data["filtering"]["notice"]

    kwargs.update(file_path=str(ignored_docs), include_ignored=True)
    complete = json.loads(tool._run(**kwargs))
    names = {entry["file_path"] for entry in complete.get("entries", complete.get("matches"))}
    assert {"current.md", ".hidden.md"} <= names
    assert complete["filtering"]["include_ignored"] is True
    assert complete["filtering"]["ignore_rules"] == "none"


def test_glob_explicit_ignored_base_reports_fallback_and_keeps_docs(
    bound_tool, ignored_docs,
):
    data = json.loads(bound_tool(GlobTool)._run(
        file_path=str(ignored_docs), pattern="**/*.md",
    ))
    assert {entry["file_path"] for entry in data["matches"]} == {
        "current.md", "nested/design.md",
    }
    assert data["filtering"]["base_ignored"] is True
    assert data["filtering"]["ignore_rules"] == "fallback"
