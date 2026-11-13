"""Glob limits prevent enumeration of excluded directory trees."""

from __future__ import annotations

import glob
import json
import os
import subprocess

import pytest

from infinidev.tools.file.glob_tool import GlobTool


def _forbid_scanning(monkeypatch, directories):
    original = os.scandir
    forbidden = {str(path) for path in directories}

    def guarded(path):
        assert os.path.normpath(path) not in forbidden, f"Enumerated excluded directory: {path}"
        return original(path)

    monkeypatch.setattr(os, "scandir", guarded)
    # Python 3.13 pathlib captures scandir in glob at import time.
    if hasattr(glob, "_StringGlobber"):
        monkeypatch.setattr(glob._StringGlobber, "scandir", staticmethod(guarded))


@pytest.mark.parametrize("git", [False, True])
def test_glob_prunes_ignored_trees_before_enumeration(
    bound_tool, workspace_dir, monkeypatch, git,
):
    if git:
        subprocess.run(["git", "init", "-q", str(workspace_dir)], check=True)
        (workspace_dir / ".gitignore").write_text("target/\n")
    for name in ("target", ".git"):
        directory = workspace_dir / name
        directory.mkdir(exist_ok=True)
        (directory / "nested").mkdir()
        (directory / "nested" / "generated.py").write_text("generated")
    (workspace_dir / "source.py").write_text("source")
    _forbid_scanning(monkeypatch, [workspace_dir / "target", workspace_dir / ".git"])

    result = json.loads(bound_tool(GlobTool)._run(
        pattern="**/*.py", file_path=str(workspace_dir),
    ))

    assert {item["file_path"] for item in result["matches"]} == {"source.py"}


def test_glob_prunes_max_depth_before_enumeration(bound_tool, workspace_dir, monkeypatch):
    deep = workspace_dir / "nested"
    deep.mkdir()
    (deep / "too_deep.py").write_text("deep")
    (workspace_dir / "source.py").write_text("source")
    _forbid_scanning(monkeypatch, [deep])

    result = json.loads(bound_tool(GlobTool)._run(
        pattern="**/*.py", file_path=str(workspace_dir), max_depth=1, include_ignored=True,
    ))

    assert [item["file_path"] for item in result["matches"]] == ["source.py"]


def test_glob_git_pruning_preserves_nested_rules_and_negation(
    bound_tool, workspace_dir, monkeypatch,
):
    subprocess.run(["git", "init", "-q", str(workspace_dir)], check=True)
    (workspace_dir / ".gitignore").write_text("src/*\n!src/keep/\n")
    keep = workspace_dir / "src" / "keep"
    keep.mkdir(parents=True)
    (keep / ".gitignore").write_text("*.py\n!important.py\n")
    (keep / "important.py").write_text("keep")
    (keep / "generated.py").write_text("ignored")
    ignored = workspace_dir / "src" / "ignored"
    ignored.mkdir()
    (ignored / "generated.py").write_text("ignored")
    _forbid_scanning(monkeypatch, [ignored, workspace_dir / ".git"])

    result = json.loads(bound_tool(GlobTool)._run(
        pattern="**/*.py", file_path=str(workspace_dir),
    ))

    assert [item["file_path"] for item in result["matches"]] == ["src/keep/important.py"]


@pytest.mark.parametrize(("pattern", "expected"), [
    ("*.py", {"root.py"}),
    ("src/*.py", {"src/one.py", "src/.hidden.py"}),
    ("src/**/*.py", {"src/one.py", "src/.hidden.py", "src/nested/two.py"}),
    ("**/nested/tw?.[p][y]", {"src/nested/two.py", "other/nested/two.py"}),
    ("**/**/two.py", {"src/nested/two.py", "other/nested/two.py"}),
    ("**/.hidden.py", {"src/.hidden.py"}),
    ("src/", set()),
])
def test_glob_preserves_path_component_matching(bound_tool, workspace_dir, pattern, expected):
    for name in ("root.py", "src/one.py", "src/nested/two.py", "other/nested/two.py",
                 "src/.hidden.py"):
        path = workspace_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source")

    result = json.loads(bound_tool(GlobTool)._run(
        pattern=pattern, file_path=str(workspace_dir), include_ignored=True,
    ))

    assert {item["file_path"] for item in result["matches"]} == expected


def test_glob_include_ignored_enters_junk_and_recursive_symlinks_stay_unfollowed(
    bound_tool, workspace_dir,
):
    target = workspace_dir / "target"
    target.mkdir()
    (target / "generated.py").write_text("source")
    (target / "cycle").symlink_to(workspace_dir, target_is_directory=True)
    (workspace_dir / "link").symlink_to(target, target_is_directory=True)
    tool = bound_tool(GlobTool)

    result = json.loads(tool._run(
        pattern="**/*.py", file_path=str(workspace_dir), include_ignored=True,
    ))
    explicit = json.loads(tool._run(
        pattern="link/*.py", file_path=str(workspace_dir), include_ignored=True,
    ))

    assert [item["file_path"] for item in result["matches"]] == ["target/generated.py"]
    assert [item["file_path"] for item in explicit["matches"]] == ["link/generated.py"]


def test_glob_content_filter_depth_and_result_limit(bound_tool, workspace_dir, monkeypatch):
    (workspace_dir / "source.py").write_text("Not a match")
    nested = workspace_dir / "src"
    nested.mkdir()
    (nested / "one.py").write_text("needle")
    (nested / "two.py").write_text("NEEDLE")
    deep = nested / "deeper"
    deep.mkdir()
    (deep / "hidden.py").write_text("needle")
    _forbid_scanning(monkeypatch, [deep])

    result = json.loads(bound_tool(GlobTool)._run(
        pattern="**/*.py", file_path=str(workspace_dir), max_depth=2,
        content_pattern="needle", case_sensitive=False, max_results=1,
    ))

    assert result["match_count"] == 1
    assert result["truncated"] is True
    assert result["matches"][0]["file_path"] in {"src/one.py", "src/two.py"}


@pytest.mark.parametrize("tracked_path", ["target/tracked.py", "target/src/tracked.py"])
def test_glob_preserves_tracked_files_below_ignored_directories(
    bound_tool, workspace_dir, monkeypatch, tracked_path,
):
    subprocess.run(["git", "init", "-q", str(workspace_dir)], check=True)
    target = workspace_dir / "target"
    target.mkdir()
    tracked = workspace_dir / tracked_path
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text("tracked")
    subprocess.run(["git", "add", tracked_path], cwd=workspace_dir, check=True)
    (workspace_dir / ".gitignore").write_text("target/\n")
    junk = target / "generated"
    junk.mkdir()
    (junk / "junk.py").write_text("untracked")
    _forbid_scanning(monkeypatch, [junk, workspace_dir / ".git"])

    result = json.loads(bound_tool(GlobTool)._run(
        pattern="**/*.py", file_path=str(workspace_dir),
    ))

    assert [item["file_path"] for item in result["matches"]] == [tracked_path]


@pytest.mark.parametrize("pattern", ["**", "src/**", "**/**"])
def test_glob_terminal_recursive_pattern_matches_runtime_pathlib(
    bound_tool, workspace_dir, pattern,
):
    source = workspace_dir / "src"
    source.mkdir()
    (source / "source.py").write_text("source")
    expected = {
        str(path.relative_to(workspace_dir))
        for path in workspace_dir.glob(pattern) if not path.is_dir()
    }

    result = json.loads(bound_tool(GlobTool)._run(
        pattern=pattern, file_path=str(workspace_dir), include_ignored=True,
    ))

    assert {item["file_path"] for item in result["matches"]} == expected
