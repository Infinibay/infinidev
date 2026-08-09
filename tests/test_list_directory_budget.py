"""Directory listings stay navigable without flooding model context."""

from __future__ import annotations

import json

from infinidev.config.settings import settings
from infinidev.tools.file.list_directory_tool import ListDirectoryTool


def _large_directory(path, count: int = 180) -> None:
    for index in range(count):
        (path / f"test_case_{index:03d}.py").write_text(f"VALUE = {index}\n")


def test_large_listing_is_valid_json_within_character_budget(
    bound_tool, workspace_dir, monkeypatch,
) -> None:
    _large_directory(workspace_dir)
    monkeypatch.setattr(settings, "MAX_DIR_LISTING_CHARS", 1_400)

    result = bound_tool(ListDirectoryTool)._run(file_path=str(workspace_dir))
    data = json.loads(result)

    assert len(result) <= settings.MAX_DIR_LISTING_CHARS
    assert data["total"] == 181  # generated files plus the fixture's sample.txt
    assert data["returned"] == len(data["entries"])
    assert data["omitted"] == data["total"] - data["returned"]
    assert data["truncated"] is True
    assert data["selection"] == "priority_hash_sample"
    assert "character_budget" in data["truncation_reasons"]


def test_budgeted_listing_is_stable_and_not_an_alphabetical_prefix(
    bound_tool, workspace_dir, monkeypatch,
) -> None:
    _large_directory(workspace_dir)
    monkeypatch.setattr(settings, "MAX_DIR_LISTING_CHARS", 1_400)
    tool = bound_tool(ListDirectoryTool)

    first = tool._run(file_path=str(workspace_dir))
    second = tool._run(file_path=str(workspace_dir))
    names = [entry["file_path"] for entry in json.loads(first)["entries"]]
    sampled_indices = [
        int(name.removeprefix("test_case_").removesuffix(".py"))
        for name in names
        if name.startswith("test_case_")
    ]

    assert first == second
    assert max(sampled_indices) >= 150


def test_high_signal_files_and_directories_survive_sampling(
    bound_tool, workspace_dir, monkeypatch,
) -> None:
    _large_directory(workspace_dir)
    (workspace_dir / "pyproject.toml").write_text("[project]\n")
    (workspace_dir / "src").mkdir()
    monkeypatch.setattr(settings, "MAX_DIR_LISTING_CHARS", 1_400)

    result = bound_tool(ListDirectoryTool)._run(file_path=str(workspace_dir))
    names = {entry["file_path"] for entry in json.loads(result)["entries"]}

    assert {"pyproject.toml", "src"} <= names


def test_pattern_can_recover_an_exact_small_listing(
    bound_tool, workspace_dir, monkeypatch,
) -> None:
    _large_directory(workspace_dir)
    (workspace_dir / "README.md").write_text("# Project\n")
    (workspace_dir / "NOTES.md").write_text("notes\n")
    monkeypatch.setattr(settings, "MAX_DIR_LISTING_CHARS", 1_400)

    result = bound_tool(ListDirectoryTool)._run(
        file_path=str(workspace_dir), pattern="*.md",
    )
    data = json.loads(result)

    assert {entry["file_path"] for entry in data["entries"]} == {
        "NOTES.md", "README.md",
    }
    assert data["truncated"] is False
