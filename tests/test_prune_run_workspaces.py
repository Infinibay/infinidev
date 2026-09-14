"""The tool that frees a campaign's disk without losing its measurements.

The leak it addresses is not self-announcing: 300 runs left 105 GB of fixture
copies in ``bench/runs``, the volume filled, and the symptom was a hundred test
errors in unrelated files. These tests pin the two things that make the tool
safe to run — it targets only the runner's own copies, and it never deletes by
default.
"""

from __future__ import annotations

from pathlib import Path

from bench.prune_run_workspaces import find_copies, prune


def _run(tmp_path: Path, arm: str, name: str, *, artifact: bool = True, size: int = 10) -> Path:
    directory = tmp_path / arm / "artifacts" / name
    workspace = directory / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "payload.txt").write_text("x" * size, encoding="utf-8")
    if artifact:
        (directory / "run.json").write_text("{}", encoding="utf-8")
    return workspace


def test_only_the_runners_own_copies_are_found(tmp_path: Path) -> None:
    _run(tmp_path, "task", "complex-plan.r0.baseline")
    # A fixture checkout named `workspace` is an input, not an output.
    fixture = tmp_path / "fixtures" / "demo" / "workspace"
    fixture.mkdir(parents=True)
    (fixture / "f.txt").write_text("keep", encoding="utf-8")

    copies = find_copies(tmp_path)

    assert len(copies) == 1
    assert copies[0].workspace.parent.parent.name == "artifacts"


def test_an_orphan_copy_is_the_one_no_artifact_justifies(tmp_path: Path) -> None:
    _run(tmp_path, "task", "kept.r0.baseline", artifact=True)
    _run(tmp_path, "task", "died.r0.baseline", artifact=False)

    copies = {c.workspace.parent.name: c for c in find_copies(tmp_path)}

    assert copies["kept.r0.baseline"].is_orphan is False
    assert copies["died.r0.baseline"].is_orphan is True


def test_nothing_is_deleted_without_apply(tmp_path: Path) -> None:
    workspace = _run(tmp_path, "task", "complex-plan.r0.baseline")

    result = prune(tmp_path, apply=False, keep_artifacts=False)

    assert result["targeted"] == 1
    assert result["removed"] == 0
    assert result["bytes_targeted"] > 0
    assert workspace.is_dir(), "a dry run must not touch the tree"


def test_apply_removes_the_copy_and_keeps_the_artifact(tmp_path: Path) -> None:
    workspace = _run(tmp_path, "task", "complex-plan.r0.baseline")
    artifact = workspace.parent / "run.json"

    result = prune(tmp_path, apply=True, keep_artifacts=False)

    assert result["removed"] == 1
    assert not workspace.exists()
    assert artifact.is_file(), "the measurement must survive the cleanup"


def test_keep_artifacts_spares_every_copy_that_recorded_a_run(tmp_path: Path) -> None:
    kept = _run(tmp_path, "task", "kept.r0.baseline", artifact=True)
    died = _run(tmp_path, "task", "died.r0.baseline", artifact=False)

    result = prune(tmp_path, apply=True, keep_artifacts=True)

    assert result["orphans"] == 1
    assert result["removed"] == 1
    assert kept.is_dir()
    assert not died.exists()
