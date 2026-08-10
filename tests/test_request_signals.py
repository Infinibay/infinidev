"""Regression tests for deterministic orchestration request signals."""

from infinidev.engine.orchestration.request_signals import (
    referenced_file_paths,
    resolve_referenced_repository,
)


def _repository_with_brief(root, name: str):
    repository = root / name
    (repository / ".git").mkdir(parents=True)
    (repository / "CONTINUE.md").write_text("Continue here.\n")
    return repository


def test_referenced_paths_preserve_nested_brief_path():
    request = "Lee infinigpu/CONTINUE.md, y continua el trabajo"

    assert referenced_file_paths(request) == ("infinigpu/CONTINUE.md",)


def test_resolves_repository_containing_the_named_brief(tmp_path):
    repository = _repository_with_brief(tmp_path, "infinigpu")

    resolved = resolve_referenced_repository(
        "Lee infinigpu/CONTINUE.md y continua el trabajo",
        str(tmp_path),
    )

    assert resolved == str(repository)


def test_rejects_paths_that_escape_the_workspace(tmp_path):
    outside = _repository_with_brief(tmp_path.parent, "outside-repository")

    resolved = resolve_referenced_repository(
        "Lee ../outside-repository/CONTINUE.md y continua",
        str(tmp_path),
    )

    assert outside.exists()
    assert resolved is None


def test_fails_closed_when_request_names_multiple_repositories(tmp_path):
    _repository_with_brief(tmp_path, "repo-a")
    _repository_with_brief(tmp_path, "repo-b")

    resolved = resolve_referenced_repository(
        "Continua desde repo-a/CONTINUE.md y repo-b/CONTINUE.md",
        str(tmp_path),
    )

    assert resolved is None
