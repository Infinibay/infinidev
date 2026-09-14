

def test_two_spellings_of_one_file_are_one_change(tmp_path):
    """`/var` and `/private/var` name the same file on macOS.

    The baseline root is captured with ``realpath`` while a tool call carries
    whatever spelling the model used. Keying them separately recorded every
    edit twice: the reviewer's prompt carried the same diff twice, and the
    change count doubled in 22 of the 32 recorded benchmark runs.
    """
    import os

    from infinidev.engine.file_change_tracker import FileChangeTracker
    from infinidev.engine.workspace_baseline import WorkspaceBaseline

    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "mod.py"
    target.write_text("value = 1\n", encoding="utf-8")

    # A symlinked spelling of the same directory, as /var and /private/var are.
    link = tmp_path / "link"
    link.symlink_to(repo, target_is_directory=True)
    linked = link / "mod.py"
    assert os.path.realpath(linked) == os.path.realpath(target)
    assert str(linked) != str(target)

    tracker = FileChangeTracker(WorkspaceBaseline.capture(str(repo)))
    # The tool edit arrives under one spelling...
    tracker.record(str(linked), "value = 1\n", "value = 2\n")
    # ...and the disk really changed, so the reconcile scan sees it too.
    linked.write_text("value = 2\n", encoding="utf-8")
    tracker.reconcile_workspace()

    paths = tracker.get_all_paths()
    assert len(paths) == 1, f"one file, one entry: {paths}"
    assert tracker.get_change_count(paths[0]) == 1
    assert tracker.get_action(paths[0]) == "modified"
