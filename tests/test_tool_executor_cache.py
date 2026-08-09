"""Regression tests for best-effort tool-result caching."""

from __future__ import annotations

from infinidev.engine.loop.models import LoopState
from infinidev.engine.tool_executor import update_opened_files_cache


def test_read_cache_accepts_dispatch_path_aliases():
    state = LoopState()

    update_opened_files_cache(
        state,
        "read_file",
        {"file": "module.py"},
        "source",
    )

    # Workspace resolution is process-local; the important contract is that
    # the alias becomes a concrete cache key rather than ``None``.
    assert len(state.opened_files) == 1
    cached_path = next(iter(state.opened_files))
    assert cached_path.endswith("module.py")
    assert state.opened_files[cached_path].content == "source"


def test_read_cache_without_path_is_ignored():
    state = LoopState()

    update_opened_files_cache(state, "read_file", {}, "source")

    assert state.opened_files == {}
