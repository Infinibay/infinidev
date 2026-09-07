"""Headless runtime initialisation for the server.

This mirrors ``cli/main.py::_bootstrap_single_prompt_runtime`` — the
canonical "set up everything the engine needs before the first turn"
routine — but calls the underlying modules directly instead of importing
``cli.main`` (whose module-level logging reconfiguration would fight with
uvicorn's). Keep this in sync with the CLI bootstrap; the steps are:

  1. initialise the SQLite DB (``init_db``)
  2. register the engine→event_bus UI hooks (``register_ui_hooks``)
  3. register behaviour-scoring hooks
  4. run the initial code index for project 1
  5. start the background index queue + file watcher

All steps after the DB are best-effort: a failure logs and continues so
the server still boots (a missing watcher just means slightly staler
code-intel, not a broken session).
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_BOOTSTRAPPED = False
_LOCK = threading.Lock()
_INDEX_QUEUE = None


def bootstrap_runtime(*, on_progress=None) -> None:
    """Idempotently initialise the engine runtime for the current cwd.

    Safe to call more than once — only the first call does work. ``cwd``
    matters: the DB, settings and code index all live under
    ``<cwd>/.infinidev`` and the indexed workspace is ``cwd`` itself, so
    the caller must ``chdir`` into the target project *before* calling.
    """
    global _BOOTSTRAPPED, _INDEX_QUEUE
    with _LOCK:
        if _BOOTSTRAPPED:
            return

        def _progress(msg: str) -> None:
            logger.info("%s", msg)
            if on_progress is not None:
                try:
                    on_progress(msg)
                except Exception:
                    pass

        # 1. DB — the one step that must succeed.
        from infinidev.db.service import init_db

        init_db()

        # 2. Engine → event_bus hooks. Without this the loop emits nothing
        #    and the WebSocket stream would be empty during execution.
        try:
            from infinidev.engine.hooks.ui_hooks import register_ui_hooks

            register_ui_hooks()
        except Exception:
            logger.warning("register_ui_hooks failed", exc_info=True)

        # 3. Behaviour hooks (scoring / nudges). Best-effort.
        try:
            from infinidev.engine.behavior.hook import register_behavior_hooks

            register_behavior_hooks()
        except Exception:
            logger.warning("register_behavior_hooks failed", exc_info=True)

        # 4. Initial index so code-intel tools work on the first turn.
        try:
            from infinidev.cli.initial_index import run_initial_index

            run_initial_index(project_id=1, on_progress=_progress)
        except Exception:
            logger.warning("run_initial_index failed", exc_info=True)

        # 5. Background index queue + file watcher (catches external edits).
        try:
            from infinidev.cli.index_queue import IndexQueue
            from infinidev.code_intel.background_indexer import set_global_queue

            q = IndexQueue(project_id=1)
            q.start()
            _INDEX_QUEUE = q
            set_global_queue(q)
            _start_file_watcher(q)
        except Exception:
            logger.warning("background indexer setup failed", exc_info=True)

        # Warm Pydantic tool-schema introspection so the first
        # LoopEngine._build_context() doesn't pay ~500ms on the
        # analysis→develop transition (see CLI bootstrap for rationale).
        try:
            from infinidev.tools import get_tools_for_role
            from infinidev.engine.tool_dispatch import build_tool_schemas

            warm = get_tools_for_role("developer", small_model=True)
            build_tool_schemas(warm, small_model=True)
        except Exception:
            pass

        _BOOTSTRAPPED = True
        _progress("Runtime ready.")


def shutdown_runtime() -> None:
    """Stop owned watchers and processes when the server exits."""
    global _BOOTSTRAPPED, _INDEX_QUEUE
    from infinidev.tools.shell.background_manager import get_background_manager

    get_background_manager().shutdown()
    if _INDEX_QUEUE is not None:
        watcher = getattr(_INDEX_QUEUE, "_file_watcher", None)
        if watcher is not None:
            watcher.stop()
        _INDEX_QUEUE.stop()
        _INDEX_QUEUE = None
    _BOOTSTRAPPED = False


def _start_file_watcher(index_queue) -> None:
    """Start the watchfiles-based workspace watcher, if available."""
    try:
        from infinidev.cli.file_watcher import FileWatcher, WATCHFILES_AVAILABLE

        if not WATCHFILES_AVAILABLE:
            return
        workspace = os.getcwd()

        def _index_on_change(changed_path: str) -> None:
            try:
                from infinidev.code_intel.background_indexer import enqueue_or_sync

                enqueue_or_sync(1, changed_path)
            except Exception:
                pass

        watcher = FileWatcher(
            workspace=workspace,
            callback=lambda _p: None,  # diff surfacing handled by event_bus
            index_callback=_index_on_change,
        )
        watcher.start()
        setattr(index_queue, "_file_watcher", watcher)
    except Exception:
        logger.debug("file watcher unavailable", exc_info=True)
