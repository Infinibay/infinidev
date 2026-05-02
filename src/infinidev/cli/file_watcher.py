"""File watcher module for Infinidev TUI.

Monitors workspace for file changes and provides callbacks for affected paths.
Only triggers refresh for visible (expanded) directories in the explorer.
"""

import os
import pathlib
from typing import Callable, Optional, Set
from threading import Thread, Event
import logging

import os
os.environ["WATCHFILES_DEBUG"] = "false"  # Suppress Rust-level debug output

try:
    from watchfiles import watch, DefaultFilter
    WATCHFILES_AVAILABLE = True
except ImportError:
    WATCHFILES_AVAILABLE = False
    DefaultFilter = object  # type: ignore[assignment, misc]
    logging.warning("watchfiles not installed. File watching disabled.")

logger = logging.getLogger(__name__)

# ``watchfiles`` itself logs ``"N changes detected"`` at INFO on every
# debounce window. With ``INFINIDEV_LOG_LEVEL=INFO`` that becomes the
# noisiest line in the entire process, especially because infinidev's
# own writes to ``.infinidev/`` (db, log, history) used to feed back
# into the watcher pre-filter. Clamp it to WARNING so the watcher's
# own steady-state activity stays silent regardless of the user's
# global log level.
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)


# Directories whose contents should never feed the watcher. The two
# crucial ones for infinidev are ``.infinidev`` (sqlite WAL/log/history
# writes during a normal turn would re-trigger the watcher infinitely)
# and ``.git`` (every ``git`` invocation rewrites refs/index). The rest
# are common build/cache dirs that explode the change rate without ever
# being interesting to the indexer.
_IGNORED_DIRS: frozenset[str] = frozenset({
    ".infinidev",
    ".git",
    "__pycache__",
    ".venv", "venv", ".tox",
    "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".eggs",
    ".idea", ".vscode",
    ".DS_Store",
})


# File suffixes that change continuously during normal infinidev
# operation: log files (every emit) and SQLite WAL/SHM/journal sidecars
# (every commit). They are never interesting to the indexer or the UI
# tree — the .db itself is filtered too because watching it has no
# downstream consumer.
_IGNORED_SUFFIXES: frozenset[str] = frozenset({
    ".log",
    ".db", ".db-wal", ".db-shm", ".db-journal",
    ".sqlite", ".sqlite-wal", ".sqlite-shm", ".sqlite-journal",
})


class _InfinidevWatchFilter(DefaultFilter):  # type: ignore[misc, valid-type]
    """Extend ``watchfiles.DefaultFilter`` to ignore infinidev's own
    state directory, generated/cache dirs, and high-churn file types
    (logs, sqlite sidecars).

    ``DefaultFilter`` already skips ``.git``, ``__pycache__``, and
    similar — we union our own list to make the contract explicit and
    add ``.infinidev``, which the upstream filter does not know about.
    The ``__call__`` override layers a suffix check on top so any
    ``*.log`` / ``*.db*`` write — even outside ``.infinidev`` — is
    swallowed before it can fan out to callbacks.
    """

    def __init__(self) -> None:
        super().__init__()
        # ``DefaultFilter.ignore_dirs`` is a tuple; widen it.
        existing = tuple(getattr(self, "ignore_dirs", ()))
        self.ignore_dirs = tuple(set(existing) | _IGNORED_DIRS)

    def __call__(self, change, path: str) -> bool:  # type: ignore[override]
        # Suffix gate first — cheaper than the parent's regex pass and
        # catches the dominant noise sources (logs, sqlite sidecars).
        # ``Path.suffix`` would miss multi-segment suffixes like
        # ``.db-wal``, so check the basename tail explicitly.
        parts = path.split("/")
        name = parts[-1]
        for suf in _IGNORED_SUFFIXES:
            if name.endswith(suf):
                return False
        # ``DefaultFilter.ignore_dirs`` is consulted by watchfiles
        # *during tree pruning* in the Rust side. Once a path has
        # already been emitted (because the watcher started inside an
        # ignored subtree, or because filtering happens after polling),
        # the parent ``__call__`` only checks the basename. Walk every
        # component so ``.infinidev/history/x.json`` is rejected even
        # when reached as a leaf event.
        for component in parts[:-1]:
            if component in _IGNORED_DIRS:
                return False
        return super().__call__(change, path)


class FileWatcher:
    """Monitors a workspace directory for file changes.
    
    Provides callbacks for changed file paths. Only triggers refresh
    if the affected path is within a visible/expanded directory.
    
    Attributes:
        workspace: The root directory to monitor.
        visible_paths: Set of paths that are currently expanded/visible.
        callback: Function called when a change is detected.
    """
    
    def __init__(
        self,
        workspace: str,
        callback: Callable[[str], None],
        visible_paths_callback: Optional[Callable[[], Set[str]]] = None,
        index_callback: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the file watcher.

        Args:
            workspace: Path to the workspace directory to monitor.
            callback: Function to call with the changed file path (visibility-gated).
            visible_paths_callback: Optional function returning set of
                currently visible/expanded paths.
            index_callback: Optional function called for ALL changed files
                regardless of visibility. Used for background indexing.
        """
        self.workspace = pathlib.Path(workspace).resolve()
        self.callback = callback
        self.index_callback = index_callback
        self.visible_paths_callback = visible_paths_callback or (lambda: set())
        
        self._running = False
        self._stop_event = Event()
        self._watch_thread: Optional[Thread] = None
        self._file_set: Set[str] = set()
    
    def _is_path_visible(self, file_path: pathlib.Path) -> bool:
        """Check if a file path is within a visible/expanded directory.
        
        Args:
            file_path: The path to check.
            
        Returns:
            True if the path is visible or matches an explicitly tracked file.
        """
        try:
            # Normalize the path relative to workspace
            relative = file_path.relative_to(self.workspace)
            
            # If we have a visible paths callback, check against it
            if self.visible_paths_callback:
                visible = self.visible_paths_callback()
                for visible_path in visible:
                    try:
                        vp = pathlib.Path(visible_path).resolve()
                        if file_path.resolve().is_relative_to(vp):
                            return True
                    except (ValueError, FileNotFoundError):
                        continue
                return False
            else:
                # No visible paths tracked, only watch workspace root
                return True
                
        except ValueError:
            # Path is not under workspace
            return False
    
    def _should_refresh(self, changed_path: str) -> bool:
        """Determine if a change should trigger a refresh.
        
        Only returns True if the file is within a visible/expanded directory.
        
        Args:
            changed_path: Path to the changed file.
            
        Returns:
            True if refresh is needed.
        """
        try:
            file_path = pathlib.Path(changed_path).resolve()
            return self._is_path_visible(file_path)
        except (ValueError, FileNotFoundError) as e:
            logger.debug(f"Could not check visibility for {changed_path}: {e}")
            return False
    
    def _run_watcher(self):
        """Internal watcher loop running in background thread."""
        logger.info(f"Starting file watcher for {self.workspace}")

        try:
            for changes in watch(
                str(self.workspace),
                stop_event=self._stop_event,
                # Custom filter: skip ``.infinidev``/cache dirs and
                # log / sqlite sidecar files. Without this, infinidev's
                # own writes to ``.infinidev/infinidev.db-wal`` and
                # ``.infinidev/infinidev.log`` re-enter the watcher in
                # a tight self-trigger loop.
                watch_filter=_InfinidevWatchFilter(),
                debounce=500,  # 500ms debounce to batch rapid changes
                debug=False,
            ):
                if self._stop_event.is_set():
                    break
                    
                # Process each change
                for change_type, file_path in changes:
                    if not self._running:
                        break
                        
                    try:
                        fp = pathlib.Path(file_path).resolve()
                        fp_str = str(fp)

                        # Visual callback: only for visible paths
                        if self._should_refresh(fp):
                            logger.debug(f"Change detected in visible path: {file_path}")
                            self.callback(fp_str)
                        else:
                            logger.debug(f"Change detected in hidden path: {file_path}")

                        # Index callback: for ALL changed files (background indexing)
                        if self.index_callback is not None:
                            try:
                                self.index_callback(fp_str)
                            except Exception as ie:
                                logger.debug(f"Index callback error for {fp_str}: {ie}")

                    except Exception as e:
                        logger.error(f"Error processing change {file_path}: {e}")
                        
        except Exception as e:
            logger.error(f"Watcher error: {e}")
        finally:
            logger.info("File watcher stopped")
    
    def start(self) -> bool:
        """Start the file watcher.
        
        Returns:
            True if watcher started successfully, False if watchfiles unavailable.
        """
        if not WATCHFILES_AVAILABLE:
            logger.error("Cannot start file watcher: watchfiles not installed")
            return False
            
        if self._running:
            logger.warning("File watcher already running")
            return True
            
        self._running = True
        self._stop_event.clear()
        self._watch_thread = Thread(target=self._run_watcher, daemon=True)
        self._watch_thread.start()
        
        logger.info(f"File watcher started for {self.workspace}")
        return True
    
    def stop(self):
        """Stop the file watcher gracefully."""
        if not self._running:
            return
            
        self._running = False
        self._stop_event.set()
        
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=2.0)
            
        logger.info("File watcher stopped")
    
    def update_visible_paths(self, visible_paths: Set[str]):
        """Update the set of visible/expanded paths.
        
        Args:
            visible_paths: Set of absolute paths that are currently expanded.
        """
        self._file_set = {pathlib.Path(p).resolve() for p in visible_paths}
        logger.debug(f"Updated visible paths: {len(visible_paths)} paths")
    
    def get_workspace(self) -> str:
        """Get the workspace path being monitored."""
        return str(self.workspace)
    
    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        return self._running
