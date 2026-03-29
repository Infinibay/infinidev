"""Initial workspace indexing — runs before the LLM starts working.

On first open (or when the index is stale), indexes all source files so
that code intelligence tools (symbols, references, etc.) are available
from the very first LLM interaction.

Uses the same hash-based skip logic as ensure_indexed(), so subsequent
runs are near-instant if nothing changed.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)


def run_initial_index(
    project_id: int = 1,
    workspace: str | None = None,
    on_progress: callable | None = None,
) -> dict[str, int]:
    """Index the workspace before the LLM starts.

    Args:
        project_id: Project ID for the index (default 1).
        workspace: Directory to index (default: cwd).
        on_progress: Optional callback(message: str) for status updates.
            Called with progress messages like "Indexing... 42 files, 380 symbols".

    Returns:
        Stats dict: {files_indexed, symbols_total, files_skipped, elapsed_ms}
    """
    workspace = workspace or os.getcwd()

    def _log(msg: str):
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    _log("Indexing workspace...")

    from infinidev.code_intel.smart_index import ensure_directory_indexed
    stats = ensure_directory_indexed(project_id, workspace)

    elapsed_ms = stats.get("elapsed_ms", 0)
    files = stats.get("files_indexed", 0)
    symbols = stats.get("symbols_total", 0)
    skipped = stats.get("files_skipped", 0)

    if files > 0:
        _log(
            f"Index ready: {files} files, {symbols} symbols "
            f"({elapsed_ms}ms, {skipped} skipped)"
        )
    else:
        _log(f"Index up to date ({elapsed_ms}ms)")

    return stats
