"""Smart reindexing — ensure files are indexed before symbol operations.

Provides `ensure_indexed()` which any tool can call before querying symbols.
Uses content hash comparison to skip unchanged files.
"""

from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)


def ensure_indexed(project_id: int, file_path: str) -> bool:
    """Ensure a file is indexed with its current content.

    Compares the stored content hash with the current file content.
    If different (or not yet indexed), re-indexes the file.

    Args:
        project_id: Project ID for the index
        file_path: Absolute or relative path to the file

    Returns:
        True if the file was (re)indexed, False if already up-to-date or skipped.
    """
    if not file_path or not os.path.isfile(file_path):
        return False

    try:
        abs_path = os.path.abspath(file_path)
    except (OSError, ValueError):
        return False

    # Read current content and hash
    try:
        with open(abs_path, "rb") as f:
            content = f.read()
    except (PermissionError, OSError):
        return False

    current_hash = hashlib.sha256(content).hexdigest()[:16]

    # Compare with stored hash
    from infinidev.code_intel.index import get_file_hash
    stored_hash = get_file_hash(project_id, abs_path)

    if stored_hash == current_hash:
        return False  # Already up-to-date

    # Needs (re)indexing
    try:
        from infinidev.code_intel.indexer import index_file
        count = index_file(project_id, abs_path)
        if count > 0:
            logger.debug("Reindexed %s: %d symbols", abs_path, count)
        return count > 0
    except Exception as exc:
        logger.warning("Failed to index %s: %s", abs_path, str(exc)[:100])
        return False


def ensure_directory_indexed(project_id: int, dir_path: str) -> dict[str, int]:
    """Ensure all supported files in a directory are indexed.

    Only indexes files that have changed since last indexing.

    Returns:
        Dict with stats: {files_indexed, files_skipped, symbols_total}
    """
    if not dir_path or not os.path.isdir(dir_path):
        return {"files_indexed": 0, "files_skipped": 0, "symbols_total": 0}

    from infinidev.code_intel.indexer import index_directory
    try:
        return index_directory(project_id, dir_path)
    except Exception as exc:
        logger.warning("Failed to index directory %s: %s", dir_path, str(exc)[:100])
        return {"files_indexed": 0, "files_skipped": 0, "symbols_total": 0}
