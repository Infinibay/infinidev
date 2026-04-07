"""Import resolution engine for heuristic analysis.

Resolves Python imports against the code index and filesystem
to classify them as stdlib, local, third-party, or unresolved.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass

from infinidev.code_intel.models import Import
from infinidev.code_intel.stdlib_modules import get_stdlib_modules

logger = logging.getLogger(__name__)

_STDLIB = get_stdlib_modules()

# Python builtins that can appear as import sources but aren't modules
_SKIP_SOURCES = frozenset({"__future__"})


@dataclass
class ResolvedImport:
    """Result of resolving an import."""
    status: str  # "stdlib", "local", "third_party", "unresolved"
    resolved_path: str = ""
    reason: str = ""


def resolve_import(
    project_id: int,
    imp: Import,
    workspace: str,
) -> ResolvedImport:
    """Resolve an import to determine its origin.

    Returns a ResolvedImport with status indicating whether the import
    is from stdlib, a local module, a third-party package, or unresolved.
    """
    source = imp.source
    if not source:
        return ResolvedImport("unresolved", reason="empty source")

    top_level = source.lstrip(".").split(".")[0]

    # 1. Stdlib check
    if top_level in _STDLIB:
        return ResolvedImport("stdlib")

    # 2. Relative imports (start with .)
    if source.startswith("."):
        return _resolve_relative(imp, workspace)

    # 3. Local module resolution
    local = _resolve_local(source, imp.name, workspace, project_id)
    if local is not None:
        return local

    # 4. Third-party check via importlib
    try:
        spec = importlib.util.find_spec(top_level)
        if spec is not None:
            return ResolvedImport("third_party", reason=f"installed package: {top_level}")
    except (ModuleNotFoundError, ValueError):
        pass

    return ResolvedImport("unresolved", reason=f"module '{source}' not found")


def _resolve_relative(imp: Import, workspace: str) -> ResolvedImport:
    """Resolve a relative import (starts with .)."""
    if not imp.file_path:
        return ResolvedImport("unresolved", reason="no file context for relative import")

    source = imp.source
    # Count leading dots
    dots = 0
    for ch in source:
        if ch == ".":
            dots += 1
        else:
            break

    # Navigate up from the file's directory
    base_dir = os.path.dirname(os.path.abspath(imp.file_path))
    for _ in range(dots - 1):
        base_dir = os.path.dirname(base_dir)

    # Resolve remaining dotted path
    remaining = source[dots:]
    if remaining:
        parts = remaining.split(".")
        candidate = os.path.join(base_dir, *parts)
    else:
        candidate = base_dir

    # Check as file or package
    for path in [candidate + ".py", os.path.join(candidate, "__init__.py")]:
        if os.path.isfile(path):
            return ResolvedImport("local", resolved_path=path)

    return ResolvedImport("unresolved", reason=f"relative import '{source}' not found")


def _resolve_local(
    source: str,
    name: str,
    workspace: str,
    project_id: int,
) -> ResolvedImport | None:
    """Try to resolve as a local module within the workspace."""
    parts = source.split(".")

    # Try direct file path: foo.bar → foo/bar.py or foo/bar/__init__.py
    candidate = os.path.join(workspace, *parts)
    for path in [candidate + ".py", os.path.join(candidate, "__init__.py")]:
        if os.path.isfile(path):
            return ResolvedImport("local", resolved_path=path)

    # Try src/ layout: src/foo/bar.py
    src_candidate = os.path.join(workspace, "src", *parts)
    for path in [src_candidate + ".py", os.path.join(src_candidate, "__init__.py")]:
        if os.path.isfile(path):
            return ResolvedImport("local", resolved_path=path)

    # Check ci_files for indexed files matching the module path
    try:
        from infinidev.code_intel.index import get_file_hash
        from infinidev.code_intel._db import execute_with_retry

        # Search for file paths ending with the module pattern
        module_suffix = os.sep.join(parts) + ".py"
        pkg_suffix = os.sep.join(parts) + os.sep + "__init__.py"

        def _search(conn):
            rows = conn.execute(
                "SELECT file_path FROM ci_files WHERE project_id = ? "
                "AND (file_path LIKE ? OR file_path LIKE ?)",
                (project_id, f"%{module_suffix}", f"%{pkg_suffix}"),
            ).fetchall()
            return [r[0] for r in rows]

        matches = execute_with_retry(_search)
        if matches:
            return ResolvedImport("local", resolved_path=matches[0])
    except Exception:
        pass

    return None
