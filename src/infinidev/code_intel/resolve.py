"""Symbol resolution — find a symbol by qualified name with smart indexing.

Resolves "ClassName.method_name" or "top_level_func" to a Symbol
with exact file location (line_start, line_end).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from infinidev.code_intel.models import Symbol
from infinidev.code_intel.smart_index import ensure_indexed

logger = logging.getLogger(__name__)


@dataclass
class ResolveResult:
    """Result of symbol resolution."""
    symbol: Symbol | None = None
    error: str = ""
    candidates: list[Symbol] | None = None


def resolve_symbol(
    project_id: int,
    symbol: str,
    file_path: str | None = None,
) -> ResolveResult:
    """Resolve a qualified symbol name to a Symbol with line range.

    Args:
        project_id: Project ID for the index
        symbol: Qualified name like "ClassName.method" or "func_name"
        file_path: Optional file hint to narrow search

    Returns:
        ResolveResult with symbol, error message, or candidate list.
    """
    from infinidev.code_intel.query import find_definition

    # Ensure file is indexed if we have a path
    if file_path:
        abs_path = os.path.abspath(file_path) if not os.path.isabs(file_path) else file_path
        ensure_indexed(project_id, abs_path)

    # Split qualified name: "Class.method" → name="method", parent="Class"
    parts = symbol.rsplit(".", 1)
    if len(parts) == 2:
        parent_name, method_name = parts
    else:
        parent_name, method_name = None, parts[0]

    # Query for the symbol
    results = find_definition(project_id, method_name)

    if not results:
        # Try indexing the workspace and retrying
        if file_path:
            return ResolveResult(error=f"Symbol '{symbol}' not found in index. File may need indexing.")
        return ResolveResult(error=f"Symbol '{symbol}' not found in index.")

    # Filter by parent if qualified
    if parent_name:
        filtered = [s for s in results if s.parent_symbol == parent_name]
        if not filtered:
            # Try with qualified_name
            filtered = [s for s in results if s.qualified_name == symbol]
        if filtered:
            results = filtered

    # Filter by file if specified
    if file_path:
        abs_path = os.path.abspath(file_path)
        file_filtered = [s for s in results if s.file_path == abs_path]
        if file_filtered:
            results = file_filtered

    if len(results) == 0:
        return ResolveResult(error=f"Symbol '{symbol}' not found after filtering.")

    if len(results) == 1:
        return ResolveResult(symbol=results[0])

    # Multiple matches — prefer methods/functions over variables
    preferred = [s for s in results if s.kind.value in ("method", "function", "class")]
    if len(preferred) == 1:
        return ResolveResult(symbol=preferred[0])

    # Ambiguous — return candidates
    return ResolveResult(
        error=f"Ambiguous symbol '{symbol}': {len(results)} matches. Specify file_path to disambiguate.",
        candidates=results,
    )
