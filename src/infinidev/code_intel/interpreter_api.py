"""Read-only code-intelligence API exposed to the sandboxed interpreter.

Gives the model a small, curated surface of query functions it can
call from within ``code_interpreter`` scripts. The goal: let the
agent combine queries in ways no fixed tool supports — "find every
method that calls both X and Y", "rank classes by method count",
"trace the call graph two levels out from method Z", etc.

Design rules:

  1. **Read-only.** Every function in this module queries the
     existing code intelligence DB (``ci_symbols``, ``ci_references``,
     ``ci_method_bodies``). Nothing mutates state. No INSERT /
     UPDATE / DELETE. No calls into the indexer. No access to the
     write side of any tool.

  2. **Plain Python data, not dataclasses.** All functions return
     ``dict`` / ``list[dict]`` — never ``Symbol`` / ``Reference`` /
     ``ParsedFailure`` etc. The model gets ``result["name"]``
     everywhere; no ``result.name`` with dataclass introspection.
     Makes the output trivially JSON-serialisable and avoids teaching
     the model the shape of every internal class.

  3. **Project scope is ambient.** ``project_id`` and
     ``workspace_path`` come from environment variables
     ``INFINIDEV_PROJECT_ID`` and ``INFINIDEV_WORKSPACE_PATH`` which
     the ``code_interpreter`` tool sets in the subprocess env. The
     model never needs to pass these — they're invisible. This also
     makes it impossible for the model to query a DIFFERENT project's
     data by accident.

  4. **No __builtins__ stripping.** The subprocess is already
     sandboxed by the ``code_interpreter`` tool (timeout, process
     isolation, no network by default). This module does not try to
     re-implement a security boundary on top of that — it just
     provides a clean API.

  5. **Import-time initialisation is cheap.** Opening a SQLite
     connection is ~1 ms; all heavy work is deferred to function
     calls. A ``code_interpreter`` script that never calls any API
     function pays ~0 ms of import cost.

Usage from a code_interpreter script (the tool auto-prepends the
import line, but it's equivalent to this):

    from infinidev.code_intel.interpreter_api import (
        find_symbols, find_references, get_source, find_similar,
        search_by_intent, extract_skeleton, list_file_symbols,
        list_files, project_stats,
    )

    # "Which methods call connectToVm but not disconnect?"
    callers_of_connect = {
        r["file_path"] + ":" + str(r["line"])
        for r in find_references("connectToVm")
    }
    callers_of_disconnect = {
        r["file_path"] + ":" + str(r["line"])
        for r in find_references("disconnect")
    }
    lifecycle = callers_of_connect - callers_of_disconnect
    print(f"{len(lifecycle)} methods connect but never disconnect")
"""

from __future__ import annotations

import os
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Ambient scope — resolved once at import time
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_project_id() -> int:
    """Pick up project_id from env; default to 1 (the CLI default)."""
    raw = os.environ.get("INFINIDEV_PROJECT_ID", "1")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 1


def _resolve_workspace_path() -> str:
    """Pick up workspace path from env; default to cwd."""
    return os.environ.get("INFINIDEV_WORKSPACE_PATH", os.getcwd())


_PROJECT_ID = _resolve_project_id()
_WORKSPACE_PATH = _resolve_workspace_path()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: convert internal objects to plain dicts
# ─────────────────────────────────────────────────────────────────────────────


def _symbol_to_dict(sym: Any) -> dict:
    """Flatten a :class:`Symbol` instance into a plain dict.

    The enum ``SymbolKind`` gets stringified via ``.value`` so the
    model sees ``"method"`` instead of ``<SymbolKind.method: 'method'>``.
    Boolean flags are preserved. Empty strings are NOT stripped — the
    model may want to filter by `parent_symbol == ""` etc.
    """
    try:
        kind = sym.kind.value if hasattr(sym.kind, "value") else str(sym.kind)
    except Exception:
        kind = ""
    return {
        "name": sym.name,
        "qualified_name": sym.qualified_name,
        "kind": kind,
        "file_path": sym.file_path,
        "line_start": sym.line_start,
        "line_end": sym.line_end,
        "signature": sym.signature,
        "docstring": sym.docstring,
        "parent_symbol": sym.parent_symbol,
        "visibility": sym.visibility,
        "is_async": bool(sym.is_async),
        "is_static": bool(sym.is_static),
        "is_abstract": bool(sym.is_abstract),
        "language": sym.language,
    }


def _reference_to_dict(ref: Any) -> dict:
    """Flatten a :class:`Reference` instance into a plain dict."""
    return {
        "name": ref.name,
        "file_path": ref.file_path,
        "line": ref.line,
        "column": ref.column,
        "context": ref.context,
        "ref_kind": ref.ref_kind,
        "language": ref.language,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public query API (read-only)
# ─────────────────────────────────────────────────────────────────────────────


def find_symbols(query: str, kind: str = "", limit: int = 50) -> list[dict]:
    """Search symbols by name across the project (FTS5 + prefix match).

    *query* can be a full name, a prefix, or multiple words — the
    underlying FTS5 index handles all three. *kind* is an optional
    filter (``"function"``, ``"method"``, ``"class"``, etc.). Returns
    the matched symbols as plain dicts, sorted with exact-name
    matches first and then alphabetically.

    Equivalent to the ``search_symbols`` tool, but available inside
    a code_interpreter script where you can pipe the results through
    arbitrary Python.
    """
    from infinidev.code_intel.query import search_symbols
    results = search_symbols(
        _PROJECT_ID, query, kind=kind or None, limit=limit,
    )
    return [_symbol_to_dict(s) for s in results]


def find_definitions(name: str, kind: str = "") -> list[dict]:
    """Find every place *name* is defined.

    Accepts both bare names (``"connectToVm"``) and qualified dotted
    names (``"VirtioSocketWatcherService.connectToVm"``). Returns an
    empty list when the symbol isn't indexed.
    """
    from infinidev.code_intel.query import find_definition
    results = find_definition(
        _PROJECT_ID, name, kind=kind or None,
    )
    return [_symbol_to_dict(s) for s in results]


def find_references(
    name: str, file_path: str = "", ref_kind: str = "", limit: int = 200,
) -> list[dict]:
    """Find every call / usage of *name* across the project.

    *ref_kind* filters by kind — ``"call"`` for method calls,
    ``"usage"`` for property reads, ``"import"`` for import references.
    Leave empty to get all. *file_path* narrows to a single file.
    """
    from infinidev.code_intel.query import find_references as _find_references
    results = _find_references(
        _PROJECT_ID, name,
        ref_kind=ref_kind or None,
        file_path=file_path or None,
        limit=limit,
    )
    return [_reference_to_dict(r) for r in results]


def list_file_symbols(file_path: str, kind: str = "") -> list[dict]:
    """Return every symbol defined in *file_path*.

    *file_path* can be relative to the workspace or absolute.
    *kind* filters to a specific kind (``"method"``, ``"class"``,
    etc.). Useful for "give me every method of file X" without
    having to scan the full search index.
    """
    from infinidev.code_intel.query import list_symbols
    # Resolve relative paths against the workspace so the model can
    # use the paths it sees in other tool results directly.
    if file_path and not os.path.isabs(file_path):
        candidate = os.path.join(_WORKSPACE_PATH, file_path)
        if os.path.exists(candidate):
            file_path = candidate
    results = list_symbols(
        _PROJECT_ID, file_path, kind=kind or None,
    )
    return [_symbol_to_dict(s) for s in results]


def get_source(qualified_name: str, file_path: str = "") -> str:
    """Return the source code of a symbol, by qualified name.

    Equivalent to the ``get_symbol_code`` tool. *file_path* is an
    optional disambiguator when two files have a symbol with the
    same name. Returns the source as a single string with line
    numbers prefixed, or an empty string when the symbol isn't found.
    """
    from infinidev.code_intel.query import find_definition
    syms = find_definition(
        _PROJECT_ID, qualified_name, file_hint=file_path or None,
    )
    if not syms:
        return ""
    sym = syms[0]
    try:
        with open(sym.file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return ""
    start = max(0, sym.line_start - 1)
    end = min(len(lines), (sym.line_end or sym.line_start + 30))
    numbered = "".join(
        f"{start + i + 1:5d}\t{line}" for i, line in enumerate(lines[start:end])
    )
    return numbered


def find_similar(
    qualified_name: str,
    file_path: str = "",
    threshold: float = 0.7,
    limit: int = 10,
) -> list[dict]:
    """Return methods whose body looks like *qualified_name*'s body.

    Uses the normalized-token Jaccard similarity from the method
    index. Each result is a dict with the match's file path, line
    range, similarity score in [0, 1], and an ``is_exact_dup`` flag
    for copy-paste duplicates (Jaccard = 1.0 via hash match).
    """
    from infinidev.code_intel.method_index import find_similar as _find_similar
    hits = _find_similar(
        _PROJECT_ID,
        qualified_name=qualified_name,
        file_path=file_path or None,
        threshold=threshold,
        limit=limit,
    )
    return [h.to_dict() for h in hits]


def search_by_intent(
    query: str, kind: str = "", limit: int = 20,
) -> list[dict]:
    """Find symbols by what they DO — searches docstrings + signatures.

    Natural-language query (``"parse timestamp"``, ``"validate email"``)
    matched via FTS5 BM25 against the docstring and signature columns
    of every indexed symbol. Complements :func:`find_symbols` (name-
    based) — use this when you don't know what the symbol is called
    but you know what it should do.

    Each result is a tuple dict with the symbol + a ``bm25`` score;
    lower scores are better matches (FTS5 convention).
    """
    from infinidev.code_intel.query import search_by_docstring
    results = search_by_docstring(
        _PROJECT_ID, query, kind=kind or None, limit=limit,
    )
    out: list[dict] = []
    for sym, rank in results:
        d = _symbol_to_dict(sym)
        d["bm25"] = rank
        out.append(d)
    return out


def extract_skeleton(file_path: str) -> dict:
    """Return the structured skeleton of a single file.

    Reads the file from disk (not the index), runs the tree-sitter
    skeleton extractor, and returns a dict with:

      * ``file_path``
      * ``language``
      * ``total_lines``, ``total_bytes``
      * ``symbols``: list of ``{kind, name, line_start, line_end, doc}``

    Supports every language the indexer supports (14 total). Returns
    an empty skeleton dict when the file doesn't exist or has no
    parseable symbols.
    """
    if file_path and not os.path.isabs(file_path):
        candidate = os.path.join(_WORKSPACE_PATH, file_path)
        if os.path.exists(candidate):
            file_path = candidate
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return {
            "file_path": file_path,
            "language": "",
            "total_lines": 0,
            "total_bytes": 0,
            "symbols": [],
        }

    total_lines = text.count("\n") + 1
    total_bytes = len(text.encode("utf-8"))
    try:
        from infinidev.code_intel.syntax_check import extract_file_skeleton
        entries, language = extract_file_skeleton(text, file_path=file_path)
    except Exception:
        entries, language = [], ""
    return {
        "file_path": file_path,
        "language": language,
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        "symbols": [e.to_dict() for e in entries],
    }


def iter_symbols(
    kind: str = "",
    parent: str = "",
    language: str = "",
    file_path: str = "",
    limit: int = 5000,
) -> list[dict]:
    """Iterate over all indexed symbols, optionally filtered.

    Unlike :func:`find_symbols` (which requires a search query and
    goes through FTS5), this function does a direct SELECT on
    ``ci_symbols`` — it's the right tool when you want to walk the
    full set of methods / classes / functions in the project and
    filter in Python.

    Filters:
      * ``kind`` — ``"method"``, ``"class"``, ``"function"``, etc.
      * ``parent`` — exact match on ``parent_symbol`` (the class
        name for methods). Empty string means top-level symbols.
      * ``language`` — ``"typescript"``, ``"python"``, etc.
      * ``file_path`` — restrict to one file (absolute or relative
        to workspace).

    All filters are AND'd together. Returns up to *limit* symbols as
    plain dicts. The default limit (5000) is high because iteration
    is the primary use case — a typical medium repo has 2-3k methods
    and you want all of them in one go.

    Example — "count methods per class in the project":

        from collections import Counter
        methods = iter_symbols(kind="method")
        by_class = Counter(m["parent_symbol"] for m in methods if m["parent_symbol"])
        for name, n in by_class.most_common(10):
            print(f"{n:3}  {name}")
    """
    from infinidev.tools.base.db import execute_with_retry
    import sqlite3

    if file_path and not os.path.isabs(file_path):
        candidate = os.path.join(_WORKSPACE_PATH, file_path)
        if os.path.exists(candidate):
            file_path = candidate

    def _q(conn: sqlite3.Connection):
        conditions = ["project_id = ?"]
        params: list[Any] = [_PROJECT_ID]
        if kind:
            conditions.append("kind = ?")
            params.append(kind)
        if parent:
            conditions.append("parent_symbol = ?")
            params.append(parent)
        if language:
            conditions.append("language = ?")
            params.append(language)
        if file_path:
            conditions.append("file_path = ?")
            params.append(file_path)
        params.append(limit)
        sql = (
            "SELECT name, qualified_name, kind, file_path, line_start, "
            "line_end, signature, type_annotation, docstring, parent_symbol, "
            "visibility, is_async, is_static, is_abstract, language "
            f"FROM ci_symbols WHERE {' AND '.join(conditions)} "
            "ORDER BY file_path, line_start LIMIT ?"
        )
        return conn.execute(sql, params).fetchall()

    try:
        rows = execute_with_retry(_q) or []
    except Exception:
        return []

    # Translate rows to plain dicts with the same shape as
    # _symbol_to_dict. We bypass the internal Symbol constructor
    # because this function doesn't need the enum conversion.
    out: list[dict] = []
    for r in rows:
        out.append({
            "name": r[0],
            "qualified_name": r[1],
            "kind": r[2],
            "file_path": r[3],
            "line_start": r[4],
            "line_end": r[5],
            "signature": r[6],
            "type_annotation": r[7],
            "docstring": r[8],
            "parent_symbol": r[9],
            "visibility": r[10],
            "is_async": bool(r[11]),
            "is_static": bool(r[12]),
            "is_abstract": bool(r[13]),
            "language": r[14],
        })
    return out


def list_files(language: str = "") -> list[str]:
    """List every indexed file in the project, optionally filtered by language.

    Returns absolute file paths, sorted. Useful for iterating — e.g.
    "for every Python file, count the methods". Skips files that
    haven't been indexed yet; call ``/reindex`` first if you want a
    full list.
    """
    from infinidev.tools.base.db import execute_with_retry
    import sqlite3

    def _q(conn: sqlite3.Connection):
        if language:
            return conn.execute(
                "SELECT DISTINCT file_path FROM ci_files "
                "WHERE project_id = ? AND language = ? ORDER BY file_path",
                (_PROJECT_ID, language),
            ).fetchall()
        return conn.execute(
            "SELECT DISTINCT file_path FROM ci_files "
            "WHERE project_id = ? ORDER BY file_path",
            (_PROJECT_ID,),
        ).fetchall()

    try:
        rows = execute_with_retry(_q) or []
    except Exception:
        return []
    return [r[0] for r in rows]


def project_stats() -> dict:
    """Summary statistics for the current project's index.

    Returns a dict with:
      * ``total_files``
      * ``total_symbols``, ``symbols_by_kind``
      * ``total_references``
      * ``total_method_bodies`` (in ci_method_bodies)
      * ``languages``: list of distinct languages seen

    Meant as a "what's here?" probe the model runs at the start of
    an analysis script to orient itself.
    """
    from infinidev.tools.base.db import execute_with_retry
    import sqlite3

    def _q(conn: sqlite3.Connection):
        out = {}
        out["total_files"] = conn.execute(
            "SELECT COUNT(*) FROM ci_files WHERE project_id = ?",
            (_PROJECT_ID,),
        ).fetchone()[0]
        out["total_symbols"] = conn.execute(
            "SELECT COUNT(*) FROM ci_symbols WHERE project_id = ?",
            (_PROJECT_ID,),
        ).fetchone()[0]
        out["total_references"] = conn.execute(
            "SELECT COUNT(*) FROM ci_references WHERE project_id = ?",
            (_PROJECT_ID,),
        ).fetchone()[0]
        out["total_method_bodies"] = conn.execute(
            "SELECT COUNT(*) FROM ci_method_bodies WHERE project_id = ?",
            (_PROJECT_ID,),
        ).fetchone()[0]
        by_kind = conn.execute(
            "SELECT kind, COUNT(*) FROM ci_symbols WHERE project_id = ? "
            "GROUP BY kind",
            (_PROJECT_ID,),
        ).fetchall()
        out["symbols_by_kind"] = {k: c for k, c in by_kind}
        langs = conn.execute(
            "SELECT DISTINCT language FROM ci_files WHERE project_id = ?",
            (_PROJECT_ID,),
        ).fetchall()
        out["languages"] = sorted([l[0] for l in langs if l[0]])
        return out

    try:
        return execute_with_retry(_q) or {}
    except Exception:
        return {}


__all__ = [
    "find_symbols",
    "find_definitions",
    "find_references",
    "list_file_symbols",
    "iter_symbols",
    "get_source",
    "find_similar",
    "search_by_intent",
    "extract_skeleton",
    "list_files",
    "project_stats",
]
