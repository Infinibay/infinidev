"""Query engine for code intelligence — find definitions, references, symbols."""

from __future__ import annotations

import sqlite3
from typing import Any

from infinidev.code_intel.models import Symbol, SymbolKind, Reference, Import
from infinidev.tools.base.db import execute_with_retry


def _row_to_symbol(row: sqlite3.Row) -> Symbol:
    return Symbol(
        name=row["name"],
        qualified_name=row["qualified_name"],
        kind=SymbolKind(row["kind"]),
        file_path=row["file_path"],
        line_start=row["line_start"],
        line_end=row["line_end"],
        column_start=row["column_start"],
        signature=row["signature"],
        type_annotation=row["type_annotation"],
        docstring=row["docstring"],
        parent_symbol=row["parent_symbol"],
        visibility=row["visibility"],
        is_async=bool(row["is_async"]),
        is_static=bool(row["is_static"]),
        is_abstract=bool(row["is_abstract"]),
        language=row["language"],
    )


def _row_to_reference(row: sqlite3.Row) -> Reference:
    return Reference(
        name=row["name"],
        file_path=row["file_path"],
        line=row["line"],
        column=row["column_num"],
        context=row["context"],
        ref_kind=row["ref_kind"],
        resolved_file=row["resolved_file"],
        resolved_line=row["resolved_line"],
        language=row["language"],
    )


def _row_to_import(row: sqlite3.Row) -> Import:
    return Import(
        source=row["source"],
        name=row["name"],
        alias=row["alias"],
        file_path=row["file_path"],
        line=row["line"],
        is_wildcard=bool(row["is_wildcard"]),
        resolved_file=row["resolved_file"],
        language=row["language"],
    )


def find_definition(
    project_id: int,
    name: str,
    *,
    kind: str | None = None,
    file_hint: str | None = None,
    limit: int = 20,
) -> list[Symbol]:
    """Find where a symbol is defined.

    Accepts both bare names (``connectToVm``) and qualified dotted
    names (``VirtioSocketWatcherService.connectToVm``). When the input
    contains a dot, the query first tries an exact match on
    ``qualified_name``; if that returns nothing, it falls back to
    splitting on the LAST dot and matching ``parent_symbol = X AND
    name = Y``. This second pass exists because some indexers populate
    qualified_name as just the bare name even for class methods (a
    historical bug we're cleaning up). The bare-name fallback at the
    end keeps backwards compatibility with callers that already pass
    short names.
    """
    def _query(conn: sqlite3.Connection):
        # Pass 1: input contains a dot → treat as qualified name.
        if "." in name:
            sql_qual = f"""
                SELECT * FROM ci_symbols
                WHERE project_id = ? AND qualified_name = ?
                {f"AND kind = ?" if kind else ""}
                ORDER BY
                    CASE WHEN kind IN ('function', 'method', 'class') THEN 0 ELSE 1 END,
                    file_path
                LIMIT ?
            """
            params_qual: list[Any] = [project_id, name]
            if kind:
                params_qual.append(kind)
            params_qual.append(limit)
            rows = conn.execute(sql_qual, params_qual).fetchall()
            if rows:
                return [_row_to_symbol(r) for r in rows]

            # Pass 2: split on the LAST dot — last segment is the
            # symbol name, prefix is the parent. Lets us match
            # `VirtioSocketWatcherService.connectToVm` against
            # `(parent_symbol='VirtioSocketWatcherService',
            #   name='connectToVm')` even when qualified_name was
            # only populated as the bare name.
            parent_part, _, leaf = name.rpartition(".")
            sql_parent = f"""
                SELECT * FROM ci_symbols
                WHERE project_id = ?
                  AND parent_symbol = ?
                  AND name = ?
                {f"AND kind = ?" if kind else ""}
                ORDER BY
                    CASE WHEN kind IN ('function', 'method', 'class') THEN 0 ELSE 1 END,
                    file_path
                LIMIT ?
            """
            params_parent: list[Any] = [project_id, parent_part, leaf]
            if kind:
                params_parent.append(kind)
            params_parent.append(limit)
            rows = conn.execute(sql_parent, params_parent).fetchall()
            if rows:
                return [_row_to_symbol(r) for r in rows]

            # Pass 3: bare leaf only — last-resort fuzzy match.
            sql_leaf = f"""
                SELECT * FROM ci_symbols
                WHERE project_id = ? AND name = ?
                {f"AND kind = ?" if kind else ""}
                ORDER BY
                    CASE WHEN kind IN ('function', 'method', 'class') THEN 0 ELSE 1 END,
                    file_path
                LIMIT ?
            """
            params_leaf: list[Any] = [project_id, leaf]
            if kind:
                params_leaf.append(kind)
            params_leaf.append(limit)
            rows = conn.execute(sql_leaf, params_leaf).fetchall()
            return [_row_to_symbol(r) for r in rows]

        # No dot — original bare-name search.
        conditions = ["project_id = ?", "name = ?"]
        params: list[Any] = [project_id, name]
        if kind:
            conditions.append("kind = ?")
            params.append(kind)
        sql = f"""\
            SELECT * FROM ci_symbols
            WHERE {' AND '.join(conditions)}
            ORDER BY
                CASE WHEN kind IN ('function', 'method', 'class') THEN 0 ELSE 1 END,
                file_path
            LIMIT ?
        """
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_symbol(r) for r in rows]

    return execute_with_retry(_query) or []


def find_references(
    project_id: int,
    name: str,
    *,
    ref_kind: str | None = None,
    file_path: str | None = None,
    limit: int = 50,
) -> list[Reference]:
    """Find all usages of a symbol."""
    def _query(conn: sqlite3.Connection):
        conditions = ["project_id = ?", "name = ?"]
        params: list[Any] = [project_id, name]

        if ref_kind:
            conditions.append("ref_kind = ?")
            params.append(ref_kind)
        if file_path:
            conditions.append("file_path = ?")
            params.append(file_path)

        sql = f"""\
            SELECT * FROM ci_references
            WHERE {' AND '.join(conditions)}
            ORDER BY file_path, line
            LIMIT ?
        """
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_reference(r) for r in rows]

    return execute_with_retry(_query) or []


def list_symbols(
    project_id: int,
    file_path: str,
    *,
    kind: str | None = None,
    limit: int = 200,
) -> list[Symbol]:
    """List all symbols defined in a file."""
    def _query(conn: sqlite3.Connection):
        conditions = ["project_id = ?", "file_path = ?"]
        params: list[Any] = [project_id, file_path]

        if kind:
            conditions.append("kind = ?")
            params.append(kind)

        sql = f"""\
            SELECT * FROM ci_symbols
            WHERE {' AND '.join(conditions)}
            ORDER BY line_start
            LIMIT ?
        """
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_symbol(r) for r in rows]

    return execute_with_retry(_query) or []


def get_signature(
    project_id: int,
    name: str,
    *,
    file_path: str | None = None,
    limit: int = 5,
) -> list[Symbol]:
    """Get the full signature of a function/method."""
    def _query(conn: sqlite3.Connection):
        conditions = ["project_id = ?", "name = ?", "kind IN ('function', 'method')"]
        params: list[Any] = [project_id, name]

        order = "file_path"
        if file_path:
            order = f"CASE WHEN file_path = '{file_path}' THEN 0 ELSE 1 END, file_path"

        sql = f"""\
            SELECT * FROM ci_symbols
            WHERE {' AND '.join(conditions)}
            ORDER BY {order}
            LIMIT ?
        """
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_symbol(r) for r in rows]

    return execute_with_retry(_query) or []


def search_symbols(
    project_id: int,
    query: str,
    *,
    kind: str | None = None,
    limit: int = 20,
) -> list[Symbol]:
    """Fuzzy search across all symbol names via FTS5."""
    def _query(conn: sqlite3.Connection):
        # Sanitize FTS query
        fts_query = query.replace('"', '').strip()
        if not fts_query:
            return []

        # Use prefix matching for partial search
        fts_terms = " ".join(f'"{t}"*' for t in fts_query.split() if t)

        conditions = ["s.project_id = ?"]
        params: list[Any] = [project_id]
        if kind:
            conditions.append("s.kind = ?")
            params.append(kind)

        sql = f"""\
            SELECT s.* FROM ci_symbols s
            WHERE s.id IN (
                SELECT rowid FROM ci_symbols_fts WHERE ci_symbols_fts MATCH ?
            )
            AND {' AND '.join(conditions)}
            ORDER BY
                CASE WHEN s.name = ? THEN 0
                     WHEN s.name LIKE ? THEN 1
                     ELSE 2 END
            LIMIT ?
        """
        params = [fts_terms, project_id] + ([kind] if kind else []) + [query, query + "%", limit]

        try:
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_symbol(r) for r in rows]
        except Exception:
            # FTS query syntax error — fallback to LIKE
            like_sql = f"""\
                SELECT * FROM ci_symbols
                WHERE project_id = ? AND name LIKE ?
                {f"AND kind = ?" if kind else ""}
                ORDER BY name
                LIMIT ?
            """
            like_params: list[Any] = [project_id, f"%{query}%"]
            if kind:
                like_params.append(kind)
            like_params.append(limit)
            rows = conn.execute(like_sql, like_params).fetchall()
            return [_row_to_symbol(r) for r in rows]

    return execute_with_retry(_query) or []


def search_by_docstring(
    project_id: int,
    query: str,
    *,
    kind: str | None = None,
    limit: int = 20,
) -> list[tuple[Symbol, float]]:
    """Find symbols by what their docstring/signature SAYS they do.

    Complements :func:`search_symbols` (which is name-based) with an
    intent-based search: instead of "find me everything called
    parseTime", you ask "find me anything that mentions parsing
    timestamps" and the symbol with that phrase in its docstring wins.

    Backed by the existing ``ci_symbols_fts`` virtual table — same
    FTS5 index that powers ``search_symbols``, but here we MATCH
    against the ``docstring`` and ``signature`` columns specifically
    and rank by BM25. The triggers on ``ci_symbols`` keep the index
    fresh on every (re)indexing pass, so callers don't need to do
    anything special.

    Returns a list of ``(Symbol, bm25_rank)`` tuples sorted by
    relevance — lower BM25 score = better match (FTS5 convention).
    Caller can ignore the rank and just use the order, or display it
    for debugging.

    *kind* filters to a specific symbol kind (``"function"``,
    ``"method"``, ``"class"``). Useful for "find me a method that does
    X" vs "find me any code that mentions X".
    """
    def _q(conn: sqlite3.Connection):
        cleaned = query.replace('"', '').strip()
        if not cleaned:
            return []

        # Each token gets a wildcard prefix so partial words match.
        # We OR them together inside the docstring/signature columns
        # so the model doesn't have to remember the exact wording.
        tokens = [t for t in cleaned.split() if t]
        if not tokens:
            return []
        # FTS5 column-restricted query: only search docstring + signature.
        # The {docstring signature ...} prefix limits the MATCH to those
        # columns; otherwise FTS would also match against `name` and
        # we'd get name-based hits polluting the intent-based ranking.
        fts_query = "{docstring signature} : (" + " OR ".join(f'"{t}"*' for t in tokens) + ")"

        conditions = ["s.project_id = ?"]
        params: list[Any] = []
        if kind:
            conditions.append("s.kind = ?")
            params.append(kind)

        sql = f"""\
            SELECT s.*, bm25(ci_symbols_fts) AS rank
            FROM ci_symbols_fts
            JOIN ci_symbols s ON s.id = ci_symbols_fts.rowid
            WHERE ci_symbols_fts MATCH ?
              AND {' AND '.join(conditions)}
            ORDER BY rank
            LIMIT ?
        """
        full_params = [fts_query, project_id] + params + [limit]
        try:
            rows = conn.execute(sql, full_params).fetchall()
        except Exception:
            # If the column-restricted syntax fails (older FTS5 builds),
            # fall back to a plain MATCH and post-filter — slower but
            # works on every SQLite version.
            simple_fts = " OR ".join(f'"{t}"*' for t in tokens)
            sql2 = f"""\
                SELECT s.*, bm25(ci_symbols_fts) AS rank
                FROM ci_symbols_fts
                JOIN ci_symbols s ON s.id = ci_symbols_fts.rowid
                WHERE ci_symbols_fts MATCH ?
                  AND {' AND '.join(conditions)}
                  AND (s.docstring != '' OR s.signature != '')
                ORDER BY rank
                LIMIT ?
            """
            try:
                rows = conn.execute(sql2, [simple_fts, project_id] + params + [limit]).fetchall()
            except Exception:
                return []

        out: list[tuple[Symbol, float]] = []
        for r in rows:
            try:
                sym = _row_to_symbol(r)
            except Exception:
                continue
            try:
                rank = float(r["rank"])
            except (KeyError, IndexError, TypeError, ValueError):
                rank = 0.0
            out.append((sym, rank))
        return out

    return execute_with_retry(_q) or []


def find_imports_of(
    project_id: int,
    name: str,
    *,
    limit: int = 50,
) -> list[Import]:
    """Find all files that import a specific symbol."""
    def _query(conn: sqlite3.Connection):
        rows = conn.execute(
            """\
            SELECT * FROM ci_imports
            WHERE project_id = ? AND name = ?
            ORDER BY file_path
            LIMIT ?
            """,
            (project_id, name, limit),
        ).fetchall()
        return [_row_to_import(r) for r in rows]

    return execute_with_retry(_query) or []


def get_index_stats(project_id: int) -> dict[str, int]:
    """Return stats about the index for a project."""
    def _query(conn: sqlite3.Connection):
        files = conn.execute(
            "SELECT COUNT(*) as c FROM ci_files WHERE project_id = ?", (project_id,)
        ).fetchone()["c"]
        symbols = conn.execute(
            "SELECT COUNT(*) as c FROM ci_symbols WHERE project_id = ?", (project_id,)
        ).fetchone()["c"]
        refs = conn.execute(
            "SELECT COUNT(*) as c FROM ci_references WHERE project_id = ?", (project_id,)
        ).fetchone()["c"]
        imports = conn.execute(
            "SELECT COUNT(*) as c FROM ci_imports WHERE project_id = ?", (project_id,)
        ).fetchone()["c"]
        return {"files": files, "symbols": symbols, "references": refs, "imports": imports}

    return execute_with_retry(_query) or {}
