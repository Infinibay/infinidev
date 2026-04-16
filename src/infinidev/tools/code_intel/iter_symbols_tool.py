"""Tool: walk all indexed symbols, optionally filtered.

Promoted from the ``code_interpreter`` bridge to a flat tool because
the bridge layer added too much indirection for small models. Where
``find_symbols`` requires an FTS5 query and rejects empty input, this
tool answers "give me ALL methods of class Foo" or "every TypeScript
function" with a direct database walk — the right primitive for the
"iterate then filter in my head" workflow that small models naturally
default to.

This complements ``search_symbols`` (FTS5 name search) and
``find_references`` (call/usage scan): use this when you don't have
a search term and just need the full set.
"""

from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class IterSymbolsInput(BaseModel):
    """All filters are optional and AND'd together — empty filters
    return every indexed symbol up to ``limit``."""

    kind: str = Field(
        default="",
        description=(
            "Filter by symbol kind: 'method', 'class', 'function', "
            "'interface', 'enum', 'variable', 'constant'. Empty for all."
        ),
    )
    parent: str = Field(
        default="",
        description=(
            "Exact match on the parent symbol name (the class for "
            "methods). Empty string only matches top-level symbols. "
            "Use this when you want every method of one specific class."
        ),
    )
    language: str = Field(
        default="",
        description=(
            "Restrict by detected language: 'typescript', 'python', "
            "'rust', 'go', 'java', etc. Empty for all languages."
        ),
    )
    file_path: str = Field(
        default="",
        description="Restrict to one specific file (absolute or relative).",
    )
    limit: int = Field(
        default=200,
        description=(
            "Maximum results to return. Default 200; raise to walk "
            "larger result sets. Hard ceiling is the database row limit."
        ),
        ge=1, le=5000,
    )


class IterSymbolsTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "iter_symbols"
    description: str = (
        "Walk all indexed symbols, optionally filtered by kind, parent "
        "class, language, or file path. The right primitive for 'every "
        "method of class X', 'all TypeScript classes', 'every method of "
        "the project'. Returns a compact list — one symbol per line "
        "with kind, qualified name, and file:line. Use this instead of "
        "search_symbols when you don't have a search term."
    )
    args_schema: Type[BaseModel] = IterSymbolsInput

    def _run(
        self,
        kind: str = "",
        parent: str = "",
        language: str = "",
        file_path: str = "",
        limit: int = 200,
    ) -> str:
        import os
        import sqlite3
        from infinidev.tools.base.db import execute_with_retry

        project_id = self.project_id
        if not project_id:
            return self._error("No project context — cannot iterate symbols.")

        # Resolve file_path against the workspace if relative — same
        # convention the other code-intel tools follow.
        if file_path and not os.path.isabs(file_path):
            workspace = self.workspace_path or os.getcwd()
            candidate = os.path.join(workspace, file_path)
            if os.path.exists(candidate):
                file_path = candidate

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 200

        # Direct SELECT on ci_symbols. We don't go through query.py
        # because the existing search_symbols function requires an FTS
        # query — that was the whole reason this tool exists.
        def _q(conn: sqlite3.Connection):
            conditions = ["project_id = ?"]
            params: list = [project_id]
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
                "SELECT name, qualified_name, kind, file_path, "
                "line_start, line_end, parent_symbol, signature, "
                "language, visibility "
                f"FROM ci_symbols WHERE {' AND '.join(conditions)} "
                "ORDER BY file_path, line_start LIMIT ?"
            )
            return conn.execute(sql, params).fetchall()

        try:
            rows = execute_with_retry(_q) or []
        except Exception as exc:
            return self._error(f"Query failed: {exc}")

        if not rows:
            filters = []
            if kind: filters.append(f"kind={kind}")
            if parent: filters.append(f"parent={parent}")
            if language: filters.append(f"language={language}")
            if file_path: filters.append(f"file={file_path}")
            f_str = ", ".join(filters) if filters else "no filters"
            return self._error(
                f"No symbols matched ({f_str}). The index may be empty "
                "for this project — try /reindex first."
            )

        lines = []
        for r in rows:
            name, qual, k, fp, ls, le, parent_sym, sig, lang, vis = r
            display_name = qual or name
            location = f"{fp}:{ls}"
            extra = ""
            if vis and vis != "public":
                extra = f" [{vis}]"
            lines.append(f"  {k:9}  {display_name}{extra}  @ {location}")

        header = (
            f"Found {len(rows)} symbol(s)"
            + (f" of kind={kind!r}" if kind else "")
            + (f" in class {parent!r}" if parent else "")
            + (f" ({language})" if language else "")
            + ":"
        )
        return header + "\n" + "\n".join(lines)
