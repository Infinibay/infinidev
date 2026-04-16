"""Tool: find symbols by what their docstring/signature says they DO.

This is the intent-based counterpart to ``search_symbols`` (which is
name-based) and ``find_similar_methods`` (which is body-based). The
three together cover the full grid of "find me code I might want to
reuse":

  • search_symbols          → I know what it's CALLED.
  • search_by_docstring     → I know what it DOES (in words).
  • find_similar_methods    → I know what it LOOKS LIKE (in code).

Backed by the existing ``ci_symbols_fts`` virtual table — no new
indexes, no new triggers. The docstring column was already being
populated by the language parsers; this tool just exposes it.
"""

from __future__ import annotations

from typing import Type
from pydantic import BaseModel, Field, field_validator

from infinidev.tools.base.base_tool import InfinibayBaseTool


class SearchByDocstringInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Words describing what the code DOES, e.g. 'parse timestamp', "
            "'validate email format', 'retry on failure with backoff'. "
            "Multiple words are OR'd together — every match counts. Use "
            "natural phrases the original author might have written in a "
            "docstring or comment, not exact identifier names."
        ),
        min_length=1,
    )
    kind: str = Field(
        default="",
        description=(
            "Optional filter by symbol kind: 'function', 'method', 'class'. "
            "Leave empty to search across all kinds."
        ),
    )
    limit: int = Field(
        default=15,
        description="Maximum number of results (1-50).",
        ge=1, le=50,
    )

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "query must not be empty — describe what the code should do"
            )
        return v


class SearchByDocstringTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "search_by_docstring"
    description: str = (
        "Find functions/methods/classes by what they DO (intent), not what "
        "they're CALLED (name). Searches docstrings and signatures via FTS5 "
        "with BM25 ranking. Use natural-language phrases like 'parse JSON "
        "from a stream' or 'validate email address'. Complements "
        "search_symbols (name-based) and find_similar_methods (body-based) — "
        "use this one when you don't know what's already in the project but "
        "you know what behaviour you need."
    )
    args_schema: Type[BaseModel] = SearchByDocstringInput

    def _run(self, query: str, kind: str = "", limit: int = 15) -> str:
        from infinidev.code_intel.query import search_by_docstring
        from infinidev.code_intel.indexer import index_directory

        project_id = self.project_id
        workspace = self.workspace_path
        if not project_id:
            return self._error("No project context — cannot query the symbol index.")

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 15

        kind_filter = (kind or "").strip().lower() or None
        if kind_filter and kind_filter not in ("function", "method", "class", "variable", "constant"):
            return self._error(
                f"Invalid kind '{kind_filter}'. Use one of: function, method, "
                "class, variable, constant — or leave empty for all."
            )

        results = search_by_docstring(
            project_id, query, kind=kind_filter, limit=limit,
        )
        if not results and workspace:
            # Auto-index on first query, same pattern as search_symbols.
            index_directory(project_id, workspace)
            results = search_by_docstring(
                project_id, query, kind=kind_filter, limit=limit,
            )

        if not results:
            return (
                f"No symbols found whose docstring or signature matches "
                f"'{query}'"
                + (f" (kind={kind_filter})" if kind_filter else "")
                + ". Most files in this project may not have docstrings — "
                "try search_symbols for name-based search instead, or "
                "code_search for full-text search across all source."
            )

        lines = [
            f"Found {len(results)} symbol(s) matching intent '{query}'"
            + (f" (kind={kind_filter})" if kind_filter else "")
            + " — sorted by BM25 relevance:"
        ]
        for sym, rank in results:
            parent = f" (in {sym.parent_symbol})" if sym.parent_symbol else ""
            sig = sym.signature or sym.name
            doc_preview = ""
            if sym.docstring:
                # First sentence or first 100 chars, whichever is shorter.
                doc = " ".join(sym.docstring.split())
                if len(doc) > 100:
                    doc = doc[:97] + "..."
                doc_preview = f"\n        — {doc}"
            kind_label = sym.kind.value if hasattr(sym.kind, "value") else str(sym.kind)
            lines.append(
                f"  {kind_label:10} {sig}  "
                f"@ {sym.file_path}:{sym.line_start}{parent}{doc_preview}"
            )

        return "\n".join(lines)
