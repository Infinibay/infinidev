"""Tool: fuzzy search across all symbol names."""

from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.code_intel.search_symbols_input import SearchSymbolsInput


class SearchSymbolsTool(InfinibayBaseTool):
    name: str = "search_symbols"
    description: str = "Search symbols by name across the project. Supports partial matching."
    args_schema: Type[BaseModel] = SearchSymbolsInput

    def _run(self, query: str, kind: str = "") -> str:
        from infinidev.code_intel.query import search_symbols
        from infinidev.code_intel.indexer import index_directory

        project_id = self.project_id
        workspace = self.workspace_path

        results = search_symbols(project_id, query, kind=kind or None)
        if not results and workspace:
            index_directory(project_id, workspace)
            results = search_symbols(project_id, query, kind=kind or None)

        if not results:
            return self._error(f"No symbols matching '{query}'")

        lines = []
        for s in results:
            parent = f" (in {s.parent_symbol})" if s.parent_symbol else ""
            sig = s.signature or s.name
            lines.append(f"{s.kind.value:10} {sig} — {s.file_path}:{s.line_start}{parent}")

        return f"Found {len(results)} symbol(s) matching '{query}':\n" + "\n".join(lines)

