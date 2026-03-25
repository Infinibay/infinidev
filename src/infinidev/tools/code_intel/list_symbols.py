"""Tool: list all symbols in a file."""

from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class ListSymbolsInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to list symbols from")
    kind: str = Field(
        default="",
        description="Optional filter: 'function', 'method', 'class', 'variable'",
    )


class ListSymbolsTool(InfinibayBaseTool):
    name: str = "list_symbols"
    description: str = (
        "List all functions, classes, methods, and variables defined in a file. "
        "Returns a structured overview without reading the entire file. "
        "Use to quickly understand a file's contents and structure."
    )
    args_schema: Type[BaseModel] = ListSymbolsInput

    def _run(self, file_path: str, kind: str = "") -> str:
        import os
        from infinidev.code_intel.query import list_symbols
        from infinidev.code_intel.indexer import index_file

        file_path = self._resolve_path(os.path.expanduser(file_path))
        project_id = self.project_id

        results = list_symbols(project_id, file_path, kind=kind or None)
        if not results:
            # Try indexing the file first
            index_file(project_id, file_path)
            results = list_symbols(project_id, file_path, kind=kind or None)

        if not results:
            return self._error(f"No symbols found in '{file_path}'")

        lines = []
        for s in results:
            indent = "  " if s.parent_symbol else ""
            vis = f" ({s.visibility})" if s.visibility != "public" else ""
            async_mark = "async " if s.is_async else ""
            sig = s.signature or s.name
            line = f"{indent}L{s.line_start:4d}  {s.kind.value:10} {async_mark}{sig}{vis}"
            if s.docstring:
                line += f"  # {s.docstring}"
            lines.append(line)

        header = f"Symbols in {file_path} ({len(results)} total):"
        return header + "\n" + "\n".join(lines)
