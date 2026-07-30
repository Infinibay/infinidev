"""Tool: list all symbols in a file."""

from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.code_intel.list_symbols_input import ListSymbolsInput


class ListSymbolsTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "list_symbols"
    description: str = "List symbols defined in a file, optionally filtered by kind."
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
            # tree-sitter may not cover this language; Ken indexes more.
            from infinidev.engine.ken_client import get_ken_client

            ken_symbols = get_ken_client().file_symbols(file_path)
            if kind:
                ken_symbols = [s for s in ken_symbols if s.kind == kind]
            if ken_symbols:
                body = "\n".join(
                    f"L{s.line:4d}  {s.kind:10} {s.qualname}"
                    + (f"  # {s.docstring}" if s.docstring else "")
                    for s in ken_symbols
                )
                return (
                    f"Symbols in {file_path} ({len(ken_symbols)} total, via Ken):\n"
                    + body
                )
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

