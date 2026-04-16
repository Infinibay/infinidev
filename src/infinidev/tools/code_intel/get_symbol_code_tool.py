"""Tool: get the full source code of a function, method, or class."""

import os
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.code_intel.get_symbol_code_input import GetSymbolCodeInput


class GetSymbolCodeTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "get_symbol_code"
    description: str = "Get source code of a symbol by name."
    args_schema: Type[BaseModel] = GetSymbolCodeInput

    def _run(self, name: str, kind: str = "", file_path: str = "") -> str:
        from infinidev.code_intel.query import find_definition
        from infinidev.code_intel.indexer import index_directory

        project_id = self.project_id
        workspace = self.workspace_path

        # Auto-index specific file if provided
        if file_path:
            resolved = self._resolve_path(file_path)
            if os.path.isfile(resolved):
                try:
                    from infinidev.code_intel.smart_index import ensure_indexed
                    ensure_indexed(project_id, resolved)
                except Exception:
                    pass

        results = find_definition(project_id, name, kind=kind or None, file_hint=file_path or None)
        if not results and workspace:
            index_directory(project_id, workspace)
            results = find_definition(project_id, name, kind=kind or None)

        if not results:
            return self._error(f"No definition found for '{name}'")

        # Get the code for each result (up to 3)
        output_parts = []
        for sym in results[:3]:
            if not os.path.isfile(sym.file_path):
                continue

            try:
                with open(sym.file_path, "r", encoding="utf-8", errors="replace") as f:
                    all_lines = f.readlines()
            except Exception:
                continue

            start = sym.line_start - 1  # 0-based
            end = (sym.line_end or sym.line_start + 20) - 1  # Default 20 lines if no end

            # Clamp
            start = max(0, start)
            end = min(len(all_lines), end + 1)

            code_lines = all_lines[start:end]
            numbered = "".join(
                f"{start + i + 1:5d}\t{line}" for i, line in enumerate(code_lines)
            )

            header = f"### {sym.file_path}:{sym.line_start}-{sym.line_end or '?'}"
            if sym.signature:
                header += f"  ({sym.signature})"
            if sym.parent_symbol:
                header += f"  in {sym.parent_symbol}"

            output_parts.append(f"{header}\n{numbered}")

        if not output_parts:
            return self._error(f"Could not read source for '{name}'")

        return "\n\n".join(output_parts)

