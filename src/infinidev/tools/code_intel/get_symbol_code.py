"""Tool: get the full source code of a function, method, or class."""

import os
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class GetSymbolCodeInput(BaseModel):
    name: str = Field(..., description="Symbol name (function, method, or class)")
    kind: str = Field(
        default="",
        description="Optional: 'function', 'method', 'class' to narrow results",
    )


class GetSymbolCodeTool(InfinibayBaseTool):
    name: str = "get_symbol_code"
    description: str = (
        "Get the full source code of a function, method, or class by name. "
        "Returns the file path, line range, and the complete source code. "
        "Combines find_definition + read_file in one call — much faster."
    )
    args_schema: Type[BaseModel] = GetSymbolCodeInput

    def _run(self, name: str, kind: str = "") -> str:
        from infinidev.code_intel.query import find_definition
        from infinidev.code_intel.indexer import index_directory

        project_id = self.project_id
        workspace = self.workspace_path

        results = find_definition(project_id, name, kind=kind or None)
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
