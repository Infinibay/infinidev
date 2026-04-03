"""Tool: find where a symbol is defined."""

from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.code_intel.find_definition_input import FindDefinitionInput


class FindDefinitionTool(InfinibayBaseTool):
    name: str = "find_definition"
    description: str = (
        "Find where a function, class, or variable is defined in the codebase. "
        "Returns file path, line number, signature, and docstring. "
        "Much faster and more precise than code_search for finding definitions."
    )
    args_schema: Type[BaseModel] = FindDefinitionInput

    def _run(self, name: str, kind: str = "") -> str:
        from infinidev.code_intel.query import find_definition
        from infinidev.code_intel.indexer import index_directory

        project_id = self.project_id
        workspace = self.workspace_path

        # Lazy index: if no symbols found, try indexing first
        results = find_definition(project_id, name, kind=kind or None)
        if not results and workspace:
            index_directory(project_id, workspace)
            results = find_definition(project_id, name, kind=kind or None)

        if not results:
            return self._error(f"No definition found for '{name}'")

        lines = []
        for s in results:
            line = f"{s.file_path}:{s.line_start}"
            if s.signature:
                line += f" — {s.signature}"
            if s.docstring:
                line += f"  # {s.docstring}"
            if s.parent_symbol:
                line += f"  (in {s.parent_symbol})"
            lines.append(line)

        return "\n".join(lines)

