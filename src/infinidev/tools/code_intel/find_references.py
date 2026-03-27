"""Tool: find all usages of a symbol."""

from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class FindReferencesInput(BaseModel):
    name: str = Field(..., description="Symbol name to find usages of")
    ref_kind: str = Field(
        default="",
        description="Optional filter: 'call', 'import', 'type_ref', 'assignment', 'usage'",
    )


class FindReferencesTool(InfinibayBaseTool):
    name: str = "find_references"
    description: str = "Find all references to a symbol in the codebase."
    args_schema: Type[BaseModel] = FindReferencesInput

    def _run(self, name: str, ref_kind: str = "") -> str:
        from infinidev.code_intel.query import find_references
        from infinidev.code_intel.indexer import index_directory

        project_id = self.project_id
        workspace = self.workspace_path

        results = find_references(project_id, name, ref_kind=ref_kind or None)
        if not results and workspace:
            index_directory(project_id, workspace)
            results = find_references(project_id, name, ref_kind=ref_kind or None)

        if not results:
            return self._error(f"No references found for '{name}'")

        lines = []
        for r in results:
            line = f"{r.file_path}:{r.line} [{r.ref_kind}] {r.context}"
            lines.append(line)

        header = f"Found {len(results)} reference(s) for '{name}':"
        return header + "\n" + "\n".join(lines)
