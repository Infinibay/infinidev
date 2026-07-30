"""Tool: find all usages of a symbol."""

from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.code_intel.find_references_input import FindReferencesInput


class FindReferencesTool(InfinibayBaseTool):
    is_read_only: bool = True
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
            # The local index only records references it parsed. Ken's call
            # graph resolves call-sites across languages it indexed, so it
            # can still answer when our index is cold or partial.
            callers = self._ken_callers(name)
            if callers:
                return callers
            return self._error(f"No references found for '{name}'")

        lines = []
        for r in results:
            line = f"{r.file_path}:{r.line} [{r.ref_kind}] {r.context}"
            lines.append(line)

        header = f"Found {len(results)} reference(s) for '{name}':"
        return header + "\n" + "\n".join(lines)

    @staticmethod
    def _ken_callers(name: str) -> str:
        """Resolved call-sites from Ken's call graph. Empty string when absent."""
        from infinidev.engine.ken_client import get_ken_client

        client = get_ken_client()
        callers = client.callers_of(name)
        if not callers:
            # Last resort: a literal worktree scan — never stale, and it
            # catches references the call graph does not model (imports,
            # annotations, string references).
            matches = client.grep(name, limit=25)
            if not matches:
                return ""
            lines = [f"{m.path}:{m.line} [literal] {m.text}" for m in matches]
            return (
                f"Found {len(matches)} literal occurrence(s) of '{name}' "
                f"(call graph had none):\n" + "\n".join(lines)
            )
        lines = [
            f"{c.file}:{c.line} [calls] {c.qualname}"
            + (f" ({c.docstring})" if c.docstring else "")
            for c in callers
        ]
        return f"Found {len(callers)} caller(s) of '{name}':\n" + "\n".join(lines)

