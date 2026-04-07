"""Tool: summary statistics for the project's code intelligence index.

Promoted from the ``code_interpreter`` bridge to a flat tool as the
"first thing to call when you don't know what's in the index". Tells
you in one call: how many files indexed, how many symbols, what
languages exist, what kinds of symbols are present. Replaces the
need to write a custom Python script for orientation.
"""

from typing import Type
from pydantic import BaseModel

from infinidev.tools.base.base_tool import InfinibayBaseTool


class ProjectStatsInput(BaseModel):
    """No parameters — the tool always reports stats for the
    currently active project."""
    pass


class ProjectStatsTool(InfinibayBaseTool):
    name: str = "project_stats"
    description: str = (
        "Summary of the project's code intelligence index: file count, "
        "symbol count, references, languages present, and a breakdown "
        "of symbols by kind. Use as the first call in any analysis "
        "task to orient yourself — tells you instantly whether the "
        "index is populated and what kinds of code are available."
    )
    args_schema: Type[BaseModel] = ProjectStatsInput

    def _run(self) -> str:
        import sqlite3
        from infinidev.tools.base.db import execute_with_retry

        project_id = self.project_id
        if not project_id:
            return self._error("No project context — cannot read stats.")

        # Single connection for all the COUNT queries — keeps lock
        # acquisition cheap (one acquire instead of five) on a busy DB.
        def _q(conn: sqlite3.Connection):
            out = {}
            out["files"] = conn.execute(
                "SELECT COUNT(*) FROM ci_files WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
            out["symbols"] = conn.execute(
                "SELECT COUNT(*) FROM ci_symbols WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
            out["references"] = conn.execute(
                "SELECT COUNT(*) FROM ci_references WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
            out["method_bodies"] = conn.execute(
                "SELECT COUNT(*) FROM ci_method_bodies WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
            by_kind = conn.execute(
                "SELECT kind, COUNT(*) FROM ci_symbols "
                "WHERE project_id = ? GROUP BY kind ORDER BY COUNT(*) DESC",
                (project_id,),
            ).fetchall()
            out["by_kind"] = [(k, c) for k, c in by_kind]
            langs = conn.execute(
                "SELECT DISTINCT language FROM ci_files WHERE project_id = ?",
                (project_id,),
            ).fetchall()
            out["languages"] = sorted([l[0] for l in langs if l[0]])
            return out

        try:
            stats = execute_with_retry(_q) or {}
        except Exception as exc:
            return self._error(f"Stats query failed: {exc}")

        if not stats or stats.get("files", 0) == 0:
            return self._error(
                "Index is empty for this project. Run /reindex or call "
                "read_file on a few source files to seed the index."
            )

        lines = ["Project intelligence summary:"]
        lines.append(f"  files indexed:    {stats['files']}")
        lines.append(f"  total symbols:    {stats['symbols']}")
        lines.append(f"  references:       {stats['references']}")
        lines.append(f"  method bodies:    {stats['method_bodies']}")
        lines.append(
            f"  languages:        {', '.join(stats['languages']) or '(none)'}"
        )
        if stats["by_kind"]:
            lines.append("  symbols by kind:")
            for k, c in stats["by_kind"][:8]:
                lines.append(f"    {c:>6}  {k}")
        return "\n".join(lines)
