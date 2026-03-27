"""Tool: run heuristic code analysis on indexed data."""

import json
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class AnalyzeCodeInput(BaseModel):
    file_path: str = Field(
        default="",
        description="File to analyze. Empty = analyze whole project.",
    )
    checks: str = Field(
        default="",
        description="Comma-separated checks: broken_imports,undefined_symbols,unused_imports,unused_definitions. Empty = all.",
    )


class AnalyzeCodeTool(InfinibayBaseTool):
    name: str = "analyze_code"
    description: str = "Detect code errors: broken imports, undefined symbols, unused imports/definitions."
    args_schema: Type[BaseModel] = AnalyzeCodeInput

    def _run(self, file_path: str = "", checks: str = "") -> str:
        from infinidev.code_intel.analyzer import analyze_code, ALL_CHECKS
        from infinidev.code_intel.smart_index import ensure_indexed

        project_id = self.project_id or 1
        workspace = self.workspace_path or ""

        # Parse checks
        check_list = None
        if checks.strip():
            check_list = [c.strip() for c in checks.split(",") if c.strip()]
            invalid = set(check_list) - set(ALL_CHECKS)
            if invalid:
                return self._error(
                    f"Unknown checks: {', '.join(invalid)}. "
                    f"Valid: {', '.join(ALL_CHECKS)}"
                )

        # Resolve and ensure indexed
        resolved_path = None
        if file_path:
            resolved_path = self._resolve_path(file_path)
            try:
                ensure_indexed(project_id, resolved_path)
            except Exception:
                pass

        scope = "file" if resolved_path else "project"
        report = analyze_code(
            project_id=project_id,
            workspace=workspace,
            file_path=resolved_path,
            scope=scope,
            checks=check_list,
        )

        # Format output
        if not report.diagnostics:
            return self._success({
                "message": "No issues found.",
                "scope": scope,
                "checks_run": check_list or ALL_CHECKS,
            })

        # Group by severity
        by_severity: dict[str, list] = {}
        for d in report.diagnostics:
            by_severity.setdefault(d.severity, []).append({
                "file": d.file_path,
                "line": d.line,
                "check": d.check,
                "message": d.message,
                "fix": d.fix_suggestion,
            })

        return self._success({
            "total": len(report.diagnostics),
            "stats": report.stats,
            "errors": by_severity.get("error", []),
            "warnings": by_severity.get("warning", []),
            "hints": by_severity.get("hint", []),
        })
