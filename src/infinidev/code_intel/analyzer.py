"""Heuristic code analyzer — detects errors using indexed data.

All checks run SQL queries against ci_symbols, ci_references, and ci_imports.
No re-parsing needed — works on data already extracted by tree-sitter.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from infinidev.tools.base.db import execute_with_retry

logger = logging.getLogger(__name__)

# Python builtins that should not be flagged as undefined
_PYTHON_BUILTINS = frozenset({
    "print", "len", "str", "int", "float", "bool", "list", "dict", "set",
    "tuple", "range", "type", "isinstance", "issubclass", "hasattr", "getattr",
    "setattr", "delattr", "super", "property", "staticmethod", "classmethod",
    "object", "id", "hash", "repr", "abs", "round", "min", "max", "sum",
    "sorted", "reversed", "enumerate", "zip", "map", "filter", "any", "all",
    "iter", "next", "input", "open", "vars", "dir", "callable", "format",
    "chr", "ord", "hex", "oct", "bin", "pow", "divmod", "complex", "bytes",
    "bytearray", "memoryview", "frozenset", "slice", "breakpoint",
    "Exception", "BaseException", "ValueError", "TypeError", "KeyError",
    "IndexError", "AttributeError", "ImportError", "ModuleNotFoundError",
    "FileNotFoundError", "OSError", "IOError", "RuntimeError", "StopIteration",
    "NotImplementedError", "PermissionError", "ConnectionError", "TimeoutError",
    "AssertionError", "NameError", "ZeroDivisionError", "OverflowError",
    "UnicodeDecodeError", "UnicodeEncodeError", "SystemExit", "RecursionError",
    "True", "False", "None", "NotImplemented", "Ellipsis",
    "__name__", "__file__", "__doc__", "__all__", "__init__",
    # Common decorators/functions from typing
    "Optional", "Union", "List", "Dict", "Set", "Tuple", "Any",
    "Callable", "Iterator", "Generator", "Sequence", "Mapping",
    "TYPE_CHECKING", "Protocol", "TypeVar", "Generic", "ClassVar",
    "Final", "Literal", "overload", "cast",
    # dataclasses
    "dataclass", "field",
})

ALL_CHECKS = ["broken_imports", "undefined_symbols", "unused_imports", "unused_definitions"]


@dataclass
class Diagnostic:
    """A single diagnostic finding."""
    file_path: str
    line: int
    severity: str  # "error", "warning", "hint"
    check: str  # check name
    message: str
    fix_suggestion: str = ""


@dataclass
class AnalysisReport:
    """Result of running code analysis."""
    diagnostics: list[Diagnostic] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    scope: str = "file"
    file_path: str | None = None


def analyze_code(
    project_id: int,
    workspace: str,
    file_path: str | None = None,
    scope: str = "file",
    checks: list[str] | None = None,
) -> AnalysisReport:
    """Run heuristic analysis on indexed code.

    Args:
        project_id: Project to analyze.
        workspace: Workspace root path.
        file_path: Specific file (None = whole project based on scope).
        scope: "file" or "project".
        checks: List of check names to run (None = all).

    Returns:
        AnalysisReport with diagnostics and stats.
    """
    if checks is None:
        checks = ALL_CHECKS

    report = AnalysisReport(scope=scope, file_path=file_path)

    for check_name in checks:
        try:
            if check_name == "broken_imports":
                diags = check_broken_imports(project_id, workspace, file_path)
            elif check_name == "undefined_symbols":
                diags = check_undefined_symbols(project_id, file_path)
            elif check_name == "unused_imports":
                diags = check_unused_imports(project_id, file_path)
            elif check_name == "unused_definitions":
                diags = check_unused_definitions(project_id, file_path)
            else:
                continue

            report.diagnostics.extend(diags)
            report.stats[check_name] = len(diags)
        except Exception as exc:
            logger.warning("Check %s failed: %s", check_name, exc)
            report.stats[check_name] = -1  # indicates error

    return report


def check_broken_imports(
    project_id: int, workspace: str, file_path: str | None = None,
) -> list[Diagnostic]:
    """Find imports that can't be resolved."""
    from infinidev.code_intel.import_resolver import resolve_import
    from infinidev.code_intel.models import Import

    def _get_imports(conn):
        sql = "SELECT source, name, alias, file_path, line, is_wildcard, resolved_file, language FROM ci_imports WHERE project_id = ?"
        params: list = [project_id]
        if file_path:
            sql += " AND file_path = ?"
            params.append(file_path)
        return conn.execute(sql, params).fetchall()

    rows = execute_with_retry(_get_imports)
    diagnostics = []

    for row in rows:
        imp = Import(
            source=row[0], name=row[1], alias=row[2],
            file_path=row[3], line=row[4],
            is_wildcard=bool(row[5]), resolved_file=row[6] or "",
            language=row[7],
        )

        if imp.is_wildcard:
            continue

        result = resolve_import(project_id, imp, workspace)
        if result.status == "unresolved":
            diagnostics.append(Diagnostic(
                file_path=imp.file_path,
                line=imp.line,
                severity="error",
                check="broken_imports",
                message=f"Cannot resolve import '{imp.source}' (symbol: {imp.name}): {result.reason}",
                fix_suggestion=f"Check if module '{imp.source}' is installed or the path is correct.",
            ))

    return diagnostics


def check_undefined_symbols(
    project_id: int, file_path: str | None = None,
) -> list[Diagnostic]:
    """Find references to symbols that have no definition."""
    def _query(conn):
        # Get all references (usage/call) for the file
        sql = """
            SELECT DISTINCT r.name, r.file_path, MIN(r.line) as first_line
            FROM ci_references r
            WHERE r.project_id = ? AND r.ref_kind IN ('usage', 'call')
        """
        params: list = [project_id]
        if file_path:
            sql += " AND r.file_path = ?"
            params.append(file_path)
        sql += " GROUP BY r.name, r.file_path"
        refs = conn.execute(sql, params).fetchall()

        # Get all defined symbol names in the project
        defined = set(
            row[0] for row in
            conn.execute("SELECT DISTINCT name FROM ci_symbols WHERE project_id = ?", (project_id,)).fetchall()
        )

        # Get all imported names per file
        imports_by_file: dict[str, set[str]] = {}
        import_rows = conn.execute(
            "SELECT file_path, name, alias FROM ci_imports WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        for fp, name, alias in import_rows:
            imports_by_file.setdefault(fp, set()).add(alias if alias else name)

        # Check for wildcard imports per file
        wildcard_files = set(
            row[0] for row in
            conn.execute(
                "SELECT DISTINCT file_path FROM ci_imports WHERE project_id = ? AND is_wildcard = 1",
                (project_id,),
            ).fetchall()
        )

        return refs, defined, imports_by_file, wildcard_files

    refs, defined, imports_by_file, wildcard_files = execute_with_retry(_query)
    diagnostics = []

    for name, fp, first_line in refs:
        # Skip files with wildcard imports (too many false positives)
        if fp in wildcard_files:
            continue

        # Skip builtins
        if name in _PYTHON_BUILTINS:
            continue

        # Skip if defined anywhere in project
        if name in defined:
            continue

        # Skip if imported in this file
        file_imports = imports_by_file.get(fp, set())
        if name in file_imports:
            continue

        # Skip short names (likely loop vars, comprehensions)
        if len(name) < 2:
            continue

        diagnostics.append(Diagnostic(
            file_path=fp,
            line=first_line,
            severity="warning",
            check="undefined_symbols",
            message=f"Symbol '{name}' is used but not defined or imported.",
            fix_suggestion=f"Add an import for '{name}' or check for typos.",
        ))

    return diagnostics


def check_unused_imports(
    project_id: int, file_path: str | None = None,
) -> list[Diagnostic]:
    """Find imports that are never referenced in the same file."""
    def _query(conn):
        sql = "SELECT name, alias, source, file_path, line FROM ci_imports WHERE project_id = ?"
        params: list = [project_id]
        if file_path:
            sql += " AND file_path = ?"
            params.append(file_path)
        imports = conn.execute(sql, params).fetchall()

        # Get references grouped by file
        refs_by_file: dict[str, set[str]] = {}
        ref_sql = "SELECT file_path, name, line FROM ci_references WHERE project_id = ?"
        ref_params: list = [project_id]
        if file_path:
            ref_sql += " AND file_path = ?"
            ref_params.append(file_path)
        for fp, name, ref_line in conn.execute(ref_sql, ref_params).fetchall():
            refs_by_file.setdefault(fp, set()).add((name, ref_line))

        return imports, refs_by_file

    imports, refs_by_file = execute_with_retry(_query)
    diagnostics = []

    for name, alias, source, fp, line in imports:
        # Skip __init__.py (re-exports)
        if fp.endswith("__init__.py"):
            continue

        # The name used in code is the alias if present
        used_name = alias if alias else name

        # Skip wildcard imports
        if name == "*":
            continue

        # Check if the imported name is referenced in the same file
        # on a line OTHER than the import line itself
        file_refs = refs_by_file.get(fp, set())
        used_elsewhere = any(
            ref_name == used_name and ref_line != line
            for ref_name, ref_line in file_refs
        )
        if not used_elsewhere:
            diagnostics.append(Diagnostic(
                file_path=fp,
                line=line,
                severity="warning",
                check="unused_imports",
                message=f"Import '{used_name}' (from {source}) is never used.",
                fix_suggestion=f"Remove: from {source} import {name}" if source != name else f"Remove: import {name}",
            ))

    return diagnostics


def check_unused_definitions(
    project_id: int, file_path: str | None = None,
) -> list[Diagnostic]:
    """Find symbols that are defined but never referenced anywhere."""
    def _query(conn):
        sql = "SELECT name, qualified_name, kind, file_path, line_start FROM ci_symbols WHERE project_id = ?"
        params: list = [project_id]
        if file_path:
            sql += " AND file_path = ?"
            params.append(file_path)
        symbols = conn.execute(sql, params).fetchall()

        # Get all referenced names across the ENTIRE project (exclude same-file-only refs for file scope)
        if file_path:
            # For file scope: only count references from OTHER files
            all_refs = set(
                row[0] for row in
                conn.execute(
                    "SELECT DISTINCT name FROM ci_references WHERE project_id = ? AND file_path != ?",
                    (project_id, file_path),
                ).fetchall()
            )
        else:
            all_refs = set(
                row[0] for row in
                conn.execute("SELECT DISTINCT name FROM ci_references WHERE project_id = ?", (project_id,)).fetchall()
            )

        # Also count import usages (symbol used as import target)
        all_imported = set(
            row[0] for row in
            conn.execute("SELECT DISTINCT name FROM ci_imports WHERE project_id = ?", (project_id,)).fetchall()
        )

        return symbols, all_refs | all_imported

    symbols, all_used = execute_with_retry(_query)
    diagnostics = []

    for name, qualified_name, kind, fp, line_start in symbols:
        # Skip dunder methods and special names
        if name.startswith("__") and name.endswith("__"):
            continue

        # Skip private symbols (convention: not meant for external use)
        if name.startswith("_"):
            continue

        # Skip main functions
        if name == "main":
            continue

        # Skip class definitions (they're often used as types)
        if kind == "class":
            continue

        # Check if the symbol is referenced anywhere
        if name not in all_used:
            diagnostics.append(Diagnostic(
                file_path=fp,
                line=line_start,
                severity="hint",
                check="unused_definitions",
                message=f"{kind.capitalize()} '{qualified_name or name}' is defined but never referenced.",
            ))

    return diagnostics


def store_diagnostics(project_id: int, file_path: str, diagnostics: list[Diagnostic]) -> None:
    """Persist diagnostics to the database, replacing old ones for the file."""
    def _store(conn):
        conn.execute(
            "DELETE FROM ci_diagnostics WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        )
        if diagnostics:
            conn.executemany(
                """INSERT INTO ci_diagnostics
                   (project_id, file_path, line, severity, check_name, message, fix_suggestion)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(project_id, d.file_path, d.line, d.severity, d.check, d.message, d.fix_suggestion)
                 for d in diagnostics],
            )
        conn.commit()

    execute_with_retry(_store)


def get_diagnostics(project_id: int, file_path: str | None = None) -> list[Diagnostic]:
    """Retrieve stored diagnostics from the database."""
    def _get(conn):
        sql = "SELECT file_path, line, severity, check_name, message, fix_suggestion FROM ci_diagnostics WHERE project_id = ?"
        params: list = [project_id]
        if file_path:
            sql += " AND file_path = ?"
            params.append(file_path)
        sql += " ORDER BY file_path, line"
        return conn.execute(sql, params).fetchall()

    rows = execute_with_retry(_get)
    return [
        Diagnostic(file_path=r[0], line=r[1], severity=r[2], check=r[3], message=r[4], fix_suggestion=r[5])
        for r in rows
    ]
