"""Tool: rename a symbol and update all references across the project."""

import os
import tempfile
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.code_intel.rename_symbol_input import RenameSymbolInput


def _atomic_write(path: str, content: str) -> None:
    """Write content atomically, preserving permissions."""
    import stat
    dir_name = os.path.dirname(path)
    original_mode = os.stat(path).st_mode
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".infinibay_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.chmod(tmp_path, stat.S_IMODE(original_mode))
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _replace_identifier(line: str, old: str, new: str) -> str:
    """Replace an identifier in a line, respecting word boundaries.

    Avoids replacing substrings: 'get' should not match 'get_name'.
    """
    import re
    # \b matches word boundary — handles most cases correctly
    return re.sub(r'\b' + re.escape(old) + r'\b', new, line)


class RenameSymbolTool(InfinibayBaseTool):
    name: str = "rename_symbol"
    description: str = "Rename a symbol and update all references and imports across the project."
    args_schema: Type[BaseModel] = RenameSymbolInput

    def _run(self, symbol: str, new_name: str, file_path: str = "") -> str:
        from infinidev.code_intel.resolve import resolve_symbol
        from infinidev.code_intel.query import find_references
        from infinidev.code_intel.smart_index import ensure_indexed

        if not symbol:
            return self._error("symbol is required")
        if not new_name:
            return self._error("new_name is required")
        if not new_name.isidentifier():
            return self._error(f"'{new_name}' is not a valid Python identifier")

        project_id = self.project_id or 1

        # Resolve the symbol
        resolved_path = ""
        if file_path:
            resolved_path = self._resolve_path(file_path)
            ensure_indexed(project_id, resolved_path)

        result = resolve_symbol(project_id, symbol, resolved_path or None)
        if result.error:
            if result.candidates:
                candidates_str = "\n".join(
                    f"  - {s.qualified_name or s.name} at {s.file_path}:{s.line_start}"
                    for s in result.candidates[:5]
                )
                return self._error(f"{result.error}\nCandidates:\n{candidates_str}")
            return self._error(result.error)

        sym = result.symbol
        if not sym or not sym.file_path:
            return self._error(f"Symbol '{symbol}' resolved but has no file path")

        old_name = sym.name
        if old_name == new_name:
            return self._error(f"Symbol is already named '{new_name}'")

        # Find all references to this symbol across the project
        refs = find_references(project_id, old_name, limit=500)

        # If it's a method (has parent), filter refs to only those likely
        # belonging to this class (same file as definition, or callers)
        # For now, we rename ALL occurrences of old_name — this is the safe
        # approach for unique names. For common names, file_path helps.

        # Group changes by file: {file_path: [(line, old_name, new_name)]}
        changes_by_file: dict[str, list[tuple[int, str, str]]] = {}

        # 1. The definition itself
        changes_by_file.setdefault(sym.file_path, []).append(
            (sym.line_start, old_name, new_name)
        )

        # 2. All references
        for ref in refs:
            changes_by_file.setdefault(ref.file_path, []).append(
                (ref.line, old_name, new_name)
            )

        # 3. Imports: find imports of this symbol
        from infinidev.code_intel.query import find_imports_of
        imports = find_imports_of(project_id, old_name, limit=100)
        for imp in imports:
            changes_by_file.setdefault(imp.file_path, []).append(
                (imp.line, old_name, new_name)
            )

        # Apply changes file by file
        files_modified = []
        total_replacements = 0

        for fpath, line_changes in changes_by_file.items():
            if not os.path.isfile(fpath):
                continue

            sandbox_err = self._validate_sandbox_path(fpath)
            if sandbox_err:
                continue

            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except Exception:
                continue

            # Deduplicate by line number
            affected_lines = set(lc[0] for lc in line_changes)
            replacements_in_file = 0

            for line_num in affected_lines:
                idx = line_num - 1
                if 0 <= idx < len(lines):
                    original = lines[idx]
                    # Replace the old name with new name (word-boundary aware)
                    # Simple approach: replace exact token occurrences
                    new_line = _replace_identifier(original, old_name, new_name)
                    if new_line != original:
                        lines[idx] = new_line
                        replacements_in_file += 1

            if replacements_in_file > 0:
                new_content = "".join(lines)
                try:
                    _atomic_write(fpath, new_content)
                    ensure_indexed(project_id, fpath)
                    files_modified.append(fpath)
                    total_replacements += replacements_in_file
                except Exception:
                    continue

        self._log_tool_usage(
            f"rename_symbol: '{old_name}' → '{new_name}' "
            f"({total_replacements} replacements in {len(files_modified)} files)"
        )

        return self._success({
            "old_name": old_name,
            "new_name": new_name,
            "files_modified": files_modified,
            "total_replacements": total_replacements,
        })


