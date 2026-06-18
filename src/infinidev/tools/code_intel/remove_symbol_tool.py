"""Tool: remove a method or function by symbol name.

Uses tree-sitter index to find and delete the symbol's source lines.
"""

import os
import tempfile
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.code_intel.remove_method_input import RemoveMethodInput


class RemoveSymbolTool(InfinibayBaseTool):
    name: str = "remove_symbol"
    description: str = "Remove a method or function by symbol name."
    args_schema: Type[BaseModel] = RemoveMethodInput

    def _run(self, symbol: str, file_path: str = "") -> str:
        from infinidev.code_intel.resolve import resolve_symbol
        from infinidev.code_intel.smart_index import ensure_indexed

        if not symbol:
            return self._error("symbol is required")

        project_id = self.project_id or 1

        resolved_path = ""
        if file_path:
            resolved_path = self._resolve_path(file_path)
            sandbox_err = self._validate_sandbox_path(resolved_path)
            if sandbox_err:
                return self._error(sandbox_err)

        if resolved_path:
            ensure_indexed(project_id, resolved_path)

        # Resolve symbol
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

        target_path = sym.file_path
        if not os.path.isfile(target_path):
            return self._error(f"File not found: {target_path}")

        sandbox_err = self._validate_sandbox_path(target_path)
        if sandbox_err:
            return self._error(sandbox_err)

        from infinidev.tools.base.permissions import check_file_permission
        perm_err = check_file_permission("edit_file", target_path)
        if perm_err:
            return self._error(perm_err)

        # Read file
        try:
            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as exc:
            return self._error(f"Cannot read {target_path}: {exc}")

        start = sym.line_start - 1  # 0-based
        end = sym.line_end if sym.line_end else start + 1
        end = min(end, len(lines))

        if start < 0 or start >= len(lines):
            return self._error(f"Symbol line range invalid: {sym.line_start}-{sym.line_end}")

        lines_removed = end - start

        # Remove lines
        result_lines = lines[:start] + lines[end:]

        # Collapse blank lines only at the deletion seam (max 2 consecutive),
        # leaving the rest of the file byte-identical. The old whole-file pass
        # silently reformatted unrelated blank runs elsewhere in the file.
        seam_start = start
        while seam_start > 0 and result_lines[seam_start - 1].strip() == "":
            seam_start -= 1
        seam_end = start
        while seam_end < len(result_lines) and result_lines[seam_end].strip() == "":
            seam_end += 1
        if seam_end - seam_start > 2:
            result_lines = (
                result_lines[:seam_start]
                + result_lines[seam_start:seam_start + 2]
                + result_lines[seam_end:]
            )

        new_content = "".join(result_lines)

        # Atomic write
        try:
            dir_name = os.path.dirname(target_path)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)
            try:
                st = os.stat(target_path)
                os.chmod(tmp_path, st.st_mode)
            except OSError:
                pass
            os.replace(tmp_path, target_path)
        except Exception as exc:
            return self._error(f"Failed to write {target_path}: {exc}")

        # Reindex
        ensure_indexed(project_id, target_path)

        self._log_tool_usage(f"remove_symbol: {symbol} from {target_path} ({lines_removed} lines)")

        return self._success({
            "path": target_path,
            "symbol": symbol,
            "lines_removed": lines_removed,
        })

