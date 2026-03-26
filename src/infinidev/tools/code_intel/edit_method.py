"""Tool: edit (replace) a method or function by symbol name.

Uses tree-sitter index to find the exact location of the symbol,
then replaces it with new code. No old_string matching needed.
"""

import os
import tempfile
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class EditMethodInput(BaseModel):
    symbol: str = Field(
        ...,
        description=(
            "Qualified symbol name: 'ClassName.method_name' for methods, "
            "'function_name' for top-level functions"
        ),
    )
    new_code: str = Field(
        ...,
        description="Complete new source code for the method/function (including def line)",
    )
    file_path: str = Field(
        default="",
        description="File path hint if symbol is ambiguous (optional)",
    )


class EditMethodTool(InfinibayBaseTool):
    name: str = "edit_method"
    description: str = (
        "Replace an entire method or function with new code. Uses the code index "
        "to find the symbol by name — no need to match exact old text. "
        "Provide the complete method including the def line. "
        "Example: edit_method(symbol='Database.execute', new_code='def execute(self, sql):\\n    ...')"
    )
    args_schema: Type[BaseModel] = EditMethodInput

    def _run(self, symbol: str, new_code: str, file_path: str = "") -> str:
        from infinidev.code_intel.resolve import resolve_symbol
        from infinidev.code_intel.smart_index import ensure_indexed

        if not symbol:
            return self._error("symbol is required")
        if not new_code:
            return self._error("new_code is required")

        project_id = self.project_id or 1

        # Resolve path
        resolved_path = ""
        if file_path:
            resolved_path = self._resolve_path(file_path)
            sandbox_err = self._validate_sandbox_path(resolved_path)
            if sandbox_err:
                return self._error(sandbox_err)

        # Ensure file is indexed
        if resolved_path:
            ensure_indexed(project_id, resolved_path)

        # Resolve symbol
        result = resolve_symbol(project_id, symbol, resolved_path or None)

        if result.error:
            # Include candidates in error for helpful feedback
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

        # Validate file
        target_path = sym.file_path
        if not os.path.isfile(target_path):
            return self._error(f"File not found: {target_path}")

        sandbox_err = self._validate_sandbox_path(target_path)
        if sandbox_err:
            return self._error(sandbox_err)

        # Read current file
        try:
            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as exc:
            return self._error(f"Cannot read {target_path}: {exc}")

        # Determine line range
        start = sym.line_start - 1  # 0-based
        end = sym.line_end if sym.line_end else start + 20  # fallback if no line_end
        end = min(end, len(lines))

        if start < 0 or start >= len(lines):
            return self._error(f"Symbol line range invalid: {sym.line_start}-{sym.line_end}")

        # Detect indentation from the original code
        original_indent = ""
        if lines[start]:
            original_indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]

        # Prepare new code with correct indentation
        new_lines = new_code.rstrip().split("\n")
        # Detect indentation of provided new_code
        if new_lines:
            new_indent = new_lines[0][: len(new_lines[0]) - len(new_lines[0].lstrip())]
            if new_indent != original_indent:
                # Re-indent to match original
                adjusted = []
                for line in new_lines:
                    if line.strip():  # non-empty
                        # Remove new_indent, add original_indent
                        stripped = line
                        if new_indent and stripped.startswith(new_indent):
                            stripped = stripped[len(new_indent):]
                        adjusted.append(original_indent + stripped)
                    else:
                        adjusted.append("")
                new_lines = adjusted

        # Ensure trailing newline
        new_content = "\n".join(new_lines) + "\n"

        # Replace lines
        result_lines = lines[:start] + [new_content] + lines[end:]
        new_file_content = "".join(result_lines)

        # Atomic write
        try:
            dir_name = os.path.dirname(target_path)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_file_content)
            # Preserve permissions
            try:
                st = os.stat(target_path)
                os.chmod(tmp_path, st.st_mode)
            except OSError:
                pass
            os.replace(tmp_path, target_path)
        except Exception as exc:
            return self._error(f"Failed to write {target_path}: {exc}")

        # Reindex after edit
        ensure_indexed(project_id, target_path)

        lines_replaced = end - start
        self._log_tool_usage(f"edit_method: {symbol} in {target_path} ({lines_replaced} lines replaced)")

        return self._success({
            "path": target_path,
            "symbol": symbol,
            "lines_replaced": lines_replaced,
            "new_size": len(new_file_content),
        })
