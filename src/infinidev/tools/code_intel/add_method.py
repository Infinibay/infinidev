"""Tool: add a method or function to a file or class.

Inserts code at the end of a class (if class_name given) or end of file.
Auto-detects indentation.
"""

import os
import tempfile
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class AddMethodInput(BaseModel):
    file_path: str = Field(..., description="Path to the target file")
    code: str = Field(
        ...,
        description="Complete method/function source code (including def line)",
    )
    class_name: str = Field(
        default="",
        description="Class to add the method to. If empty, appends to end of file.",
    )


class AddMethodTool(InfinibayBaseTool):
    name: str = "add_method"
    description: str = (
        "Add a new method or function to a file. If class_name is provided, "
        "inserts at the end of that class with correct indentation. "
        "If no class_name, appends to the end of the file."
    )
    args_schema: Type[BaseModel] = AddMethodInput

    def _run(self, file_path: str, code: str, class_name: str = "") -> str:
        from infinidev.code_intel.resolve import resolve_symbol
        from infinidev.code_intel.smart_index import ensure_indexed

        if not file_path:
            return self._error("file_path is required")
        if not code:
            return self._error("code is required")

        resolved_path = self._resolve_path(file_path)
        sandbox_err = self._validate_sandbox_path(resolved_path)
        if sandbox_err:
            return self._error(sandbox_err)

        if not os.path.isfile(resolved_path):
            return self._error(f"File not found: {resolved_path}")

        project_id = self.project_id or 1

        # Read current content
        try:
            with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                lines = content.split("\n")
        except Exception as exc:
            return self._error(f"Cannot read {resolved_path}: {exc}")

        # Prepare the code to insert
        code_lines = code.rstrip().split("\n")
        insert_line = len(lines)  # default: end of file
        indent = ""

        if class_name:
            # Find the class and its end line
            ensure_indexed(project_id, resolved_path)
            result = resolve_symbol(project_id, class_name, resolved_path)

            if result.error or not result.symbol:
                return self._error(f"Class '{class_name}' not found in {file_path}: {result.error}")

            sym = result.symbol
            if sym.line_end:
                insert_line = sym.line_end  # insert at end of class
            else:
                insert_line = sym.line_start + 10  # fallback

            insert_line = min(insert_line, len(lines))

            # Detect class body indentation (typically 4 spaces for Python)
            indent = "    "  # default
            for i in range(sym.line_start, min(sym.line_start + 5, len(lines))):
                line = lines[i] if i < len(lines) else ""
                stripped = line.lstrip()
                if stripped.startswith("def ") or stripped.startswith("async def "):
                    indent = line[: len(line) - len(stripped)]
                    break

            # Re-indent code to match class body
            code_indent = ""
            if code_lines:
                first = code_lines[0]
                code_indent = first[: len(first) - len(first.lstrip())]

            if code_indent != indent:
                adjusted = []
                for line in code_lines:
                    if line.strip():
                        if code_indent and line.startswith(code_indent):
                            line = line[len(code_indent):]
                        adjusted.append(indent + line)
                    else:
                        adjusted.append("")
                code_lines = adjusted

        # Insert the code
        insert_text = "\n" + "\n".join(code_lines) + "\n"
        new_lines = lines[:insert_line] + [insert_text] + lines[insert_line:]
        new_content = "\n".join(new_lines)

        # Atomic write
        try:
            dir_name = os.path.dirname(resolved_path)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)
            try:
                st = os.stat(resolved_path)
                os.chmod(tmp_path, st.st_mode)
            except OSError:
                pass
            os.replace(tmp_path, resolved_path)
        except Exception as exc:
            return self._error(f"Failed to write {resolved_path}: {exc}")

        # Reindex
        ensure_indexed(project_id, resolved_path)

        self._log_tool_usage(f"add_method: added to {class_name or 'file'} in {resolved_path}")

        return self._success({
            "path": resolved_path,
            "inserted_at_line": insert_line + 1,
            "class_name": class_name or "(top-level)",
            "lines_added": len(code_lines),
        })
