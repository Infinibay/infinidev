"""Tool: move a symbol (function/method/class) to another file or class."""

import os
import tempfile
import stat
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.code_intel.move_symbol_input import MoveSymbolInput


def _atomic_write(path: str, content: str) -> None:
    """Write content atomically, preserving permissions."""
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


def _file_to_module(file_path: str, workspace: str) -> str:
    """Convert a file path to a Python module path.

    e.g. /workspace/src/infinidev/tools/file.py → infinidev.tools.file
    """
    try:
        rel = os.path.relpath(file_path, workspace)
    except ValueError:
        return ""

    # Strip .py extension
    if rel.endswith(".py"):
        rel = rel[:-3]
    # Strip __init__
    if rel.endswith("__init__"):
        rel = rel[:-9].rstrip(os.sep)

    # Handle src/ layout
    if rel.startswith("src" + os.sep):
        rel = rel[4:]

    # Convert path separators to dots
    module = rel.replace(os.sep, ".")

    return module


def _reindent(code: str, target_indent: str) -> str:
    """Re-indent code to match target indentation."""
    lines = code.split("\n")
    if not lines:
        return code

    # Detect current indent from first non-empty line
    current_indent = ""
    for line in lines:
        if line.strip():
            current_indent = line[: len(line) - len(line.lstrip())]
            break

    if current_indent == target_indent:
        return code

    result = []
    for line in lines:
        if line.strip():
            if current_indent and line.startswith(current_indent):
                line = target_indent + line[len(current_indent):]
            elif not current_indent:
                line = target_indent + line
        result.append(line)

    return "\n".join(result)


class MoveSymbolTool(InfinibayBaseTool):
    name: str = "move_symbol"
    description: str = "Move a symbol to another file or class, updating imports across the project."
    args_schema: Type[BaseModel] = MoveSymbolInput

    def _run(
        self,
        symbol: str,
        target_file: str,
        target_class: str = "",
        after_line: int = 0,
    ) -> str:
        from infinidev.code_intel.resolve import resolve_symbol
        from infinidev.code_intel.query import find_imports_of
        from infinidev.code_intel.smart_index import ensure_indexed

        if not symbol:
            return self._error("symbol is required")
        if not target_file:
            return self._error("target_file is required")

        project_id = self.project_id or 1
        target_path = self._resolve_path(target_file)

        sandbox_err = self._validate_sandbox_path(target_path)
        if sandbox_err:
            return self._error(sandbox_err)

        # Resolve symbol
        result = resolve_symbol(project_id, symbol, None)
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

        source_path = sym.file_path
        if not os.path.isfile(source_path):
            return self._error(f"Source file not found: {source_path}")

        sandbox_err = self._validate_sandbox_path(source_path)
        if sandbox_err:
            return self._error(sandbox_err)

        # Same file check
        if os.path.abspath(source_path) == os.path.abspath(target_path) and not target_class:
            return self._error("Source and target are the same file. Specify target_class to move within the file.")

        # ── Step 1: Extract source code ──────────────────────────────────
        try:
            with open(source_path, "r", encoding="utf-8", errors="replace") as f:
                source_lines = f.readlines()
        except Exception as exc:
            return self._error(f"Cannot read {source_path}: {exc}")

        start = sym.line_start - 1  # 0-based
        end = sym.line_end if sym.line_end else start + 1
        end = min(end, len(source_lines))

        if start < 0 or start >= len(source_lines):
            return self._error(f"Symbol line range invalid: {sym.line_start}-{sym.line_end}")

        extracted_lines = source_lines[start:end]
        extracted_code = "".join(extracted_lines)

        # ── Step 2: Remove from source file ──────────────────────────────
        remaining_lines = source_lines[:start] + source_lines[end:]

        # Clean up excessive blank lines
        cleaned = []
        blank_count = 0
        for line in remaining_lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:
                    cleaned.append(line)
            else:
                blank_count = 0
                cleaned.append(line)

        source_content = "".join(cleaned)

        # ── Step 3: Prepare code for target ──────────────────────────────
        if target_class:
            # Re-indent for class body
            ensure_indexed(project_id, target_path)
            class_result = resolve_symbol(project_id, target_class, target_path)
            if class_result.error or not class_result.symbol:
                return self._error(f"Target class '{target_class}' not found in {target_file}: {class_result.error}")

            target_sym = class_result.symbol
            # Detect class body indentation
            try:
                with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                    target_lines = f.readlines()
            except Exception as exc:
                return self._error(f"Cannot read {target_path}: {exc}")

            indent = "    "  # default
            for i in range(target_sym.line_start, min(target_sym.line_start + 10, len(target_lines))):
                line = target_lines[i] if i < len(target_lines) else ""
                stripped = line.lstrip()
                if stripped.startswith("def ") or stripped.startswith("async def "):
                    indent = line[: len(line) - len(stripped)]
                    break

            # Re-indent extracted code
            code_to_insert = _reindent(extracted_code, indent)

            # Determine insert position
            if after_line > 0:
                insert_idx = after_line
            else:
                # End of class
                insert_idx = target_sym.line_end if target_sym.line_end else target_sym.line_start + 1
                insert_idx = min(insert_idx, len(target_lines))
        else:
            # Top-level: strip to base indentation
            code_to_insert = _reindent(extracted_code, "")

            if not os.path.isfile(target_path):
                # Target doesn't exist yet — will create
                target_lines = []
                insert_idx = 0
            else:
                try:
                    with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                        target_lines = f.readlines()
                except Exception as exc:
                    return self._error(f"Cannot read {target_path}: {exc}")

                if after_line > 0:
                    insert_idx = after_line
                else:
                    insert_idx = len(target_lines)

        # ── Step 4: Insert into target file ──────────────────────────────
        insert_lines = code_to_insert.splitlines(keepends=True)
        if insert_lines and not insert_lines[-1].endswith("\n"):
            insert_lines[-1] += "\n"

        # Add blank line separator
        if insert_idx > 0 and target_lines and insert_idx <= len(target_lines):
            prev_line = target_lines[insert_idx - 1] if insert_idx > 0 else ""
            if prev_line.strip():
                insert_lines = ["\n"] + insert_lines
        if insert_idx < len(target_lines):
            next_line = target_lines[insert_idx] if insert_idx < len(target_lines) else ""
            if next_line.strip():
                insert_lines = insert_lines + ["\n"]

        new_target_lines = target_lines[:insert_idx] + insert_lines + target_lines[insert_idx:]
        target_content = "".join(new_target_lines)

        # ── Step 5: Write both files atomically ──────────────────────────
        try:
            _atomic_write(source_path, source_content)
        except Exception as exc:
            return self._error(f"Failed to write source {source_path}: {exc}")

        try:
            if not os.path.exists(target_path):
                parent = os.path.dirname(target_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(target_content)
            else:
                _atomic_write(target_path, target_content)
        except Exception as exc:
            # Rollback source
            try:
                _atomic_write(source_path, "".join(source_lines))
            except Exception:
                pass
            return self._error(f"Failed to write target {target_path}: {exc}")

        # ── Step 6: Update imports across the project ────────────────────
        imports_updated = 0
        if source_path != target_path:
            imports_updated = self._update_imports(
                project_id, sym.name, source_path, target_path
            )

        # ── Step 7: Re-index modified files ──────────────────────────────
        ensure_indexed(project_id, source_path)
        ensure_indexed(project_id, target_path)

        lines_moved = end - start
        self._log_tool_usage(
            f"move_symbol: '{symbol}' from {source_path} to {target_path}"
            f"{' in ' + target_class if target_class else ''}"
            f" ({lines_moved} lines, {imports_updated} imports updated)"
        )

        return self._success({
            "symbol": symbol,
            "source_file": source_path,
            "target_file": target_path,
            "target_class": target_class or "(top-level)",
            "inserted_at_line": insert_idx + 1,
            "lines_moved": lines_moved,
            "imports_updated": imports_updated,
        })

    def _update_imports(
        self, project_id: int, symbol_name: str, old_file: str, new_file: str,
    ) -> int:
        """Update import statements that reference the moved symbol."""
        from infinidev.code_intel.query import find_imports_of
        from infinidev.code_intel.smart_index import ensure_indexed

        imports = find_imports_of(project_id, symbol_name, limit=200)
        updated_count = 0

        # Compute new module path from new_file
        workspace = self.workspace_path or os.getcwd()
        new_module = _file_to_module(new_file, workspace)
        old_module = _file_to_module(old_file, workspace)

        if not new_module or not old_module:
            return 0

        for imp in imports:
            # Only update imports that reference the old module
            if imp.source != old_module and not imp.source.endswith("." + old_module.split(".")[-1]):
                continue

            if not os.path.isfile(imp.file_path):
                continue

            sandbox_err = self._validate_sandbox_path(imp.file_path)
            if sandbox_err:
                continue

            try:
                with open(imp.file_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except Exception:
                continue

            idx = imp.line - 1
            if 0 <= idx < len(lines):
                original = lines[idx]
                new_line = original.replace(old_module, new_module)
                if new_line != original:
                    lines[idx] = new_line
                    try:
                        _atomic_write(imp.file_path, "".join(lines))
                        ensure_indexed(project_id, imp.file_path)
                        updated_count += 1
                    except Exception:
                        continue

        return updated_count


