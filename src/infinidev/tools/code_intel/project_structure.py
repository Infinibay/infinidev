"""Tool: show project structure with semantic descriptions from the code index."""

import os
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class ProjectStructureInput(BaseModel):
    path: str = Field(
        default=".",
        description="Directory to show structure of. Defaults to project root.",
    )
    depth: int = Field(
        default=2,
        description="How many levels deep to show (1-5). Default 2.",
    )


class ProjectStructureTool(InfinibayBaseTool):
    name: str = "project_structure"
    description: str = "Show project directory tree with file descriptions."
    args_schema: Type[BaseModel] = ProjectStructureInput

    def _run(
        self,
        path: str = ".",
        depth: int = 2,
        # Aliases
        directory: str = "",
        dir: str = "",
        folder: str = "",
        subdir: str = "",
    ) -> str:
        # Accept aliases
        path = directory or dir or folder or subdir or path

        path = self._resolve_path(os.path.expanduser(path))
        if not os.path.isdir(path):
            return self._error(f"Directory not found: {path}")

        depth = max(1, min(5, depth))

        from infinidev.code_intel.query import list_symbols, get_index_stats
        from infinidev.code_intel.indexer import SKIP_DIRS
        from infinidev.code_intel.parsers import detect_language

        project_id = self.project_id
        lines: list[str] = []

        def _describe_file(fpath: str) -> str:
            """Get a short description of a file from its indexed symbols."""
            syms = list_symbols(project_id, fpath, limit=10)
            if not syms:
                return ""
            # Summarize: list top-level symbols
            parts = []
            for s in syms:
                if not s.parent_symbol:  # Top-level only
                    if s.kind.value in ("class", "interface", "enum"):
                        parts.append(f"{s.kind.value} {s.name}")
                    elif s.kind.value == "function":
                        parts.append(f"fn {s.name}()")
                    elif s.kind.value in ("constant", "variable"):
                        parts.append(s.name)
                if len(parts) >= 5:
                    break
            return ", ".join(parts)

        def _describe_dir(dirpath: str) -> str:
            """Get a short description of a directory from its files."""
            count = 0
            kinds: dict[str, int] = {}
            for fname in os.listdir(dirpath):
                fpath = os.path.join(dirpath, fname)
                if os.path.isfile(fpath):
                    lang = detect_language(fpath)
                    if lang:
                        count += 1
                        kinds[lang] = kinds.get(lang, 0) + 1
            if not kinds:
                return ""
            lang_summary = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items(), key=lambda x: -x[1]))
            return f"({count} files: {lang_summary})"

        def _walk(dirpath: str, current_depth: int, prefix: str) -> None:
            if current_depth > depth:
                return

            try:
                entries = sorted(os.listdir(dirpath))
            except PermissionError:
                return

            dirs = []
            files = []
            for entry in entries:
                if entry.startswith("."):
                    continue
                full = os.path.join(dirpath, entry)
                if os.path.isdir(full):
                    if entry not in SKIP_DIRS:
                        dirs.append(entry)
                else:
                    files.append(entry)

            for d in dirs:
                full = os.path.join(dirpath, d)
                desc = _describe_dir(full)
                lines.append(f"{prefix}{d}/  {desc}")
                _walk(full, current_depth + 1, prefix + "  ")

            for f in files:
                full = os.path.join(dirpath, f)
                desc = _describe_file(full)
                if desc:
                    lines.append(f"{prefix}{f}  — {desc}")
                else:
                    lines.append(f"{prefix}{f}")

        _walk(path, 1, "")

        if not lines:
            return self._error(f"Empty directory: {path}")

        header = f"Project structure of {path} (depth={depth}):"
        return header + "\n" + "\n".join(lines)
