"""Tool for reading file contents with optional line-range selection."""

import json
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file.read_file_input import ReadFileInput


_TEXT_SAFE = frozenset(
    set(range(0x20, 0x7F))        # printable ASCII
    | {0x09, 0x0A, 0x0D}          # tab, newline, carriage return
    | set(range(0x80, 0x100))     # high bytes (UTF-8 continuations, Latin-1)
)


def _is_binary_file(file_path: str, sample_size: int = 8192) -> bool:
    """Detect whether *file_path* is a binary file by content heuristics.

    Reads the first *sample_size* bytes and checks:
    1. Well-known binary magic signatures (ELF, PNG, JPEG, PDF, Zip, etc.).
    2. Proportion of non-text bytes — if more than 10 % of the sample consists
       of control characters (outside normal whitespace and UTF-8 high-bytes),
       the file is treated as binary.

    Returns ``True`` for binary, ``False`` for text.  On read errors, returns
    ``False`` (let the caller handle the error).
    """
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(sample_size)
    except Exception:
        return False

    if not chunk:
        return False  # empty file is text

    # 1. Magic signature check (covers the most common binary formats)
    _MAGIC = (
        b"\x7fELF",              # ELF executables
        b"\x89PNG\r\n\x1a\n",    # PNG
        b"\xff\xd8\xff",          # JPEG
        b"GIF87a", b"GIF89a",    # GIF
        b"PK\x03\x04",           # ZIP / XLSX / DOCX / JAR
        b"PK\x05\x06",           # ZIP (empty)
        b"\x1f\x8b",             # gzip
        b"BZh",                  # bzip2
        b"\xfd7zXZ\x00",         # xz
        b"\x50\x4b\x03\x04",    # ZIP
        b"%PDF",                  # PDF
        b"\xd0\xcf\x11\xe0",    # MS OLE2 (DOC, XLS, PPT)
        b"RIFF",                 # RIFF container (WAV, AVI, WebP)
        b"\x00\x00\x01\x00",    # ICO
        b"\x00\x00\x02\x00",    # CUR
        b"MZ",                   # DOS/PE executables (EXE, DLL)
        b"\xca\xfe\xba\xbe",    # Mach-O / Java class (universal)
        b"\xcf\xfa\xed\xfe",    # Mach-O (little-endian)
        b"\xce\xfa\xed\xfe",    # Mach-O (32-bit LE)
        b"SQLite format 3",     # SQLite
        b"\x04\x22\x4d\x18",   # LZ4
        b"\x28\xb5\x2f\xfd",   # Zstandard
    )
    for sig in _MAGIC:
        if chunk.startswith(sig):
            return True

    # 2. Byte-distribution heuristic — count non-text control bytes
    non_text = sum(1 for b in chunk if b not in _TEXT_SAFE)
    return (non_text / len(chunk)) > 0.10


class ReadFileTool(InfinibayBaseTool):
    name: str = "read_file"
    description: str = "Read file contents with line numbers. Auto-indexes for code intelligence."
    args_schema: Type[BaseModel] = ReadFileInput

    def _run(
        self,
        file_path: str,
        offset: int | None = None,
        limit: int | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        line_range: str | None = None,
    ) -> str:
        # Coerce offset/limit to int (LLMs may send strings like "10")
        if offset is not None:
            try:
                offset = int(offset)
            except (ValueError, TypeError):
                offset = None
        if limit is not None:
            try:
                limit = int(limit)
            except (ValueError, TypeError):
                limit = None
        if start_line is not None:
            try:
                start_line = int(start_line)
            except (ValueError, TypeError):
                start_line = None
        if end_line is not None:
            try:
                end_line = int(end_line)
            except (ValueError, TypeError):
                end_line = None

        # Parse line_range (e.g. "10-50", "10:50", "10,50")
        if line_range is not None and offset is None:
            import re
            m = re.match(r"(\d+)\s*[-:,]\s*(\d+)", str(line_range))
            if m:
                start_line = int(m.group(1))
                end_line = int(m.group(2))
            else:
                # Single number
                try:
                    start_line = int(line_range)
                except (ValueError, TypeError):
                    pass

        # Accept start_line/end_line as aliases for offset/limit
        if start_line is not None and offset is None:
            offset = start_line
        if end_line is not None and offset is not None and limit is None:
            limit = max(1, end_line - (offset or 1) + 1)
        file_path = self._resolve_path(os.path.expanduser(file_path))

        if self._is_pod_mode():
            return self._run_in_pod(file_path, offset, limit)

        # Sandbox check (resolves symlinks, enforces directory boundaries)
        sandbox_err = self._validate_sandbox_path(file_path)
        if sandbox_err:
            return self._error(sandbox_err)

        if not os.path.exists(file_path):
            return self._error(f"File not found: {file_path}")
        if not os.path.isfile(file_path):
            return self._error(f"Not a file: {file_path}")

        # Check if file is binary (non-text) by content analysis
        if _is_binary_file(file_path):
            return self._error(
                f"Cannot read '{file_path}': file appears to be binary (not a text file). "
                "Use a specialised tool or command to inspect binary files."
            )

        file_size = os.path.getsize(file_path)
        if file_size > settings.MAX_FILE_SIZE_BYTES:
            return self._error(
                f"File too large: {file_size} bytes "
                f"(max {settings.MAX_FILE_SIZE_BYTES} bytes)"
            )

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        except PermissionError:
            return self._error(f"Permission denied: {file_path}")
        except Exception as e:
            return self._error(f"Error reading file: {e}")

        total_lines = len(all_lines)

        # Apply offset/limit for partial reads
        if offset is not None or limit is not None:
            start = max((offset or 1) - 1, 0)  # convert 1-based to 0-based
            end = start + limit if limit is not None else total_lines
            selected = all_lines[start:end]
            # Format with line numbers for easy reference
            numbered = []
            for i, line in enumerate(selected, start=start + 1):
                numbered.append(f"{i:>6}\t{line.rstrip()}")
            content = "\n".join(numbered)
            desc = f"lines {start + 1}-{min(end, total_lines)} of {total_lines}"
        else:
            # Full read — still add line numbers for consistency
            numbered = []
            for i, line in enumerate(all_lines, start=1):
                numbered.append(f"{i:>6}\t{line.rstrip()}")
            content = "\n".join(numbered)
            desc = f"{total_lines} lines"

        self._log_tool_usage(f"Read {file_path} ({desc})")

        # Auto-index the file for code intelligence (best-effort)
        try:
            from infinidev.code_intel.smart_index import ensure_indexed
            project_id = self.project_id
            if project_id:
                ensure_indexed(project_id, file_path)
        except Exception:
            pass  # Never fail a read because of indexing

        return content

    def _run_in_pod(
        self, file_path: str, offset: int | None, limit: int | None,
    ) -> str:
        """Read file via infinibay-file-helper inside the pod."""
        req = {"op": "read", "file_path": file_path}
        if offset is not None:
            req["offset"] = offset
        if limit is not None:
            req["limit"] = limit

        try:
            result = self._exec_in_pod(
                ["infinibay-file-helper"],
                stdin_data=json.dumps(req),
            )
        except RuntimeError as e:
            return self._error(f"Pod execution failed: {e}")

        if result.exit_code != 0:
            return self._error(f"File helper error: {result.stderr.strip()}")

        try:
            resp = json.loads(result.stdout)
        except json.JSONDecodeError:
            return self._error(f"Invalid response from file helper: {result.stdout[:200]}")

        if not resp.get("ok"):
            return self._error(resp.get("error", "Unknown error"))

        data = resp["data"]
        self._log_tool_usage(f"Read {file_path} (pod, {data.get('total_lines', '?')} lines)")
        return data["content"]

