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

# Files larger than this (in lines) get a structured skeleton instead of
# the raw content when read without an explicit line range. The threshold
# is intentionally conservative — even modern small models start to
# struggle around the 1k-line mark, and Claude Opus refuses to inline
# anything bigger than ~30k tokens worth of code.
_LARGE_FILE_LINE_THRESHOLD = 800

# Head/tail preview sizes for files in unsupported languages (where the
# tree-sitter skeleton extractor can't help).
_FALLBACK_HEAD_LINES = 60
_FALLBACK_TAIL_LINES = 30


def _format_head_tail_preview(
    all_lines: list[str],
    *,
    file_path: str,
    total_lines: int,
    total_bytes: int,
) -> str:
    """Render a head+tail preview for files that have no tree-sitter skeleton.

    Used as the fallback when ``extract_file_skeleton`` returns nothing
    (unsupported language, parse failure, no symbols at all). Same shape
    as the structured skeleton: tells the model the file is huge, shows
    a small head and tail so it can recognise the file type, and ends
    with the same "use read_file with a line range" hint.
    """
    head_lines = all_lines[:_FALLBACK_HEAD_LINES]
    tail_start = max(_FALLBACK_HEAD_LINES, total_lines - _FALLBACK_TAIL_LINES)
    tail_lines = all_lines[tail_start:] if tail_start < total_lines else []

    out: list[str] = []
    out.append("⚠ FILE TOO LARGE TO READ IN FULL — returning head+tail preview.")
    out.append(f"  file:  {file_path}")
    out.append(f"  size:  {total_lines} lines, {total_bytes} bytes")
    out.append(
        "  note:  no tree-sitter skeleton available for this file type — "
        "showing first and last lines only."
    )
    out.append("")
    out.append(f"── First {len(head_lines)} lines ──")
    for i, line in enumerate(head_lines, start=1):
        out.append(f"{i:>6}\t{line.rstrip()}")
    if tail_lines:
        out.append("")
        out.append(
            f"  ... ({tail_start - _FALLBACK_HEAD_LINES} lines hidden) ..."
        )
        out.append("")
        out.append(f"── Last {len(tail_lines)} lines ──")
        for i, line in enumerate(tail_lines, start=tail_start + 1):
            out.append(f"{i:>6}\t{line.rstrip()}")
    out.append("")
    out.append("── How to read this file ──")
    out.append(
        "  This file is too large to load in full. To inspect specific parts,"
    )
    out.append(
        "  call read_file again with explicit start_line and end_line."
    )
    out.append("")
    out.append(
        "  • read_file(file_path=..., start_line=N, end_line=M)"
    )
    out.append(
        "      → read a specific line range, e.g. (1, 200) or (500, 700)."
    )
    out.append("")
    out.append(
        "  Pick the line range you actually need and call read_file with it. "
        "Don't try to read the whole file."
    )
    return "\n".join(out)


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
    description: str = (
        "Read file contents with line numbers. Auto-indexes for code "
        "intelligence. Accepts start_line/end_line for partial reads. "
        "For files larger than ~800 lines, returns a structured "
        "skeleton (classes, methods, functions, line ranges) instead "
        "of the full content — call again with start_line/end_line, "
        "or use get_symbol_code, to zoom in on specific parts."
    )
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
            from infinidev.engine.static_analysis_timer import measure as _sa_measure
            with _sa_measure("tool_io"):
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
        elif total_lines > _LARGE_FILE_LINE_THRESHOLD:
            # ── Large-file skeleton mode ──────────────────────────────
            # The model asked for the whole file but it's too big to inline
            # without burning context. Return a structured skeleton built
            # from tree-sitter (when the language is supported) or a
            # head+tail preview (otherwise). Both end with an explicit
            # hint pointing the model at read_file(start_line, end_line)
            # or get_symbol_code,
            # because small models don't discover those tools on their own.
            try:
                from infinidev.code_intel.syntax_check import (
                    extract_file_skeleton, render_skeleton_text,
                )
                skeleton, language = extract_file_skeleton(
                    "".join(all_lines), file_path=file_path,
                )
            except Exception:
                skeleton, language = [], ""
            if skeleton:
                content = render_skeleton_text(
                    skeleton,
                    file_path=file_path,
                    total_lines=total_lines,
                    total_bytes=file_size,
                    language=language or "",
                )
                desc = (
                    f"skeleton ({len(skeleton)} symbols, "
                    f"{total_lines}-line file)"
                )
            else:
                content = _format_head_tail_preview(
                    all_lines,
                    file_path=file_path,
                    total_lines=total_lines,
                    total_bytes=file_size,
                )
                desc = f"head+tail preview ({total_lines}-line file)"
        else:
            # Full read — still add line numbers for consistency
            numbered = []
            for i, line in enumerate(all_lines, start=1):
                numbered.append(f"{i:>6}\t{line.rstrip()}")
            content = "\n".join(numbered)
            desc = f"{total_lines} lines"

        self._log_tool_usage(f"Read {file_path} ({desc})")

        # Auto-index the file for code intelligence (best-effort).
        # Goes through ``background_indexer.enqueue_or_sync``: if the
        # process has a running IndexQueue (registered by cli.main at
        # startup), this is a non-blocking ``queue.put`` of ~µs and the
        # worker thread does the actual ensure_indexed off the hot path.
        # Without a queue (tests, scripts, isolated tool calls) it
        # falls back to synchronous ensure_indexed so behaviour matches
        # the pre-async version exactly.
        #
        # Wrapped in the static-analysis timer for visibility — when the
        # queue is active the file_indexing bucket should drop to ~µs/call
        # because the actual parse is happening on a worker thread.
        try:
            from infinidev.code_intel.background_indexer import enqueue_or_sync
            from infinidev.engine.static_analysis_timer import measure as _sa_measure
            project_id = self.project_id
            if project_id:
                with _sa_measure("file_indexing"):
                    enqueue_or_sync(project_id, file_path)
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

