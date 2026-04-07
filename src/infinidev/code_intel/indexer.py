"""File indexer — parses source files and stores symbols in the index."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

from infinidev.code_intel import index as ci_index
from infinidev.code_intel.parsers import detect_language, get_parser

logger = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    "target", "bin", "obj", ".dart_tool", "vendor",
    ".next", ".nuxt", "coverage", ".cache", ".infinidev",
    ".eggs", ".ruff_cache", ".hypothesis", ".nox",
    "site-packages", ".cargo", "bower_components",
    "unsloth_compiled_cache",
}

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".lock", ".sum", ".map",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".doc", ".docx",
}

MAX_FILE_SIZE = 1_000_000  # 1 MB


def _file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def index_file(project_id: int, file_path: str) -> int:
    """Index a single file. Returns number of symbols extracted.

    Skips if the file content hash hasn't changed.
    """
    if not os.path.isfile(file_path):
        return 0

    language = detect_language(file_path)
    if not language:
        return 0

    # Config files: track hash for change detection, no symbol parsing
    if language == "config":
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            content_hash = _file_hash(content)
            existing_hash = ci_index.get_file_hash(project_id, file_path)
            if existing_hash == content_hash:
                return 0
            ci_index.mark_file_indexed(project_id, file_path, language, content_hash, 0)
            return 0
        except (PermissionError, OSError):
            return 0

    parser = get_parser(language)
    if parser is None:
        return 0

    try:
        with open(file_path, "rb") as f:
            content = f.read()
    except (PermissionError, OSError):
        return 0

    if len(content) > MAX_FILE_SIZE:
        return 0

    content_hash = _file_hash(content)

    # Skip if unchanged
    existing_hash = ci_index.get_file_hash(project_id, file_path)
    if existing_hash == content_hash:
        return 0

    # Parse
    from tree_sitter import Parser, Language

    # Get the right language module
    ts_language = _get_ts_language(language)
    if ts_language is None:
        return 0

    ts_parser = Parser(ts_language)
    tree = ts_parser.parse(content)

    symbols = parser.extract_symbols(tree, content, file_path)
    references = parser.extract_references(tree, content, file_path)
    imports = parser.extract_imports(tree, content, file_path)

    # Store
    ci_index.store_file_symbols(project_id, file_path, symbols, references, imports)
    ci_index.mark_file_indexed(project_id, file_path, language, content_hash, len(symbols))

    # Method body fingerprints — populates ci_method_bodies for the
    # find_similar_methods tool. Runs after store_file_symbols because
    # it reads the freshly inserted ci_symbols rows for line ranges.
    # Best-effort: any failure here must not block the main index.
    try:
        from infinidev.code_intel.method_index import index_methods_for_file
        index_methods_for_file(project_id, file_path, content, language)
    except Exception as exc:
        logger.debug("method_index hook failed for %s: %s", file_path, exc)

    # File integrity check — the single-source-of-truth for "did a
    # change on disk leave this file in a broken syntactic state?".
    # This fires on EVERY path that calls index_file: direct writes
    # by file tools (create_file, replace_lines, etc.), the background
    # file watcher when an external edit or shell redirect lands, and
    # the /reindex slash command. Because ``ensure_indexed`` short-
    # circuits on unchanged content hashes, the check only runs when
    # something actually changed — no duplicate work across trigger
    # paths.
    #
    # On valid content, ``push_notification`` with an empty list
    # auto-heals the queue for this file. On broken content, it pushes
    # a dedup'd entry the engine will drain into the next prompt.
    # Best-effort — never let a check_syntax bug break the indexer.
    try:
        from infinidev.code_intel.syntax_check import check_syntax
        from infinidev.code_intel.file_change_notifications import push_notification
        try:
            text = content.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        if text:
            issues = check_syntax(text, file_path=file_path)
            push_notification(file_path, language, issues)
    except Exception as exc:
        logger.debug("integrity check failed for %s: %s", file_path, exc)

    return len(symbols)


def reindex_file(project_id: int, file_path: str) -> int:
    """Force reindex of a file (clears old data first)."""
    ci_index.clear_file(project_id, file_path)

    # Remove cached hash so index_file doesn't skip
    def _clear_hash(conn):
        conn.execute(
            "DELETE FROM ci_files WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        )
        conn.commit()

    from infinidev.code_intel._db import execute_with_retry
    execute_with_retry(_clear_hash)

    return index_file(project_id, file_path)


def index_directory(
    project_id: int,
    dir_path: str,
    *,
    verbose: bool = False,
) -> dict[str, int]:
    """Index all supported source files in a directory tree.

    Returns dict with stats: files_indexed, symbols_total, files_skipped, elapsed_ms.
    """
    start = time.time()
    files_indexed = 0
    symbols_total = 0
    files_skipped = 0

    dir_path = os.path.abspath(dir_path)

    for root, dirs, files in os.walk(dir_path):
        # Skip ignored directories and nested git repos (not the root)
        filtered = []
        for d in dirs:
            if d in SKIP_DIRS or d.startswith("."):
                continue
            # Skip subdirectories that are independent git repos
            sub = os.path.join(root, d)
            if sub != dir_path and os.path.isdir(os.path.join(sub, ".git")):
                continue
            filtered.append(d)
        dirs[:] = filtered

        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext.lower() in SKIP_EXTENSIONS:
                files_skipped += 1
                continue

            fpath = os.path.join(root, fname)
            try:
                count = index_file(project_id, fpath)
                if count > 0:
                    files_indexed += 1
                    symbols_total += count
                    if verbose:
                        logger.info("Indexed %s (%d symbols)", fpath, count)
                else:
                    files_skipped += 1
            except Exception as exc:
                logger.debug("Failed to index %s: %s", fpath, exc)
                files_skipped += 1

    elapsed = int((time.time() - start) * 1000)
    stats = {
        "files_indexed": files_indexed,
        "symbols_total": symbols_total,
        "files_skipped": files_skipped,
        "elapsed_ms": elapsed,
    }
    logger.info(
        "Index complete: %d files, %d symbols in %dms",
        files_indexed, symbols_total, elapsed,
    )
    return stats


def _get_ts_language(language: str):
    """Get the tree-sitter Language object for a language name.

    Routes the dedicated grammars (python, js, ts, rust, c) to their
    bespoke imports, and delegates the long-tail languages (go, java,
    ruby, csharp, php, kotlin, bash, cpp, tsx) to the centralised
    loader registry in ``code_intel/syntax_check.py``. The centralised
    registry is the same one the file-skeleton extractor uses, so
    the indexer and the skeleton extractor stay in lockstep — adding
    a new language is one entry in that registry, not two.
    """
    from tree_sitter import Language

    try:
        if language == "python":
            import tree_sitter_python as ts
            return Language(ts.language())
        elif language == "javascript":
            import tree_sitter_javascript as ts
            return Language(ts.language())
        elif language == "typescript":
            import tree_sitter_typescript as ts
            return Language(ts.language_typescript())
        elif language == "rust":
            import tree_sitter_rust as ts
            return Language(ts.language())
        elif language == "c":
            import tree_sitter_c as ts
            return Language(ts.language())
    except ImportError:
        logger.warning("tree-sitter grammar not installed for %s", language)
        return None

    # Long-tail languages — go through the shared loader registry that
    # also powers the skeleton extractor. Returning None for an unknown
    # language is the same contract as the original code path, so the
    # indexer treats it as "skip this file".
    try:
        from infinidev.code_intel.syntax_check import _load_parser
        parser = _load_parser(language)
        if parser is not None:
            # _load_parser returns a Parser; extract its language attribute.
            return getattr(parser, "language", None)
    except Exception:
        pass
    return None
