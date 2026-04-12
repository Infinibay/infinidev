"""Parser registry — maps languages to their tree-sitter parsers.

**Parser versioning.**  Each language-specific parser has a
``PARSER_VERSION`` integer that should be bumped whenever its
extraction logic changes in a way that would produce different
output for the same input file — typically a bug fix like
"accept type_identifier for TypeScript class names" or "capture
abstract class declarations correctly".

The version is stored in ``ci_files.parser_version`` alongside the
``content_hash``.  The incremental indexer's skip check compares
both fields: if the content hash matches but the parser version is
older, the file is re-parsed.  This prevents stale symbols from
surviving parser bug fixes indefinitely (which is exactly what
happened with ErrorHandler.ts classes being indexed with empty
names by a pre-fix parser — the content hash never changed so the
incremental skip kept returning the broken data for months).

To bump a parser's version: edit the corresponding entry in
``PARSER_VERSIONS`` below and include a one-line note about the
change in the git commit message.  Running ``/reindex`` on any
project will then re-process all files for that language.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infinidev.code_intel.parsers.base import LanguageParser


# Parser version per language.  Bump when extraction logic changes.
#
# Version history:
#   python=1:      initial release
#   javascript=2:  fix class_declaration name extraction accepting
#                  type_identifier (was only accepting identifier,
#                  which caused empty names on TypeScript classes
#                  extending the JS parser — same fix propagates here).
#   typescript=2:  inherits JS parser fix plus adds interface / enum
#                  / type_alias extraction.
#   rust=1:        initial release
#   c=1:           initial release
#   generic=1:     initial release (covers Go, Java, Ruby, C#, PHP,
#                  Kotlin, Bash, C++, TSX).
PARSER_VERSIONS: dict[str, int] = {
    "python":     1,
    "javascript": 2,
    "typescript": 2,
    "rust":       1,
    "c":          1,
    "config":     1,
    # Generic parser entries — all share the same version because
    # the GenericParser class is one implementation.
    "go":         1,
    "java":       1,
    "ruby":       1,
    "csharp":     1,
    "php":        1,
    "kotlin":     1,
    "bash":       1,
    "cpp":        1,
    "tsx":        1,
    "dart":       1,
}


def get_parser_version(language: str) -> int:
    """Return the current parser version for a language.

    Returns ``0`` if the language isn't in ``PARSER_VERSIONS`` — this
    effectively disables versioned skip-check for unknown languages
    (content-hash match alone will still allow skip).
    """
    return PARSER_VERSIONS.get(language, 0)


# File extension → language name
EXTENSIONS: dict[str, str] = {
    ".py":   "python",
    ".pyi":  "python",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".mjs":  "javascript",
    ".cjs":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "tsx",
    ".rs":   "rust",
    ".c":    "c",
    ".h":    "c",
    ".cc":   "cpp",
    ".cpp":  "cpp",
    ".cxx":  "cpp",
    ".hpp":  "cpp",
    ".hh":   "cpp",
    ".rb":   "ruby",
    ".dart": "dart",
    ".cs":   "csharp",
    ".go":   "go",
    ".java": "java",
    ".kt":   "kotlin",
    ".kts":  "kotlin",
    ".php":  "php",
    ".sh":   "bash",
    ".bash": "bash",
    ".zsh":  "bash",
    # Config files — tracked for change detection, no symbol parsing
    ".toml": "config",
    ".json": "config",
    ".yaml": "config",
    ".yml":  "config",
    ".cfg":  "config",
    ".ini":  "config",
}


def get_parser(language: str) -> LanguageParser | None:
    """Return the parser for a language, or None if unsupported.

    Returns a dedicated parser for the 5 languages with bespoke
    extractors (Python, JavaScript, TypeScript, Rust, C). Falls back
    to :class:`GenericParser` for any language listed in
    ``GENERIC_LANGUAGES`` (Go, Java, Ruby, C#, PHP, Kotlin, Bash,
    C++, TSX) — those go through the config-driven walker that
    reuses the same per-language node-type tables the file-skeleton
    extractor maintains. The generic parser does symbols + imports
    but not references; that's an acceptable tradeoff for unblocking
    9 languages with one module.
    """
    if language == "python":
        from infinidev.code_intel.parsers.python_parser import PythonParser
        return PythonParser()
    if language == "javascript":
        from infinidev.code_intel.parsers.javascript_parser import JavaScriptParser
        return JavaScriptParser()
    if language == "typescript":
        from infinidev.code_intel.parsers.typescript_parser import TypeScriptParser
        return TypeScriptParser()
    if language == "rust":
        from infinidev.code_intel.parsers.rust_parser import RustParser
        return RustParser()
    if language == "c":
        from infinidev.code_intel.parsers.c_parser import CParser
        return CParser()
    # Generic fallback for the long tail of languages.
    from infinidev.code_intel.parsers.generic_parser import GenericParser, GENERIC_LANGUAGES
    if language in GENERIC_LANGUAGES:
        return GenericParser(language)
    return None


def detect_language(file_path: str) -> str | None:
    """Detect language from file extension."""
    import os
    _, ext = os.path.splitext(file_path)
    return EXTENSIONS.get(ext.lower())
