"""Parser registry — maps languages to their tree-sitter parsers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infinidev.code_intel.parsers.base import LanguageParser

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
