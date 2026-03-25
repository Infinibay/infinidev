"""Parser registry — maps languages to their tree-sitter parsers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infinidev.code_intel.parsers.base import LanguageParser

# File extension → language name
EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".rb": "ruby",
    ".dart": "dart",
    ".cs": "csharp",
    ".go": "go",
    ".java": "java",
}


def get_parser(language: str) -> LanguageParser | None:
    """Return the parser for a language, or None if unsupported."""
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
    return None


def detect_language(file_path: str) -> str | None:
    """Detect language from file extension."""
    import os
    _, ext = os.path.splitext(file_path)
    return EXTENSIONS.get(ext.lower())
