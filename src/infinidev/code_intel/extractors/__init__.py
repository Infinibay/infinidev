"""Per-language symbol extractors.

Each module in this package walks a tree-sitter AST for one language
and returns the set of qualified top-level symbol names. Splitting by
language means adding support for a new language (e.g. Go, Rust) does
not require editing ``syntax_check.py`` — just drop a new extractor
here and register it in ``LANGUAGE_EXTRACTORS``.
"""

from __future__ import annotations

from typing import Any, Callable

from infinidev.code_intel.extractors.python import extract_python_symbols
from infinidev.code_intel.extractors.javascript import extract_js_symbols

# Registry: language key → extractor callable. ``syntax_check`` looks up
# by normalized language name; unknown languages fall back to no-op.
LANGUAGE_EXTRACTORS: dict[str, Callable[[Any, bytes], set[str]]] = {
    "python": extract_python_symbols,
    "javascript": extract_js_symbols,
    "typescript": extract_js_symbols,
    "tsx": extract_js_symbols,
    "jsx": extract_js_symbols,
}

__all__ = [
    "LANGUAGE_EXTRACTORS",
    "extract_python_symbols",
    "extract_js_symbols",
]
