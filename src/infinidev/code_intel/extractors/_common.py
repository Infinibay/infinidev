"""Helpers shared by per-language symbol extractors."""

from __future__ import annotations

from typing import Any


def node_name(node: Any, source: bytes) -> str | None:
    """Return the text of a node's identifier child, or None.

    Covers the three identifier-ish node types tree-sitter exposes for
    most languages (``identifier``, ``type_identifier``,
    ``property_identifier``). Shared across extractors because the
    identifier-lookup pattern is the same everywhere; only the
    *container* node types differ per language.
    """
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "property_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return None
