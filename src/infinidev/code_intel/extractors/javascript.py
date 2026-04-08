"""JavaScript / TypeScript top-level symbol extraction."""

from __future__ import annotations

from typing import Any

from infinidev.code_intel.extractors._common import node_name

JS_DEF_NODES = (
    "function_declaration",
    "class_declaration",
    "method_definition",
    "lexical_declaration",
)


def extract_js_symbols(root: Any, source: bytes) -> set[str]:
    """Walk a JavaScript/TypeScript tree and return symbol names.

    Same shape as the Python extractor: methods inside classes get
    qualified with the class name. Used for ``.js``, ``.jsx``, ``.ts``
    and ``.tsx`` — all four grammars share these node types.
    """
    out: set[str] = set()

    def walk(node: Any, parent_name: str = "") -> None:
        if node.type in JS_DEF_NODES:
            name = node_name(node, source)
            if name:
                qual = f"{parent_name}.{name}" if parent_name else name
                out.add(qual)
                new_parent = qual if "class" in node.type else parent_name
                for child in node.children:
                    walk(child, new_parent)
                return
        for child in node.children:
            walk(child, parent_name)

    walk(root)
    return out
