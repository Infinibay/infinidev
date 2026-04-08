"""Python top-level symbol extraction."""

from __future__ import annotations

from typing import Any

from infinidev.code_intel.extractors._common import node_name

PYTHON_DEF_NODES = ("function_definition", "class_definition")


def extract_python_symbols(root: Any, source: bytes) -> set[str]:
    """Walk a Python tree and return a set of qualified symbol names.

    Methods defined inside a ``class`` body are qualified with the
    class name (``MyClass.my_method``) so the caller can distinguish
    them from free functions.
    """
    out: set[str] = set()

    def walk(node: Any, parent_name: str = "") -> None:
        if node.type in PYTHON_DEF_NODES:
            name = node_name(node, source)
            if name:
                qual = f"{parent_name}.{name}" if parent_name else name
                out.add(qual)
                # Recurse into class/function body so methods are
                # captured under the class name.
                new_parent = qual if node.type == "class_definition" else parent_name
                for child in node.children:
                    if child.type == "block":
                        for grand in child.children:
                            walk(grand, new_parent)
                return
        for child in node.children:
            walk(child, parent_name)

    walk(root)
    return out
