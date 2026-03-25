"""TypeScript parser — extends JavaScript parser with TS-specific nodes."""

from __future__ import annotations

from tree_sitter import Node, Tree

from infinidev.code_intel.models import Symbol, SymbolKind, Reference, Import
from infinidev.code_intel.parsers.javascript_parser import JavaScriptParser, _node_text


class TypeScriptParser(JavaScriptParser):
    """Extends JS parser with TypeScript-specific constructs."""

    def _walk_symbols(self, node: Node, source: bytes, fp: str, symbols: list, parent: str) -> None:
        if node.type == "interface_declaration":
            symbols.append(self._parse_interface(node, source, fp, parent))
            return

        elif node.type == "type_alias_declaration":
            symbols.append(self._parse_type_alias(node, source, fp))
            return

        elif node.type == "enum_declaration":
            symbols.append(self._parse_enum(node, source, fp))
            return

        # Delegate to JS parser for everything else
        super()._walk_symbols(node, source, fp, symbols, parent)

    def _parse_interface(self, node: Node, source: bytes, fp: str, parent: str) -> Symbol:
        name = ""
        for child in node.children:
            if child.type in ("type_identifier", "identifier"):
                name = _node_text(child, source)
                break

        return Symbol(
            name=name, qualified_name=f"{parent}.{name}" if parent else name,
            kind=SymbolKind.interface, file_path=fp,
            line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
            signature=f"interface {name}",
            parent_symbol=parent, language="typescript",
        )

    def _parse_type_alias(self, node: Node, source: bytes, fp: str) -> Symbol:
        name = ""
        for child in node.children:
            if child.type in ("type_identifier", "identifier"):
                name = _node_text(child, source)
                break

        return Symbol(
            name=name, qualified_name=name, kind=SymbolKind.type_alias,
            file_path=fp, line_start=node.start_point[0] + 1,
            signature=f"type {name}", language="typescript",
        )

    def _parse_enum(self, node: Node, source: bytes, fp: str) -> Symbol:
        name = ""
        for child in node.children:
            if child.type in ("identifier",):
                name = _node_text(child, source)
                break

        return Symbol(
            name=name, qualified_name=name, kind=SymbolKind.enum,
            file_path=fp, line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=f"enum {name}", language="typescript",
        )
