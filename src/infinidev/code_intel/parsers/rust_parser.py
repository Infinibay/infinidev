"""Rust parser using tree-sitter."""

from __future__ import annotations

from tree_sitter import Node, Tree

from infinidev.code_intel.models import Symbol, SymbolKind, Reference, Import


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


class RustParser:
    """Extract symbols, references, and imports from Rust source."""

    def extract_symbols(self, tree: Tree, source: bytes, file_path: str) -> list[Symbol]:
        symbols: list[Symbol] = []
        self._walk(tree.root_node, source, file_path, symbols, parent="")
        return symbols

    def _walk(self, node: Node, source: bytes, fp: str, symbols: list, parent: str) -> None:
        if node.type == "function_item":
            symbols.append(self._parse_function(node, source, fp, parent))
        elif node.type == "struct_item":
            symbols.append(self._parse_struct(node, source, fp))
        elif node.type == "enum_item":
            symbols.append(self._parse_enum(node, source, fp))
        elif node.type == "trait_item":
            symbols.append(self._parse_trait(node, source, fp))
        elif node.type == "impl_item":
            # Extract impl target name, use as parent for methods
            impl_name = ""
            for child in node.children:
                if child.type == "type_identifier":
                    impl_name = _node_text(child, source)
                    break
            for child in node.children:
                if child.type == "declaration_list":
                    self._walk(child, source, fp, symbols, parent=impl_name)
            return
        elif node.type == "const_item":
            symbols.append(self._parse_const(node, source, fp))
        elif node.type == "static_item":
            symbols.append(self._parse_const(node, source, fp))
        elif node.type == "type_item":
            symbols.append(self._parse_type_alias(node, source, fp))

        for child in node.children:
            self._walk(child, source, fp, symbols, parent)

    def _parse_function(self, node: Node, source: bytes, fp: str, parent: str) -> Symbol:
        name = ""
        params = ""
        ret = ""
        is_pub = False
        is_async = False

        text = _node_text(node, source)
        if text.startswith("pub "):
            is_pub = True
        if "async " in text.split("(")[0]:
            is_async = True

        for child in node.children:
            if child.type == "identifier":
                name = _node_text(child, source)
            elif child.type == "parameters":
                params = _node_text(child, source)
            elif child.type == "type_identifier" or (child.type == "generic_type" and not ret):
                ret = _node_text(child, source)

        kind = SymbolKind.method if parent else SymbolKind.function
        sig = f"fn {name}{params}"
        if ret:
            sig += f" -> {ret}"

        return Symbol(
            name=name, qualified_name=f"{parent}::{name}" if parent else name,
            kind=kind, file_path=fp,
            line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
            signature=sig, type_annotation=ret,
            visibility="public" if is_pub else "private",
            is_async=is_async, parent_symbol=parent,
            language="rust",
        )

    def _parse_struct(self, node: Node, source: bytes, fp: str) -> Symbol:
        name = ""
        is_pub = _node_text(node, source).startswith("pub ")
        for child in node.children:
            if child.type == "type_identifier":
                name = _node_text(child, source)
                break
        return Symbol(
            name=name, qualified_name=name, kind=SymbolKind.class_,
            file_path=fp, line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=f"struct {name}",
            visibility="public" if is_pub else "private",
            language="rust",
        )

    def _parse_enum(self, node: Node, source: bytes, fp: str) -> Symbol:
        name = ""
        is_pub = _node_text(node, source).startswith("pub ")
        for child in node.children:
            if child.type == "type_identifier":
                name = _node_text(child, source)
                break
        return Symbol(
            name=name, qualified_name=name, kind=SymbolKind.enum,
            file_path=fp, line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=f"enum {name}",
            visibility="public" if is_pub else "private",
            language="rust",
        )

    def _parse_trait(self, node: Node, source: bytes, fp: str) -> Symbol:
        name = ""
        is_pub = _node_text(node, source).startswith("pub ")
        for child in node.children:
            if child.type == "type_identifier":
                name = _node_text(child, source)
                break
        return Symbol(
            name=name, qualified_name=name, kind=SymbolKind.interface,
            file_path=fp, line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=f"trait {name}",
            visibility="public" if is_pub else "private",
            language="rust",
        )

    def _parse_const(self, node: Node, source: bytes, fp: str) -> Symbol:
        name = ""
        for child in node.children:
            if child.type == "identifier":
                name = _node_text(child, source)
                break
        is_pub = _node_text(node, source).startswith("pub ")
        return Symbol(
            name=name, qualified_name=name, kind=SymbolKind.constant,
            file_path=fp, line_start=node.start_point[0] + 1,
            signature=f"const {name}",
            visibility="public" if is_pub else "private",
            language="rust",
        )

    def _parse_type_alias(self, node: Node, source: bytes, fp: str) -> Symbol:
        name = ""
        for child in node.children:
            if child.type == "type_identifier":
                name = _node_text(child, source)
                break
        return Symbol(
            name=name, qualified_name=name, kind=SymbolKind.type_alias,
            file_path=fp, line_start=node.start_point[0] + 1,
            signature=f"type {name}", language="rust",
        )

    def extract_references(self, tree: Tree, source: bytes, file_path: str) -> list[Reference]:
        def_locs: set[tuple[int, int]] = set()
        self._collect_defs(tree.root_node, source, def_locs)
        refs: list[Reference] = []
        lines = source.decode("utf-8", errors="replace").splitlines()
        self._walk_refs(tree.root_node, source, file_path, refs, def_locs, lines)
        return refs

    def _collect_defs(self, node: Node, source: bytes, locs: set) -> None:
        if node.type in ("function_item", "struct_item", "enum_item", "trait_item",
                         "const_item", "static_item", "type_item"):
            for child in node.children:
                if child.type in ("identifier", "type_identifier"):
                    locs.add((child.start_point[0], child.start_point[1]))
                    break
        for child in node.children:
            self._collect_defs(child, source, locs)

    def _walk_refs(self, node: Node, source: bytes, fp: str, refs: list, defs: set, lines: list) -> None:
        if node.type == "identifier":
            loc = (node.start_point[0], node.start_point[1])
            if loc not in defs:
                name = _node_text(node, source)
                if len(name) >= 2 and name not in ("self", "Self", "super", "crate"):
                    line_num = node.start_point[0]
                    context = lines[line_num] if line_num < len(lines) else ""
                    refs.append(Reference(
                        name=name, file_path=fp, line=line_num + 1,
                        column=node.start_point[1], context=context.strip()[:200],
                        ref_kind="usage", language="rust",
                    ))
        for child in node.children:
            self._walk_refs(child, source, fp, refs, defs, lines)

    def extract_imports(self, tree: Tree, source: bytes, file_path: str) -> list[Import]:
        imports: list[Import] = []
        self._walk_imports(tree.root_node, source, file_path, imports)
        return imports

    def _walk_imports(self, node: Node, source: bytes, fp: str, imports: list) -> None:
        if node.type == "use_declaration":
            text = _node_text(node, source).replace("use ", "").rstrip(";").strip()
            # Simple: use std::io::Read;
            parts = text.rsplit("::", 1)
            if len(parts) == 2:
                source_path, name = parts
                imports.append(Import(
                    source=source_path, name=name, file_path=fp,
                    line=node.start_point[0] + 1, language="rust",
                ))
            elif parts:
                imports.append(Import(
                    source=parts[0], name=parts[0].split("::")[-1],
                    file_path=fp, line=node.start_point[0] + 1, language="rust",
                ))
        for child in node.children:
            self._walk_imports(child, source, fp, imports)
