"""C parser using tree-sitter."""

from __future__ import annotations

from tree_sitter import Node, Tree

from infinidev.code_intel.models import Symbol, SymbolKind, Reference, Import


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


class CParser:
    """Extract symbols, references, and imports from C source."""

    def extract_symbols(self, tree: Tree, source: bytes, file_path: str) -> list[Symbol]:
        symbols: list[Symbol] = []
        self._walk(tree.root_node, source, file_path, symbols)
        return symbols

    def _walk(self, node: Node, source: bytes, fp: str, symbols: list) -> None:
        if node.type == "function_definition":
            sym = self._parse_function(node, source, fp)
            if sym:
                symbols.append(sym)
        elif node.type == "declaration":
            # Could be function declaration, variable, typedef
            sym = self._parse_declaration(node, source, fp)
            if sym:
                symbols.append(sym)
        elif node.type == "struct_specifier":
            name = self._find_name(node, source, "type_identifier")
            if name:
                symbols.append(Symbol(
                    name=name, qualified_name=name, kind=SymbolKind.class_,
                    file_path=fp, line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=f"struct {name}", language="c",
                ))
        elif node.type == "enum_specifier":
            name = self._find_name(node, source, "type_identifier")
            if name:
                symbols.append(Symbol(
                    name=name, qualified_name=name, kind=SymbolKind.enum,
                    file_path=fp, line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=f"enum {name}", language="c",
                ))
        elif node.type == "type_definition":
            name = self._find_name(node, source, "type_identifier")
            if name:
                symbols.append(Symbol(
                    name=name, qualified_name=name, kind=SymbolKind.type_alias,
                    file_path=fp, line_start=node.start_point[0] + 1,
                    signature=f"typedef {name}", language="c",
                ))
        elif node.type == "preproc_def":
            name = self._find_name(node, source, "identifier")
            if name:
                symbols.append(Symbol(
                    name=name, qualified_name=name, kind=SymbolKind.constant,
                    file_path=fp, line_start=node.start_point[0] + 1,
                    signature=f"#define {name}", language="c",
                ))

        for child in node.children:
            self._walk(child, source, fp, symbols)

    def _parse_function(self, node: Node, source: bytes, fp: str) -> Symbol | None:
        name = ""
        params = ""
        ret_type = ""
        is_static = _node_text(node, source).lstrip().startswith("static ")

        for child in node.children:
            if child.type == "function_declarator":
                for sub in child.children:
                    if sub.type == "identifier":
                        name = _node_text(sub, source)
                    elif sub.type == "parameter_list":
                        params = _node_text(sub, source)
            elif child.type in ("primitive_type", "type_identifier", "sized_type_specifier"):
                ret_type = _node_text(child, source)

        if not name:
            return None

        sig = f"{ret_type} {name}{params}" if ret_type else f"{name}{params}"
        return Symbol(
            name=name, qualified_name=name, kind=SymbolKind.function,
            file_path=fp, line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=sig.strip(), type_annotation=ret_type,
            visibility="private" if is_static else "public",
            is_static=is_static, language="c",
        )

    def _parse_declaration(self, node: Node, source: bytes, fp: str) -> Symbol | None:
        """Parse top-level declarations (global variables, function prototypes)."""
        text = _node_text(node, source).strip()
        # Skip if it looks like a function prototype (handled by function_definition)
        if "(" in text and ")" in text and not text.endswith(";"):
            return None
        return None  # Skip complex declarations for now

    def _find_name(self, node: Node, source: bytes, name_type: str) -> str:
        for child in node.children:
            if child.type == name_type:
                return _node_text(child, source)
        return ""

    def extract_references(self, tree: Tree, source: bytes, file_path: str) -> list[Reference]:
        def_locs: set[tuple[int, int]] = set()
        self._collect_defs(tree.root_node, source, def_locs)
        refs: list[Reference] = []
        lines = source.decode("utf-8", errors="replace").splitlines()
        self._walk_refs(tree.root_node, source, file_path, refs, def_locs, lines)
        return refs

    def _collect_defs(self, node: Node, source: bytes, locs: set) -> None:
        if node.type == "function_declarator":
            for child in node.children:
                if child.type == "identifier":
                    locs.add((child.start_point[0], child.start_point[1]))
        elif node.type == "preproc_def":
            for child in node.children:
                if child.type == "identifier":
                    locs.add((child.start_point[0], child.start_point[1]))
        for child in node.children:
            self._collect_defs(child, source, locs)

    def _walk_refs(self, node: Node, source: bytes, fp: str, refs: list, defs: set, lines: list) -> None:
        if node.type == "identifier":
            loc = (node.start_point[0], node.start_point[1])
            if loc not in defs:
                name = _node_text(node, source)
                if len(name) >= 2 and name not in ("int", "char", "void", "float", "double",
                                                     "long", "short", "unsigned", "signed",
                                                     "NULL", "sizeof", "return", "if", "else",
                                                     "for", "while", "do", "switch", "case",
                                                     "break", "continue", "struct", "enum",
                                                     "typedef", "static", "extern", "const"):
                    line_num = node.start_point[0]
                    context = lines[line_num] if line_num < len(lines) else ""
                    ref_kind = "usage"
                    if node.parent and node.parent.type == "call_expression":
                        ref_kind = "call"
                    refs.append(Reference(
                        name=name, file_path=fp, line=line_num + 1,
                        column=node.start_point[1], context=context.strip()[:200],
                        ref_kind=ref_kind, language="c",
                    ))
        for child in node.children:
            self._walk_refs(child, source, fp, refs, defs, lines)

    def extract_imports(self, tree: Tree, source: bytes, file_path: str) -> list[Import]:
        imports: list[Import] = []
        self._walk_imports(tree.root_node, source, file_path, imports)
        return imports

    def _walk_imports(self, node: Node, source: bytes, fp: str, imports: list) -> None:
        if node.type == "preproc_include":
            path = ""
            for child in node.children:
                if child.type in ("string_literal", "system_lib_string"):
                    path = _node_text(child, source).strip('<>"')
            if path:
                name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                imports.append(Import(
                    source=path, name=name, file_path=fp,
                    line=node.start_point[0] + 1, language="c",
                ))
        for child in node.children:
            self._walk_imports(child, source, fp, imports)
