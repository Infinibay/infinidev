"""JavaScript parser using tree-sitter."""

from __future__ import annotations

import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser, Node, Tree

from infinidev.code_intel.models import Symbol, SymbolKind, Reference, Import

JS_LANGUAGE = Language(tsjs.language())


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_docstring(node: Node, source: bytes) -> str:
    """Get preceding comment as docstring."""
    prev = node.prev_named_sibling
    if prev and prev.type == "comment":
        text = _node_text(prev, source).strip("/* \n\t").strip("// ").strip()
        return text.split("\n")[0][:200]
    return ""


class JavaScriptParser:
    """Extract symbols, references, and imports from JavaScript source."""

    def extract_symbols(self, tree: Tree, source: bytes, file_path: str) -> list[Symbol]:
        symbols: list[Symbol] = []
        self._walk_symbols(tree.root_node, source, file_path, symbols, parent="")
        return symbols

    def _walk_symbols(self, node: Node, source: bytes, fp: str, symbols: list, parent: str) -> None:
        if node.type == "function_declaration":
            symbols.append(self._parse_function(node, source, fp, parent))

        elif node.type == "class_declaration":
            sym = self._parse_class(node, source, fp, parent)
            symbols.append(sym)
            for child in node.children:
                if child.type == "class_body":
                    self._walk_symbols(child, source, fp, symbols, parent=sym.name)
            return

        elif node.type == "method_definition":
            symbols.append(self._parse_method(node, source, fp, parent))

        elif node.type in ("lexical_declaration", "variable_declaration"):
            for child in node.children:
                if child.type == "variable_declarator":
                    sym = self._parse_variable(child, node, source, fp)
                    if sym:
                        symbols.append(sym)

        elif node.type == "export_statement":
            # Process exported declarations
            for child in node.children:
                self._walk_symbols(child, source, fp, symbols, parent)
            return

        for child in node.children:
            self._walk_symbols(child, source, fp, symbols, parent)

    def _parse_function(self, node: Node, source: bytes, fp: str, parent: str) -> Symbol:
        name = ""
        params = ""
        is_async = "async" in _node_text(node, source).split("(")[0]

        for child in node.children:
            if child.type == "identifier":
                name = _node_text(child, source)
            elif child.type == "formal_parameters":
                params = _node_text(child, source)

        return Symbol(
            name=name, qualified_name=f"{parent}.{name}" if parent else name,
            kind=SymbolKind.method if parent else SymbolKind.function,
            file_path=fp, line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=f"function {name}{params}",
            is_async=is_async,
            docstring=_get_docstring(node, source),
            parent_symbol=parent, language="javascript",
        )

    def _parse_class(self, node: Node, source: bytes, fp: str, parent: str) -> Symbol:
        name = ""
        bases = ""
        for child in node.children:
            if child.type == "identifier":
                name = _node_text(child, source)
            elif child.type == "class_heritage":
                bases = _node_text(child, source)

        sig = f"class {name} {bases}" if bases else f"class {name}"
        return Symbol(
            name=name, qualified_name=f"{parent}.{name}" if parent else name,
            kind=SymbolKind.class_, file_path=fp,
            line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
            signature=sig, docstring=_get_docstring(node, source),
            parent_symbol=parent, language="javascript",
        )

    def _parse_method(self, node: Node, source: bytes, fp: str, parent: str) -> Symbol:
        name = ""
        params = ""
        is_async = False
        is_static = False
        kind = SymbolKind.method

        for child in node.children:
            if child.type in ("property_identifier", "identifier"):
                name = _node_text(child, source)
            elif child.type == "formal_parameters":
                params = _node_text(child, source)

        text = _node_text(node, source)
        if text.startswith("static "):
            is_static = True
        if "async " in text.split("(")[0]:
            is_async = True
        if text.startswith("get ") or text.startswith("static get "):
            kind = SymbolKind.property_
        if text.startswith("set ") or text.startswith("static set "):
            kind = SymbolKind.property_

        visibility = "private" if name.startswith("#") else "public"

        return Symbol(
            name=name, qualified_name=f"{parent}.{name}" if parent else name,
            kind=kind, file_path=fp,
            line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
            signature=f"{name}{params}",
            is_async=is_async, is_static=is_static, visibility=visibility,
            parent_symbol=parent, language="javascript",
        )

    def _parse_variable(self, node: Node, decl_node: Node, source: bytes, fp: str) -> Symbol | None:
        name = ""
        is_arrow = False
        for child in node.children:
            if child.type == "identifier":
                name = _node_text(child, source)
            elif child.type == "arrow_function":
                is_arrow = True

        if not name:
            return None

        decl_keyword = ""
        for child in decl_node.children:
            if child.type in ("const", "let", "var"):
                decl_keyword = child.type
                break

        if is_arrow:
            kind = SymbolKind.function
            sig = f"{decl_keyword} {name} = (...) =>"
        elif decl_keyword == "const" and name.isupper():
            kind = SymbolKind.constant
            sig = f"const {name}"
        else:
            kind = SymbolKind.variable
            sig = f"{decl_keyword} {name}"

        return Symbol(
            name=name, qualified_name=name, kind=kind, file_path=fp,
            line_start=node.start_point[0] + 1, signature=sig,
            language="javascript",
        )

    def extract_references(self, tree: Tree, source: bytes, file_path: str) -> list[Reference]:
        def_locs: set[tuple[int, int]] = set()
        self._collect_defs(tree.root_node, source, def_locs)
        refs: list[Reference] = []
        lines = source.decode("utf-8", errors="replace").splitlines()
        self._walk_refs(tree.root_node, source, file_path, refs, def_locs, lines)
        return refs

    def _collect_defs(self, node: Node, source: bytes, locs: set) -> None:
        if node.type in ("function_declaration", "class_declaration", "method_definition"):
            for child in node.children:
                if child.type in ("identifier", "property_identifier"):
                    locs.add((child.start_point[0], child.start_point[1]))
        elif node.type == "variable_declarator":
            for child in node.children:
                if child.type == "identifier":
                    locs.add((child.start_point[0], child.start_point[1]))
                    break
        for child in node.children:
            self._collect_defs(child, source, locs)

    def _walk_refs(self, node: Node, source: bytes, fp: str, refs: list, defs: set, lines: list) -> None:
        if node.type == "identifier":
            loc = (node.start_point[0], node.start_point[1])
            if loc not in defs:
                name = _node_text(node, source)
                if len(name) >= 2 and name not in ("this", "true", "false", "null", "undefined"):
                    line_num = node.start_point[0]
                    context = lines[line_num] if line_num < len(lines) else ""
                    ref_kind = "usage"
                    if node.parent and node.parent.type == "call_expression" and node == node.parent.children[0]:
                        ref_kind = "call"
                    refs.append(Reference(
                        name=name, file_path=fp, line=line_num + 1,
                        column=node.start_point[1], context=context.strip()[:200],
                        ref_kind=ref_kind, language="javascript",
                    ))
        for child in node.children:
            self._walk_refs(child, source, fp, refs, defs, lines)

    def extract_imports(self, tree: Tree, source: bytes, file_path: str) -> list[Import]:
        imports: list[Import] = []
        self._walk_imports(tree.root_node, source, file_path, imports)
        return imports

    def _walk_imports(self, node: Node, source: bytes, fp: str, imports: list) -> None:
        if node.type == "import_statement":
            source_str = ""
            names: list[tuple[str, str]] = []  # (name, alias)

            for child in node.children:
                if child.type == "string":
                    source_str = _node_text(child, source).strip("'\"")
                elif child.type == "import_clause":
                    for sub in child.children:
                        if sub.type == "identifier":
                            names.append((_node_text(sub, source), ""))
                        elif sub.type == "named_imports":
                            for spec in sub.children:
                                if spec.type == "import_specifier":
                                    name = alias = ""
                                    for s in spec.children:
                                        if s.type == "identifier":
                                            if not name:
                                                name = _node_text(s, source)
                                            else:
                                                alias = _node_text(s, source)
                                    if name:
                                        names.append((name, alias))
                        elif sub.type == "namespace_import":
                            for s in sub.children:
                                if s.type == "identifier":
                                    names.append((_node_text(s, source), ""))

            for name, alias in names:
                imports.append(Import(
                    source=source_str, name=name, alias=alias,
                    file_path=fp, line=node.start_point[0] + 1,
                    language="javascript",
                ))

        for child in node.children:
            self._walk_imports(child, source, fp, imports)
