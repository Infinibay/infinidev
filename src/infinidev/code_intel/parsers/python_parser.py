"""Python parser using tree-sitter."""

from __future__ import annotations

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node, Tree

from infinidev.code_intel.models import Symbol, SymbolKind, Reference, Import

PY_LANGUAGE = Language(tspython.language())


def _node_text(node: Node, source: bytes) -> str:
    """Extract text content of a node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_docstring(body_node: Node, source: bytes) -> str:
    """Extract the first-line docstring from a function/class body."""
    if body_node is None or body_node.child_count == 0:
        return ""
    first = body_node.children[0]
    if first.type == "expression_statement" and first.child_count > 0:
        expr = first.children[0]
        if expr.type == "string":
            doc = _node_text(expr, source).strip("\"'").strip()
            # First line only
            return doc.split("\n")[0][:200]
    return ""


def _get_parent_class(node: Node) -> str:
    """Walk up the tree to find the enclosing class name."""
    current = node.parent
    while current:
        if current.type == "class_definition":
            for child in current.children:
                if child.type == "name":
                    return _node_text(child, current.text)
        current = current.parent
    return ""


class PythonParser:
    """Extract symbols, references, and imports from Python source."""

    def extract_symbols(self, tree: Tree, source: bytes, file_path: str) -> list[Symbol]:
        symbols: list[Symbol] = []
        self._walk_for_symbols(tree.root_node, source, file_path, symbols, parent="")
        return symbols

    def _walk_for_symbols(
        self, node: Node, source: bytes, file_path: str,
        symbols: list[Symbol], parent: str,
    ) -> None:
        if node.type == "function_definition":
            symbols.append(self._parse_function(node, source, file_path, parent))
        elif node.type == "class_definition":
            sym = self._parse_class(node, source, file_path, parent)
            symbols.append(sym)
            # Recurse into class body with class name as parent
            for child in node.children:
                self._walk_for_symbols(child, source, file_path, symbols, parent=sym.name)
            return  # Already recursed
        elif node.type == "expression_statement" and parent == "":
            # Top-level assignments → variables/constants
            for child in node.children:
                if child.type == "assignment":
                    sym = self._parse_assignment(child, source, file_path)
                    if sym:
                        symbols.append(sym)
        elif node.type == "decorated_definition":
            # Recurse into the decorated definition
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    self._walk_for_symbols(child, source, file_path, symbols, parent)

        # Recurse into all children
        for child in node.children:
            self._walk_for_symbols(child, source, file_path, symbols, parent)

    def _parse_function(self, node: Node, source: bytes, file_path: str, parent: str) -> Symbol:
        name = ""
        params_text = ""
        return_type = ""
        is_async = False
        is_static = False
        is_abstract = False
        docstring = ""

        # Check for async
        if node.parent and node.parent.type == "decorated_definition":
            decorated = node.parent
        else:
            decorated = None

        # Check decorators
        decorators = []
        if decorated:
            for child in decorated.children:
                if child.type == "decorator":
                    dec_text = _node_text(child, source).strip("@").strip()
                    decorators.append(dec_text)

        if "staticmethod" in decorators:
            is_static = True
        if "abstractmethod" in decorators:
            is_abstract = True

        # Check async
        prev = node.prev_named_sibling
        if prev and prev.type == "async":
            is_async = True
        # Also check in parent context
        full_text = _node_text(node, source)
        if full_text.startswith("async "):
            is_async = True

        for child in node.children:
            if child.type == "identifier":
                name = _node_text(child, source)
            elif child.type == "parameters":
                params_text = _node_text(child, source)
            elif child.type == "type":
                return_type = _node_text(child, source)
            elif child.type == "block":
                docstring = _get_docstring(child, source)

        # Determine kind
        kind = SymbolKind.method if parent else SymbolKind.function
        if "property" in decorators:
            kind = SymbolKind.property_

        # Visibility
        visibility = "public"
        if name.startswith("__") and name.endswith("__"):
            visibility = "public"  # dunder methods are public
        elif name.startswith("__"):
            visibility = "private"
        elif name.startswith("_"):
            visibility = "protected"

        # Signature
        sig = f"def {name}{params_text}"
        if return_type:
            sig += f" -> {return_type}"

        qualified = f"{parent}.{name}" if parent else name

        return Symbol(
            name=name,
            qualified_name=qualified,
            kind=kind,
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            column_start=node.start_point[1],
            signature=sig,
            type_annotation=return_type,
            docstring=docstring,
            parent_symbol=parent,
            visibility=visibility,
            is_async=is_async,
            is_static=is_static,
            is_abstract=is_abstract,
            language="python",
        )

    def _parse_class(self, node: Node, source: bytes, file_path: str, parent: str) -> Symbol:
        name = ""
        bases = ""
        docstring = ""

        for child in node.children:
            if child.type == "identifier":
                name = _node_text(child, source)
            elif child.type == "argument_list":
                bases = _node_text(child, source)
            elif child.type == "block":
                docstring = _get_docstring(child, source)

        sig = f"class {name}{bases}" if bases else f"class {name}"
        qualified = f"{parent}.{name}" if parent else name

        return Symbol(
            name=name,
            qualified_name=qualified,
            kind=SymbolKind.class_,
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            column_start=node.start_point[1],
            signature=sig,
            docstring=docstring,
            parent_symbol=parent,
            language="python",
        )

    def _parse_assignment(self, node: Node, source: bytes, file_path: str) -> Symbol | None:
        """Parse a top-level assignment as a variable or constant."""
        if node.child_count < 1:
            return None

        left = node.children[0]
        if left.type != "identifier":
            return None

        name = _node_text(left, source)
        # UPPER_CASE → constant
        kind = SymbolKind.constant if name.isupper() else SymbolKind.variable

        # Try to get type annotation
        type_ann = ""
        for child in node.children:
            if child.type == "type":
                type_ann = _node_text(child, source)

        return Symbol(
            name=name,
            qualified_name=name,
            kind=kind,
            file_path=file_path,
            line_start=node.start_point[0] + 1,
            column_start=node.start_point[1],
            type_annotation=type_ann,
            language="python",
        )

    def extract_references(self, tree: Tree, source: bytes, file_path: str) -> list[Reference]:
        """Extract all identifier references that are not definitions."""
        # First, collect all definition locations
        def_locations: set[tuple[int, int]] = set()  # (line, col)
        self._collect_def_locations(tree.root_node, source, def_locations)

        # Then walk all identifiers
        refs: list[Reference] = []
        lines = source.decode("utf-8", errors="replace").splitlines()
        self._walk_for_refs(tree.root_node, source, file_path, refs, def_locations, lines)
        return refs

    def _collect_def_locations(self, node: Node, source: bytes, locations: set) -> None:
        if node.type in ("function_definition", "class_definition"):
            for child in node.children:
                if child.type == "name":
                    locations.add((child.start_point[0], child.start_point[1]))
        elif node.type == "assignment":
            left = node.children[0] if node.children else None
            if left and left.type == "identifier":
                locations.add((left.start_point[0], left.start_point[1]))
        elif node.type == "parameter":
            for child in node.children:
                if child.type == "identifier":
                    locations.add((child.start_point[0], child.start_point[1]))

        for child in node.children:
            self._collect_def_locations(child, source, locations)

    def _walk_for_refs(
        self, node: Node, source: bytes, file_path: str,
        refs: list[Reference], def_locations: set, lines: list[str],
    ) -> None:
        if node.type == "identifier":
            loc = (node.start_point[0], node.start_point[1])
            if loc not in def_locations:
                name = _node_text(node, source)
                # Skip builtins and very short names
                if len(name) >= 2 and name not in ("self", "cls", "True", "False", "None"):
                    line_num = node.start_point[0]
                    context = lines[line_num] if line_num < len(lines) else ""

                    # Classify ref_kind
                    ref_kind = "usage"
                    parent = node.parent
                    if parent:
                        if parent.type == "call" and node == parent.children[0]:
                            ref_kind = "call"
                        elif parent.type in ("type", "type_identifier"):
                            ref_kind = "type_ref"

                    refs.append(Reference(
                        name=name,
                        file_path=file_path,
                        line=line_num + 1,
                        column=node.start_point[1],
                        context=context.strip()[:200],
                        ref_kind=ref_kind,
                        language="python",
                    ))

        for child in node.children:
            self._walk_for_refs(child, source, file_path, refs, def_locations, lines)

    def extract_imports(self, tree: Tree, source: bytes, file_path: str) -> list[Import]:
        imports: list[Import] = []
        self._walk_for_imports(tree.root_node, source, file_path, imports)
        return imports

    def _walk_for_imports(self, node: Node, source: bytes, file_path: str, imports: list[Import]) -> None:
        if node.type == "import_statement":
            # import foo, import foo.bar
            for child in node.children:
                if child.type == "dotted_name":
                    name = _node_text(child, source)
                    imports.append(Import(
                        source=name,
                        name=name.split(".")[-1],
                        file_path=file_path,
                        line=node.start_point[0] + 1,
                        language="python",
                    ))
                elif child.type == "aliased_import":
                    dotted = alias = None
                    for sub in child.children:
                        if sub.type == "dotted_name":
                            dotted = _node_text(sub, source)
                        elif sub.type == "identifier":
                            alias = _node_text(sub, source)
                    if dotted:
                        imports.append(Import(
                            source=dotted,
                            name=dotted.split(".")[-1],
                            alias=alias or "",
                            file_path=file_path,
                            line=node.start_point[0] + 1,
                            language="python",
                        ))

        elif node.type == "import_from_statement":
            # from foo import bar, baz
            module = ""
            for child in node.children:
                if child.type == "dotted_name" and not module:
                    module = _node_text(child, source)
                elif child.type == "relative_import":
                    module = _node_text(child, source)
                elif child.type == "dotted_name" and module:
                    name = _node_text(child, source)
                    imports.append(Import(
                        source=module,
                        name=name,
                        file_path=file_path,
                        line=node.start_point[0] + 1,
                        language="python",
                    ))
                elif child.type == "aliased_import":
                    name = alias = None
                    for sub in child.children:
                        if sub.type == "dotted_name":
                            name = _node_text(sub, source)
                        elif sub.type == "identifier" and name:
                            alias = _node_text(sub, source)
                    if name:
                        imports.append(Import(
                            source=module,
                            name=name,
                            alias=alias or "",
                            file_path=file_path,
                            line=node.start_point[0] + 1,
                            language="python",
                        ))
                elif child.type == "wildcard_import":
                    imports.append(Import(
                        source=module,
                        name="*",
                        is_wildcard=True,
                        file_path=file_path,
                        line=node.start_point[0] + 1,
                        language="python",
                    ))

        for child in node.children:
            self._walk_for_imports(child, source, file_path, imports)


def parse_file(source: bytes, file_path: str) -> tuple[list[Symbol], list[Reference], list[Import]]:
    """Convenience function: parse a Python file and return all extracted data."""
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(source)
    p = PythonParser()
    return p.extract_symbols(tree, source, file_path), p.extract_references(tree, source, file_path), p.extract_imports(tree, source, file_path)
