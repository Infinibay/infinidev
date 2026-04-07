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


def _find_enclosing_name(node: Node, source: bytes) -> str:
    """Walk up from *node* to find the nearest named enclosing scope.

    Used by the JS/TS parser to rescue methods defined inside object
    literals — things like::

        const migration = {
            up() { ... },
            down() { ... },
        }

        export function createHelpers() {
            return {
                isAuthenticated() { ... },
                hasRole() { ... },
            }
        }

    Tree-sitter treats these ``method_definition`` nodes as children
    of an ``object`` rather than a ``class_body``, so the default
    walker leaves them with ``parent_symbol=""``. This helper walks
    up from the method node looking for the nearest ancestor that
    names a scope:

      * **variable_declarator** — ``const foo = { ... }``. Return the
        first identifier child (``foo``).
      * **function_declaration** — factory that returns an object.
        Return the function's identifier child.
      * **pair** — ``{ foo: { bar() {} } }``. Return the key's text
        (``foo``) as the parent name for ``bar``.
      * **assignment_expression** — ``module.exports = { ... }`` or
        similar. Return the left-hand side's identifier text.

    Returns ``""`` when no named ancestor is found (e.g. an anonymous
    object literal passed as a function argument — there's genuinely
    no sensible parent name for those).
    """
    current = node.parent
    while current is not None:
        t = current.type

        if t == "variable_declarator":
            for child in current.children:
                if child.type in ("identifier", "property_identifier"):
                    return _node_text(child, source)
            return ""

        if t == "function_declaration":
            for child in current.children:
                if child.type == "identifier":
                    return _node_text(child, source)
            return ""

        if t == "pair":
            # Object key:value pair. The first child is the key —
            # could be a string, identifier, or property_identifier.
            for child in current.children:
                if child.type in ("property_identifier", "identifier"):
                    return _node_text(child, source)
                if child.type == "string":
                    # Strip the surrounding quotes for clean parent names.
                    return _node_text(child, source).strip('"\'`')
            return ""

        if t == "assignment_expression":
            # LHS is the first child. Could be an identifier
            # (``foo = {}``) or a member_expression
            # (``module.exports = {}``).
            if current.children:
                lhs = current.children[0]
                if lhs.type == "identifier":
                    return _node_text(lhs, source)
                if lhs.type == "member_expression":
                    # Use the last property_identifier — for
                    # ``module.exports = {}`` that's "exports".
                    last_prop = ""
                    for sub in lhs.children:
                        if sub.type == "property_identifier":
                            last_prop = _node_text(sub, source)
                    return last_prop
            return ""

        # Don't walk through another class — a method_definition that
        # sits inside a class_body should have been handled by the
        # caller with parent=ClassName. If we reach a class_declaration
        # while walking up, something went wrong upstream — return
        # empty so we don't produce a weird qualified name.
        if t in ("class_declaration", "abstract_class_declaration"):
            return ""

        current = current.parent

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

        elif node.type in ("class_declaration", "abstract_class_declaration"):
            # ``abstract_class_declaration`` is the TypeScript-specific node
            # type for ``abstract class Foo {}``. The body extraction and
            # name handling are identical to a regular class — the only
            # difference is the wrapper type. Treating both the same lets
            # abstract classes' methods get parent_symbol propagation
            # without duplicating any walker logic.
            sym = self._parse_class(node, source, fp, parent)
            symbols.append(sym)
            for child in node.children:
                if child.type == "class_body":
                    self._walk_symbols(child, source, fp, symbols, parent=sym.name)
            return

        elif node.type == "method_definition":
            # A method_definition node can appear in three contexts:
            #
            #   1. Inside a class_body — handled via the class_declaration
            #      branch above which recurses with parent=ClassName.
            #      By the time we reach this branch, ``parent`` is already
            #      set correctly by the caller.
            #
            #   2. Inside an object literal assigned to a const or
            #      returned from a factory:
            #        const migration = { up() {}, down() {} }
            #        return { isAuthenticated() {} }
            #      These appear in an ``object`` node, not a class_body.
            #      The walker reaches them via default recursion with
            #      parent="" — which leaves every method orphaned.
            #
            #   3. Inside an anonymous class expression assigned to
            #      a variable — same problem as #2.
            #
            # Fix: when ``parent`` is empty at this point, walk UP the
            # tree to find the nearest NAMED enclosing scope and use
            # its name as the parent. Named scopes are:
            #   • variable_declarator (``const foo = { ... }``)
            #   • function_declaration (factory returning the object)
            #   • pair (``{ foo: { bar() {} } }`` — use the key name)
            #
            # This closes the 3 % gap we saw against the backend-refactor
            # index, where factory functions and migration templates
            # produced methods with parent_symbol="".
            effective_parent = parent
            if not effective_parent:
                effective_parent = _find_enclosing_name(node, source)
            symbols.append(self._parse_method(node, source, fp, effective_parent))

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
            # TypeScript uses "type_identifier" for the class name; vanilla
            # JS uses "identifier". Accept both — without this, every
            # TS class produces ``name=""`` and every method inside it
            # gets ``parent_symbol=""``, which silently breaks
            # ``get_symbol_code('Class.method')`` and every other tool
            # that resolves dotted names.
            if child.type in ("identifier", "type_identifier"):
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
        elif node.type == "property_identifier":
            # Method/property access through a member_expression:
            #   this.connect(), obj.foo, console.log(), svc.start()
            #
            # The earlier version only walked plain "identifier" nodes,
            # which are always the BASE of a dotted path — never the
            # method name after the dot. Result: zero references to
            # method calls in class-heavy code (the Virtio audit showed
            # 0 hits for connectToVm on a 4245-line file that calls
            # it 6+ times). Fix: also record property_identifier nodes,
            # classifying them as "call" when the surrounding
            # member_expression is the callee of a call_expression,
            # and "usage" otherwise (property read).
            #
            # Definitions (method_definition → property_identifier) are
            # already excluded via the defs set populated in
            # _collect_defs.
            loc = (node.start_point[0], node.start_point[1])
            if loc not in defs:
                name = _node_text(node, source)
                if len(name) >= 2:
                    line_num = node.start_point[0]
                    context = lines[line_num] if line_num < len(lines) else ""
                    ref_kind = "usage"
                    # The property_identifier lives inside a
                    # member_expression. The member_expression in turn
                    # may be the FIRST child of a call_expression if
                    # this is a method call. Walk up two levels to
                    # detect the call case.
                    parent = node.parent
                    if parent and parent.type == "member_expression":
                        gp = parent.parent
                        if gp and gp.type == "call_expression" and parent == gp.children[0]:
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
