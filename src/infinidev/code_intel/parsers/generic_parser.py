"""Generic config-driven parser for languages without a dedicated parser.

This module fills the gap that this session's audit exposed:
``EXTENSIONS`` in ``parsers/__init__.py`` claims to support 14
languages, but ``get_parser()`` only returned a real parser for 5
of them (Python, JavaScript, TypeScript, Rust, C). Files in the
remaining 9 languages (Go, Java, Ruby, C#, PHP, Kotlin, Bash, C++,
TSX) hit ``index_file`` and were silently dropped at the
``parser is None`` check.

Rather than write 9 separate ~200-LOC parsers (each duplicating the
same tree-walking pattern with slightly different node-type names),
this module provides ONE walker driven by the per-language config
dict already maintained in ``code_intel/syntax_check.py`` for the
file-skeleton extractor. Adding a new language is one entry in that
config; this parser picks it up automatically.

What this parser extracts:

  * **Symbols** (classes, functions, methods, top-level globals).
    Methods inside class bodies get ``parent_symbol`` and
    ``qualified_name`` set correctly via the same in_class flag the
    skeleton walker uses.
  * **Imports** (just the source line text, with the full statement
    stored as the ``source`` field — language-specific structured
    parsing is left to the dedicated parsers).

What this parser does NOT extract:
  * **References** (call sites, identifier usages). Reference
    extraction is the second-most expensive part of indexing and is
    much more language-specific than symbols. The dedicated parsers
    do this; the generic parser leaves ``ci_references`` empty for
    its languages until someone wants to add it.

So languages routed through this parser have working
``get_symbol_code``, ``search_symbols``, ``find_definition``,
``find_similar_methods``, and skeleton mode — but ``find_references``
returns "no results" until a proper extractor is written.
"""

from __future__ import annotations

import logging
from typing import Any

from tree_sitter import Node, Tree

from infinidev.code_intel.models import Symbol, SymbolKind, Reference, Import
from infinidev.code_intel.syntax_check import (
    _LANGUAGE_SKELETON_CONFIG,
    _python_docstring,
    _js_leading_comment,
)

logger = logging.getLogger(__name__)


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _signature_first_line(node: Node, source: bytes, max_len: int = 200) -> str:
    """First non-blank line of *node*'s text, capped at *max_len*.

    Used as a cheap signature for methods/classes when the language
    doesn't have a clear "signature only" subnode. Strips trailing
    `{` and trailing whitespace so the result reads like a function
    declaration without the body.
    """
    text = _node_text(node, source)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            stripped = stripped.rstrip("{").rstrip()
            return stripped[:max_len]
    return ""


# Per-language helpers for extracting the "name" of a definition node.
#
# Most tree-sitter grammars expose the name as one of these node types,
# but the right one depends on the parent node's grammar. Java
# method_declaration has the return type FIRST as type_identifier and
# the method name second as identifier — naively grabbing the first
# identifier-like child returns the return type. Same problem in C#
# (return type) and Kotlin (no return type usually but type annotations
# can confuse the order).
#
# Solution: keep the catch-all helper for the simple case AND a
# language-specific override for the awkward grammars where order
# matters. Both tried in priority order.

_NAME_NODE_TYPES: tuple[str, ...] = (
    "identifier", "type_identifier", "property_identifier",
    "field_identifier", "constant", "variable_name",
    "name", "simple_identifier",
)

# Methods/functions in these languages have their NAME as a child of
# type "identifier" specifically — distinct from type_identifier which
# is the return type. The map says "for this node type, ONLY accept
# these exact child types as the name".
_NAME_TYPES_BY_NODE: dict[str, tuple[str, ...]] = {
    # Java: method_declaration → modifiers + type + identifier + formal_parameters
    "method_declaration": ("identifier",),
    "constructor_declaration": ("identifier",),
    # C#: same shape as Java for methods
    "local_function_statement": ("identifier",),
    # Go: method_declaration has a "field_identifier" name child
    # (and a "parameter_list" receiver that we handle separately)
    # Function declarations in Go use "identifier"
    # Kotlin: function_declaration → modifiers + simple_identifier + parameters
    "function_declaration": ("identifier", "simple_identifier", "field_identifier"),
}


def _extract_name(node: Node, source: bytes, max_depth: int = 3) -> str:
    """Find the first identifier-like child of *node*, BFS to a small depth.

    For node types in :data:`_NAME_TYPES_BY_NODE`, only the listed
    child types are accepted — this fixes Java/C# where the first
    type-like child is the return type, not the method name.

    The depth cap prevents grabbing an identifier from a function
    body (way down the tree) when the actual name is in the header
    (depth 1-2).
    """
    accepted = _NAME_TYPES_BY_NODE.get(node.type, _NAME_NODE_TYPES)
    queue: list[tuple[Node, int]] = [(node, 0)]
    while queue:
        n, depth = queue.pop(0)
        if depth > max_depth:
            continue
        for child in n.children:
            if child.type in accepted:
                return _node_text(child, source)
        for child in n.children:
            queue.append((child, depth + 1))
    return ""


def _extract_go_receiver_type(node: Node, source: bytes) -> str:
    """For a Go ``method_declaration`` node, extract the receiver type name.

    Go method syntax is ``func (r *Receiver) Name() { ... }`` — the
    receiver is a ``parameter_list`` child whose first ``parameter_declaration``
    child has a ``type_identifier`` (or ``pointer_type``→``type_identifier``)
    that names the type the method is attached to. We use that name as
    the method's ``parent_symbol`` so dotted lookups
    ``Receiver.Name`` resolve correctly even though Go has no class
    body to walk into.

    Returns "" when the node has no receiver (i.e. it's a regular
    ``function_declaration``, not a method).
    """
    for child in node.children:
        if child.type != "parameter_list":
            continue
        # First parameter_list child of method_declaration is the receiver.
        for sub in child.children:
            if sub.type != "parameter_declaration":
                continue
            # Walk down to find a type_identifier
            stack = [sub]
            while stack:
                n = stack.pop()
                if n.type == "type_identifier":
                    return _node_text(n, source)
                stack.extend(n.children)
        return ""
    return ""


# Per-language token markers for detecting visibility / async / static.
# Cheap textual prefix check on the first ~80 chars of the node text.
def _detect_modifiers(node: Node, source: bytes, language: str) -> dict[str, Any]:
    """Return a dict of {visibility, is_async, is_static, is_abstract}.

    Heuristic — looks at the first line of the node text for keyword
    markers. Each language uses different conventions:

      * Python:    `_name` private, `__name` strongly private
      * Java/C#:   `public`, `private`, `protected`, `static`, `abstract`
      * Kotlin:    `public` (default), `private`, `protected`, `internal`
      * Go:        Capitalised first letter = exported (public)
      * Rust:      `pub` prefix = public
      * Ruby:      `private` / `protected` keyword scopes (hard to track)
      * PHP:       `public`, `private`, `protected`
      * Bash:      no visibility concept
    """
    out = {
        "visibility": "public",
        "is_async": False,
        "is_static": False,
        "is_abstract": False,
    }
    text = _node_text(node, source)
    head = text[:120].lower()

    if language in ("java", "csharp", "c_sharp", "kotlin", "php"):
        if "private " in head:
            out["visibility"] = "private"
        elif "protected " in head:
            out["visibility"] = "protected"
        elif "internal " in head:
            out["visibility"] = "internal"
        if "static " in head:
            out["is_static"] = True
        if "abstract " in head:
            out["is_abstract"] = True
        if "async " in head:
            out["is_async"] = True
    elif language == "rust":
        if not text.lstrip().startswith("pub"):
            out["visibility"] = "private"
        if "async " in head.split("(")[0]:
            out["is_async"] = True
    elif language == "go":
        # Capitalised first letter of name → exported (public). Done by caller
        # since it needs the name. We just leave defaults here.
        pass
    elif language == "python":
        # The name-based check happens in the caller, but we mark
        # async if we see the keyword.
        if head.lstrip().startswith("async "):
            out["is_async"] = True
    return out


def _kind_for_node_type(node_type: str, in_class: bool, config: dict[str, Any]) -> SymbolKind:
    """Map a tree-sitter node type to a SymbolKind.

    Same priority order the skeleton walker uses: in_class flag wins
    over the bare class/function distinction so a function_item inside
    an impl_item gets classified as a method.
    """
    if node_type in config.get("class_nodes", ()):
        # Differentiate trait/interface/enum/struct from plain class
        if "trait" in node_type:
            return SymbolKind.interface
        if "interface" in node_type:
            return SymbolKind.interface
        if "enum" in node_type:
            return SymbolKind.enum
        if "struct" in node_type or "record" in node_type:
            return SymbolKind.class_
        return SymbolKind.class_
    if in_class and (
        node_type in config.get("method_nodes", ())
        or node_type in config.get("function_nodes", ())
    ):
        return SymbolKind.method
    if node_type in config.get("function_nodes", ()):
        return SymbolKind.function
    if node_type in config.get("global_nodes", ()):
        return SymbolKind.variable
    return SymbolKind.variable


# Languages that don't have a dedicated parser get routed here.
# Listed explicitly so the get_parser() factory in parsers/__init__.py
# can ask "is this a generic-parsed language?" without importing
# the syntax_check module just for the config.
GENERIC_LANGUAGES: frozenset[str] = frozenset({
    "go", "java", "ruby", "csharp", "c_sharp", "php", "kotlin",
    "bash", "shell", "cpp", "tsx",
})


class GenericParser:
    """Symbol/import extractor driven by ``_LANGUAGE_SKELETON_CONFIG``.

    Instantiated per-call by ``get_parser(language)`` with the target
    language captured at construction. The same instance is safe to
    reuse across files of the same language but is NOT thread-safe
    across mixed languages — give each thread its own instance.
    """

    def __init__(self, language: str) -> None:
        self.language = language
        self.config = _LANGUAGE_SKELETON_CONFIG.get(language, {})

    # ── Public API (matches the dedicated parsers' shape) ───────────────

    def extract_symbols(self, tree: Tree, source: bytes, file_path: str) -> list[Symbol]:
        if not self.config:
            return []
        symbols: list[Symbol] = []
        self._walk(
            tree.root_node, source, file_path, symbols,
            parent="", in_class=False, in_function=False,
        )
        return symbols

    def extract_references(self, tree: Tree, source: bytes, file_path: str) -> list[Reference]:
        # Generic parser does NOT extract references — see module docstring.
        # Returning an empty list keeps `find_references` returning a
        # clean "no results" instead of crashing the indexer.
        return []

    def extract_imports(self, tree: Tree, source: bytes, file_path: str) -> list[Import]:
        if not self.config:
            return []
        out: list[Import] = []
        self._walk_imports(tree.root_node, source, file_path, out)
        return out

    # ── Internal walker ─────────────────────────────────────────────────

    def _walk(
        self,
        node: Node,
        source: bytes,
        fp: str,
        symbols: list[Symbol],
        parent: str,
        in_class: bool,
        in_function: bool,
    ) -> None:
        """Same invariants as the skeleton walker in syntax_check.py:

          1. Skip ERROR nodes — never descend into broken regions.
          2. Globals are top-level only — never collect them inside a
             class or function body.
          3. Methods are class-body-only — same node type at module
             level vs inside a class body produces different kinds.
          4. Passthrough nodes (export_statement, namespace_declaration,
             mod_item) descend without recording the wrapper.
        """
        if node.type == "ERROR":
            return

        cfg = self.config
        t = node.type

        # Passthrough wrappers — descend at the same logical level.
        if t in cfg.get("passthrough", ()):
            for child in node.children:
                self._walk(child, source, fp, symbols, parent, in_class, in_function)
            return

        # Class-like declarations.
        if t in cfg.get("class_nodes", ()):
            name = _extract_name(node, source)
            if name:
                sym = self._make_class_symbol(node, source, fp, name, parent)
                symbols.append(sym)
                # Walk class body — children of body nodes get in_class=True.
                body_types = cfg.get("class_body", ())
                for child in node.children:
                    if child.type in body_types:
                        for grand in child.children:
                            self._walk(
                                grand, source, fp, symbols,
                                parent=name, in_class=True, in_function=False,
                            )
            return

        # Method or function definition.
        is_method_node = t in cfg.get("method_nodes", ())
        is_function_node = t in cfg.get("function_nodes", ())
        if is_method_node or is_function_node:
            name = _extract_name(node, source)
            if name:
                # in_class flag wins: same node type inside a class is a method.
                kind = SymbolKind.method if in_class else SymbolKind.function
                effective_parent = parent

                # Go-specific: methods are top-level node types (not
                # inside a class body), but ``method_declaration`` has
                # a receiver parameter that names the struct/type
                # they're attached to. Use the receiver type as the
                # parent so dotted lookups (Receiver.Name) work even
                # though Go has no class body for the walker to descend.
                #
                # The naive _extract_name BFS finds the receiver
                # variable's identifier (`s` in `func (s *Server) Start`)
                # before the actual method name (`Start`, which is a
                # field_identifier child of method_declaration). Override
                # the name extraction to look at direct children only and
                # prefer field_identifier over identifier — that gives us
                # the method name and skips the receiver var.
                if (
                    self.language == "go"
                    and t == "method_declaration"
                    and not in_class
                ):
                    receiver_type = _extract_go_receiver_type(node, source)
                    if receiver_type:
                        effective_parent = receiver_type
                        kind = SymbolKind.method
                        # Override the name with a direct-child lookup
                        # for field_identifier (the method name).
                        for direct in node.children:
                            if direct.type == "field_identifier":
                                name = _node_text(direct, source)
                                break

                sym = self._make_function_symbol(
                    node, source, fp, name, effective_parent, kind,
                )
                symbols.append(sym)
            # Don't recurse into a function body looking for more symbols —
            # nested functions are rare and tracking them adds complexity
            # for marginal benefit. Set in_function=True so any global
            # detection further down the tree gets suppressed.
            return

        # Module-level globals (const/var/let, top-level only).
        if (
            t in cfg.get("global_nodes", ())
            and not in_class
            and not in_function
            and not parent
        ):
            name = _extract_name(node, source)
            if name:
                symbols.append(Symbol(
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.constant if name.isupper() else SymbolKind.variable,
                    file_path=fp,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=_signature_first_line(node, source, 120),
                    language=self.language,
                ))
            return

        # Default: walk all children with the same flags.
        for child in node.children:
            self._walk(
                child, source, fp, symbols,
                parent, in_class, in_function,
            )

    # ── Symbol builders ─────────────────────────────────────────────────

    def _make_class_symbol(
        self, node: Node, source: bytes, fp: str, name: str, parent: str,
    ) -> Symbol:
        # Pick the right SymbolKind based on the node type.
        kind = SymbolKind.class_
        t = node.type
        if "interface" in t or "trait" in t:
            kind = SymbolKind.interface
        elif "enum" in t:
            kind = SymbolKind.enum
        elif "type_alias" in t:
            kind = SymbolKind.type_alias

        mods = _detect_modifiers(node, source, self.language)
        # Pick the right doc strategy.
        doc = ""
        doc_strategy = self.config.get("doc_strategy", "")
        try:
            if doc_strategy == "python":
                doc = _python_docstring(node, source)
            elif doc_strategy == "leading_comment":
                doc = _js_leading_comment(node, source, node)
        except Exception:
            doc = ""

        return Symbol(
            name=name,
            qualified_name=f"{parent}.{name}" if parent else name,
            kind=kind,
            file_path=fp,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=_signature_first_line(node, source, 200),
            docstring=doc,
            parent_symbol=parent,
            visibility=mods["visibility"],
            is_static=mods["is_static"],
            is_abstract=mods["is_abstract"],
            language=self.language,
        )

    def _make_function_symbol(
        self,
        node: Node, source: bytes, fp: str,
        name: str, parent: str, kind: SymbolKind,
    ) -> Symbol:
        mods = _detect_modifiers(node, source, self.language)

        # Go-specific: capitalised first letter = exported (public).
        # The general default is "public", so we override only when the
        # name signals private under Go's convention.
        if self.language == "go":
            if name and name[0].islower():
                mods["visibility"] = "private"

        # Python-specific: leading underscore = private convention.
        if self.language == "python":
            if name.startswith("_") and not name.startswith("__"):
                mods["visibility"] = "private"

        doc = ""
        doc_strategy = self.config.get("doc_strategy", "")
        try:
            if doc_strategy == "python":
                doc = _python_docstring(node, source)
            elif doc_strategy == "leading_comment":
                doc = _js_leading_comment(node, source, node)
        except Exception:
            doc = ""

        return Symbol(
            name=name,
            qualified_name=f"{parent}.{name}" if parent else name,
            kind=kind,
            file_path=fp,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=_signature_first_line(node, source, 200),
            docstring=doc,
            parent_symbol=parent,
            visibility=mods["visibility"],
            is_async=mods["is_async"],
            is_static=mods["is_static"],
            is_abstract=mods["is_abstract"],
            language=self.language,
        )

    # ── Imports ─────────────────────────────────────────────────────────

    def _walk_imports(
        self, node: Node, source: bytes, fp: str, out: list[Import],
    ) -> None:
        if node.type in self.config.get("import_nodes", ()):
            text = _node_text(node, source).strip()
            # Bare import line — store the raw statement as the source.
            # Per-language structured parsing is left to dedicated parsers.
            out.append(Import(
                source=text[:200],
                name="",
                alias="",
                file_path=fp,
                line=node.start_point[0] + 1,
                language=self.language,
            ))
            return
        for child in node.children:
            self._walk_imports(child, source, fp, out)


__all__ = ["GenericParser", "GENERIC_LANGUAGES"]
