"""Pre-write syntax validation using tree-sitter.

A single, dependency-free entry point for "would this string parse cleanly
in language X?". Used by file-write tools (replace_lines, create_file,
write_file, multi_edit_file) to surface syntax issues as advisory warnings
on the tool response — false positives (Jest mocks, TSX experimental
syntax) mean we never block the write.

Design notes:
  * Tree-sitter is error-tolerant: it parses anything and marks invalid
    regions with ``ERROR`` nodes and missing-token nodes
    (``node.is_missing == True``). Walking the tree once and collecting
    those is enough to catch the vast majority of small-model edit
    mistakes (unbalanced brackets, broken indentation, dangling commas,
    truncated function bodies, etc.).
  * No language-specific knowledge lives here — language detection is
    delegated to ``code_intel.parsers.detect_language``, and the actual
    parsers are loaded lazily via ``_load_parser`` so importing this
    module costs nothing if it isn't used.
  * Returns a list of :class:`SyntaxIssue` (empty = valid). Callers
    decide what to do with issues — typically render them as a tool error
    so the model sees the line numbers and a hint to retry.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from infinidev.code_intel.parsers import detect_language


@dataclass(frozen=True)
class SyntaxIssue:
    """A single syntax problem found by tree-sitter."""

    line: int           # 1-based line number
    column: int         # 1-based column number
    message: str        # Short, human-readable description
    snippet: str = ""   # ~80 chars of source around the error, for context

    def format(self) -> str:
        prefix = f"line {self.line}:{self.column}"
        if self.snippet:
            return f"{prefix}  {self.message}\n    {self.snippet}"
        return f"{prefix}  {self.message}"


# ── Parser loading ───────────────────────────────────────────────────────

# Each loader returns a tree-sitter ``Language`` object built from the
# corresponding pip package. Loaders are lazy so importing this module
# costs nothing if a given language is never used. Adding a new
# language is two lines: a loader and an entry in ``_LANGUAGE_LOADERS``.
def _python_lang() -> Any:
    from infinidev.code_intel.parsers.python_parser import PY_LANGUAGE
    return PY_LANGUAGE


def _javascript_lang() -> Any:
    from infinidev.code_intel.parsers.javascript_parser import JS_LANGUAGE
    return JS_LANGUAGE


def _typescript_lang() -> Any:
    from tree_sitter import Language
    import tree_sitter_typescript as ts_ts
    return Language(ts_ts.language_typescript())


def _tsx_lang() -> Any:
    from tree_sitter import Language
    import tree_sitter_typescript as ts_ts
    return Language(ts_ts.language_tsx())


def _go_lang() -> Any:
    from tree_sitter import Language
    import tree_sitter_go as ts_go
    return Language(ts_go.language())


def _rust_lang() -> Any:
    from tree_sitter import Language
    import tree_sitter_rust as ts_rust
    return Language(ts_rust.language())


def _java_lang() -> Any:
    from tree_sitter import Language
    import tree_sitter_java as ts_java
    return Language(ts_java.language())


def _c_lang() -> Any:
    from tree_sitter import Language
    import tree_sitter_c as ts_c
    return Language(ts_c.language())


def _ruby_lang() -> Any:
    from tree_sitter import Language
    import tree_sitter_ruby as ts_ruby
    return Language(ts_ruby.language())


def _csharp_lang() -> Any:
    from tree_sitter import Language
    import tree_sitter_c_sharp as ts_cs
    return Language(ts_cs.language())


def _php_lang() -> Any:
    from tree_sitter import Language
    import tree_sitter_php as ts_php
    return Language(ts_php.language_php())


def _kotlin_lang() -> Any:
    from tree_sitter import Language
    import tree_sitter_kotlin as ts_kt
    return Language(ts_kt.language())


def _bash_lang() -> Any:
    from tree_sitter import Language
    import tree_sitter_bash as ts_bash
    return Language(ts_bash.language())


_LANGUAGE_LOADERS: dict[str, Any] = {
    "python":     _python_lang,
    "javascript": _javascript_lang,
    "typescript": _typescript_lang,
    "tsx":        _tsx_lang,
    "go":         _go_lang,
    "rust":       _rust_lang,
    "java":       _java_lang,
    "c":          _c_lang,
    "cpp":        _c_lang,    # C parser is good enough for C++ structure
    "ruby":       _ruby_lang,
    "csharp":     _csharp_lang,
    "c_sharp":    _csharp_lang,
    "php":        _php_lang,
    "kotlin":     _kotlin_lang,
    "bash":       _bash_lang,
    "shell":      _bash_lang,
}


@lru_cache(maxsize=8)
def _load_parser(language: str) -> Any | None:
    """Return a tree-sitter ``Parser`` for *language*, or None if unsupported.

    Cached: parsers are reused across calls. Returns None for languages
    without a registered loader (the caller treats that as
    "skip syntax check, write through"). Never raises.
    """
    loader = _LANGUAGE_LOADERS.get(language)
    if loader is None:
        return None
    try:
        from tree_sitter import Parser
        lang = loader()
        parser = Parser()
        # tree-sitter ≥0.22 uses attribute assignment; older uses set_language.
        try:
            parser.language = lang
        except AttributeError:
            parser.set_language(lang)  # type: ignore[attr-defined]
        return parser
    except Exception:
        return None


# ── Issue collection ─────────────────────────────────────────────────────

def _walk_for_errors(node: Any, source_lines: list[str], issues: list[SyntaxIssue]) -> None:
    """DFS the tree, collecting ERROR and missing-token nodes."""
    # Tree-sitter marks invalid regions with type=="ERROR" and missing
    # tokens (e.g. expected ':' in a Python def) with is_missing.
    if node.type == "ERROR" or getattr(node, "is_missing", False):
        line = node.start_point[0] + 1   # 0-based → 1-based
        col = node.start_point[1] + 1
        if getattr(node, "is_missing", False):
            msg = f"missing token (parser expected '{node.type}' here)"
        else:
            msg = "syntax error — invalid region for the language grammar"
        snippet = ""
        if 0 < line <= len(source_lines):
            raw = source_lines[line - 1]
            snippet = raw.rstrip("\n")[:120]
        issues.append(SyntaxIssue(line=line, column=col, message=msg, snippet=snippet))

    for child in node.children:
        _walk_for_errors(child, source_lines, issues)


# ── Public API ───────────────────────────────────────────────────────────

def check_syntax(text: str, language: str | None = None, file_path: str | None = None) -> list[SyntaxIssue]:
    """Validate that *text* parses cleanly as *language*.

    Either *language* or *file_path* (whose extension is used to detect
    the language) must be provided. Returns an empty list when:
      * the language is unknown / unsupported,
      * the tree-sitter grammar isn't installed,
      * the text parses cleanly.

    Returns a list of :class:`SyntaxIssue` describing each error region
    when the text contains errors. Order matches a top-down tree walk,
    so the first issue is usually closest to the root cause.

    Never raises — any internal failure is treated as "skip the check
    and let the write proceed", because false positives here would block
    legitimate edits.
    """
    if not text:
        return []
    if language is None and file_path:
        language = detect_language(file_path)
    if not language or language == "config":
        return []

    parser = _load_parser(language)
    if parser is None:
        return []

    try:
        tree = parser.parse(bytes(text, "utf-8"))
    except Exception:
        return []

    if not tree.root_node.has_error:
        # has_error is True iff the tree contains any ERROR or missing nodes.
        # Fast-path: skip the walk when the file is clean.
        return []

    issues: list[SyntaxIssue] = []
    _walk_for_errors(tree.root_node, text.splitlines(), issues)
    return issues


def format_issues(issues: list[SyntaxIssue], *, max_show: int = 5) -> str:
    """Format issues as a multi-line, human-readable error string for tool output.

    Truncates to *max_show* entries to avoid flooding the model's context;
    the count of suppressed issues is appended.
    """
    if not issues:
        return ""
    shown = issues[:max_show]
    lines = [issue.format() for issue in shown]
    if len(issues) > max_show:
        lines.append(f"... and {len(issues) - max_show} more issues")
    return "\n".join(lines)


# ── Top-level symbol extraction ──────────────────────────────────────────
#
# Used by ``validate_no_silent_deletion`` to detect when an edit removes
# functions or classes the model probably didn't mean to delete. Returns
# a SET of fully-qualified names (e.g. ``"Database._execute_create_table"``)
# so additions and removals can be compared with simple set arithmetic.

# Per-language symbol extractors live in ``code_intel/extractors/``.
# Re-exported here under their legacy names so existing call sites
# (``_extract_python_symbols`` etc.) keep working unchanged.
from infinidev.code_intel.extractors import (
    extract_python_symbols as _extract_python_symbols,
    extract_js_symbols as _extract_js_symbols,
)
from infinidev.code_intel.extractors._common import node_name as _node_name


def extract_top_level_symbols(
    text: str,
    language: str | None = None,
    file_path: str | None = None,
) -> set[str]:
    """Return the set of top-level symbol names defined in *text*.

    Symbols nested inside a class are returned qualified
    (``Database.execute``). Returns an empty set when the language is
    unsupported, the parser is unavailable, or the text is empty —
    callers must treat an empty set as "I have no information" rather
    than "no symbols".
    """
    if not text:
        return set()
    if language is None and file_path:
        language = detect_language(file_path)
    if not language or language == "config":
        return set()

    parser = _load_parser(language)
    if parser is None:
        return set()

    try:
        source = bytes(text, "utf-8")
        tree = parser.parse(source)
    except Exception:
        return set()

    if language == "python":
        return _extract_python_symbols(tree.root_node, source)
    if language in ("javascript", "typescript"):
        return _extract_js_symbols(tree.root_node, source)
    return set()


# ── File skeleton extraction ─────────────────────────────────────────────
#
# When a file is too large to inline in a model's context, we return a
# structured skeleton instead: imports, globals, classes (with their
# methods), and free functions, each with line ranges and a short doc
# extracted from the source. The model then uses read_file with a
# line range or get_symbol_code to zoom in on the parts it actually
# needs — the same
# workflow a human uses with ctags or an outline panel.
#
# This sits next to extract_top_level_symbols on purpose: both walk the
# same tree-sitter trees and share the same parser cache, so the
# skeleton is essentially free once the file is already indexed.

@dataclass(frozen=True)
class SkeletonEntry:
    """One symbol in a file skeleton.

    Line numbers are 1-based and inclusive on both ends. ``parent`` is
    empty for top-level entries; for methods inside a class it carries
    the class name (so ``parent="Database"``, ``name="execute"`` means
    ``Database.execute``). ``doc`` is at most ~120 chars — anything
    longer is truncated with an ellipsis.
    """

    kind: str           # "class" | "function" | "method" | "const" | "var" | "import"
    name: str
    line_start: int
    line_end: int
    doc: str = ""
    parent: str = ""

    @property
    def qualified_name(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    def to_dict(self) -> dict:
        d = {
            "kind": self.kind,
            "name": self.qualified_name,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }
        if self.doc:
            d["doc"] = self.doc
        return d


def _truncate_doc(s: str, limit: int = 120) -> str:
    """Collapse whitespace and cap *s* to *limit* chars with an ellipsis."""
    if not s:
        return ""
    cleaned = " ".join(s.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _python_docstring(node: Any, source: bytes) -> str:
    """Return the first string literal in a Python ``def`` / ``class`` body, or ''."""
    # Body is a "block" child containing statements; first statement that
    # is an "expression_statement" containing a "string" is the docstring.
    for child in node.children:
        if child.type != "block":
            continue
        for stmt in child.children:
            if stmt.type == "expression_statement":
                for s in stmt.children:
                    if s.type == "string":
                        raw = source[s.start_byte:s.end_byte].decode("utf-8", errors="replace")
                        # Strip Python string quoting (''' " """ ' "")
                        for q in ('"""', "'''", '"', "'"):
                            if raw.startswith(q) and raw.endswith(q) and len(raw) >= 2 * len(q):
                                raw = raw[len(q):-len(q)]
                                break
                        return _truncate_doc(raw)
                return ""
            # First non-string statement: no docstring.
            if stmt.type not in ("comment",):
                return ""
        return ""
    return ""


def _js_leading_comment(node: Any, source: bytes, root: Any) -> str:
    """Return the comment block immediately preceding *node*, or ''.

    Walks the parent's children looking for the comment that ends right
    before the node starts (within ~3 lines). Strips ``//`` and ``/* */``
    markers and trims to a single short line.
    """
    parent = node.parent if hasattr(node, "parent") else None
    if parent is None:
        parent = root
    prev_comment = None
    for child in parent.children:
        if child == node:
            break
        if child.type == "comment":
            prev_comment = child
    if prev_comment is None:
        return ""
    # Comment must end within 2 lines of the node start to count as "leading".
    if node.start_point[0] - prev_comment.end_point[0] > 2:
        return ""
    raw = source[prev_comment.start_byte:prev_comment.end_byte].decode("utf-8", errors="replace")
    # Strip /* */ and //
    raw = raw.strip()
    if raw.startswith("/*"):
        raw = raw[2:]
        if raw.endswith("*/"):
            raw = raw[:-2]
        # Strip leading * on each line
        lines = [ln.strip().lstrip("*").strip() for ln in raw.splitlines()]
        raw = " ".join(ln for ln in lines if ln)
    elif raw.startswith("//"):
        raw = raw[2:].strip()
    return _truncate_doc(raw)


# ── Generic skeleton extractor ───────────────────────────────────────────
#
# Each language is described by a small config dict instead of its own
# bespoke walker function. This means adding a new language is one new
# config entry — no new tree-walking logic, no new edge cases.
#
# The walker honours one critical rule: it NEVER descends into a
# class/function body looking for "globals". Once we enter a class body
# we only collect methods; once we enter a function body we collect
# nothing. This is what was wrong with the previous JS walker — it
# treated function-local ``const`` declarations as module-level
# globals, polluting the skeleton.

# Per-language config keys:
#   class_nodes:    tuple of node types that define a class/struct/interface
#   function_nodes: tuple of node types that define a top-level function
#   method_nodes:   tuple of node types for methods inside a class body
#   import_nodes:   tuple of node types for top-level imports
#   class_body:     tuple of node types whose children are class members
#   global_nodes:   tuple of node types for module-level const/var (or empty)
#   doc_strategy:   "python" | "leading_comment" | "" (none)
#   passthrough:    tuple of node types whose children should be walked
#                   but which themselves don't add a symbol (e.g. JS
#                   ``export_statement``, Rust ``mod_item``).
_LANGUAGE_SKELETON_CONFIG: dict[str, dict[str, Any]] = {
    "python": {
        "class_nodes":    ("class_definition",),
        "function_nodes": ("function_definition",),
        "method_nodes":   ("function_definition",),
        "import_nodes":   ("import_statement", "import_from_statement"),
        "class_body":     ("block",),
        "global_nodes":   ("expression_statement",),  # contains assignment
        "doc_strategy":   "python",
        "passthrough":    (),
    },
    "javascript": {
        "class_nodes":    ("class_declaration", "abstract_class_declaration"),
        "function_nodes": ("function_declaration", "generator_function_declaration"),
        "method_nodes":   ("method_definition",),
        "import_nodes":   ("import_statement",),
        "class_body":     ("class_body",),
        "global_nodes":   ("lexical_declaration", "variable_declaration"),
        "doc_strategy":   "leading_comment",
        "passthrough":    ("export_statement",),
    },
    "typescript": {
        "class_nodes":    (
            "class_declaration", "abstract_class_declaration",
            "interface_declaration", "type_alias_declaration",
        ),
        "function_nodes": ("function_declaration", "generator_function_declaration"),
        "method_nodes":   ("method_definition", "method_signature", "abstract_method_signature"),
        "import_nodes":   ("import_statement",),
        "class_body":     ("class_body", "interface_body", "object_type"),
        "global_nodes":   ("lexical_declaration", "variable_declaration"),
        "doc_strategy":   "leading_comment",
        "passthrough":    ("export_statement", "ambient_declaration"),
    },
    "go": {
        "class_nodes":    ("type_declaration",),  # struct/interface
        "function_nodes": ("function_declaration", "method_declaration"),
        "method_nodes":   (),  # Go has no class body — methods are file-level
        "import_nodes":   ("import_declaration",),
        "class_body":     (),
        "global_nodes":   ("var_declaration", "const_declaration"),
        "doc_strategy":   "leading_comment",
        "passthrough":    (),
    },
    "rust": {
        "class_nodes":    ("struct_item", "enum_item", "trait_item", "impl_item"),
        "function_nodes": ("function_item",),
        "method_nodes":   ("function_item",),
        "import_nodes":   ("use_declaration",),
        "class_body":     ("declaration_list",),
        "global_nodes":   ("const_item", "static_item"),
        "doc_strategy":   "leading_comment",
        "passthrough":    ("mod_item", "extern_crate_declaration"),
    },
    "java": {
        "class_nodes":    ("class_declaration", "interface_declaration", "enum_declaration", "record_declaration"),
        "function_nodes": (),
        "method_nodes":   ("method_declaration", "constructor_declaration"),
        "import_nodes":   ("import_declaration",),
        "class_body":     ("class_body", "interface_body", "enum_body"),
        "global_nodes":   (),
        "doc_strategy":   "leading_comment",
        "passthrough":    (),
    },
    "c": {
        "class_nodes":    ("struct_specifier", "union_specifier", "enum_specifier"),
        "function_nodes": ("function_definition",),
        "method_nodes":   (),
        "import_nodes":   ("preproc_include",),
        "class_body":     ("field_declaration_list",),
        "global_nodes":   ("declaration",),
        "doc_strategy":   "leading_comment",
        "passthrough":    ("preproc_def",),
    },
    "ruby": {
        "class_nodes":    ("class", "module",),
        "function_nodes": ("method",),
        "method_nodes":   ("method",),
        "import_nodes":   (),
        "class_body":     ("body_statement",),
        "global_nodes":   (),
        "doc_strategy":   "leading_comment",
        "passthrough":    (),
    },
    "csharp": {
        "class_nodes":    ("class_declaration", "interface_declaration", "struct_declaration", "enum_declaration", "record_declaration"),
        "function_nodes": (),
        "method_nodes":   ("method_declaration", "constructor_declaration", "property_declaration"),
        "import_nodes":   ("using_directive",),
        "class_body":     ("declaration_list",),
        "global_nodes":   (),
        "doc_strategy":   "leading_comment",
        "passthrough":    ("namespace_declaration", "file_scoped_namespace_declaration"),
    },
    "php": {
        "class_nodes":    ("class_declaration", "interface_declaration", "trait_declaration", "enum_declaration"),
        "function_nodes": ("function_definition",),
        "method_nodes":   ("method_declaration",),
        "import_nodes":   ("namespace_use_declaration",),
        "class_body":     ("declaration_list",),
        "global_nodes":   (),
        "doc_strategy":   "leading_comment",
        "passthrough":    ("namespace_definition",),
    },
    "kotlin": {
        "class_nodes":    ("class_declaration", "object_declaration"),
        "function_nodes": ("function_declaration",),
        "method_nodes":   ("function_declaration",),
        "import_nodes":   ("import_header",),
        "class_body":     ("class_body",),
        "global_nodes":   ("property_declaration",),
        "doc_strategy":   "leading_comment",
        "passthrough":    (),
    },
    "bash": {
        "class_nodes":    (),
        "function_nodes": ("function_definition",),
        "method_nodes":   (),
        "import_nodes":   (),
        "class_body":     (),
        "global_nodes":   ("variable_assignment",),
        "doc_strategy":   "leading_comment",
        "passthrough":    (),
    },
}

# Aliases — same config under multiple names so detect_language can use
# whichever spelling. Keep these in sync with EXTENSIONS in parsers/__init__.py.
_LANGUAGE_SKELETON_CONFIG["tsx"] = _LANGUAGE_SKELETON_CONFIG["typescript"]
_LANGUAGE_SKELETON_CONFIG["cpp"] = _LANGUAGE_SKELETON_CONFIG["c"]
_LANGUAGE_SKELETON_CONFIG["c_sharp"] = _LANGUAGE_SKELETON_CONFIG["csharp"]
_LANGUAGE_SKELETON_CONFIG["shell"] = _LANGUAGE_SKELETON_CONFIG["bash"]


def _generic_skeleton(
    root: Any,
    source: bytes,
    config: dict[str, Any],
) -> list[SkeletonEntry]:
    """Walk a tree-sitter tree using a per-language config and return entries.

    This is the only walker. It honours four invariants:

      1. **Globals are top-level only**. Once we descend into a class
         body or a function body, we never collect ``global_nodes``.
         This prevents function-local ``const`` declarations from being
         mis-reported as module globals (the bug that polluted the
         first VirtioSocketWatcher attempt).
      2. **Methods are class-body-only**. Function definitions found at
         module level are functions; the same node type inside a class
         body is a method. The ``in_class`` flag carries this state.
      3. **Passthrough nodes don't add symbols**. JS ``export_statement``
         and Rust ``mod_item`` are wrappers — their children are real
         declarations.
      4. **ERROR nodes are not walked**. tree-sitter is error-tolerant
         and the JS-on-TS fallback used to dump function bodies into
         the symbol list because the class declaration parsed as ERROR.
         We now skip ERROR nodes entirely; if a class fails to parse,
         we just lose that class instead of producing garbage.
    """
    out: list[SkeletonEntry] = []
    class_nodes    = config["class_nodes"]
    function_nodes = config["function_nodes"]
    method_nodes   = config["method_nodes"]
    import_nodes   = config["import_nodes"]
    class_body     = config["class_body"]
    global_nodes   = config["global_nodes"]
    doc_strategy   = config["doc_strategy"]
    passthrough    = config["passthrough"]

    def doc_for(node: Any) -> str:
        if doc_strategy == "python":
            return _python_docstring(node, source)
        if doc_strategy == "leading_comment":
            return _js_leading_comment(node, source, root)
        return ""

    def add(node: Any, kind: str, name: str, parent: str = "", doc: str = "") -> None:
        out.append(SkeletonEntry(
            kind=kind, name=name,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            doc=doc, parent=parent,
        ))

    def walk(node: Any, parent_name: str = "", in_class: bool = False, in_function: bool = False) -> None:
        # Skip ERROR nodes — they can wrap arbitrary garbage and would
        # otherwise cause us to misclassify their children. Tree-sitter's
        # error tolerance is great for parsing, terrible for skeleton
        # extraction without this guard.
        if node.type == "ERROR":
            return

        t = node.type

        # Passthrough wrappers (export, namespace, mod) — descend into
        # children at the same logical level, do not record the wrapper.
        if t in passthrough:
            for child in node.children:
                walk(child, parent_name, in_class, in_function)
            return

        # Imports — top-level only.
        if t in import_nodes and not parent_name and not in_class and not in_function:
            text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            add(node, "import", _truncate_doc(text, 80))
            return

        # Class-like declarations (class/struct/interface/enum/trait/impl).
        if t in class_nodes:
            name = _node_name(node, source) or "?"
            doc = doc_for(node)
            add(node, "class", name, parent_name, doc)
            # Walk only the class body, marking children as in_class so
            # method nodes get classified correctly and globals stop.
            for child in node.children:
                if child.type in class_body:
                    for grand in child.children:
                        walk(grand, name, in_class=True, in_function=False)
            return

        # Methods inside a class body. Some languages (Python, Rust, Kotlin)
        # use the same node type for free functions and methods, so the
        # method/function distinction is purely positional (in_class flag).
        if in_class and (t in method_nodes or t in function_nodes):
            name = _node_name(node, source) or "?"
            doc = doc_for(node)
            add(node, "method", name, parent_name, doc)
            return

        # Free functions at top level (or any non-class scope).
        if t in function_nodes and not in_class:
            name = _node_name(node, source) or "?"
            doc = doc_for(node)
            add(node, "function", name, parent_name, doc)
            return

        # Module-level const / var. NEVER inside a class or function.
        if t in global_nodes and not in_class and not in_function and not parent_name:
            # Try to extract a name from the declarator subtree.
            name = _extract_first_identifier(node, source) or ""
            if name:
                # const if name is ALL_CAPS or token starts with "const"
                first_token = source[node.start_byte:node.start_byte + 6].decode(
                    "utf-8", errors="replace"
                ).lstrip()
                kind = "const" if (name.isupper() or first_token.startswith("const")) else "var"
                add(node, kind, name)
            return

        # Default recursion — walk all children with the same flags.
        # Once we enter a function we set in_function=True so nested
        # declarations don't pollute the global list.
        next_in_function = in_function or (t in function_nodes)
        for child in node.children:
            walk(child, parent_name, in_class, next_in_function)

    walk(root)
    return out


def _extract_first_identifier(node: Any, source: bytes) -> str | None:
    """Find the first ``identifier`` descendant of *node*, breadth-first.

    Used by the global-variable extractor to fish a name out of a
    declaration subtree without knowing the exact grammar shape.
    Restricted to a small depth so a function-body assignment misclassified
    as a global doesn't fish the wrong name out.
    """
    queue: list[tuple[Any, int]] = [(node, 0)]
    while queue:
        n, depth = queue.pop(0)
        if depth > 4:
            continue
        if n.type in ("identifier", "type_identifier", "property_identifier", "field_identifier", "variable_name"):
            return source[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
        for c in n.children:
            queue.append((c, depth + 1))
    return None


def extract_file_skeleton(
    text: str,
    language: str | None = None,
    file_path: str | None = None,
) -> tuple[list[SkeletonEntry], str]:
    """Return a structured skeleton of *text* + the detected language.

    The skeleton is a flat list of :class:`SkeletonEntry`, one per
    top-level definition (and one per method inside a class). Returns
    ``([], "")`` for unsupported languages so the caller can fall back
    to a head-and-tail preview.

    Never raises — any internal failure returns ``([], language or "")``.
    """
    if not text:
        return [], language or ""
    if language is None and file_path:
        language = detect_language(file_path)
    if not language or language == "config":
        return [], language or ""

    parser = _load_parser(language)
    if parser is None:
        return [], language

    try:
        source = bytes(text, "utf-8")
        tree = parser.parse(source)
    except Exception:
        return [], language

    config = _LANGUAGE_SKELETON_CONFIG.get(language)
    if config is None:
        return [], language

    try:
        return _generic_skeleton(tree.root_node, source, config), language
    except Exception:
        return [], language


def render_skeleton_text(
    skeleton: list[SkeletonEntry],
    *,
    file_path: str,
    total_lines: int,
    total_bytes: int,
    language: str,
    max_per_section: int = 60,
) -> str:
    """Render a skeleton as the human/LLM-readable block returned by read_file.

    Groups entries by kind (imports, globals, classes, functions),
    truncates very long sections, and ends with a "next action hint"
    that explicitly names the tools the model should call to dig
    further. The hint is the most important part — it converts "the
    file is too big" from a wall into a ramp toward the code-intel
    tools that already exist."""
    imports = [s for s in skeleton if s.kind == "import"]
    globals_ = [s for s in skeleton if s.kind in ("const", "var")]
    classes = [s for s in skeleton if s.kind == "class"]
    methods = [s for s in skeleton if s.kind == "method"]
    functions = [s for s in skeleton if s.kind == "function"]

    # Group methods under their parent class for display.
    methods_by_class: dict[str, list[SkeletonEntry]] = {}
    for m in methods:
        methods_by_class.setdefault(m.parent, []).append(m)

    out: list[str] = []
    out.append(
        f"⚠ FILE TOO LARGE TO READ IN FULL — returning structured skeleton."
    )
    out.append(
        f"  file:     {file_path}"
    )
    out.append(
        f"  size:     {total_lines} lines, {total_bytes} bytes ({language})"
    )
    out.append(
        f"  symbols:  {len(classes)} classes, {len(functions)} functions, "
        f"{len(methods)} methods, {len(globals_)} globals, {len(imports)} imports"
    )
    out.append("")

    def _section(title: str, entries: list[SkeletonEntry]) -> None:
        if not entries:
            return
        out.append(f"── {title} ({len(entries)}) ──")
        shown = entries[:max_per_section]
        for e in shown:
            line_range = f"L{e.line_start}-{e.line_end}"
            doc_part = f"  — {e.doc}" if e.doc else ""
            if e.kind == "import":
                # The "name" field for imports is the import statement text.
                out.append(f"  {line_range}  {e.name}")
            else:
                out.append(f"  {line_range}  {e.qualified_name}{doc_part}")
        if len(entries) > max_per_section:
            out.append(
                f"  ... and {len(entries) - max_per_section} more "
                f"(call list_symbols for the full list)"
            )
        out.append("")

    _section("imports", imports)
    _section("globals (constants and module-level variables)", globals_)

    if classes:
        out.append(f"── classes ({len(classes)}) ──")
        for c in classes[:max_per_section]:
            doc_part = f"  — {c.doc}" if c.doc else ""
            out.append(f"  L{c.line_start}-{c.line_end}  class {c.name}{doc_part}")
            cms = methods_by_class.get(c.name, [])
            for m in cms[:max_per_section]:
                m_doc = f"  — {m.doc}" if m.doc else ""
                out.append(f"      L{m.line_start}-{m.line_end}  .{m.name}{m_doc}")
            if len(cms) > max_per_section:
                out.append(
                    f"      ... and {len(cms) - max_per_section} more methods"
                )
        if len(classes) > max_per_section:
            out.append(
                f"  ... and {len(classes) - max_per_section} more classes"
            )
        out.append("")

    _section("functions (top-level)", functions)

    out.append("── How to read this file ──")
    out.append(
        "  This file is too large to load in full. To inspect specific parts,"
    )
    out.append(
        "  use one of these tools — DO NOT call read_file again without"
    )
    out.append(
        "  start_line/end_line, that will return this same skeleton."
    )
    out.append("")
    out.append(
        "  • read_file(file_path=..., start_line=N, end_line=M)"
    )
    out.append(
        "      → read a specific line range. Use the L<start>-<end> values above."
    )
    out.append(
        "  • get_symbol_code(file_path=..., name='ClassName.methodName')"
    )
    out.append(
        "      → read one symbol's full code. Cheaper than a line range when"
    )
    out.append(
        "        you only care about one method."
    )
    out.append(
        "  • search_symbols(query=...)"
    )
    out.append(
        "      → fuzzy-find symbols by name across the project."
    )
    out.append("")
    out.append(
        "  Pick the symbol or line range you actually need from the list above"
    )
    out.append(
        "  and call ONE of those tools. Don't try to read the whole file."
    )

    return "\n".join(out)
