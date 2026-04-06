"""Pre-write syntax validation using tree-sitter.

A single, dependency-free entry point for "would this string parse cleanly
in language X?". Used by file-write tools (replace_lines, create_file,
edit_file, multi_edit_file, apply_patch) to refuse edits that would leave
the file in a broken-syntax state — *before* writing to disk.

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

# Map from language name to a callable that returns the tree-sitter
# Language object. Add more entries as the corresponding parser modules
# start exporting their LANGUAGE constant. Anything not in the map causes
# check_syntax to short-circuit and return [].
def _python_lang() -> Any:
    from infinidev.code_intel.parsers.python_parser import PY_LANGUAGE
    return PY_LANGUAGE


def _javascript_lang() -> Any:
    from infinidev.code_intel.parsers.javascript_parser import JS_LANGUAGE
    return JS_LANGUAGE


_LANGUAGE_LOADERS: dict[str, Any] = {
    "python": _python_lang,
    "javascript": _javascript_lang,
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

# Tree-sitter node types per language that we treat as "top-level symbol".
# Each language gets a small list of node types whose ``name`` child is the
# symbol name. Methods inside classes get qualified with the class name.
_PYTHON_DEF_NODES = ("function_definition", "class_definition")
_JS_DEF_NODES = (
    "function_declaration", "class_declaration",
    "method_definition", "lexical_declaration",
)


def _node_name(node: Any, source: bytes) -> str | None:
    """Find a node's identifier child and return its text, or None."""
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "property_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return None


def _extract_python_symbols(root: Any, source: bytes) -> set[str]:
    """Walk a Python tree and return a set of qualified symbol names."""
    out: set[str] = set()

    def walk(node: Any, parent_name: str = "") -> None:
        if node.type in _PYTHON_DEF_NODES:
            name = _node_name(node, source)
            if name:
                qual = f"{parent_name}.{name}" if parent_name else name
                out.add(qual)
                # Recurse into class/function body so methods are captured
                # under the class name.
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


def _extract_js_symbols(root: Any, source: bytes) -> set[str]:
    """Walk a JavaScript/TypeScript tree and return symbol names."""
    out: set[str] = set()

    def walk(node: Any, parent_name: str = "") -> None:
        if node.type in _JS_DEF_NODES:
            name = _node_name(node, source)
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
