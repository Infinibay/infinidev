"""Per-method body fingerprints for fuzzy similarity search.

This module is the second half of the read-large-files story. The
skeleton extractor (``code_intel.syntax_check.extract_file_skeleton``)
gives a model the *shape* of a big file. This module gives the model
something the skeleton can't: a project-wide view of "are there other
methods that look like this one?". Use cases:

  * Refactoring: before extracting a method into a new file, see whether
    it's actually a duplicate of one that already exists elsewhere.
  * Code review: spot copy-pasted methods that should be consolidated.
  * Onboarding: when reading a new file, surface "this method has
    siblings in 3 other places" so the model knows to look there too.

Design overview
---------------

For each function/method we store three things in ``ci_method_bodies``:

  1. **body_hash** — sha256(normalized_body)[:16]. Catches exact
     copy-paste even when whitespace, comments, or local identifiers
     have been changed. Cheap to query: indexed by ``body_hash``, two
     methods with the same hash are guaranteed near-identical.

  2. **body_norm** — space-separated tokens of the normalized body.
     Used for Jaccard similarity over token sets. Catches "almost
     duplicate" cases where the algorithm is the same but variable
     names differ or one method has an extra branch.

  3. **body_size** — line count (after stripping comments and blanks).
     Used as a cheap pre-filter: comparing two methods of vastly
     different sizes is wasteful — they're not similar by construction.

The normalization pipeline:

  source body
    → strip comments (// /* */ # — picked by language)
    → strip blank lines
    → replace identifiers with placeholders (V1, V2, V3...) so
      `for (let i ...)` and `for (let j ...)` collapse to the same
      shape
    → collapse all whitespace to single spaces
    → lowercase

Strings are kept literal because two methods with different string
constants are doing different things. Numbers are kept literal for
the same reason.

The Jaccard pre-filter uses the **token-set** of the normalized body
(words separated by whitespace), giving a number in [0, 1]. We default
to ``threshold=0.7`` which catches "obviously similar" without
flooding the result list with weak matches.

Performance notes
-----------------

  * Indexing cost is bounded: one hash + one tokenization per method,
    O(method_size). Negligible compared to the tree-sitter parse that
    already ran in the indexer.
  * Query cost is the bottleneck: ``find_similar`` reads ALL methods
    of similar size from the project (typically a few thousand) and
    computes Jaccard pairwise. For projects up to ~10k methods this
    runs in <100 ms; bigger codebases would need an inverted index
    over trigrams, which we'll add when someone hits the wall.
  * The size pre-filter (``± 30 %`` line count) drops most candidates
    before the Jaccard math runs.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Iterable

from infinidev.tools.base.db import execute_with_retry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

# Tokens that should NOT be replaced with placeholders — language
# keywords are part of the structure, not user-chosen names. Keeping
# them stable makes the normalized form much more discriminative
# (a `for` loop and a `while` loop should NOT collapse to the same
# shape just because both use V1, V2, V3 placeholders).
#
# This is a union of common keywords across the supported languages.
# A token is treated as a placeholder candidate iff it's a valid
# identifier AND not in this set AND not a builtin literal
# (true/false/null/None/etc).
_KEYWORDS: frozenset[str] = frozenset({
    # python
    "def", "class", "if", "elif", "else", "for", "while", "in", "not",
    "and", "or", "is", "return", "yield", "lambda", "import", "from",
    "as", "with", "try", "except", "finally", "raise", "pass", "break",
    "continue", "global", "nonlocal", "del", "async", "await", "True",
    "False", "None", "self", "cls",
    # javascript / typescript
    "function", "var", "let", "const", "new", "this", "super",
    "extends", "implements", "interface", "type", "enum", "public",
    "private", "protected", "static", "readonly", "abstract",
    "switch", "case", "default", "throw", "true", "false", "null",
    "undefined", "void", "of", "instanceof", "typeof", "delete",
    "export",
    # go
    "func", "package", "struct", "map", "chan", "go", "defer", "select",
    "fallthrough", "range", "make", "len", "cap", "append", "nil",
    # rust
    "fn", "impl", "trait", "mut", "ref", "Self", "match", "loop", "use",
    "mod", "pub", "crate", "Some", "None", "Ok", "Err", "where", "dyn",
    "unsafe", "move", "box",
    # java / c#
    "void", "int", "long", "short", "byte", "char", "float", "double",
    "boolean", "String", "throws", "extends", "implements", "abstract",
    "synchronized", "volatile", "transient", "final", "native",
    "namespace", "using", "internal", "sealed", "override", "virtual",
    "out", "ref", "params", "is", "as",
    # c / cpp
    "include", "define", "typedef", "struct", "union", "enum", "extern",
    "static", "inline", "register", "auto", "const", "volatile",
    "sizeof", "goto", "do",
    # ruby
    "do", "end", "begin", "rescue", "ensure", "module", "require",
    "puts", "nil", "self",
    # php
    "echo", "print", "array", "isset", "unset", "empty", "global",
})

_LITERAL_TOKENS: frozenset[str] = frozenset({
    "true", "false", "null", "none", "nil", "undefined", "void",
    "True", "False", "None",
})

# Per-language comment tokens. Keys are the language name as produced
# by ``code_intel.parsers.detect_language``. Languages not in the map
# get the default of "// and /* */ and #" which covers everything.
_LINE_COMMENT_BY_LANG: dict[str, tuple[str, ...]] = {
    "python":     ("#",),
    "ruby":       ("#",),
    "bash":       ("#",),
    "shell":      ("#",),
    "javascript": ("//",),
    "typescript": ("//",),
    "tsx":        ("//",),
    "go":         ("//",),
    "rust":       ("//",),
    "java":       ("//",),
    "c":          ("//",),
    "cpp":        ("//",),
    "csharp":     ("//",),
    "c_sharp":    ("//",),
    "kotlin":     ("//",),
    "php":        ("//", "#"),
}

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_PY_DOCSTRING_RE = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\')', re.DOTALL)
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z_0-9]*\b")


def _strip_comments(text: str, language: str) -> str:
    """Remove comments from *text* using language-aware rules.

    Block comments (/* ... */) are stripped for any language, even
    Python — they don't appear in valid Python so the regex is a no-op
    there. Python docstrings ('''...''' / \"\"\"...\"\"\") are stripped
    separately so they don't dominate the normalized form.

    This is heuristic on purpose: we don't want to invoke tree-sitter
    a second time per method when a regex pass costs ~µs and is
    accurate enough for the similarity use case.
    """
    # Block comments first — applies to C-family, JS, Go, Rust, Java, PHP, etc.
    text = _BLOCK_COMMENT_RE.sub(" ", text)
    if language == "python":
        text = _PY_DOCSTRING_RE.sub(" ", text)

    # Line comments — language-aware, default to // and #.
    line_markers = _LINE_COMMENT_BY_LANG.get(language, ("//", "#"))
    out: list[str] = []
    for line in text.splitlines():
        stripped = line
        for marker in line_markers:
            idx = stripped.find(marker)
            if idx >= 0:
                # Don't strip a marker that's inside a string. Cheap
                # heuristic: count quotes before the marker; an odd
                # count means we're inside a string.
                prefix = stripped[:idx]
                if (prefix.count('"') - prefix.count('\\"')) % 2 == 0 and \
                   (prefix.count("'") - prefix.count("\\'")) % 2 == 0:
                    stripped = prefix
        out.append(stripped)
    return "\n".join(out)


def _normalize_identifiers(text: str) -> str:
    """Replace user-chosen identifiers with stable placeholders V1, V2, ...

    Keywords and literal tokens (true/false/null/etc) are kept as-is.
    Identifiers are renamed in order of first appearance, so two
    methods that differ only in variable names produce the same
    normalized form.

    The placeholder uses 'V' (for "variable") so it doesn't collide
    with any real keyword in the supported languages.
    """
    name_map: dict[str, str] = {}
    counter = [0]

    def _replace(m: re.Match) -> str:
        tok = m.group(0)
        if tok in _KEYWORDS or tok in _LITERAL_TOKENS:
            return tok
        if tok in name_map:
            return name_map[tok]
        counter[0] += 1
        placeholder = f"V{counter[0]}"
        name_map[tok] = placeholder
        return placeholder

    return _IDENT_RE.sub(_replace, text)


def normalize_body(body: str, language: str) -> tuple[str, int]:
    """Return ``(normalized_text, body_size_in_lines)`` for *body*.

    *body* is the raw source slice (extracted with file content +
    line_start/line_end of a method). *language* is the detected
    language; used to pick comment markers and the docstring rule.

    The returned ``body_size`` is the number of non-blank lines AFTER
    comment removal — this is what the size pre-filter compares.
    """
    if not body:
        return "", 0

    no_comments = _strip_comments(body, language)
    # Drop blank lines for the size count.
    non_blank = [ln for ln in no_comments.splitlines() if ln.strip()]
    body_size = len(non_blank)
    if body_size == 0:
        return "", 0

    rebuilt = "\n".join(non_blank)
    renamed = _normalize_identifiers(rebuilt)
    # Collapse all whitespace to single spaces and lowercase.
    collapsed = re.sub(r"\s+", " ", renamed).strip().lower()
    return collapsed, body_size


def body_hash(normalized: str) -> str:
    """SHA-256 (truncated to 16 hex chars) of a normalized body string.

    Truncation is fine for our use case: 16 hex chars = 64 bits = 1 in
    10^19 collision odds, way below the noise floor for any project we
    might index. Saves a few bytes per row in the DB and shows up in
    logs without wrapping.
    """
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity over the token sets of two normalized bodies.

    Returns 0.0 when either input is empty. Tokens are split on
    whitespace — both *a* and *b* are expected to be the output of
    :func:`normalize_body`. The result is in [0, 1] where 1.0 means
    identical token sets.

    Set similarity (rather than multiset) was chosen because the
    placeholder pass already collapses repeated identifiers; comparing
    multisets would over-weight loop bodies with many identical tokens.
    """
    if not a or not b:
        return 0.0
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DB read/write
# ─────────────────────────────────────────────────────────────────────────────

# Methods smaller than this are skipped entirely. A 3-line getter is
# always going to "look like" every other 3-line getter, polluting the
# similarity index with noise. The threshold is intentionally low —
# we want to catch real duplicates, not just trivial accessors.
_MIN_INDEXED_BODY_LINES = 6


@dataclass(frozen=True)
class MethodFingerprint:
    """One row of ``ci_method_bodies``, in code form."""

    project_id: int
    file_path: str
    qualified_name: str
    kind: str           # "function" | "method"
    line_start: int
    line_end: int
    body_size: int
    body_hash: str
    body_norm: str
    language: str


def _slice_body(file_lines: list[str], line_start: int, line_end: int) -> str:
    """Return the source slice for a 1-based inclusive line range.

    Defensive: the indexer occasionally produces line ranges that
    overshoot the file by one line (off-by-one in the parser). We
    clamp to the file length silently.
    """
    if line_start < 1:
        line_start = 1
    if line_end < line_start:
        return ""
    if line_end > len(file_lines):
        line_end = len(file_lines)
    return "".join(file_lines[line_start - 1:line_end])


def index_methods_for_file(
    project_id: int,
    file_path: str,
    file_content: bytes,
    language: str,
) -> int:
    """Compute and store fingerprints for all methods/functions in a file.

    Reads the symbols already inserted by ``store_file_symbols`` from
    ``ci_symbols``, slices each method body out of *file_content*,
    normalizes it, and writes the result to ``ci_method_bodies``.

    This is called by the indexer right after symbols are stored, so
    it always runs against the latest content. Old fingerprints for
    the file are deleted first (atomic replace).

    Returns the number of fingerprints inserted.
    """
    try:
        text = file_content.decode("utf-8", errors="replace")
    except Exception:
        return 0
    file_lines = text.splitlines(keepends=True)

    def _select_methods(conn: sqlite3.Connection) -> list[tuple]:
        return conn.execute(
            """
            SELECT name, qualified_name, kind, line_start, line_end
            FROM ci_symbols
            WHERE project_id = ? AND file_path = ?
              AND kind IN ('function', 'method')
              AND line_start IS NOT NULL AND line_end IS NOT NULL
            """,
            (project_id, file_path),
        ).fetchall()

    rows = execute_with_retry(_select_methods)
    if not rows:
        # Still clear any stale fingerprints in case the file used to
        # have methods and no longer does.
        _clear_file_fingerprints(project_id, file_path)
        return 0

    fingerprints: list[MethodFingerprint] = []
    for row in rows:
        # row is a sqlite3.Row — accessible by name or index.
        name = row["name"] if hasattr(row, "keys") else row[0]
        qual = row["qualified_name"] if hasattr(row, "keys") else row[1]
        kind = row["kind"] if hasattr(row, "keys") else row[2]
        line_start = row["line_start"] if hasattr(row, "keys") else row[3]
        line_end = row["line_end"] if hasattr(row, "keys") else row[4]

        if not line_start or not line_end:
            continue

        # Use qualified name when available; fall back to bare name.
        # Qualified names are how methods get correlated across files.
        full_name = qual or name
        if not full_name:
            continue

        body = _slice_body(file_lines, int(line_start), int(line_end))
        normalized, body_size = normalize_body(body, language)
        if body_size < _MIN_INDEXED_BODY_LINES:
            continue
        if not normalized:
            continue

        fingerprints.append(MethodFingerprint(
            project_id=project_id,
            file_path=file_path,
            qualified_name=full_name,
            kind=kind,
            line_start=int(line_start),
            line_end=int(line_end),
            body_size=body_size,
            body_hash=body_hash(normalized),
            body_norm=normalized,
            language=language,
        ))

    _replace_file_fingerprints(project_id, file_path, fingerprints)
    return len(fingerprints)


def _clear_file_fingerprints(project_id: int, file_path: str) -> None:
    def _delete(conn: sqlite3.Connection):
        conn.execute(
            "DELETE FROM ci_method_bodies WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        )
        conn.commit()
    try:
        execute_with_retry(_delete)
    except Exception as exc:
        logger.debug("clear fingerprints failed for %s: %s", file_path, exc)


def _replace_file_fingerprints(
    project_id: int,
    file_path: str,
    fingerprints: list[MethodFingerprint],
) -> None:
    """Atomically replace all fingerprints for a file."""
    def _store(conn: sqlite3.Connection):
        conn.execute(
            "DELETE FROM ci_method_bodies WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        )
        if fingerprints:
            conn.executemany(
                """
                INSERT OR REPLACE INTO ci_method_bodies
                (project_id, file_path, qualified_name, kind, line_start, line_end,
                 body_size, body_hash, body_norm, language)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (fp.project_id, fp.file_path, fp.qualified_name, fp.kind,
                     fp.line_start, fp.line_end, fp.body_size,
                     fp.body_hash, fp.body_norm, fp.language)
                    for fp in fingerprints
                ],
            )
        conn.commit()
    try:
        execute_with_retry(_store)
    except Exception as exc:
        logger.debug("store fingerprints failed for %s: %s", file_path, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Query API
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimilarMethodHit:
    """One result row from :func:`find_similar`.

    *similarity* is in [0, 1]; 1.0 means body_hash exact match,
    anything below comes from Jaccard over normalized tokens.
    """

    qualified_name: str
    file_path: str
    line_start: int
    line_end: int
    body_size: int
    similarity: float
    is_exact_dup: bool       # body_hash matched
    language: str

    def to_dict(self) -> dict:
        return {
            "qualified_name": self.qualified_name,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "body_size": self.body_size,
            "similarity": round(self.similarity, 3),
            "is_exact_dup": self.is_exact_dup,
            "language": self.language,
        }


def fetch_fingerprint(
    project_id: int,
    qualified_name: str,
    file_path: str | None = None,
) -> MethodFingerprint | None:
    """Look up the fingerprint for one method, by qualified name.

    When *file_path* is given, the result is restricted to that file
    (useful when two classes in different files share a method name
    like ``Service.start``). Otherwise the first match wins.
    """
    def _query(conn: sqlite3.Connection) -> list[tuple]:
        if file_path is not None:
            sql = """
                SELECT project_id, file_path, qualified_name, kind, line_start,
                       line_end, body_size, body_hash, body_norm, language
                FROM ci_method_bodies
                WHERE project_id = ? AND qualified_name = ? AND file_path = ?
                LIMIT 1
            """
            return conn.execute(sql, (project_id, qualified_name, file_path)).fetchall()
        sql = """
            SELECT project_id, file_path, qualified_name, kind, line_start,
                   line_end, body_size, body_hash, body_norm, language
            FROM ci_method_bodies
            WHERE project_id = ? AND qualified_name = ?
            LIMIT 1
        """
        return conn.execute(sql, (project_id, qualified_name)).fetchall()

    rows = execute_with_retry(_query)
    if not rows:
        return None
    row = rows[0]
    return MethodFingerprint(
        project_id=row[0], file_path=row[1], qualified_name=row[2],
        kind=row[3], line_start=row[4], line_end=row[5],
        body_size=row[6], body_hash=row[7], body_norm=row[8],
        language=row[9],
    )


def find_similar(
    project_id: int,
    qualified_name: str,
    *,
    file_path: str | None = None,
    threshold: float = 0.7,
    limit: int = 10,
    size_tolerance: float = 0.4,
) -> list[SimilarMethodHit]:
    """Find methods in the project that look like *qualified_name*.

    The pipeline:

      1. Fetch the target method's fingerprint from ``ci_method_bodies``.
      2. Find every other method in the same project whose ``body_size``
         is within ``size_tolerance`` of the target (default ±40 %).
      3. Rank by Jaccard over the normalized token sets.
      4. Methods with the same ``body_hash`` are surfaced first with
         ``similarity=1.0`` and ``is_exact_dup=True``.
      5. Return the top *limit* whose similarity is ≥ *threshold*.

    *size_tolerance* keeps the candidate set small even on large
    projects: a 50-line method only gets compared to other methods of
    30-70 lines. For projects with millions of methods this is still
    O(N) but the constant is tiny because most candidates are filtered
    out before we touch the token sets.
    """
    target = fetch_fingerprint(project_id, qualified_name, file_path)
    if target is None:
        return []

    min_size = int(target.body_size * (1 - size_tolerance))
    max_size = int(target.body_size * (1 + size_tolerance)) + 1

    def _query(conn: sqlite3.Connection) -> list[tuple]:
        return conn.execute(
            """
            SELECT qualified_name, file_path, line_start, line_end, body_size,
                   body_hash, body_norm, language
            FROM ci_method_bodies
            WHERE project_id = ?
              AND NOT (file_path = ? AND qualified_name = ?)
              AND body_size BETWEEN ? AND ?
            """,
            (project_id, target.file_path, target.qualified_name, min_size, max_size),
        ).fetchall()

    rows = execute_with_retry(_query)
    if not rows:
        return []

    hits: list[SimilarMethodHit] = []
    for r in rows:
        other_name, other_path, ls, le, sz, h, norm, lang = r
        is_dup = (h == target.body_hash)
        sim = 1.0 if is_dup else jaccard(target.body_norm, norm)
        if sim >= threshold:
            hits.append(SimilarMethodHit(
                qualified_name=other_name,
                file_path=other_path,
                line_start=ls,
                line_end=le,
                body_size=sz,
                similarity=sim,
                is_exact_dup=is_dup,
                language=lang,
            ))

    # Sort: exact duplicates first, then by similarity descending,
    # then by line count descending (longer matches are more meaningful
    # than short ones at the same similarity score).
    hits.sort(key=lambda h: (not h.is_exact_dup, -h.similarity, -h.body_size))
    return hits[:limit]


__all__ = [
    "MethodFingerprint",
    "SimilarMethodHit",
    "normalize_body",
    "body_hash",
    "jaccard",
    "index_methods_for_file",
    "fetch_fingerprint",
    "find_similar",
]
