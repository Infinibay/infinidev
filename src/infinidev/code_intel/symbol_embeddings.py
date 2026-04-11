"""Per-symbol and per-file embedding pipeline for ContextRank fuzzy search.

ContextRank v3 replaces the old substring-based mention detection
(``_compute_mention_scores`` in ranker.py) with dense embeddings.  At
index time, each indexed symbol and each source file gets an
embedding stored inline on its ``ci_symbols`` / ``ci_files`` row.  At
rank time, the ranker computes the embedding of the user's input
once, then does a vectorized cosine sweep against all stored
embeddings to find semantic matches.

Compared to the old substring approach this:

1. **Handles typos.**  ``AuthServise`` still matches ``AuthService``
   because all-MiniLM-L6-v2 encodes the subword structure, not the
   literal string.
2. **Handles synonyms.**  "authentication service" matches
   ``AuthenticationHandler`` via shared semantic space, even though
   the literal strings don't overlap.
3. **Eliminates stop-word and kind-filter lists.**  No more
   _COMMON_WORDS, _STEM_SKIP, or "only functions/classes/enums"
   restrictions — the similarity score filters noise naturally.

The embedding pipeline is:

- **Symbol text:**
    ``{kind} {name} — {docstring_first_line or signature}``
  e.g. ``function validate_token — Verifies JWT signature and expiration``
  The ``kind`` prefix anchors the embedding in the symbol's role, and
  the trailing description gives the model enough context to
  distinguish ``class User`` (a data model) from ``function User``
  (a factory function).

- **File text:**
    ``{language} {basename_stem} — {module_docstring or top-N symbol names}``
  e.g. ``python auth_service — AuthService, validate_token, refresh_session``
  This lets the ranker match whole files by purpose even when no
  individual symbol name is distinctive.

Incremental behaviour is inherited from ``indexer.py`` — this module
only runs when the indexer calls it (which happens only when the
file's content hash has changed), so re-indexing skips unchanged
files automatically.

Failures are best-effort: any exception while embedding is logged at
debug level but never propagated.  The indexer continues, and the
symbols remain present in ``ci_symbols`` without embeddings.  The
ranker's fuzzy channel handles missing embeddings by simply skipping
the row — no symbol is rejected, only the fuzzy channel's coverage
drops.
"""

from __future__ import annotations

import logging
import os

from infinidev.code_intel._db import execute_with_retry
from infinidev.tools.base.embeddings import compute_embedding

logger = logging.getLogger(__name__)

# Kinds that carry enough meaning to be worth embedding.  Imports are
# excluded because their "name" is already just the module path —
# indexing them would create duplicates of the imported symbol's own
# embedding.  Everything else is fair game: the old canal 3 used to
# restrict to a narrow whitelist (function/method/class/interface),
# but with fuzzy matching, variables and constants ("DEFAULT_TIMEOUT")
# are equally valuable to surface.
_EMBEDDABLE_KINDS = frozenset({
    "function",
    "method",
    "class",
    "interface",
    "enum",
    "type_alias",
    "struct",
    "variable",
    "constant",
    "trait",
    "module",
})

# Maximum docstring characters to include in the embedding text.  The
# all-MiniLM-L6-v2 model truncates at 256 tokens, so there's no point
# embedding 2000-char docstrings — the tail would be discarded anyway.
# 200 chars leaves room for the kind + name prefix.
_MAX_DESC_CHARS = 200

# Top-N symbol names to include in a file's embedding text when the
# file has no module docstring.  Names alone are a decent proxy for
# "what this file is about" when nothing else is available.
_FILE_TOP_N_SYMBOLS = 8


def _first_line(text: str, max_chars: int = _MAX_DESC_CHARS) -> str:
    """Take the first meaningful line of a docstring, capped in length.

    Tree-sitter docstrings often come with leading/trailing whitespace
    and multi-line bodies.  For an embedding anchor we only need the
    first line — it's where developers typically put the summary
    ("Verifies JWT signature and expiration") before diving into
    parameter details.
    """
    if not text:
        return ""
    # Strip common comment/docstring markers
    text = text.strip().strip('"""').strip("'''").strip("/*").strip("*/").strip()
    # First non-empty line
    for line in text.splitlines():
        line = line.strip().lstrip("*").strip()
        if line:
            return line[:max_chars]
    return ""


def _kind_str(sym) -> str:
    """Extract the kind as a plain lowercase string.

    ``Symbol.kind`` is a ``SymbolKind`` enum in infinidev
    (see ``code_intel/parsers``) and ``str()`` on an enum returns
    ``"SymbolKind.variable"``, not ``"variable"``.  We pull the
    ``.value`` when present and fall back to ``.name`` or the
    str representation for defensive robustness in tests.
    """
    k = getattr(sym, "kind", None)
    if k is None:
        return ""
    val = getattr(k, "value", None)
    if val is not None:
        return str(val)
    return str(k)


def _build_symbol_text(sym) -> str:
    """Construct the embedding text for a symbol.

    Format: ``{kind} {name} — {desc}``

    ``desc`` is the first line of the docstring when present,
    falling back to the signature, falling back to the name alone.
    The em-dash separator is intentional — it gives sentence-
    transformers a clear break between the identity (kind + name)
    and the description.
    """
    kind = _kind_str(sym).replace("_", " ")
    name = str(getattr(sym, "name", "") or "")

    desc = _first_line(str(getattr(sym, "docstring", "") or ""))
    if not desc:
        sig = str(getattr(sym, "signature", "") or "").strip()
        if sig:
            desc = sig[:_MAX_DESC_CHARS]

    if desc:
        return f"{kind} {name} — {desc}"
    return f"{kind} {name}"


def _build_file_text(file_path: str, language: str, symbols: list) -> str:
    """Construct the embedding text for a whole file.

    Format: ``{language} {basename_stem} — {top N symbol names}``

    The stem is what developers actually say aloud ("the auth_service
    file"), so it's the natural anchor for "which file did I mean".
    The trailing symbol names capture intent when the stem is generic
    (e.g. ``index.ts`` means nothing on its own — but ``index.ts`` with
    symbols ``[AuthService, validate_token, refresh_session]`` is
    clearly the auth module's entry point).
    """
    basename = os.path.basename(file_path)
    stem = basename.rsplit(".", 1)[0] if "." in basename else basename

    # Collect distinctive symbol names
    names: list[str] = []
    for sym in symbols:
        name = str(getattr(sym, "name", "") or "")
        kind = _kind_str(sym)
        if not name or kind == "import":
            continue
        # Skip dunders and very short names — they don't help identify intent
        if name.startswith("__") or len(name) < 3:
            continue
        names.append(name)
        if len(names) >= _FILE_TOP_N_SYMBOLS:
            break

    if names:
        return f"{language} {stem} — {', '.join(names)}"
    return f"{language} {stem}"


def embed_file_symbols(
    project_id: int, file_path: str, symbols: list, language: str,
) -> None:
    """Compute and store embeddings for all symbols in a file + the file itself.

    Called from the indexer immediately after ``store_file_symbols``,
    so the ``ci_symbols`` rows for this file already exist and can be
    matched by ``(project_id, file_path, name, line_start)``.

    Failures are best-effort — any exception is caught, logged at
    debug level, and suppressed.  The indexer continues.
    """
    if not symbols:
        # Still update the file-level embedding so directory-level
        # queries work even for empty files.
        pass

    try:
        # ── File-level embedding ──────────────────────────────────
        file_text = _build_file_text(file_path, language, symbols)
        file_emb = compute_embedding(file_text)

        def _update_file(conn):
            if file_emb is not None:
                conn.execute(
                    "UPDATE ci_files "
                    "SET embedding = ?, embedding_text = ? "
                    "WHERE project_id = ? AND file_path = ?",
                    (file_emb, file_text, project_id, file_path),
                )
                conn.commit()
        execute_with_retry(_update_file)
    except Exception as exc:
        logger.debug("file embedding failed for %s: %s", file_path, exc)

    # ── Per-symbol embeddings ─────────────────────────────────────
    # Each symbol gets one compute_embedding call.  ChromaDB's
    # default embed fn is fast on short strings (~1ms each), so a
    # 100-symbol file completes in ~100ms — well inside the indexer's
    # budget.  Batching at the ChromaDB level would help on larger
    # files but adds complexity the current volume doesn't justify.
    for sym in symbols:
        kind = _kind_str(sym)
        if kind not in _EMBEDDABLE_KINDS:
            continue
        name = str(getattr(sym, "name", "") or "")
        if not name:
            continue

        try:
            text = _build_symbol_text(sym)
            emb = compute_embedding(text)
            if emb is None:
                continue

            line_start = int(getattr(sym, "line_start", 0) or 0)

            # Match on (project_id, file_path, name, line_start) — the
            # combination is unique per symbol even for overloaded names
            # in languages that allow them.
            def _update(conn, _emb=emb, _text=text, _name=name, _line=line_start):
                conn.execute(
                    "UPDATE ci_symbols "
                    "SET embedding = ?, embedding_text = ? "
                    "WHERE project_id = ? "
                    "  AND file_path = ? "
                    "  AND name = ? "
                    "  AND line_start = ?",
                    (_emb, _text, project_id, file_path, _name, _line),
                )
                conn.commit()
            execute_with_retry(_update)
        except Exception as exc:
            logger.debug(
                "symbol embedding failed for %s in %s: %s",
                name, file_path, exc,
            )
