"""Deterministic automated checks for the post-development review engine.

Collects non-LLM signals — orphaned references, missing docstrings,
test-count deltas, per-file symbol lists, and hunk +/- stats — that
ground the judge against hallucinated claims. Split out of
``review_engine`` (behavior unchanged); ``review_engine`` re-imports
``collect_automated_checks`` so its call sites and the
``review_engine.collect_automated_checks`` monkeypatch still resolve.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def collect_automated_checks(
    changed_files: list[str],
    file_tracker: Any = None,
    verification_passed: bool | None = None,
    file_changes_summary: str = "",
) -> dict[str, Any]:
    """Gather deterministic check results to feed into the reviewer.

    Runs orphaned-reference and missing-docstring checks against the
    code-intel index for the given changed files. Resilient: any single
    check that errors returns an empty list for that key.

    Additional deterministic signals folded into the result:

    * ``test_counts`` — per test-file before/after/delta, computed by
      regex over the raw file content (see :mod:`.test_counter`). Grounds
      the judge against developer-report claims like "added 10 tests".
    * ``file_symbols`` — per changed file, the list of top-level symbols
      currently in the index. Catches extractor claims that a symbol was
      added/modified when no such symbol exists in the file.
    * ``hunk_stats`` — per file, the count of added/removed lines parsed
      from ``file_changes_summary``. A cheap sanity check on the
      extractor's ``line_range`` field.
    """
    from infinidev.code_intel.analyzer import (
        check_missing_docstrings, check_orphaned_references,
    )
    from infinidev.code_intel.smart_index import ensure_indexed

    project_id = 1  # CLI uses a single project by default
    result: dict[str, Any] = {
        "verification_passed": verification_passed,
        "orphaned_references": [],
        "missing_docstrings": [],
        "test_counts": {},
        "file_symbols": {},
        "hunk_stats": {},
    }

    # Reindex changed files so the checks see post-edit state
    import os as _os
    abs_changed = [_os.path.abspath(p) for p in (changed_files or [])]
    for p in abs_changed:
        try:
            ensure_indexed(project_id, p)
        except Exception as exc:
            logger.debug("Reindex failed for %s before checks: %s", p, exc)

    # Orphaned references — only if tracker recorded deletions
    if file_tracker is not None:
        try:
            deleted = file_tracker.get_deleted_symbols()
            if deleted:
                # Make sure source files are reindexed too
                for src in deleted.keys():
                    try:
                        ensure_indexed(project_id, src)
                    except Exception:
                        pass
                diags = check_orphaned_references(project_id, deleted)
                result["orphaned_references"] = [
                    {"file": d.file_path, "line": d.line, "message": d.message}
                    for d in diags
                ]
        except Exception as exc:
            logger.debug("Orphaned references collection failed: %s", exc)

    # Missing docstrings — one query per changed file
    for p in abs_changed:
        try:
            diags = check_missing_docstrings(project_id, p)
            for d in diags:
                result["missing_docstrings"].append(
                    {"file": d.file_path, "line": d.line, "message": d.message}
                )
        except Exception as exc:
            logger.debug("Missing-docstrings check failed for %s: %s", p, exc)

    # ── Deterministic signals for the judge ─────────────────────────────
    # These kill whole classes of LLM hallucination: "added 10 tests"
    # checked against regex count, "added function foo" checked against
    # the index, and "line_range 42-58" checked against actual +/- counts.
    try:
        result["test_counts"] = _compute_test_counts(abs_changed, file_tracker)
    except Exception as exc:
        logger.debug("test_counts check failed: %s", exc)

    try:
        result["file_symbols"] = _compute_file_symbols(project_id, abs_changed)
    except Exception as exc:
        logger.debug("file_symbols check failed: %s", exc)

    if file_changes_summary:
        try:
            result["hunk_stats"] = _compute_hunk_stats(file_changes_summary)
        except Exception as exc:
            logger.debug("hunk_stats check failed: %s", exc)

    return result


# ─────────────────────────────────────────────────────────────────────────
# Deterministic signal helpers
# ─────────────────────────────────────────────────────────────────────────


def _compute_test_counts(abs_changed: list[str], file_tracker: Any) -> dict[str, dict[str, int]]:
    """Count test cases before/after for each changed test file.

    Pulls before-content from the ``file_tracker``'s ``_originals`` and
    after-content from ``_current`` (both set as edits flow through
    ``FileChangeTracker.record``). Files not in the tracker fall back to
    reading the current file from disk, with ``before`` treated as empty.
    """
    from infinidev.engine.analysis.test_counter import (
        count_tests_for_files, looks_like_test_file,
    )

    entries: list[tuple[str, str | None, str | None]] = []
    for path in abs_changed:
        if not looks_like_test_file(path):
            continue
        before: str | None = None
        after: str | None = None
        if file_tracker is not None:
            try:
                # FileChangeTracker stores by abspath internally.
                before = file_tracker._originals.get(path)
                after = file_tracker._current.get(path)
            except Exception:
                pass
        if after is None:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    after = f.read()
            except OSError:
                after = None
        entries.append((path, before, after))

    return count_tests_for_files(entries)


def _compute_file_symbols(project_id: int, abs_changed: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Return the top-level symbols currently indexed for each changed file.

    Uses the post-edit index (reindex runs before this helper). The judge
    uses this to validate extractor claims like ``symbols_added: ["foo"]``
    — if ``foo`` isn't in the file's symbol list, the judge knows the
    extractor hallucinated.

    Full before/after symbol delta is follow-up work; getting pre-edit
    symbols would require a separate tree-sitter pass on the tracker's
    ``_originals`` content and isn't worth the complexity for this phase.
    """
    from infinidev.code_intel.query import list_symbols

    out: dict[str, list[dict[str, Any]]] = {}
    for path in abs_changed:
        try:
            syms = list_symbols(project_id, path, limit=100)
        except Exception:
            continue
        if not syms:
            continue
        out[path] = [
            {"name": s.name, "kind": s.kind, "line": s.line_start}
            for s in syms
            # Filter to top-level-ish kinds; drop locals/params.
            if s.kind in ("class", "function", "method", "interface", "enum")
        ]
    return out


_HUNK_HEADER_RE = re.compile(r"^###\s+(.+?)\s+\(", re.MULTILINE)


def _compute_hunk_stats(file_changes_summary: str) -> dict[str, dict[str, int]]:
    """Count +/- lines per file by parsing the diff summary.

    The summary is the output of ``engine.get_changed_files_summary()``:
    sections headed by ``### path (action)`` followed by a fenced diff.
    We walk the string section by section, counting lines that start with
    ``+`` or ``-`` but aren't diff headers (``+++``/``---``).
    """
    stats: dict[str, dict[str, int]] = {}
    # Split on the file headers, keeping the path as the section key.
    positions = [(m.start(), m.group(1).strip()) for m in _HUNK_HEADER_RE.finditer(file_changes_summary)]
    if not positions:
        return stats
    positions.append((len(file_changes_summary), ""))  # sentinel end

    for i in range(len(positions) - 1):
        start, path = positions[i]
        end, _ = positions[i + 1]
        block = file_changes_summary[start:end]
        added = 0
        removed = 0
        for line in block.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                removed += 1
        stats[path] = {"added": added, "removed": removed}
    return stats
