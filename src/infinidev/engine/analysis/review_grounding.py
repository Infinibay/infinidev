"""Grounding helpers for the post-development review engine.

Pure functions that verify the judge's claims against ground truth:
quote-in-diff containment, stable issue-id hashing, line coercion, and
changed-file extraction from the diff summary. Split out of
``review_engine`` to keep that module focused on orchestration. The
functions are unchanged, only relocated; ``review_engine`` re-imports
them so existing call sites and tests keep working.
"""

from __future__ import annotations

import re
from typing import Any


# Matches the per-file header emitted by engine.get_changed_files_summary()
# (see engine/loop/engine.py ~line 294): "### path (action)" optionally
# followed by ", no diff". This is the canonical source of truth for the
# file set — we grep it rather than asking the LLM to enumerate.
_FILE_HEADER_RE = re.compile(r"^###\s+(.+?)\s+\(([^)]+)\)\s*$", re.MULTILINE)


def _coerce_line(value: Any) -> int | None:
    """Coerce the judge's ``line`` field to an int or None.

    The LLM sometimes returns strings ("42"), sometimes floats, sometimes
    a range like "42-58". We accept the first integer we find; anything
    else becomes None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"-?\d+", s)
    if m is None:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _quote_is_grounded(
    quoted: str,
    file_path: str,
    file_contents: dict[str, str],
    file_changes_summary: str,
) -> bool:
    """Verify ``quoted`` appears verbatim in the cited file or the diff.

    Normalizes both sides by collapsing runs of whitespace so minor
    formatting drift doesn't trigger a false demotion. The string must
    still appear in the post-edit file contents (if we have them for
    ``file_path``) or anywhere in ``file_changes_summary``.

    Absent content and an absent quote both return False — the caller
    treats that as "couldn't verify" and demotes.
    """
    if not quoted:
        return False
    needle = _normalize_whitespace(quoted)
    if not needle:
        return False

    content = file_contents.get(file_path) if file_path else None
    if content and needle in _normalize_whitespace(content):
        return True
    if file_changes_summary and needle in _normalize_whitespace(file_changes_summary):
        return True
    return False


_WS_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces for fuzzy containment."""
    return _WS_RE.sub(" ", text).strip()


def _compute_issue_id(issue: dict) -> str:
    """Assign a 10-char sha1 hash derived from the issue's grounded fields.

    Stable across the extractor's retry-once loop and across rework
    rounds — the same issue re-emitted gets the same id, enabling
    dedup and cross-round tracking.

    Uses the POST-validation severity so a demoted issue changes id;
    a demoted issue is effectively a different claim.
    """
    import hashlib
    desc = str(issue.get("description") or "")[:80]
    desc = _WS_RE.sub(" ", desc).strip().lower()
    parts = [
        str(issue.get("file") or "").strip(),
        str(issue.get("line") if issue.get("line") is not None else ""),
        str(issue.get("category") or "").strip().lower(),
        str(issue.get("severity") or "").strip().lower(),
        desc,
    ]
    key = "|".join(parts).encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:10]


def _extract_changed_files_from_summary(summary: str) -> list[tuple[str, str]]:
    """Parse ``get_changed_files_summary`` output into (path, action) pairs.

    Deterministic ground truth for the diff's file set. Used to validate
    and backfill the extractor LLM's ``changes[]`` output, which is
    empirically prone to dropping files on large diffs.
    """
    if not summary:
        return []
    pairs: list[tuple[str, str]] = []
    for m in _FILE_HEADER_RE.finditer(summary):
        path = m.group(1).strip()
        action_raw = m.group(2).strip()
        # "modified, no diff" → action="modified"
        action = action_raw.split(",", 1)[0].strip()
        pairs.append((path, action))
    return pairs
