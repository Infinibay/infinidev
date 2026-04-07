"""Dynamic guidance: warn when the model just wrote code that already exists.

Unlike the static detectors in :mod:`detectors`, which return a boolean
and let the library render a fixed message, this detector produces its
own rendered text with the specific methods it found. That's why it
lives in a separate module and is called through a separate path in
:mod:`hooks`.

The trigger conditions:

  1. The model just wrote one or more files in the previous step
     (``state.recently_written_files`` is non-empty — populated by
     ``engine.loop.engine._run_inner_loop`` from the behavior
     tracker's ``files_edited`` set).
  2. The indexer has already re-indexed those files (it runs
     synchronously inside ``index_file`` after every write via the
     standard indexing path), so ``ci_method_bodies`` contains the
     freshly-normalized fingerprints.
  3. For any method in the written file whose body is large enough
     to fingerprint (≥ 6 normalized lines), there exists another
     method in a DIFFERENT file whose body matches by hash or has
     Jaccard ≥ ``_STRONG_JACCARD_THRESHOLD``.

When it fires, the detector:

  * Records each warned-about file in ``state.similarity_warned_files``
    so the same warning is never emitted twice for the same file
    (per task).
  * Clears ``state.recently_written_files`` regardless of whether
    the check produced a warning — the state field is a per-step
    trigger, not a running buffer.
  * Returns the fully-rendered ``<guidance>`` XML block, ready for
    ``state.pending_guidance``.

The emitted message names up to three (target, match) pairs with file
paths and line ranges so the model can navigate directly. Anything
beyond three is counted ("... and N more") to keep the prompt small.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infinidev.engine.loop.loop_state import LoopState

logger = logging.getLogger(__name__)


# Jaccard threshold for "strong similarity". Higher than the default
# used by the user-facing find_similar_methods tool (0.7) because we
# only want to warn on genuinely duplicated logic — not on "two
# methods with vaguely similar shapes". 0.85 catches copy-paste with
# variable renaming, light comment changes, and small tweaks.
_STRONG_JACCARD_THRESHOLD = 0.85

# Maximum number of (target, match) pairs we include in the rendered
# message. More than this and the prompt gets noisy; the model can
# always ask for more via find_similar_methods explicitly.
_MAX_PAIRS_IN_MESSAGE = 3

# Maximum number of target methods we check per detector run. Writing
# a whole new module with 50 methods would otherwise trigger 50
# similarity queries — slow AND noisy. We pick the first N methods
# with a long-enough body.
_MAX_TARGETS_PER_FILE = 8

# Only look at the first N files in a single detector run. Most steps
# write 1 file; bursts of 5+ are rare and can carry over to the next
# step without losing information (the state field is a queue, not a
# one-shot trigger).
_MAX_FILES_PER_RUN = 2

# Minimum normalized body size to bother checking. Matches the
# _MIN_INDEXED_BODY_LINES in method_index but re-declared here so
# changes to the index don't silently change the warning threshold.
_MIN_BODY_SIZE = 6

# Files with more than this many indexed methods are assumed to be
# legitimate monoliths (large services) — we skip the similarity
# check entirely because (a) it's expensive and (b) a file with 30+
# methods was written on purpose and is not a refactoring mistake.
_MAX_METHODS_FOR_CHECK = 25


def _resolve_project_id(state: "LoopState") -> int | None:
    """Best-effort fetch of the current project_id from the tool context.

    The similarity detector doesn't get a project_id directly because
    the guidance system's contract is ``(messages, state)`` — keeping
    it that way would require plumbing project_id into every detector
    signature. Instead we read it from the process-global tool
    context, which is set once per task by the loop engine.
    """
    try:
        from infinidev.tools.base.context import get_current_project_id
        return get_current_project_id()
    except Exception:
        return None


def _fetch_fingerprints_for_file(
    project_id: int, file_path: str,
) -> list[tuple[str, int, int, int, str, str]]:
    """Return (qualified_name, line_start, line_end, body_size, body_hash, body_norm)
    for every method fingerprint currently stored for *file_path*.

    These are exactly the rows that the ``index_methods_for_file`` hook
    wrote right after the latest write, so comparing them against the
    rest of the project catches same-session duplication.

    Returns at most ``_MAX_TARGETS_PER_FILE`` entries, largest first,
    so we focus the similarity check on the methods that actually
    matter (trivial small helpers are uninformative).

    When the file has more than ``_MAX_METHODS_FOR_CHECK`` indexed
    methods total, returns an empty list — the detector skips big
    monolith files entirely because (a) the similarity check would
    be expensive across all of them and (b) a 30-method file is
    deliberate work, not an accidental duplication of something else.
    """
    from infinidev.tools.base.db import execute_with_retry
    import sqlite3

    def _q(conn: sqlite3.Connection):
        # Fast count first so we can skip big monoliths cheaply.
        total = conn.execute(
            "SELECT COUNT(*) FROM ci_method_bodies "
            "WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        ).fetchone()[0]
        if total > _MAX_METHODS_FOR_CHECK:
            return []
        return conn.execute(
            """
            SELECT qualified_name, line_start, line_end, body_size,
                   body_hash, body_norm
            FROM ci_method_bodies
            WHERE project_id = ? AND file_path = ?
              AND body_size >= ?
            ORDER BY body_size DESC
            LIMIT ?
            """,
            (project_id, file_path, _MIN_BODY_SIZE, _MAX_TARGETS_PER_FILE),
        ).fetchall()

    try:
        return execute_with_retry(_q) or []
    except Exception as exc:
        logger.debug("similarity_detector: fingerprint fetch failed: %s", exc)
        return []


def _find_matches_batched(
    project_id: int,
    file_path: str,
    targets: list[tuple[str, int, int, int, str, str]],
) -> list[tuple[str, str, int, int, float, bool]]:
    """Batched similarity query: ONE fetch per file instead of per method.

    Algorithm:

      1. **Exact-hash fast path**. Collect every ``body_hash`` from the
         targets and run ONE ``WHERE body_hash IN (...)`` query.
         Indexed, O(log N) per lookup. Matches here are already
         similarity=1.0 and skip the Jaccard math entirely.

      2. **Bounded Jaccard pass**. For targets without an exact hit,
         compute the global min/max size window across all remaining
         targets, fetch every candidate in that window with ONE
         query, then iterate in memory comparing each target against
         each candidate. This replaces N individual queries (one per
         target) with a single query whose result set is reused for
         all targets.

    Both passes exclude matches in *file_path* itself so a model
    adding a second method to a utility file doesn't get warned about
    the first one.

    Returns list of ``(other_file, other_name, line_start, line_end,
    similarity, is_exact_dup)`` across all targets, sorted with exact
    duplicates first. No deduplication within the result list —
    callers render them grouped by target.
    """
    if not targets:
        return []
    from infinidev.code_intel.method_index import jaccard
    from infinidev.tools.base.db import execute_with_retry
    import sqlite3

    # (qualified_name, line_start, line_end, body_size, body_hash, body_norm)
    target_hashes = {t[4]: t for t in targets}
    hash_list = list(target_hashes.keys())

    # Global size window covering ALL remaining targets. A target of
    # 20 lines with 0.6-1.4 tolerance wants 12-28; a target of 40
    # wants 24-56; the union is 12-56.
    min_size = min(int(t[3] * 0.6) for t in targets)
    max_size = max(int(t[3] * 1.4) + 1 for t in targets)

    def _q(conn: sqlite3.Connection):
        # Pass 1: exact hash matches across all targets in one query.
        placeholders = ",".join("?" for _ in hash_list)
        exact = conn.execute(
            f"""
            SELECT qualified_name, file_path, line_start, line_end,
                   body_hash
            FROM ci_method_bodies
            WHERE project_id = ?
              AND file_path != ?
              AND body_hash IN ({placeholders})
            """,
            [project_id, file_path, *hash_list],
        ).fetchall()

        # Pass 2: size-windowed candidates for the Jaccard math.
        candidates = conn.execute(
            """
            SELECT qualified_name, file_path, line_start, line_end,
                   body_size, body_hash, body_norm
            FROM ci_method_bodies
            WHERE project_id = ?
              AND file_path != ?
              AND body_size BETWEEN ? AND ?
            """,
            (project_id, file_path, min_size, max_size),
        ).fetchall()
        return exact, candidates

    try:
        exact, candidates = execute_with_retry(_q)
    except Exception as exc:
        logger.debug("similarity_detector: batched query failed: %s", exc)
        return []

    # hits elements: (target_name, other_file, other_name,
    #                 line_start, line_end, similarity, is_exact_dup)
    hits: list[tuple[str, str, str, int, int, float, bool]] = []

    for row in exact:
        other_name, other_path, ls, le, h = row
        target = target_hashes.get(h)
        if target is None:
            continue  # shouldn't happen but defensive
        hits.append((target[0], other_path, other_name, ls, le, 1.0, True))

    # Jaccard pass — for each candidate, find the BEST target match.
    # Skip candidates whose hash already produced an exact hit.
    exact_hashes = {h for h in target_hashes.keys()}
    for cand in candidates:
        c_name, c_path, c_ls, c_le, c_size, c_hash, c_norm = cand
        if c_hash in exact_hashes:
            continue
        best_sim = 0.0
        best_target_name = ""
        for t in targets:
            t_size = t[3]
            if not (int(t_size * 0.6) <= c_size <= int(t_size * 1.4) + 1):
                continue
            sim = jaccard(t[5], c_norm)
            if sim > best_sim:
                best_sim = sim
                best_target_name = t[0]
        if best_sim >= _STRONG_JACCARD_THRESHOLD:
            hits.append((best_target_name, c_path, c_name, c_ls, c_le, best_sim, False))

    hits.sort(key=lambda h: (not h[6], -h[5]))
    return hits


def check_similarity_after_write(state: "LoopState") -> str | None:
    """Run the similarity check on freshly written files and return guidance text.

    Returns ``None`` when nothing fires (most steps). When it does
    fire, returns a fully-rendered guidance block ready to be assigned
    to ``state.pending_guidance``.

    Side effects (only when the detector has something to do):
      * Clears ``state.recently_written_files`` — the field is a
        per-step trigger, not an accumulator. Cleared even on no-hit
        runs so we don't re-check the same files every step.
      * Adds warned-about paths to ``state.similarity_warned_files``
        so we never emit the same warning twice per task.

    Safe to call every step. Short-circuits with ``None`` when there's
    nothing written, the project_id isn't resolvable, or no matches
    cross the threshold.
    """
    if not state.recently_written_files:
        return None

    project_id = _resolve_project_id(state)
    if project_id is None:
        # Can't query the index without a project id — clear the
        # trigger list so we don't retry every step.
        state.recently_written_files = []
        return None

    # Pop the first _MAX_FILES_PER_RUN files off the queue. Remaining
    # files stay in the state and will be checked on the next step,
    # so burst writes don't all get processed in one expensive run.
    written = list(state.recently_written_files)
    to_check = written[:_MAX_FILES_PER_RUN]
    remaining = written[_MAX_FILES_PER_RUN:]
    state.recently_written_files = remaining

    warned_set = set(state.similarity_warned_files)
    pairs: list[tuple[str, str, str, str, int, int, float, bool]] = []
    # (target_file, target_name, other_file, other_name, line_start,
    #  line_end, similarity, is_exact_dup)

    for file_path in to_check:
        if file_path in warned_set:
            continue

        targets = _fetch_fingerprints_for_file(project_id, file_path)
        if not targets:
            continue

        # One batched query per file — covers all targets at once.
        # Tuples: (target_name, other_file, other_name, ls, le, sim, is_dup)
        all_matches = _find_matches_batched(project_id, file_path, targets)
        if not all_matches:
            continue

        for (t_name, other_path, other_name, ls, le, sim, is_dup) in all_matches:
            pairs.append((
                file_path, t_name,
                other_path, other_name, ls, le, sim, is_dup,
            ))
            if len(pairs) >= _MAX_PAIRS_IN_MESSAGE + 3:
                break

        state.similarity_warned_files.append(file_path)
        if len(pairs) >= _MAX_PAIRS_IN_MESSAGE + 3:
            break

    if not pairs:
        return None

    return _render_similarity_guidance(pairs)


def _render_similarity_guidance(
    pairs: list[tuple[str, str, str, str, int, int, float, bool]],
) -> str:
    """Render the (target, match) pairs as a <guidance> block.

    Same XML shape as the static library entries so the prompt builder
    doesn't need to distinguish the two sources — one ``pending_guidance``
    slot, one rendering path downstream.
    """
    exact_count = sum(1 for p in pairs if p[7])
    partial_count = sum(1 for p in pairs if not p[7])

    lines: list[str] = []
    lines.append('<guidance pattern="similarity_after_write">')
    lines.append("## You may be reimplementing existing code")
    lines.append("")
    lines.append(
        "The code you just wrote contains methods that look very similar "
        "to methods already in this project. Before continuing, consider "
        "whether you should be calling the existing helper instead, or "
        "consolidating both implementations into a shared function."
    )
    lines.append("")
    if exact_count:
        lines.append(
            f"Found {exact_count} EXACT duplicate(s) "
            "(identical after stripping comments and renaming variables)"
            + (f" and {partial_count} strong partial match(es)." if partial_count else ".")
        )
    else:
        lines.append(f"Found {partial_count} strong partial match(es) (≥85% similarity).")
    lines.append("")

    shown = pairs[:_MAX_PAIRS_IN_MESSAGE]
    for (tgt_file, tgt_name, other_file, other_name, ls, le, sim, is_dup) in shown:
        tag = "EXACT DUP" if is_dup else f"{sim:.0%}"
        lines.append(f"  [{tag}] you wrote: {tgt_name}")
        lines.append(f"          in:       {tgt_file}")
        lines.append(f"          matches:  {other_name}")
        lines.append(f"          at:       {other_file}:{ls}-{le}")
        lines.append("")

    if len(pairs) > _MAX_PAIRS_IN_MESSAGE:
        lines.append(f"  ... and {len(pairs) - _MAX_PAIRS_IN_MESSAGE} more matches.")
        lines.append("")

    lines.append("## What to do")
    lines.append("")
    lines.append(
        "1. Read the existing method with get_symbol_code or partial_read."
    )
    lines.append(
        "2. If they do the same thing, call the existing one from your new "
        "code and delete your duplicate. Use edit_symbol or replace_lines."
    )
    lines.append(
        "3. If they're different enough to justify, leave a comment "
        "explaining why — the next reader will wonder."
    )
    lines.append(
        "4. To see all matches for your method, call "
        "find_similar_methods with the qualified name."
    )
    lines.append("</guidance>")

    return "\n".join(lines)


__all__ = ["check_similarity_after_write"]
