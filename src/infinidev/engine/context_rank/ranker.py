"""ContextRanker v3 — multi-channel scoring for files, symbols, and findings.

Combines 4 independent scoring channels + 3 post-processing boosts:

**Independent channels** (produce scores from scratch):
  1. Reactive — current session tool calls with recency decay and
     productivity pattern multiplier (read+edit boosted 2×, re-read
     without edit penalised as "confusion" 0.7×, edit-only 1.5×)
  2. Predictive (historical) — embedding sim vs past contexts →
     interactions, with day-based session decay, sim² contribution,
     and per-target productivity multiplier from cr_session_scores
     (edited-in-past-session = 1.5×, exploratory-only = 0.6×)
  3. Fuzzy symbol search — cosine similarity against per-symbol
     and per-file embeddings stored at index time.  Handles typos
     ("AuthServise" → AuthService) and synonyms ("authentication
     service" → AuthenticationHandler).  Replaces v2's substring-
     based "mention detection".
  4. Semantic findings — multi-signal (cosine + topic word match +
     tag match), dedup by finding id, ORDER BY confidence DESC.

**Post-processing boosts** (modify existing scores):
  5. Co-occurrence — files frequently accessed alongside top-scored
     files in the last 90 days (cr_session_scores self-join).
  6. Import graph — 1-hop propagation through ``ci_imports``;
     importers/downstream boosted more than imported/upstream
     because downstream consumers are the non-obvious surface.
  7. Freshness — filesystem mtime amplifies already-ranked files
     (multiplicative, 1.0-1.3×).  Does not rescue unranked files.

A confidence gate suppresses the entire ``<context-rank>`` section
when the top raw score is below ``CONTEXT_RANK_MIN_CONFIDENCE``.
Outliers within a passing ranking are filtered via MAD on the
bottom half of scores (see ``_filter_outliers``).

v3 removed five things vs v2:
  - canal 5 (docstring BM25): folded into canal 3 via the symbol
    embedding text, which includes the first line of each docstring
  - canal 6 (popularity): scores too low to clear the confidence
    gate, marginal value over list_directory
  - canal 10 (directory expansion): rarely fired, JS/TS/Python bias
  - ``_LEVEL_WEIGHTS`` (escalera bonuses): replaced by sim² so
    confident matches win without arbitrary level ordering
  - ``log(1+count)`` frequency boost in reactive: rewarded confusion,
    replaced by productivity pattern multiplier
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any

import numpy as np

from infinidev.code_intel._db import execute_with_retry
from infinidev.config.settings import settings
from infinidev.engine.context_rank.models import ContextRankResult, RankedItem

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

# Phase 2 v3 removed _LEVEL_WEIGHTS (task_input / step_title /
# step_description).  The arbitrary 1.0/1.5/2.0 weights imposed a
# fixed ordering that wasn't grounded in data — a task_input that
# matches at 0.85 is more trustworthy than a step_description that
# matches at 0.55, not the other way around.  The new contribution
# formula uses `sim²` alone, which rewards high-confidence matches
# naturally without privileging any level.  The escalera is still
# used for *logging* contexts at three granularities, but the ranker
# treats them uniformly.

# Identifiers shorter than this are too common to be useful mentions
_MIN_IDENT_LEN = 4

# ── Alpha blend (reactive vs predictive) ──
# `alpha = 0` → pure predictive (historical), `alpha = 1` → pure reactive
# (this session).  Grows linearly with iteration so the loop starts
# biased toward historical memory and shifts toward current-session
# evidence as the task progresses.
#
# `ALPHA_ITERATION_SATURATE` — number of iterations at which alpha
# reaches `ALPHA_MAX`.  Chosen to be "roughly one step's worth of
# tool calls": most steps finish in 4-8 tool calls, so by iteration
# 8 the current session has produced enough signal to dominate.
_ALPHA_ITERATION_SATURATE = 8
# Never give reactive full weight — always keep a floor of historical
# signal so cross-session memory cannot be completely drowned by a
# few noisy in-session reads.
_ALPHA_MAX = 0.85
# If the current session has fewer than this many reactive signals,
# the reactive channel is too sparse to trust on its own.  We halve
# its contribution so predictive/historical still dominates.
_ALPHA_REACTIVE_MIN_SIGNALS = 3
# Multiplier applied when reactive signals are below the threshold.
_ALPHA_SPARSE_REACTIVE_MULT = 0.5

# ── Fuzzy symbol channel (v3: replaces substring mention) ──
# The v3 mention channel uses dense embeddings stored on ci_symbols
# and ci_files (populated at index time by symbol_embeddings.py).
# At rank time it does a vectorized cosine sweep against all
# embedded symbols/files in the project — handles typos, synonyms,
# and descriptive queries uniformly.
#
# Below this cosine similarity a symbol match is too weak to surface.
# Empirically calibrated against a live 9232-symbol TypeScript
# project.  We tried lowering to 0.40 to rescue one descriptive
# query whose top real match was at 0.445, but doing so introduced
# a false positive on conversational noise queries that matched
# literal symbol names (e.g. "what's the weather today" matching a
# symbol literally named `today`).  The scores for genuine weak
# matches and noise matches are indistinguishable with a linear
# threshold — 0.45 is the point that gives the best hit rate
# overall, at the cost of occasionally missing a relevant match
# by a few thousandths.
_FUZZY_SYMBOL_MIN_SIM = 0.45
# Score scale: sim ∈ [0.45, 1.0] × 5.0 → [2.25, 5.0], putting fuzzy
# scores in the same range as the old substring mention scores so
# the `max()` merge treats them comparably.
_FUZZY_SYMBOL_SCALE = 5.0
# Symbol hits get a small bonus over the same-file hit so the symbol
# itself is preferred in the symbol-type output when both would rank.
_FUZZY_SYMBOL_BONUS = 0.5

# File-level fuzzy matching uses a slightly lower threshold because
# file embeddings are shorter (stem + top-N symbol names) and their
# cosine values naturally run lower.
_FUZZY_FILE_MIN_SIM = 0.4
_FUZZY_FILE_SCALE = 4.5

# ── Finding channel weights ──
# Cosine similarity is in [0, 1]; multiplying by this scale brings
# finding semantic scores into the same range as mention scores
# (0-5 region) so the `max()` merge compares apples to apples.
_FINDING_SEMANTIC_SCALE = 3.0
# Below this similarity, the finding is not semantically related
# enough to surface.  Calibrated against all-MiniLM-L6-v2 embeddings.
_FINDING_SEMANTIC_MIN_SIM = 0.5
# Score for literal topic-word matches: `BASE + ratio * BONUS`.
_FINDING_TOPIC_BASE = 3.0
_FINDING_TOPIC_BONUS = 1.5
# Score for tag matches: `BASE + hits * BONUS`.
_FINDING_TAG_BASE = 4.0
_FINDING_TAG_BONUS = 0.5

# v3 removed the _DOCSTRING_* constants along with canal 5
# (_compute_docstring_scores).  Docstring matching is now absorbed
# into canal 3 via the symbol embedding text, which includes
# `{kind} {name} — {docstring_first_line}`.  The embedding captures
# both the symbol identity and the documented intent in one shot.

# ── Co-occurrence boost ──
# Only boost co-occurrences of files whose base score is at least this.
# Below this threshold, the anchor file is too marginal to trust as a
# source of co-occurrence signal.  Lowered from 1.0 in v3 so the
# channel can fire on lower-confidence foundations from early-session
# rankings.
_COOC_ANCHOR_MIN_SCORE = 0.6
# Only look at the top N anchor files — DB query per anchor, so this
# caps the work regardless of how many files made the cut.
_COOC_MAX_ANCHORS = 5
# Minimum number of sessions a pair must co-occur in to count.
_COOC_MIN_SESSIONS = 2
# Score propagation factor: co-occurring file gets `anchor_score *
# COOC_PROPAGATION * min(co_sessions / COOC_SATURATE, 1.0)`.
_COOC_PROPAGATION = 0.4
# Session count at which co-occurrence confidence saturates.
_COOC_SATURATE = 5.0
# Minimum propagated score to actually add to the ranking.  Below
# this, the boost is too weak to be worth the prompt tokens.
_COOC_MIN_PROPAGATED = 0.3

# ── Import graph boost ──
# Only propagate scores from files at or above this threshold.  Lowered
# from 1.5 in v3 so the channel fires on new-project tasks where no
# other channel produces strong scores yet.
_IMPORT_ANCHOR_MIN_SCORE = 0.8
# Look at the top N anchor files (each fires one UNION query).
_IMPORT_MAX_ANCHORS = 3
# When A imports B, A (the importer, downstream) gets
# `B.score * IMPORT_IN_PROPAGATION`.  Higher than OUT because downstream
# consumers are the non-obvious surface the model wouldn't discover by
# just opening B — the model needs these to know what it might break.
_IMPORT_IN_PROPAGATION = 0.5
# When A imports B, B (the dependency, upstream) gets
# `A.score * IMPORT_OUT_PROPAGATION`.  Lower because upstream deps are
# already visible in A's import block the moment the model opens the
# file — less useful to surface them separately.
_IMPORT_OUT_PROPAGATION = 0.3

# ── Freshness boost ──
# Multiplicative boost applied to already-ranked files based on
# their filesystem mtime.  Boosts files currently being worked on.
#
# Linear decay from `FRESH_MAX_MULT` today to `1.0` at
# `FRESH_DECAY_DAYS` days ago.  This is multiplicative *on top of*
# an existing score, so the effect is proportional — a file that
# was already ranked 4.0 and got touched today becomes 5.2, a file
# ranked 0.3 becomes 0.39 (still below the confidence floor).
# This is intentional: freshness amplifies, it does not rescue.
_FRESH_MAX_MULT = 1.3
_FRESH_DECAY_DAYS = 100  # smaller = faster decay; 100 gives ~30d visible effect
_FRESH_SECONDS_PER_DAY = 86400


# ── Outlier detection constants ──
# MAD-based outlier detection uses ``median + K * MAD * MAD_SCALE``
# as the threshold.  Items above that threshold are flagged as
# outliers and shown alone (if there are few enough of them).

# MAD_SCALE = 1 / Φ⁻¹(0.75) ≈ 1.4826 — converts Median Absolute
# Deviation to an equivalent standard deviation assuming a normal
# distribution.  This constant is mathematically derived and should
# not be changed.
_MAD_NORMAL_CONSISTENCY = 1.4826

# Minimum number of items required to attempt outlier detection.
# With fewer items, MAD is too noisy to be reliable.
_OUTLIER_MIN_ITEMS = 3

# When MAD is degenerate (bottom half has identical scores), fall back
# to a simple ratio test: top score must exceed baseline median by at
# least this factor to be considered an outlier.  Purely empirical.
_OUTLIER_FALLBACK_RATIO = 1.8

# MAD values below this threshold count as "degenerate" (baseline too
# tight) and trigger the fallback ratio test.
_MAD_DEGENERATE_THRESHOLD = 0.05

# Minimum relative magnitude for an item to qualify as an outlier:
# the score must be at least this many times the baseline median.
# Prevents low-variance baselines from flagging tiny deltas as
# "statistically significant" outliers when they aren't meaningful.
# 1.5 means an outlier must be at least 50% above the noise median.
_OUTLIER_MIN_RATIO = 1.5


def _percentile_to_mad_multiplier(percentile: float | str) -> float:
    """Convert a user-friendly percentile to the MAD multiplier K.

    Accepts a number (95) or a percentage string ("95%").  Returns
    the corresponding K such that ``K × MAD × 1.4826`` equals the
    standard-deviation distance at that percentile of a normal
    distribution.

        90   →  1.28   (top 10%)
        95   →  1.645  (top 5%, default)
        99   →  2.326  (top 1%)
        99.7 →  2.748  (top 0.3%, classic 3-sigma)
    """
    from statistics import NormalDist

    if isinstance(percentile, str):
        s = percentile.strip().rstrip("%")
        try:
            p = float(s)
        except ValueError:
            p = 95.0
    else:
        p = float(percentile)

    # Clamp to a sensible range
    if p <= 0 or p >= 100:
        p = 95.0

    # Φ⁻¹(p/100) — inverse normal CDF at the requested percentile
    try:
        return NormalDist().inv_cdf(p / 100.0)
    except Exception:
        return 1.645  # fallback: 95th percentile


# ── Public API ───────────────────────────────────────────────────────

def rank(
    current_input: str,
    session_id: str,
    task_id: str,
    iteration: int,
    *,
    top_k_files: int | None = None,
    top_k_symbols: int | None = None,
    top_k_findings: int | None = None,
    cached_embedding: bytes | None = None,
    cached_simplified_embedding: bytes | None = None,
    project_id: int | None = None,
) -> ContextRankResult:
    """Compute ranked resources for prompt injection.

    Multi-channel scoring: merges 6 independent channels, applies
    4 post-processing boosts, then gates on confidence.
    """
    if top_k_files is None:
        top_k_files = settings.CONTEXT_RANK_TOP_K_FILES
    if top_k_symbols is None:
        top_k_symbols = settings.CONTEXT_RANK_TOP_K_SYMBOLS
    if top_k_findings is None:
        top_k_findings = settings.CONTEXT_RANK_TOP_K_FINDINGS

    if project_id is None:
        try:
            from infinidev.tools.base.context import get_current_project_id
            project_id = get_current_project_id() or 1
        except Exception:
            project_id = 1

    # Resolve workspace once from the canonical source (context) and
    # fall back to the process cwd only if it's not set.  Passing the
    # workspace explicitly to every channel avoids re-reading os.getcwd()
    # inside hot paths and removes a hidden dependency on process state.
    workspace = _resolve_workspace()

    # ── Shared embeddings ────────────────────────────────────────
    # Two task embeddings are needed:
    #   * raw (current_input as-is): used by canals 2 (predictive) and
    #     4 (findings), both of which compare against corpus embeddings
    #     that were also stored from raw text.
    #   * simplified (stop-words filtered): used by canal 3 (fuzzy
    #     symbol search) because its corpus — symbol names + docstring
    #     first lines — is already concentrated, so a raw conversational
    #     query dilutes the match. See _simplify_query docstring.
    # Both are normally precomputed by ContextRankHooks.start() in a
    # background thread and passed in via the cached_* kwargs.  The
    # sync fallback below only runs if the precompute hasn't finished
    # or the caller didn't populate the cache.
    query_embedding = cached_embedding
    if query_embedding is None:
        try:
            from infinidev.tools.base.embeddings import compute_embedding
            query_embedding = compute_embedding(current_input)
        except Exception:
            query_embedding = None

    simplified_embedding = cached_simplified_embedding
    if simplified_embedding is None:
        try:
            simplified_text = _simplify_query(current_input)
            if simplified_text and simplified_text != current_input:
                from infinidev.tools.base.embeddings import compute_embedding
                simplified_embedding = compute_embedding(simplified_text)
            else:
                simplified_embedding = query_embedding
        except Exception:
            simplified_embedding = query_embedding

    # ── Channel 1: Reactive (current session) ────────────────────
    reactive = _compute_reactive_scores(session_id, task_id, iteration)

    # ── Channel 2: Predictive / Historical (past sessions) ───────
    predictive = _compute_predictive_scores(
        current_input, session_id, cached_embedding=query_embedding,
        workspace=workspace,
    )

    # Blend reactive + predictive with adaptive alpha
    alpha = _compute_alpha(iteration, len(reactive))
    blended = _blend_reactive_predictive(reactive, predictive, alpha)

    # ── Channel 3: Fuzzy symbol search (embedding cosine) ────────
    mentions = _compute_mention_scores(
        current_input, project_id, workspace,
        cached_simplified_embedding=simplified_embedding,
    )

    # ── Channel 4: Semantic findings (embedding similarity) ──────
    findings = _compute_finding_scores(query_embedding, current_input, project_id)

    # Canal 5 (docstring BM25) was removed in v3 — the semantic
    # match via fuzzy symbol embeddings now covers docstring
    # matching, since the symbol embedding text includes the
    # docstring's first line.

    # ── Merge all independent channels (max per target) ──────────
    combined = _merge_channels(blended, mentions, findings)

    # ── Post-processing boosts ───────────────────────────────────
    combined = _apply_cooccurrence_boost(combined)
    combined = _apply_import_boost(combined, project_id, workspace)
    combined = _apply_freshness_boost(combined, workspace)

    # ── Confidence gate ──────────────────────────────────────────
    # Check the max score across ALL raw channels (pre-merge), not
    # just the blended result — alpha blending dilutes reactive scores
    # when there's no predictive data, but the raw signal may be strong.
    min_confidence = getattr(settings, "CONTEXT_RANK_MIN_CONFIDENCE", 0.5)
    all_raw = [
        s for ch in (reactive, predictive, mentions, findings)
        for s, _, _ in ch.values()
    ]
    if not combined or (not all_raw or max(all_raw) < min_confidence):
        return ContextRankResult()

    return ContextRankResult(
        files=_filter_outliers(_top_k(combined, "file", top_k_files)),
        symbols=_filter_outliers(_top_k(combined, "symbol", top_k_symbols)),
        findings=_filter_outliers(_top_k(combined, "finding", top_k_findings)),
    )


# ── Channel 1: Reactive scoring ─────────────────────────────────────

# Per-target access pattern multipliers.  Replaces the old log(1+count)
# frequency boost which rewarded confusion — the more the model re-read
# a file, the higher the score, which is backwards.  The new pattern
# classification is productivity-aware:
#
#   read + edit  → 2.0  (model read AND edited → the file was useful)
#   edit only    → 1.5  (edited without a prior read → unusual but
#                        high-signal, the target was definitely worked on)
#   3+ reads,
#     no edit    → 0.7  (confusion: re-read N times without editing →
#                        mild penalty to push this below productive reads)
#   otherwise    → 1.0  (1-2 reads, exploratory — neutral)
#
# Write events are identified by interaction weight >= 2.0, matching
# the convention in logger._TOOL_EVENT_MAP where file_write / symbol_write
# events have weight >= 2.0.
_REACTIVE_PATTERN_READ_AND_EDIT = 2.0
_REACTIVE_PATTERN_EDIT_ONLY = 1.5
_REACTIVE_PATTERN_CONFUSION = 0.7
_REACTIVE_PATTERN_NEUTRAL = 1.0


def _compute_reactive_scores(
    session_id: str, task_id: str, current_iteration: int,
) -> dict[str, tuple[float, str, str]]:
    """Score nodes based on tool calls in the current session.

    Drains the async interaction writer first so the query sees all
    pending rows.

    Per-target score:
        score = Σ(weight × exp(-λ × Δi)) × pattern_mult(interactions)

    The pattern multiplier rewards productive access patterns (read
    followed by edit, or edit followed by verification read) and
    penalises repeated reads without any edit.  Order between read
    and edit is intentionally ignored — both read→edit and edit→read
    count as productive.
    """
    from infinidev.engine.context_rank.logger import flush as _cr_flush
    _cr_flush()
    decay_lambda = settings.CONTEXT_RANK_REACTIVE_DECAY
    many_reads_threshold = settings.CONTEXT_RANK_REACTIVE_MANY_READS
    try:
        def _query(conn):
            return conn.execute(
                "SELECT target, target_type, iteration, weight "
                "FROM cr_interactions "
                "WHERE session_id = ? AND task_id = ?",
                (session_id, task_id),
            ).fetchall()
        rows = execute_with_retry(_query)
    except Exception:
        logger.debug("Reactive scoring query failed", exc_info=True)
        return {}

    accum: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = row["target"]
        if target not in accum:
            accum[target] = {
                "target_type": row["target_type"],
                "weighted_sum": 0.0,
                "read_count": 0,
                "write_count": 0,
            }
        delta = current_iteration - row["iteration"]
        decay = math.exp(-decay_lambda * max(delta, 0))
        accum[target]["weighted_sum"] += row["weight"] * decay
        if row["weight"] >= 2.0:
            accum[target]["write_count"] += 1
        else:
            accum[target]["read_count"] += 1

    result: dict[str, tuple[float, str, str]] = {}
    for target, info in accum.items():
        read_count = info["read_count"]
        write_count = info["write_count"]
        has_write = write_count > 0
        has_read = read_count > 0

        if has_write and has_read:
            pattern_mult = _REACTIVE_PATTERN_READ_AND_EDIT
            reason = "read + edited this session"
        elif has_write:
            pattern_mult = _REACTIVE_PATTERN_EDIT_ONLY
            reason = "edited this session"
        elif read_count >= many_reads_threshold:
            pattern_mult = _REACTIVE_PATTERN_CONFUSION
            reason = f"re-read {read_count}× without editing"
        else:
            pattern_mult = _REACTIVE_PATTERN_NEUTRAL
            reason = "read this session"

        score = info["weighted_sum"] * pattern_mult
        result[target] = (score, info["target_type"], reason)
    return result


# ── Channel 2: Predictive scoring (historical escalera) ─────────────

def _compute_predictive_scores(
    current_input: str, exclude_session: str,
    *, cached_embedding: bytes | None = None,
    workspace: str | None = None,
) -> dict[str, tuple[float, str, str]]:
    """Score nodes via embedding similarity to historical contexts.

    Phase 2 v3 rewrite applies four interrelated fixes:

    (2a) **Age-filtered SQL fetch.**  The old query used
         ``ORDER BY created_at DESC LIMIT 500`` which is a temporal
         *sample*, not a relevance sample — 500 recent contexts in a
         busy project can crowd out all older-but-relevant ones.
         New query adds ``WHERE created_at > ?`` (180-day cutoff)
         and raises the LIMIT to 2000.

    (2b) **Day-based session decay.**  Old: ``session_decay ** order``
         where ``order`` was the rank of the session in the fetch
         result.  That's position, not age.  New:
         ``session_decay ** (days_ago / 7)`` — a weekly decay keyed
         to real elapsed time, independent of fetch ordering.

    (2c) **sim² contribution, no level weights.**  Old:
         ``sim × level_weight × s_decay`` with level_weight in
         {1.0, 1.5, 2.0} depending on context_type.  New:
         ``sim² × s_decay``.  Squaring naturally rewards confident
         matches (0.8² = 0.64 vs 0.5² = 0.25) and removes the
         arbitrary escalera weight ordering.  See the comment on
         ``_LEVEL_WEIGHTS`` removal above.

    (2d) **Productivity-aware interaction propagation.**  Old: every
         interaction linked to a matched context propagated with
         uniform multiplier.  New: LEFT JOIN with ``cr_session_scores``
         to fetch the per-target ``productivity`` multiplier
         (populated by snapshot_session_scores at task end), so
         historical sessions where the model actually edited the
         target contribute 1.5× and exploratory-only sessions
         contribute 0.6×.  Also filters ``was_error = 0`` so failed
         tool calls don't poison the signal.

    The embedding is expected to be provided via ``cached_embedding``
    — ``rank()`` computes it once and shares it across channels.
    """
    from time import time as _now
    from infinidev.tools.base.embeddings import embedding_from_blob
    from infinidev.tools.base.dedup import _cosine_similarity

    min_sim = settings.CONTEXT_RANK_MIN_SIMILARITY
    session_decay = settings.CONTEXT_RANK_SESSION_DECAY
    max_age_days = settings.CONTEXT_RANK_CONTEXT_MAX_AGE_DAYS
    fetch_limit = settings.CONTEXT_RANK_CONTEXT_FETCH_LIMIT

    if cached_embedding is None:
        return {}
    query_vec = np.frombuffer(cached_embedding, dtype=np.float32)

    now = _now()
    age_cutoff = now - (max_age_days * 86400)

    try:
        def _fetch_contexts(conn):
            # (2a) Widen the fetch and filter by age in SQL.  The
            # index idx_cr_contexts_created_at makes the range scan
            # cheap even on large tables.
            return conn.execute(
                "SELECT id, session_id, context_type, embedding, created_at "
                "FROM cr_contexts "
                "WHERE embedding IS NOT NULL "
                "  AND session_id != ? "
                "  AND created_at > ? "
                "ORDER BY created_at DESC LIMIT ?",
                (exclude_session, age_cutoff, fetch_limit),
            ).fetchall()
        ctx_rows = execute_with_retry(_fetch_contexts)
    except Exception:
        return {}

    if not ctx_rows:
        return {}

    matched_contexts: list[tuple[int, float, float]] = []
    for row in ctx_rows:
        ctx_vec = embedding_from_blob(row["embedding"])
        sim = float(_cosine_similarity(query_vec, ctx_vec))
        if sim < min_sim:
            continue
        # (2b) Session decay by actual days, not result position.
        days_ago = max(0.0, (now - row["created_at"]) / 86400)
        s_decay = session_decay ** (days_ago / 7.0)
        # (2c) sim² contribution — high-confidence matches dominate
        # without needing arbitrary level weights.
        contribution = (sim * sim) * s_decay
        matched_contexts.append((row["id"], contribution, sim))

    if not matched_contexts:
        return {}

    context_ids = [m[0] for m in matched_contexts]
    contrib_by_id = {m[0]: (m[1], m[2]) for m in matched_contexts}

    try:
        def _fetch_interactions(conn):
            placeholders = ",".join("?" * len(context_ids))
            # (2d) JOIN on cr_session_scores for per-target productivity.
            # COALESCE default 1.0 handles targets not yet snapshotted
            # (the current session's interactions don't have entries yet).
            # Also excludes errored interactions via was_error = 0.
            return conn.execute(
                f"SELECT i.context_id, i.target, i.target_type, i.weight, "
                f"       COALESCE(s.productivity, 1.0) AS productivity "
                f"FROM cr_interactions i "
                f"LEFT JOIN cr_session_scores s "
                f"  ON s.session_id  = i.session_id "
                f" AND s.target      = i.target "
                f" AND s.target_type = i.target_type "
                f"WHERE i.context_id IN ({placeholders}) "
                f"  AND i.was_error = 0",
                context_ids,
            ).fetchall()
        int_rows = execute_with_retry(_fetch_interactions)
    except Exception:
        return {}

    accum: dict[str, dict[str, Any]] = {}
    for row in int_rows:
        ctx_id = row["context_id"]
        contribution, sim = contrib_by_id[ctx_id]
        target = _normalize_path(row["target"], workspace) if row["target_type"] == "file" else row["target"]
        if target not in accum:
            accum[target] = {"target_type": row["target_type"], "score": 0.0, "best_sim": 0.0}
        # Edit vs Read asymmetry (kept from v2): writes get 2x in the
        # predictive channel because editing a file in a past task is
        # stronger evidence of relevance than just reading it.
        edit_mult = 2.0 if row["weight"] >= 2.0 else 1.0
        # Productivity from the snapshotted session (1.5 if the target
        # was edited in that session, 0.6 if it was repeatedly read
        # without editing, 1.0 for neutral single reads).
        productivity = row["productivity"]
        accum[target]["score"] += contribution * row["weight"] * edit_mult * productivity
        accum[target]["best_sim"] = max(accum[target]["best_sim"], sim)

    result: dict[str, tuple[float, str, str]] = {}
    for target, info in accum.items():
        result[target] = (
            info["score"], info["target_type"],
            f"predicted (similarity={info['best_sim']:.2f} to past contexts)",
        )
    return result


# ── Channel 3: Fuzzy symbol search (v3) ────────────────────────────

# Common English words kept for canal 4 (findings) topic-word match.
# Canal 3 uses this as a base for _QUERY_STOP_WORDS below.
_COMMON_WORDS: frozenset[str] = frozenset({
    "type", "file", "test", "tests", "error", "value", "data", "name",
    "time", "date", "size", "text", "info", "item", "list", "page",
    "user", "role", "form", "code", "path", "line", "mode", "next",
    "prev", "root", "done", "open", "save", "load", "send", "read",
    "init", "self", "this", "that", "from", "into", "main", "core",
    "util", "args", "kwargs", "true", "false", "none", "null", "undefined",
    "call", "exit", "work", "task", "node", "edge", "tree", "log",
    "new", "get", "set", "put", "del", "run", "map", "key", "val",
})


# ── Query simplification via Zipf frequency ─────────────────────
# Queries passed to canal 3 go through _simplify_query which drops
# conversational noise words to concentrate the distinctive tokens
# in the resulting embedding.
#
# The core insight: ``all-MiniLM-L6-v2`` averages the vectors of all
# input tokens.  When 4 of 5 tokens are generic ("show", "me", "the",
# "class"), the single distinctive token ("ErorHandler") contributes
# only 1/5 of the resulting vector, and the query's cosine against
# its intended target collapses into the same ~0.46 band as every
# other generic query.  Dropping the stop words rebalances the
# average so the distinctive token dominates.
#
# v3 approach: use ``wordfreq.zipf_frequency`` for the stop word
# decision instead of a hardcoded list.  Reasons:
#   - Multi-language by construction (wordfreq covers 40+ languages
#     from the same API; a future commit can detect the query's
#     language and filter per-language stop words).
#   - Robust to contractions, plurals, tenses — wordfreq tokenizes
#     and looks up each form properly instead of mishandling edges
#     like "what's" or "don't".
#   - No list maintenance: as the corpus is updated, the definition
#     of "common" updates automatically, without ContextRank
#     needing to ship a new stop word list.
#
# Zipf frequency scale: 0 (unknown word) to ~8 (most frequent).
# Anything ≥ 5.0 is in the top ~10k most common words of the
# detected language (the, is, how, today, show, ...), which is
# our stop-word cutoff.  Words below 5.0 are either domain content
# (firewall=3.24, validate=3.40, machine=4.90) or unknown
# identifiers (zipf 0).
#
# Only applied to canal 3 — canal 2 (predictive) keeps the raw
# embedding because its historical contexts were also stored raw,
# and filtering would introduce an asymmetry in the cosine space.
_QUERY_STOP_ZIPF = 5.0

# Token pattern used by _simplify_query.  Matches alphanumeric
# identifiers (including CamelCase and snake_case) but splits on
# apostrophes, so "what's" becomes ["what", "s"] — both of which
# get filtered as high-frequency words.  Does NOT match dots, so
# "Agent.handleEvent" becomes ["Agent", "handleEvent"] which keeps
# both halves distinct for the Zipf lookup.  Also allows non-ASCII
# unicode letters via \p{L}-style fallback so that Spanish
# ("autenticación"), French ("autenticación"), German
# ("Authentifizierung"), etc. tokenize correctly.
_QUERY_TOKEN_RE = re.compile(r"[^\W\d_][\w]*", re.UNICODE)

# Minimum number of tokens that must survive filtering before we
# use the simplified query.  Below this, the filter has removed
# too much and we fall back to the raw query (a diluted embedding
# is better than a 1-word embedding that happens to match a
# literal symbol name — see the Q6 "what's the weather today"
# regression where "weather" alone survived and the ranker
# matched a symbol literally named ``today``).
_QUERY_MIN_TOKENS_AFTER = 2

# Supported languages for query simplification.  wordfreq covers
# 40+ languages and langdetect covers 55+, so adding new ones is
# just appending the ISO code here.  These six cover the overwhelming
# majority of developer queries in practice (English plus major
# European languages).  If langdetect returns a language not in this
# set (e.g. Russian, Arabic, Chinese — all valid codes but beyond
# what we've calibrated the 5.0 Zipf threshold on), we fall back to
# English.  The fallback is safe: English Zipf frequencies for
# unknown-language words are 0, so non-English tokens all pass
# through untouched — equivalent to no simplification for them.
_QUERY_SUPPORTED_LANGS: frozenset[str] = frozenset({
    "en",  # English
    "es",  # Spanish
    "pt",  # Portuguese
    "fr",  # French
    "de",  # German
    "it",  # Italian
})

# Cached at module level to avoid the 100ms langdetect init per call.
_LANGDETECT_INITED = False


def _init_langdetect() -> None:
    """Seed langdetect's random state for deterministic detection.

    langdetect uses a non-deterministic algorithm by default (short
    ambiguous inputs can classify differently across calls).  Seeding
    the DetectorFactory makes every invocation reproducible, which
    matters for tests and for consistent ranker behaviour.
    """
    global _LANGDETECT_INITED
    if _LANGDETECT_INITED:
        return
    try:
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
        _LANGDETECT_INITED = True
    except Exception:
        pass


def _detect_query_language(text: str) -> str:
    """Return the ISO code for the query's language, falling back to ``en``.

    Langdetect needs ≥ 3 words to classify reliably.  For shorter
    queries we skip detection entirely and use English — the cost
    of mis-detecting a 2-word query is higher than the benefit.
    For longer queries, if detection fails or returns a language we
    don't support, we also fall back to English.
    """
    if not text or len(text.split()) < 3:
        return "en"
    _init_langdetect()
    try:
        from langdetect import detect
        lang = detect(text)
        if lang in _QUERY_SUPPORTED_LANGS:
            return lang
    except Exception:
        pass
    return "en"


def _simplify_query(text: str) -> str:
    """Drop high-frequency noise words from a user query.

    Uses ``wordfreq.zipf_frequency`` to decide which words are
    frequent enough to be conversational noise vs rare enough to
    carry signal.  The threshold is ``_QUERY_STOP_ZIPF = 5.0``,
    which roughly corresponds to "top 10000 most common words" in
    the detected language — things like ``the``, ``is``, ``how``
    in English; ``el``, ``cómo``, ``es`` in Spanish; ``le``,
    ``comment``, ``est`` in French.

    Language is detected via ``langdetect`` for queries of ≥ 3
    words; shorter queries default to English.  If the detected
    language isn't in ``_QUERY_SUPPORTED_LANGS``, we fall back to
    English — the Zipf lookup will return 0 for most of the query's
    tokens and effectively pass the query through unchanged, which
    is safer than aggressively filtering with the wrong lexicon.

    Unknown words (zipf 0, e.g. ``ErorHandler``, ``JWTValidator``,
    or any CamelCase identifier not in the corpus) always pass
    through — they're almost certainly code-derived tokens that
    we specifically want to preserve, regardless of language.

    Case is preserved on retained tokens so CamelCase identifiers
    keep their subword structure for the embedding model.  If the
    filter would leave fewer than ``_QUERY_MIN_TOKENS_AFTER``
    tokens, returns the original text unchanged — a diluted
    embedding is better than one built from a single token that
    accidentally matches a literal symbol name.
    """
    if not text:
        return text
    try:
        from wordfreq import zipf_frequency
    except Exception:
        # wordfreq is an optional runtime dependency — if not
        # installed, skip simplification and use the raw query.
        logger.debug("wordfreq not available; skipping query simplification")
        return text

    lang = _detect_query_language(text)

    tokens = _QUERY_TOKEN_RE.findall(text)
    kept: list[str] = []
    for tok in tokens:
        z = zipf_frequency(tok.lower(), lang)
        # Unknown words (zipf == 0) → keep (likely identifier).
        # Rare words (zipf < threshold) → keep (content word).
        # Frequent words (zipf >= threshold) → drop (stop word).
        if z == 0.0 or z < _QUERY_STOP_ZIPF:
            kept.append(tok)

    if len(kept) < _QUERY_MIN_TOKENS_AFTER:
        return text
    return " ".join(kept)


def _compute_mention_scores(
    current_input: str, project_id: int, workspace: str | None = None,
    *, cached_simplified_embedding: bytes | None = None,
) -> dict[str, tuple[float, str, str]]:
    """Fuzzy semantic search over per-symbol and per-file embeddings.

    v3 replaces the old substring-based mention detection entirely.
    At index time, ``symbol_embeddings.embed_file_symbols`` populates
    ``ci_symbols.embedding`` and ``ci_files.embedding`` with 384-dim
    float32 vectors computed from text like
    ``{kind} {name} — {docstring_first_line}``.  At rank time this
    function does a vectorized cosine sweep and keeps matches above
    the empirical thresholds ``_FUZZY_SYMBOL_MIN_SIM=0.45`` and
    ``_FUZZY_FILE_MIN_SIM=0.4``.

    Benefits over v2 substring matching:

    - **Typo tolerance.** "AuthServise" (typo) still matches
      AuthService because the embedding model sees subword tokens.
    - **Synonym tolerance.** "authentication service" matches
      AuthenticationHandler through shared semantic space.
    - **No stop-word lists.** ``_COMMON_WORDS`` and ``_STEM_SKIP``
      are gone — the similarity threshold does the filtering.
    - **No kind whitelist.** Variables and constants
      (``DEFAULT_TIMEOUT``, ``ROUTER_PREFIX``) participate too,
      restricted only by what the symbol_embeddings module
      considered embeddable (see ``_EMBEDDABLE_KINDS`` there).

    **Query simplification (v3.1):** The channel uses a simplified
    task embedding (stop-words filtered) to prevent conversational
    noise from diluting the averaged query vector. The simplification
    + embedding happens in ``ContextRankHooks.start`` (background
    thread) so this function never blocks on a second embed call.

    Performance: for 10k embedded symbols, one np.stack + one
    matmul is ~5ms; for 1k embedded files, ~1ms.  No embedding call
    on the critical path.
    """
    if not current_input:
        return {}

    if cached_simplified_embedding is None:
        return {}

    query_vec = np.frombuffer(cached_simplified_embedding, dtype=np.float32)
    # Normalize the query vector once — the stored symbol vectors
    # get normalized row-wise in the matmul.
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return {}
    q_unit = query_vec / q_norm

    result: dict[str, tuple[float, str, str]] = {}

    # ── Symbols ────────────────────────────────────────────────
    try:
        def _fetch_symbols(conn):
            # Filter out anonymous symbols (name IS NULL or empty).
            # Some TypeScript/JS parsers record anonymous defaults
            # and default-exported arrow functions with empty names;
            # surfacing them to the model produces "contains ''"
            # reasons that aren't actionable — the model can't
            # read_symbol or edit_symbol something without a name.
            return conn.execute(
                "SELECT name, qualified_name, file_path, kind, embedding "
                "FROM ci_symbols "
                "WHERE project_id = ? "
                "  AND embedding IS NOT NULL "
                "  AND name IS NOT NULL "
                "  AND name != ''",
                (project_id,),
            ).fetchall()
        sym_rows = execute_with_retry(_fetch_symbols)
    except Exception:
        logger.debug("fuzzy symbol fetch failed", exc_info=True)
        sym_rows = []

    if sym_rows:
        try:
            mat = np.stack([
                np.frombuffer(r["embedding"], dtype=np.float32) for r in sym_rows
            ])
            # Row-normalize the matrix.  Adding a small epsilon so
            # zero-vector rows (shouldn't exist but defensive) don't
            # NaN the division.
            row_norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
            m_unit = mat / row_norms
            sims = m_unit @ q_unit  # shape (N,)

            for row, sim in zip(sym_rows, sims):
                sim_val = float(sim)
                if sim_val < _FUZZY_SYMBOL_MIN_SIM:
                    continue
                base_score = sim_val * _FUZZY_SYMBOL_SCALE
                name = row["name"]

                # File gets the max symbol-level score
                fp = _normalize_path(row["file_path"], workspace)
                if fp:
                    existing = result.get(fp, (0.0, "", ""))
                    if base_score > existing[0]:
                        result[fp] = (
                            base_score, "file",
                            f"contains '{name}' (fuzzy sim={sim_val:.2f})",
                        )

                # Symbol entry gets a small bonus so the symbol itself
                # outranks its file in the symbol-type output.
                sym_key = row["qualified_name"] or name
                if sym_key:
                    sym_score = base_score + _FUZZY_SYMBOL_BONUS
                    existing = result.get(sym_key, (0.0, "", ""))
                    if sym_score > existing[0]:
                        result[sym_key] = (
                            sym_score, "symbol",
                            f"fuzzy semantic match (sim={sim_val:.2f})",
                        )
        except Exception:
            logger.debug("fuzzy symbol cosine failed", exc_info=True)

    # ── Files ──────────────────────────────────────────────────
    try:
        def _fetch_files(conn):
            return conn.execute(
                "SELECT file_path, embedding FROM ci_files "
                "WHERE project_id = ? AND embedding IS NOT NULL",
                (project_id,),
            ).fetchall()
        file_rows = execute_with_retry(_fetch_files)
    except Exception:
        logger.debug("fuzzy file fetch failed", exc_info=True)
        file_rows = []

    if file_rows:
        try:
            mat = np.stack([
                np.frombuffer(r["embedding"], dtype=np.float32) for r in file_rows
            ])
            row_norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
            m_unit = mat / row_norms
            sims = m_unit @ q_unit

            for row, sim in zip(file_rows, sims):
                sim_val = float(sim)
                if sim_val < _FUZZY_FILE_MIN_SIM:
                    continue
                fp = _normalize_path(row["file_path"], workspace)
                if not fp:
                    continue
                score = sim_val * _FUZZY_FILE_SCALE
                existing = result.get(fp, (0.0, "", ""))
                if score > existing[0]:
                    result[fp] = (
                        score, "file",
                        f"fuzzy file match (sim={sim_val:.2f})",
                    )
        except Exception:
            logger.debug("fuzzy file cosine failed", exc_info=True)

    return result


# ── Channel 4: Semantic findings match ──────────────────────────────

def _compute_finding_scores(
    cached_embedding: bytes | None,
    current_input: str,
    project_id: int,
) -> dict[str, tuple[float, str, str]]:
    """Score findings by two independent signals:

    1. **Semantic** — cosine similarity of query embedding vs finding
       embedding.  Catches synonyms and paraphrases.
    2. **Literal** — topic words or tags mentioned in the input.
       Stronger signal for exact matches.

    The max of the two signals is kept per finding.

    The embedding is expected to be provided via ``cached_embedding``
    — ``rank()`` computes it once and shares it across channels.
    """
    import json
    from infinidev.tools.base.embeddings import embedding_from_blob
    from infinidev.tools.base.dedup import _cosine_similarity

    query_vec = (
        np.frombuffer(cached_embedding, dtype=np.float32)
        if cached_embedding else None
    )

    input_lower = current_input.lower()
    padded = " " + input_lower + " "

    try:
        def _fetch(conn):
            # ORDER BY confidence DESC so the LIMIT 500 keeps the most
            # trusted findings when a project accumulates more.
            # Dedup is keyed by id (not topic) so two findings with the
            # same topic don't overwrite each other in the result dict.
            return conn.execute(
                "SELECT id, topic, content, embedding, tags_json "
                "FROM findings "
                "WHERE project_id = ? AND status = 'active' "
                "ORDER BY confidence DESC, updated_at DESC "
                "LIMIT 500",
                (project_id,),
            ).fetchall()
        rows = execute_with_retry(_fetch)
    except Exception:
        return {}

    result: dict[str, tuple[float, str, str]] = {}
    for row in rows:
        topic = row["topic"] or ""
        if not topic:
            continue
        finding_id = row["id"]

        scores: list[tuple[float, str]] = []

        # ── Signal 1: semantic similarity ────────────────────────
        if query_vec is not None and row["embedding"]:
            try:
                f_vec = embedding_from_blob(row["embedding"])
                sim = float(_cosine_similarity(query_vec, f_vec))
                if sim >= _FINDING_SEMANTIC_MIN_SIM:
                    scores.append((
                        sim * _FINDING_SEMANTIC_SCALE,
                        f"semantic match (sim={sim:.2f})",
                    ))
            except Exception:
                pass

        # ── Signal 2: topic word matching ────────────────────────
        topic_words = [
            w for w in re.split(r'\W+', topic.lower())
            if len(w) >= _MIN_IDENT_LEN and w not in _COMMON_WORDS
        ]
        if topic_words:
            matched = sum(1 for w in topic_words if f" {w} " in padded or f" {w}" in padded)
            if matched >= max(2, len(topic_words) // 2):
                ratio = matched / len(topic_words)
                scores.append((
                    _FINDING_TOPIC_BASE + ratio * _FINDING_TOPIC_BONUS,
                    f"{matched}/{len(topic_words)} topic words match",
                ))

        # ── Signal 3: tag matching ───────────────────────────────
        # Tags can be multi-word ("authentication flow").  A previous
        # implementation only matched the exact bi-gram in the input,
        # which missed the common case where the user writes just one
        # of the words ("authentication").  Fix: split each tag on
        # whitespace, consider a tag a hit when ALL of its non-common
        # words (≥ _MIN_IDENT_LEN) appear as whole words in the input.
        try:
            tags = json.loads(row["tags_json"] or "[]")
            tag_hits: list[str] = []
            for t in tags:
                if not isinstance(t, str):
                    continue
                tag_words = [
                    w for w in re.split(r'\W+', t.lower())
                    if len(w) >= _MIN_IDENT_LEN and w not in _COMMON_WORDS
                ]
                if not tag_words:
                    continue
                if all(f" {w} " in padded for w in tag_words):
                    tag_hits.append(t)
            if tag_hits:
                scores.append((
                    _FINDING_TAG_BASE + len(tag_hits) * _FINDING_TAG_BONUS,
                    f"tags match: {', '.join(tag_hits[:3])}",
                ))
        except Exception:
            pass

        # Keep the max signal.  Dict key is the finding id — prevents
        # two findings with the same topic from overwriting each other.
        if scores:
            best_score, best_reason = max(scores, key=lambda x: x[0])
            preview = (row["content"] or "")[:60]
            result[f"finding:{finding_id}"] = (
                best_score, "finding",
                f"[{topic}] {best_reason}: {preview}",
            )

    return result


# Canal 5 (docstring BM25) was removed in v3.  The fuzzy symbol
# channel now handles docstring matching via dense embeddings — the
# symbol embedding text includes the docstring's first line, so
# "authentication service" can match a function whose docstring
# says "Authenticates users via JWT" even when the function name
# itself is something unrelated like ``_verify``.  See
# ``symbol_embeddings._build_symbol_text`` for the embedding format.


# ── Post-processing: Co-occurrence boost ────────────────────────────

def _apply_cooccurrence_boost(
    scores: dict[str, tuple[float, str, str]],
) -> dict[str, tuple[float, str, str]]:
    """Boost files frequently accessed alongside high-scoring files.

    Phase 2 v3 adds a time cutoff (``CONTEXT_RANK_COOC_MAX_AGE_DAYS``,
    default 90) so co-occurrence pairs from long-refactored-away
    modules don't keep boosting files forever.  The cutoff is applied
    to the co-occurring side's ``created_at``.
    """
    from time import time as _now

    # Get top file targets to find co-occurring files for
    top_files = [
        (t, s) for t, (s, tt, _) in scores.items()
        if tt == "file" and s >= _COOC_ANCHOR_MIN_SCORE
    ]
    if not top_files:
        return scores

    top_files.sort(key=lambda x: x[1], reverse=True)
    top_files = top_files[:_COOC_MAX_ANCHORS]

    max_age_days = getattr(settings, "CONTEXT_RANK_COOC_MAX_AGE_DAYS", 90)
    age_cutoff = _now() - (max_age_days * 86400)

    for target, target_score in top_files:
        try:
            def _query(conn, t=target, ac=age_cutoff):
                return conn.execute(
                    "SELECT b.target, COUNT(DISTINCT b.session_id) as co_sessions "
                    "FROM cr_session_scores a "
                    "JOIN cr_session_scores b "
                    "  ON a.session_id = b.session_id "
                    " AND a.target != b.target "
                    "WHERE a.target = ? "
                    "  AND a.target_type = 'file' "
                    "  AND b.target_type = 'file' "
                    "  AND b.created_at > ? "
                    "GROUP BY b.target "
                    "HAVING co_sessions >= ?",
                    (t, ac, _COOC_MIN_SESSIONS),
                ).fetchall()
            rows = execute_with_retry(_query)
        except Exception:
            continue

        for row in rows:
            co_target = row["target"]
            if co_target in scores:
                continue  # Don't boost already-scored files
            confidence = min(row["co_sessions"] / _COOC_SATURATE, 1.0)
            co_score = target_score * _COOC_PROPAGATION * confidence
            if co_score > _COOC_MIN_PROPAGATED:
                scores[co_target] = (
                    co_score, "file", f"co-occurs with {os.path.basename(target)}",
                )
    return scores


# ── Post-processing: Import graph propagation ───────────────────────

def _apply_import_boost(
    scores: dict[str, tuple[float, str, str]],
    project_id: int,
    workspace: str | None = None,
) -> dict[str, tuple[float, str, str]]:
    """1-hop propagation through the import graph.

    Paths from ``ci_imports.file_path`` / ``resolved_file`` may be
    absolute depending on the indexer's resolution — this function
    normalises them via ``_normalize_path`` before merging into
    ``scores`` so the same file cannot appear twice under two keys
    (one from a semantic channel at relative path, one from here at
    absolute path).
    """
    top_files = [
        (t, s) for t, (s, tt, _) in scores.items()
        if tt == "file" and s >= _IMPORT_ANCHOR_MIN_SCORE
    ]
    if not top_files:
        return scores

    top_files.sort(key=lambda x: x[1], reverse=True)
    top_files = top_files[:_IMPORT_MAX_ANCHORS]

    for target, target_score in top_files:
        try:
            def _query(conn, t=target, pid=project_id):
                # Single UNION query fetches both directions:
                #   direction='in'  → files that import target (downstream)
                #   direction='out' → files that target imports (upstream)
                return conn.execute(
                    "SELECT file_path AS fp, 'in' AS direction FROM ci_imports "
                    "  WHERE project_id = ? AND resolved_file = ? "
                    "UNION "
                    "SELECT resolved_file AS fp, 'out' AS direction FROM ci_imports "
                    "  WHERE project_id = ? AND file_path = ? AND resolved_file != ''",
                    (pid, t, pid, t),
                ).fetchall()
            rows = execute_with_retry(_query)
        except Exception:
            continue

        basename = os.path.basename(target)
        for row in rows:
            fp = row["fp"]
            if not fp:
                continue
            fp_n = _normalize_path(fp, workspace)
            if fp_n in scores:
                continue
            if row["direction"] == "in":
                scores[fp_n] = (
                    target_score * _IMPORT_IN_PROPAGATION,
                    "file", f"imports {basename}",
                )
            else:
                scores[fp_n] = (
                    target_score * _IMPORT_OUT_PROPAGATION,
                    "file", f"imported by {basename}",
                )
    return scores


# ── Post-processing: Git freshness ──────────────────────────────────

def _apply_freshness_boost(
    scores: dict[str, tuple[float, str, str]],
    workspace: str | None = None,
) -> dict[str, tuple[float, str, str]]:
    """Boost files modified recently on disk.

    Uses ``os.stat().st_mtime`` instead of ``git log`` for speed —
    subprocess spawning would cost 5-20ms per file.  Filesystem mtime
    is a close enough proxy for "recently worked on": it tracks when
    the file was last touched, which is what we care about for
    relevance ranking.

    The boost is **multiplicative on top of an already-computed score**,
    not additive — freshness *amplifies* existing rankings instead of
    rescuing marginal files from the noise floor.  A file that another
    channel already ranked highly and is being worked on now climbs
    faster; a file nobody ranked doesn't magically appear just because
    it was touched recently.
    """
    import time as _time
    now = _time.time()
    if workspace is None:
        workspace = _resolve_workspace()

    for target in list(scores):
        score, tt, reason = scores[target]
        if tt != "file" or "." not in os.path.basename(target):
            continue
        # Resolve to absolute path
        abs_path = target if os.path.isabs(target) else os.path.join(workspace, target)
        try:
            mtime = os.stat(abs_path).st_mtime
        except OSError:
            continue
        days_ago = (now - mtime) / _FRESH_SECONDS_PER_DAY
        # Linear decay from _FRESH_MAX_MULT today to 1.0 at ~30 days
        freshness = max(1.0, _FRESH_MAX_MULT - (days_ago / _FRESH_DECAY_DAYS))
        if freshness > 1.0:
            scores[target] = (score * freshness, tt, reason)
    return scores


# ── Post-processing: Directory expansion ────────────────────────────

# ── Path normalization ───────────────────────────────────────────────

def _resolve_workspace() -> str:
    """Return the canonical workspace directory.

    Prefers ``get_current_workspace_path()`` (the value infinidev's
    tool context was initialised with) and falls back to ``os.getcwd()``
    only when the context has no workspace set.  Callers should pass
    the result explicitly to channels instead of calling os.getcwd()
    themselves, so that a mid-task cwd change (``os.chdir`` from a
    tool, test fixture, etc.) cannot desync path normalization.
    """
    try:
        from infinidev.tools.base.context import get_current_workspace_path
        ws = get_current_workspace_path()
        if ws:
            return ws
    except Exception:
        pass
    return os.getcwd()


def _normalize_path(path: str, workspace: str | None = None) -> str:
    """Normalize a file path to relative (strips workspace prefix).

    When ``workspace`` is None the function resolves it lazily — this
    is the back-compat path for callers that don't thread workspace
    through.  New code should pass it explicitly.
    """
    if not path:
        return path
    if os.path.isabs(path):
        ws = workspace if workspace is not None else _resolve_workspace()
        if path.startswith(ws + "/"):
            return path[len(ws) + 1:]
    return path


# ── Merge & helpers ──────────────────────────────────────────────────

def _blend_reactive_predictive(
    reactive: dict[str, tuple[float, str, str]],
    predictive: dict[str, tuple[float, str, str]],
    alpha: float,
) -> dict[str, tuple[float, str, str]]:
    """Blend reactive + predictive with adaptive alpha."""
    all_targets = set(reactive) | set(predictive)
    combined: dict[str, tuple[float, str, str]] = {}
    for key in all_targets:
        r_score, r_type, r_reason = reactive.get(key, (0.0, "", ""))
        p_score, p_type, p_reason = predictive.get(key, (0.0, "", ""))
        target_type = r_type or p_type
        score = alpha * r_score + (1 - alpha) * p_score
        reason = r_reason if r_score > p_score else (p_reason or r_reason)
        combined[key] = (score, target_type, reason)
    return combined


def _merge_channels(
    *channels: dict[str, tuple[float, str, str]],
) -> dict[str, tuple[float, str, str]]:
    """Merge multiple scoring channels with max() semantics per target."""
    merged: dict[str, tuple[float, str, str]] = {}
    for channel in channels:
        for target, (score, target_type, reason) in channel.items():
            if target not in merged or score > merged[target][0]:
                merged[target] = (score, target_type, reason)
    return merged


def _compute_alpha(iteration: int, reactive_signal_count: int) -> float:
    """Adaptive blend factor: 0 = pure prediction, 1 = pure reactive.

    At iteration 0 the current session has produced zero signal, so
    the ranking should lean entirely on historical memory.  As the
    session accumulates tool calls, the in-session reactive scores
    become more informative and the blend shifts toward reactive.

    Two caps prevent runaway:
      * Alpha never exceeds ``_ALPHA_MAX`` — a floor of historical
        signal is always preserved.
      * If the session has fewer than ``_ALPHA_REACTIVE_MIN_SIGNALS``
        interactions, reactive evidence is too sparse to trust alone
        and its contribution is halved via ``_ALPHA_SPARSE_REACTIVE_MULT``.
    """
    base_alpha = min(_ALPHA_MAX, iteration / _ALPHA_ITERATION_SATURATE)
    if reactive_signal_count < _ALPHA_REACTIVE_MIN_SIGNALS:
        base_alpha *= _ALPHA_SPARSE_REACTIVE_MULT
    return base_alpha


def _top_k(
    combined: dict[str, tuple[float, str, str]],
    target_type: str,
    k: int,
) -> list[RankedItem]:
    """Select the top *k* items of a given target type."""
    filtered = [
        (target, score, reason)
        for target, (score, tt, reason) in combined.items()
        if tt == target_type and score > 0
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [
        RankedItem(target=t, target_type=target_type, score=s, reason=r)
        for t, s, r in filtered[:k]
    ]


def _filter_outliers(items: list[RankedItem]) -> list[RankedItem]:
    """If a few items score dramatically higher than the rest, show only those.

    Uses Median Absolute Deviation (MAD) computed on the **bottom half**
    of the sorted scores to establish a noise baseline.  Items above
    ``median_bottom + K × MAD × 1.4826`` are outliers, where ``K`` is
    derived from ``CONTEXT_RANK_OUTLIER_PERCENTILE`` (user-friendly
    setting: 90, 95, 99, etc).  Computing MAD on the bottom half
    (not the full set) prevents outliers from inflating their own
    reference distribution.
    """
    n = len(items)
    if n < _OUTLIER_MIN_ITEMS:
        return items

    # Read user-friendly settings
    percentile = getattr(settings, "CONTEXT_RANK_OUTLIER_PERCENTILE", 99)
    max_count = getattr(settings, "CONTEXT_RANK_OUTLIER_MAX_COUNT", 3)
    min_top = getattr(settings, "CONTEXT_RANK_OUTLIER_MIN_TOP_SCORE", 1.0)
    mad_multiplier = _percentile_to_mad_multiplier(percentile)

    sorted_items = sorted(items, key=lambda it: it.score, reverse=True)
    top = sorted_items[0].score
    if top < min_top:
        return items

    # Bottom half: the lower ⌈n/2⌉ items (the "noise baseline").
    # Computing MAD on this subset avoids outliers polluting the baseline.
    bottom_count = max(2, (n + 1) // 2)
    bottom_scores = [it.score for it in sorted_items[-bottom_count:]]

    def _median(sorted_vals: list[float]) -> float:
        m = len(sorted_vals)
        return sorted_vals[m // 2] if m % 2 else (sorted_vals[m // 2 - 1] + sorted_vals[m // 2]) / 2

    b_sorted = sorted(bottom_scores)
    b_median = _median(b_sorted)

    deviations = sorted(abs(s - b_median) for s in bottom_scores)
    mad = _median(deviations)

    if mad < _MAD_DEGENERATE_THRESHOLD:
        # Baseline is too tight (all identical).  Fall back to a simple
        # ratio test — top must be significantly above the baseline.
        if top >= b_median * _OUTLIER_FALLBACK_RATIO and b_median > 0:
            threshold = (top + b_median) / 2  # midpoint
        else:
            return items
    else:
        threshold = b_median + mad_multiplier * mad * _MAD_NORMAL_CONSISTENCY

    # Secondary filter: outliers must ALSO be meaningfully above the
    # baseline in absolute magnitude.  Prevents low-variance baselines
    # (e.g. [3.2, 3.0, 2.8]) from flagging marginal values (3.5) as
    # outliers when they're really just slightly above noise.
    min_magnitude = b_median * _OUTLIER_MIN_RATIO if b_median > 0 else 0

    outliers = [
        it for it in sorted_items
        if it.score >= threshold and it.score >= min_magnitude
    ]
    if 1 <= len(outliers) <= max_count:
        return outliers
    return items
