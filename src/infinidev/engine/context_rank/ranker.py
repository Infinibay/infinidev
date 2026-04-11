"""ContextRanker v2 — multi-channel scoring for files, symbols, and findings.

Combines 6 independent scoring channels + 4 post-processing boosts:

**Independent channels** (produce scores from scratch):
  1. Reactive — current session tool calls with recency decay
  2. Predictive (historical) — embedding sim vs past contexts → interactions
  3. Mention detection — FTS5 lookup of identifiers mentioned in the input
  4. Semantic findings — cosine sim of input vs finding embeddings
  5. Semantic docstrings — BM25 via FTS5 on symbol docstrings/signatures
  6. Popularity — files accessed in 3+ sessions get a base boost

**Post-processing boosts** (modify existing scores):
  7. Co-occurrence — files frequently accessed alongside top-scored files
  8. Import graph — 1-hop propagation through ``ci_imports``
  9. Git freshness — recent git modifications boost score
  10. Directory expansion — replace directory targets with index files

A confidence gate suppresses the entire ``<context-rank>`` section when
the top score is below ``CONTEXT_RANK_MIN_CONFIDENCE``.
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

# ── Escalera level weights ──
# Higher = more specific context match contributes more to score.
# task_input is broadest (applies to whole task), step_description is
# most specific (scoped to a single plan step).
_LEVEL_WEIGHTS: dict[str, float] = {
    "task_input": 1.0,
    "step_title": 1.5,
    "step_description": 2.0,
}

# Identifiers shorter than this are too common to be useful mentions
_MIN_IDENT_LEN = 4

# Index files tried when expanding directory targets
_INDEX_FILES = ("index.ts", "index.tsx", "index.js", "index.py", "__init__.py", "mod.rs")

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

# ── Mention channel weights ──
# Base score for a symbol name appearing literally in the input.
# Longer names are more distinctive (less likely to be false
# positives) so the score grows with name length.
_MENTION_BASE_SCORE = 3.0
# Per-character bonus: `score = BASE + name_len / DIVISOR`, capped at
# `CAP`.  With DIVISOR=20 and CAP=5.0: names up to 40 chars still earn
# extra score, beyond that it saturates.
_MENTION_NAME_LEN_DIVISOR = 20.0
_MENTION_SCORE_CAP = 5.0
# Base score for matching a file basename ("foo.ts" appears in input).
# Stronger than a stem match because the extension disambiguates.
_MENTION_BASENAME_BASE = 4.0
# Base score for matching a file stem ("foo" appears with word boundary).
_MENTION_STEM_BASE = 3.2
# Penalty applied to stem matches whose stem is in `_STEM_SKIP` (e.g.
# "config", "utils").  Still scored (the stem literally appears) but
# treated as weaker evidence.
_MENTION_STEM_SKIP_BASE = 2.5
# Per-character bonus for basename/stem matches.
_MENTION_FILENAME_LEN_DIVISOR = 25.0
# Max LIKE patterns in the basename query — cap to avoid pathological
# queries when the input has many distinct words.
_MENTION_MAX_LIKE_PATTERNS = 20

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

# ── Docstring channel weights ──
# Minimum docstring length to consider — shorter docstrings like
# "Get X" match everything and produce noise.
_DOCSTRING_MIN_LENGTH = 30
# Minimum number of non-common input words that must appear in the
# docstring.  Single-word matches are too noisy in practice.
_DOCSTRING_MIN_HITS = 2
# BM25 result cap — top-5 is where BM25 ranking is most reliable,
# past that the tail is noisy.
_DOCSTRING_BM25_LIMIT = 5
# Base + density-scaled bonus for file and symbol scores.
_DOCSTRING_FILE_BASE = 3.0
_DOCSTRING_FILE_BONUS = 0.5
_DOCSTRING_SYMBOL_BASE = 3.5
_DOCSTRING_SYMBOL_BONUS = 0.5

# ── Popularity channel weights ──
# Minimum distinct sessions for a file to get any popularity score.
# Below this, it's a one-off read — no signal.
_POPULARITY_MIN_SESSIONS = 3
# Multiplier on log(session_count).  Logarithmic so a file in 20
# sessions doesn't crush a file with strong topical relevance — just
# nudges the ranking.  0.3 puts popularity scores in ~0.3-1.0 range.
_POPULARITY_LOG_SCALE = 0.3

# ── Co-occurrence boost ──
# Only boost co-occurrences of files whose base score is at least this.
# Below this threshold, the anchor file is too marginal to trust as a
# source of co-occurrence signal.
_COOC_ANCHOR_MIN_SCORE = 1.0
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
# Only propagate scores from files at or above this threshold.
_IMPORT_ANCHOR_MIN_SCORE = 1.5
# Look at the top N anchor files (each fires one UNION query).
_IMPORT_MAX_ANCHORS = 3
# When A imports B, A gets `B.score * IMPORT_IN_PROPAGATION`.
# Lower than "out" because knowing the importer is less useful than
# knowing the imported dependency.
_IMPORT_IN_PROPAGATION = 0.3
# When A imports B, B gets `A.score * IMPORT_OUT_PROPAGATION`.
# Higher because the dependency is usually needed to understand A.
_IMPORT_OUT_PROPAGATION = 0.5

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

# ── Directory expansion ──
# Score applied to an index file when expanded from its parent dir.
# Slightly lower than the original dir score because the index is a
# concrete but approximate substitute.
_DIR_EXPAND_DAMPEN = 0.8

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

    # ── Shared embedding ─────────────────────────────────────────
    # Compute the task embedding once and pass it to every channel
    # that needs it.  Previously each channel fell back to
    # `cached_embedding or compute_embedding(current_input)` which
    # could fire up to 2 redundant embedding calls (~267ms each) on
    # the first rank, when the background thread hadn't yet populated
    # the cache.
    query_embedding = cached_embedding
    if query_embedding is None:
        try:
            from infinidev.tools.base.embeddings import compute_embedding
            query_embedding = compute_embedding(current_input)
        except Exception:
            query_embedding = None

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

    # ── Channel 3: Mention detection (FTS5 symbol lookup) ────────
    mentions = _compute_mention_scores(current_input, project_id, workspace)

    # ── Channel 4: Semantic findings (embedding similarity) ──────
    findings = _compute_finding_scores(query_embedding, current_input, project_id)

    # ── Channel 5: Semantic docstrings (BM25) ────────────────────
    docstrings = _compute_docstring_scores(current_input, project_id, workspace)

    # ── Channel 6: Cross-session popularity ──────────────────────
    popularity = _compute_popularity_scores()

    # ── Merge all independent channels (max per target) ──────────
    combined = _merge_channels(blended, mentions, findings, docstrings, popularity)

    # ── Post-processing boosts ───────────────────────────────────
    combined = _apply_cooccurrence_boost(combined)
    combined = _apply_import_boost(combined, project_id)
    combined = _apply_freshness_boost(combined, workspace)
    combined = _expand_directory_targets(combined, workspace)

    # ── Confidence gate ──────────────────────────────────────────
    # Check the max score across ALL raw channels (pre-merge), not
    # just the blended result — alpha blending dilutes reactive scores
    # when there's no predictive data, but the raw signal may be strong.
    min_confidence = getattr(settings, "CONTEXT_RANK_MIN_CONFIDENCE", 0.5)
    all_raw = [
        s for ch in (reactive, predictive, mentions, findings, docstrings, popularity)
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

def _compute_reactive_scores(
    session_id: str, task_id: str, current_iteration: int,
) -> dict[str, tuple[float, str, str]]:
    """Score nodes based on tool calls in the current session."""
    decay_lambda = settings.CONTEXT_RANK_REACTIVE_DECAY
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
            accum[target] = {"target_type": row["target_type"], "weighted_sum": 0.0, "count": 0}
        delta = current_iteration - row["iteration"]
        decay = math.exp(-decay_lambda * max(delta, 0))
        accum[target]["weighted_sum"] += row["weight"] * decay
        accum[target]["count"] += 1

    result: dict[str, tuple[float, str, str]] = {}
    for target, info in accum.items():
        freq_boost = math.log(1 + info["count"])
        score = info["weighted_sum"] * freq_boost
        result[target] = (score, info["target_type"], f"accessed {info['count']}x this session")
    return result


# ── Channel 2: Predictive scoring (historical escalera) ─────────────

def _compute_predictive_scores(
    current_input: str, exclude_session: str,
    *, cached_embedding: bytes | None = None,
    workspace: str | None = None,
) -> dict[str, tuple[float, str, str]]:
    """Score nodes via embedding similarity to historical contexts.

    Edit interactions (weight >= 2.0) get a 2x multiplier in predictive
    scoring — a file *edited* in a similar past task is more likely
    relevant than one just read.

    The embedding is expected to be provided via ``cached_embedding``
    — ``rank()`` computes it once and shares it across channels.  When
    None is passed the channel returns empty rather than re-computing,
    since the caller already tried and failed.
    """
    from infinidev.tools.base.embeddings import embedding_from_blob
    from infinidev.tools.base.dedup import _cosine_similarity

    min_sim = settings.CONTEXT_RANK_MIN_SIMILARITY
    session_decay = settings.CONTEXT_RANK_SESSION_DECAY

    if cached_embedding is None:
        return {}
    query_vec = np.frombuffer(cached_embedding, dtype=np.float32)

    try:
        def _fetch_contexts(conn):
            return conn.execute(
                "SELECT id, session_id, context_type, embedding "
                "FROM cr_contexts "
                "WHERE embedding IS NOT NULL AND session_id != ? "
                "ORDER BY created_at DESC LIMIT 500",
                (exclude_session,),
            ).fetchall()
        ctx_rows = execute_with_retry(_fetch_contexts)
    except Exception:
        return {}

    if not ctx_rows:
        return {}

    matched_contexts: list[tuple[int, float, float]] = []
    session_order: dict[str, int] = {}
    order_counter = 0
    for row in ctx_rows:
        sid = row["session_id"]
        if sid not in session_order:
            session_order[sid] = order_counter
            order_counter += 1
        ctx_vec = embedding_from_blob(row["embedding"])
        sim = float(_cosine_similarity(query_vec, ctx_vec))
        if sim < min_sim:
            continue
        level_weight = _LEVEL_WEIGHTS.get(row["context_type"], 1.0)
        s_decay = session_decay ** session_order[sid]
        contribution = sim * level_weight * s_decay
        matched_contexts.append((row["id"], contribution, sim))

    if not matched_contexts:
        return {}

    context_ids = [m[0] for m in matched_contexts]
    contrib_by_id = {m[0]: (m[1], m[2]) for m in matched_contexts}

    try:
        def _fetch_interactions(conn):
            placeholders = ",".join("?" * len(context_ids))
            return conn.execute(
                f"SELECT context_id, target, target_type, weight "
                f"FROM cr_interactions "
                f"WHERE context_id IN ({placeholders})",
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
        # Edit vs Read asymmetry: writes get 2x in predictive channel
        predictive_mult = 2.0 if row["weight"] >= 2.0 else 1.0
        accum[target]["score"] += contribution * row["weight"] * predictive_mult
        accum[target]["best_sim"] = max(accum[target]["best_sim"], sim)

    result: dict[str, tuple[float, str, str]] = {}
    for target, info in accum.items():
        result[target] = (
            info["score"], info["target_type"],
            f"predicted (similarity={info['best_sim']:.2f} to past contexts)",
        )
    return result


# ── Channel 3: Mention detection ────────────────────────────────────

# Common English words that happen to be valid short symbol names.
# We exclude these from the "reverse" mention matching because they'd
# false-positive against almost any natural-language input.
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


# Words that often mean the English concept, not a file/module name.
# Only used to filter STEM matches — if the user writes "system.ts"
# literally, that exact form still matches (it's not a stem-only match).
_STEM_SKIP: frozenset[str] = frozenset({
    "system", "systems", "module", "modules", "common", "shared",
    "base", "helper", "helpers", "manager", "handler", "service",
    "client", "server", "config", "setup", "index", "const", "style",
    "styles", "route", "routes", "view", "views", "model", "models",
    "store", "stores", "util", "utils", "info", "item", "items",
})


def _compute_mention_scores(
    current_input: str, project_id: int, workspace: str | None = None,
) -> dict[str, tuple[float, str, str]]:
    """Find known symbols from the project that appear in the input.

    Inverse lookup: instead of trying to extract identifiers from the
    input with regex, we ask the DB "which of the symbols you know
    about appear literally in this text?".  One SQL query with
    ``instr()`` does all the work — no regex, no stop-word lists per
    language, no tokenization bugs.
    """
    if not current_input or len(current_input) < _MIN_IDENT_LEN:
        return {}

    input_lower = current_input.lower()
    # Pad with spaces so word-boundary heuristics work at start/end
    padded = " " + input_lower + " "

    try:
        def _fetch(conn):
            # Find symbols whose name or qualified_name appears in the input.
            # Constraints:
            #   - Only function/method/class/interface/enum (distinctive)
            #   - LENGTH(name) >= 4 to avoid matching short common words
            #   - instr() returns position (0 if not found)
            return conn.execute(
                "SELECT DISTINCT name, qualified_name, file_path, kind, "
                "LENGTH(name) as name_len "
                "FROM ci_symbols "
                "WHERE project_id = ? "
                "  AND kind IN ('function', 'method', 'class', 'interface', 'enum', 'type_alias') "
                "  AND LENGTH(name) >= 4 "
                "  AND (instr(?, LOWER(name)) > 0 "
                "       OR (qualified_name != '' AND instr(?, LOWER(qualified_name)) > 0)) "
                "LIMIT 100",
                (project_id, padded, padded),
            ).fetchall()
        rows = execute_with_retry(_fetch)
    except Exception:
        logger.debug("Mention detection query failed", exc_info=True)
        return {}

    result: dict[str, tuple[float, str, str]] = {}
    for row in rows:
        name = row["name"]
        name_lc = name.lower()
        # Skip common English words that happen to be valid symbol names
        if name_lc in _COMMON_WORDS:
            continue
        # Score scales with name length — longer names are more distinctive
        score = min(
            _MENTION_SCORE_CAP,
            _MENTION_BASE_SCORE + (row["name_len"] / _MENTION_NAME_LEN_DIVISOR),
        )

        fp = _normalize_path(row["file_path"], workspace)
        if fp:
            existing = result.get(fp, (0.0, "", ""))
            if score > existing[0]:
                result[fp] = (score, "file", f"contains '{name}' mentioned in input")

        sym_key = row["qualified_name"] or name
        if sym_key:
            existing = result.get(sym_key, (0.0, "", ""))
            if score > existing[0]:
                result[sym_key] = (score, "symbol", f"'{name}' mentioned in input")

    # Also check file basenames against the input.
    # Extract candidate words from the input once, then query only
    # files whose path contains any of those words.  This replaces
    # the full-scan approach (2000 rows → Python filter).
    input_words = set(
        w for w in re.split(r'\W+', input_lower)
        if len(w) >= _MIN_IDENT_LEN and w not in _COMMON_WORDS
    )
    file_rows = []
    if input_words:
        try:
            def _fetch_files(conn):
                # Build an OR of LIKE patterns — SQLite uses the
                # ci_files index on file_path so this is fast even
                # with 10+ patterns.
                words_list = list(input_words)[:_MENTION_MAX_LIKE_PATTERNS]
                placeholders = " OR ".join("file_path LIKE ?" for _ in words_list)
                params = [project_id] + [f"%{w}%" for w in words_list]
                return conn.execute(
                    f"SELECT DISTINCT file_path FROM ci_files "
                    f"WHERE project_id = ? AND ({placeholders}) "
                    f"LIMIT 200",
                    params,
                ).fetchall()
            file_rows = execute_with_retry(_fetch_files)
        except Exception:
            file_rows = []

    for row in file_rows:
        fp = row["file_path"]
        if not fp:
            continue
        basename = os.path.basename(fp)  # trust Python's extraction
        stem = basename.rsplit(".", 1)[0] if "." in basename else basename
        stem_lc = stem.lower()
        if len(stem) < _MIN_IDENT_LEN or stem_lc in _COMMON_WORDS:
            continue

        # The SQL already confirmed the basename substring is in the
        # input.  We now distinguish between "basename.ext literal" and
        # "stem with word boundary" for scoring.
        basename_match = "." in basename and basename.lower() in padded
        stem_match = (
            f" {stem_lc} " in padded or
            f" {stem_lc}." in padded or
            f" {stem_lc}?" in padded or
            f" {stem_lc}," in padded or
            f" {stem_lc}/" in padded
        )

        if basename_match or stem_match:
            fp_n = _normalize_path(fp, workspace)
            base_score = _MENTION_BASENAME_BASE if basename_match else _MENTION_STEM_BASE
            if not basename_match and stem_lc in _STEM_SKIP:
                base_score = _MENTION_STEM_SKIP_BASE
            score = min(
                _MENTION_SCORE_CAP,
                base_score + (len(stem) / _MENTION_FILENAME_LEN_DIVISOR),
            )
            existing = result.get(fp_n, (0.0, "", ""))
            if score > existing[0]:
                reason = (
                    f"filename '{basename}' mentioned in input"
                    if basename_match else
                    f"file stem '{stem}' mentioned in input"
                )
                result[fp_n] = (score, "file", reason)

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
            return conn.execute(
                "SELECT topic, content, embedding, tags_json "
                "FROM findings "
                "WHERE project_id = ? AND status = 'active' "
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
        try:
            tags = json.loads(row["tags_json"] or "[]")
            tag_hits = [
                t for t in tags
                if isinstance(t, str) and len(t) >= _MIN_IDENT_LEN
                and f" {t.lower()} " in padded
            ]
            if tag_hits:
                scores.append((
                    _FINDING_TAG_BASE + len(tag_hits) * _FINDING_TAG_BONUS,
                    f"tags match: {', '.join(tag_hits[:3])}",
                ))
        except Exception:
            pass

        # Keep the max signal
        if scores:
            best_score, best_reason = max(scores, key=lambda x: x[0])
            preview = (row["content"] or "")[:60]
            result[topic] = (best_score, "finding", f"{best_reason}: {preview}")

    return result


# ── Channel 5: Semantic docstring match (BM25) ─────────────────────

def _compute_docstring_scores(
    current_input: str, project_id: int, workspace: str | None = None,
) -> dict[str, tuple[float, str, str]]:
    """Score symbols by docstring/signature relevance via BM25.

    Noise control:
    - Only top ``_DOCSTRING_BM25_LIMIT`` BM25 matches — past the top,
      BM25 ranking becomes unreliable in our experience.
    - Only function/method/class/interface kinds (symbols whose
      docstring is likely to describe intent, not a type alias).
    - Requires >= ``_DOCSTRING_MIN_HITS`` non-common input words
      present in the docstring (single-word matches are too noisy).
    - Skips docstrings shorter than ``_DOCSTRING_MIN_LENGTH`` chars
      (usually "Get X" / "Returns Y" boilerplate that matches too much).
    """
    try:
        from infinidev.code_intel.query import search_by_docstring
        matches = search_by_docstring(
            project_id, current_input, limit=_DOCSTRING_BM25_LIMIT,
        )
    except Exception:
        return {}

    _RELEVANT_KINDS = {"function", "method", "class", "interface"}

    # Extract meaningful input words (≥ _MIN_IDENT_LEN chars, not common)
    input_words = [
        w for w in re.split(r'\W+', current_input.lower())
        if len(w) >= _MIN_IDENT_LEN and w not in _COMMON_WORDS
    ]
    if len(input_words) < _DOCSTRING_MIN_HITS:
        return {}

    result: dict[str, tuple[float, str, str]] = {}
    for sym, _bm25_rank in matches:
        kind = sym.kind.value if hasattr(sym.kind, "value") else str(sym.kind)
        if kind not in _RELEVANT_KINDS:
            continue

        # Require meaningful docstring
        doc = (sym.docstring or "").lower()
        if len(doc) < _DOCSTRING_MIN_LENGTH:
            continue

        # Count how many input words appear in the docstring
        hits = sum(1 for w in input_words if w in doc)
        if hits < _DOCSTRING_MIN_HITS:
            continue

        # Scale score with hit density
        density = hits / max(len(input_words), 1)
        score_file = _DOCSTRING_FILE_BASE + density * _DOCSTRING_FILE_BONUS
        score_sym = _DOCSTRING_SYMBOL_BASE + density * _DOCSTRING_SYMBOL_BONUS

        fp = _normalize_path(sym.file_path, workspace)
        if fp:
            existing = result.get(fp, (0.0, "", ""))
            if score_file > existing[0]:
                result[fp] = (score_file, "file",
                              f"contains '{sym.name}' ({hits} word match in docstring)")
        key = sym.qualified_name or sym.name
        if key:
            existing = result.get(key, (0.0, "", ""))
            if score_sym > existing[0]:
                result[key] = (score_sym, "symbol",
                               f"{hits} word docstring match")
    return result


# ── Channel 6: Cross-session popularity ─────────────────────────────

def _compute_popularity_scores() -> dict[str, tuple[float, str, str]]:
    """Base score for files accessed in many different sessions."""
    try:
        def _query(conn):
            return conn.execute(
                "SELECT target, COUNT(DISTINCT session_id) as session_count "
                "FROM cr_session_scores "
                "WHERE target_type = 'file' "
                "GROUP BY target "
                "HAVING session_count >= ?",
                (_POPULARITY_MIN_SESSIONS,),
            ).fetchall()
        rows = execute_with_retry(_query)
    except Exception:
        return {}

    result: dict[str, tuple[float, str, str]] = {}
    for row in rows:
        count = row["session_count"]
        score = math.log(count) * _POPULARITY_LOG_SCALE
        result[row["target"]] = (score, "file", f"accessed in {count} sessions")
    return result


# ── Post-processing: Co-occurrence boost ────────────────────────────

def _apply_cooccurrence_boost(
    scores: dict[str, tuple[float, str, str]],
) -> dict[str, tuple[float, str, str]]:
    """Boost files frequently accessed alongside high-scoring files."""
    # Get top file targets to find co-occurring files for
    top_files = [
        (t, s) for t, (s, tt, _) in scores.items()
        if tt == "file" and s >= _COOC_ANCHOR_MIN_SCORE
    ]
    if not top_files:
        return scores

    top_files.sort(key=lambda x: x[1], reverse=True)
    top_files = top_files[:_COOC_MAX_ANCHORS]

    for target, target_score in top_files:
        try:
            def _query(conn, t=target):
                return conn.execute(
                    "SELECT b.target, COUNT(DISTINCT b.session_id) as co_sessions "
                    "FROM cr_session_scores a "
                    "JOIN cr_session_scores b ON a.session_id = b.session_id AND a.target != b.target "
                    "WHERE a.target = ? AND a.target_type = 'file' AND b.target_type = 'file' "
                    "GROUP BY b.target "
                    "HAVING co_sessions >= ?",
                    (t, _COOC_MIN_SESSIONS),
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
) -> dict[str, tuple[float, str, str]]:
    """1-hop propagation through the import graph."""
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
                #   direction='in'  → files that import target
                #   direction='out' → files that target imports
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
            if not fp or fp in scores:
                continue
            if row["direction"] == "in":
                scores[fp] = (
                    target_score * _IMPORT_IN_PROPAGATION,
                    "file", f"imports {basename}",
                )
            else:
                scores[fp] = (
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

def _expand_directory_targets(
    scores: dict[str, tuple[float, str, str]],
    workspace: str | None = None,
) -> dict[str, tuple[float, str, str]]:
    """Replace directory targets with their entry-point files."""
    if workspace is None:
        workspace = _resolve_workspace()
    for target in list(scores):
        score, tt, reason = scores[target]
        if tt != "file":
            continue
        # Heuristic: no extension in basename → likely a directory
        if "." in os.path.basename(target):
            continue
        full_path = os.path.join(workspace, target) if not os.path.isabs(target) else target
        if not os.path.isdir(full_path):
            continue
        del scores[target]
        for idx_file in _INDEX_FILES:
            candidate = os.path.join(full_path, idx_file)
            if os.path.exists(candidate):
                rel = os.path.relpath(candidate, workspace)
                if rel not in scores:
                    scores[rel] = (
                        score * _DIR_EXPAND_DAMPEN,
                        "file", f"index of {os.path.basename(target)}/",
                    )
                break
    return scores


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
