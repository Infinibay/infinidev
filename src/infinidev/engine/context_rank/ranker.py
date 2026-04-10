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

_LEVEL_WEIGHTS: dict[str, float] = {
    "task_input": 1.0,
    "step_title": 1.5,
    "step_description": 2.0,
}

# Identifiers shorter than this are too common to be useful mentions
_MIN_IDENT_LEN = 4

# Index files tried when expanding directory targets
_INDEX_FILES = ("index.ts", "index.tsx", "index.js", "index.py", "__init__.py", "mod.rs")


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

    # ── Channel 1: Reactive (current session) ────────────────────
    reactive = _compute_reactive_scores(session_id, task_id, iteration)

    # ── Channel 2: Predictive / Historical (past sessions) ───────
    predictive = _compute_predictive_scores(
        current_input, session_id, cached_embedding=cached_embedding,
    )

    # Blend reactive + predictive with adaptive alpha
    alpha = _compute_alpha(iteration, len(reactive))
    blended = _blend_reactive_predictive(reactive, predictive, alpha)

    # ── Channel 3: Mention detection (FTS5 symbol lookup) ────────
    mentions = _compute_mention_scores(current_input, project_id)

    # ── Channel 4: Semantic findings (embedding similarity) ──────
    findings = _compute_finding_scores(cached_embedding, current_input, project_id)

    # ── Channel 5: Semantic docstrings (BM25) ────────────────────
    docstrings = _compute_docstring_scores(current_input, project_id)

    # ── Channel 6: Cross-session popularity ──────────────────────
    popularity = _compute_popularity_scores()

    # ── Merge all independent channels (max per target) ──────────
    combined = _merge_channels(blended, mentions, findings, docstrings, popularity)

    # ── Post-processing boosts ───────────────────────────────────
    combined = _apply_cooccurrence_boost(combined)
    combined = _apply_import_boost(combined, project_id)
    combined = _apply_freshness_boost(combined)
    combined = _expand_directory_targets(combined)

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
        files=_top_k(combined, "file", top_k_files),
        symbols=_top_k(combined, "symbol", top_k_symbols),
        findings=_top_k(combined, "finding", top_k_findings),
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
) -> dict[str, tuple[float, str, str]]:
    """Score nodes via embedding similarity to historical contexts.

    Edit interactions (weight >= 2.0) get a 2x multiplier in predictive
    scoring — a file *edited* in a similar past task is more likely
    relevant than one just read.
    """
    from infinidev.tools.base.embeddings import compute_embedding, embedding_from_blob
    from infinidev.tools.base.dedup import _cosine_similarity

    min_sim = settings.CONTEXT_RANK_MIN_SIMILARITY
    session_decay = settings.CONTEXT_RANK_SESSION_DECAY

    query_emb_bytes = cached_embedding or compute_embedding(current_input)
    if query_emb_bytes is None:
        return {}
    query_vec = np.frombuffer(query_emb_bytes, dtype=np.float32)

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
        target = _normalize_path(row["target"]) if row["target_type"] == "file" else row["target"]
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
    current_input: str, project_id: int,
) -> dict[str, tuple[float, str, str]]:
    """Find known symbols from the project that appear in the input.

    Inverse lookup: instead of trying to extract identifiers from the
    input with regex, we ask the DB "which of the symbols you know
    about appear literally in this text?".  One SQL query with
    ``instr()`` does all the work — no regex, no stop-word lists per
    language, no tokenization bugs.
    """
    if not current_input or len(current_input) < 4:
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
        score = min(5.0, 3.0 + (row["name_len"] / 20.0))

        fp = _normalize_path(row["file_path"])
        if fp:
            existing = result.get(fp, (0.0, "", ""))
            if score > existing[0]:
                result[fp] = (score, "file", f"contains '{name}' mentioned in input")

        sym_key = row["qualified_name"] or name
        if sym_key:
            existing = result.get(sym_key, (0.0, "", ""))
            if score > existing[0]:
                result[sym_key] = (score, "symbol", f"'{name}' mentioned in input")

    # Also check file basenames against the input
    try:
        def _fetch_files(conn):
            return conn.execute(
                "SELECT DISTINCT file_path FROM ci_files "
                "WHERE project_id = ? "
                "LIMIT 2000",
                (project_id,),
            ).fetchall()
        file_rows = execute_with_retry(_fetch_files)
    except Exception:
        file_rows = []

    for row in file_rows:
        fp = row["file_path"]
        if not fp:
            continue
        basename = os.path.basename(fp)
        stem = basename.rsplit(".", 1)[0] if "." in basename else basename
        stem_lc = stem.lower()
        if len(stem) < 4 or stem_lc in _COMMON_WORDS:
            continue

        # Basename match: "registry.ts" literally in input. This is a
        # strong signal — user clearly knows the filename.
        basename_match = "." in basename and basename.lower() in padded
        # Stem match: word boundaries on both sides. For generic stems
        # (server, system, ...) we get many matches but other channels
        # (historical, co-occurrence) rank the truly relevant ones higher.
        stem_match = (
            f" {stem_lc} " in padded or
            f" {stem_lc}." in padded or
            f" {stem_lc}?" in padded or
            f" {stem_lc}," in padded or
            f" {stem_lc}/" in padded
        )

        if basename_match or stem_match:
            fp_n = _normalize_path(fp)
            # Basename matches score higher (more specific)
            base_score = 4.0 if basename_match else 3.2
            # Generic stems get a lower score to avoid flooding results
            if not basename_match and stem_lc in _STEM_SKIP:
                base_score = 2.5
            score = min(5.0, base_score + (len(stem) / 25.0))
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
    """
    import json
    from infinidev.tools.base.embeddings import compute_embedding, embedding_from_blob
    from infinidev.tools.base.dedup import _cosine_similarity

    query_emb = cached_embedding or compute_embedding(current_input)
    query_vec = np.frombuffer(query_emb, dtype=np.float32) if query_emb else None

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
                if sim >= 0.5:
                    scores.append((sim * 3.0, f"semantic match (sim={sim:.2f})"))
            except Exception:
                pass

        # ── Signal 2: topic word matching ────────────────────────
        topic_words = [w for w in re.split(r'\W+', topic.lower()) if len(w) >= 4 and w not in _COMMON_WORDS]
        if topic_words:
            matched = sum(1 for w in topic_words if f" {w} " in padded or f" {w}" in padded)
            if matched >= max(2, len(topic_words) // 2):
                ratio = matched / len(topic_words)
                scores.append((3.0 + ratio * 1.5, f"{matched}/{len(topic_words)} topic words match"))

        # ── Signal 3: tag matching ───────────────────────────────
        try:
            tags = json.loads(row["tags_json"] or "[]")
            tag_hits = [t for t in tags if isinstance(t, str) and len(t) >= 4 and f" {t.lower()} " in padded]
            if tag_hits:
                scores.append((4.0 + len(tag_hits) * 0.5, f"tags match: {', '.join(tag_hits[:3])}"))
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
    current_input: str, project_id: int,
) -> dict[str, tuple[float, str, str]]:
    """Score symbols by docstring/signature relevance via BM25.

    Noise control:
    - Only top 5 BM25 matches (not 20) — BM25 ranks reliably at the top
    - Only function/method/class/interface kinds
    - Requires >= 2 non-common input words present in the docstring
      (single-word matches are too noisy)
    - Skips very short docstrings (< 30 chars, usually "Get X" style)
    """
    try:
        from infinidev.code_intel.query import search_by_docstring
        matches = search_by_docstring(project_id, current_input, limit=5)
    except Exception:
        return {}

    _RELEVANT_KINDS = {"function", "method", "class", "interface"}

    # Extract meaningful input words (≥4 chars, not common)
    input_words = [
        w for w in re.split(r'\W+', current_input.lower())
        if len(w) >= 4 and w not in _COMMON_WORDS
    ]
    if len(input_words) < 2:
        return {}

    result: dict[str, tuple[float, str, str]] = {}
    for sym, _bm25_rank in matches:
        kind = sym.kind.value if hasattr(sym.kind, "value") else str(sym.kind)
        if kind not in _RELEVANT_KINDS:
            continue

        # Require meaningful docstring
        doc = (sym.docstring or "").lower()
        if len(doc) < 30:
            continue

        # Count how many input words appear in the docstring
        hits = sum(1 for w in input_words if w in doc)
        if hits < 2:
            continue

        # Scale score with hit density
        density = hits / max(len(input_words), 1)
        score_file = 3.0 + density * 0.5  # up to 3.5
        score_sym = 3.5 + density * 0.5   # up to 4.0

        fp = _normalize_path(sym.file_path)
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
    """Base score for files accessed in 3+ different sessions."""
    try:
        def _query(conn):
            return conn.execute(
                "SELECT target, COUNT(DISTINCT session_id) as session_count "
                "FROM cr_session_scores "
                "WHERE target_type = 'file' "
                "GROUP BY target "
                "HAVING session_count >= 3",
            ).fetchall()
        rows = execute_with_retry(_query)
    except Exception:
        return {}

    result: dict[str, tuple[float, str, str]] = {}
    for row in rows:
        count = row["session_count"]
        score = math.log(count) * 0.3
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
        if tt == "file" and s >= 1.0
    ]
    if not top_files:
        return scores

    top_files.sort(key=lambda x: x[1], reverse=True)
    top_files = top_files[:5]  # Limit to top 5 to control DB queries

    for target, target_score in top_files:
        try:
            def _query(conn, t=target):
                return conn.execute(
                    "SELECT b.target, COUNT(DISTINCT b.session_id) as co_sessions "
                    "FROM cr_session_scores a "
                    "JOIN cr_session_scores b ON a.session_id = b.session_id AND a.target != b.target "
                    "WHERE a.target = ? AND a.target_type = 'file' AND b.target_type = 'file' "
                    "GROUP BY b.target "
                    "HAVING co_sessions >= 2",
                    (t,),
                ).fetchall()
            rows = execute_with_retry(_query)
        except Exception:
            continue

        for row in rows:
            co_target = row["target"]
            if co_target in scores:
                continue  # Don't boost already-scored files
            co_score = target_score * 0.4 * min(row["co_sessions"] / 5.0, 1.0)
            if co_score > 0.3:
                scores[co_target] = (co_score, "file", f"co-occurs with {os.path.basename(target)}")
    return scores


# ── Post-processing: Import graph propagation ───────────────────────

def _apply_import_boost(
    scores: dict[str, tuple[float, str, str]],
    project_id: int,
) -> dict[str, tuple[float, str, str]]:
    """1-hop propagation through the import graph."""
    top_files = [
        (t, s) for t, (s, tt, _) in scores.items()
        if tt == "file" and s >= 1.5
    ]
    if not top_files:
        return scores

    top_files.sort(key=lambda x: x[1], reverse=True)
    top_files = top_files[:3]  # Limit queries

    for target, target_score in top_files:
        try:
            def _query(conn, t=target, pid=project_id):
                # Files that import the scored file
                importers = conn.execute(
                    "SELECT DISTINCT file_path FROM ci_imports "
                    "WHERE project_id = ? AND resolved_file = ?",
                    (pid, t),
                ).fetchall()
                # Files imported by the scored file
                imported = conn.execute(
                    "SELECT DISTINCT resolved_file FROM ci_imports "
                    "WHERE project_id = ? AND file_path = ? AND resolved_file != ''",
                    (pid, t),
                ).fetchall()
                return importers, imported
            importers, imported = execute_with_retry(_query)
        except Exception:
            continue

        basename = os.path.basename(target)
        for row in importers:
            fp = row["file_path"]
            if fp not in scores:
                score = target_score * 0.3
                scores[fp] = (score, "file", f"imports {basename}")
        for row in imported:
            fp = row["resolved_file"]
            if fp and fp not in scores:
                score = target_score * 0.5
                scores[fp] = (score, "file", f"imported by {basename}")
    return scores


# ── Post-processing: Git freshness ──────────────────────────────────

def _apply_freshness_boost(
    scores: dict[str, tuple[float, str, str]],
) -> dict[str, tuple[float, str, str]]:
    """Boost files with recent git modifications."""
    import subprocess
    import time as _time

    file_targets = [t for t, (_, tt, _) in scores.items() if tt == "file" and "." in os.path.basename(t)]
    if not file_targets:
        return scores

    # Batch git log for efficiency
    for target in file_targets[:10]:  # Cap to 10 files
        try:
            result = subprocess.run(
                ["git", "log", "--format=%at", "-1", "--", target],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                mtime = int(result.stdout.strip())
                days_ago = (_time.time() - mtime) / 86400
                freshness = max(1.0, 1.3 - (days_ago / 100))
                if freshness > 1.0:
                    old_score, tt, reason = scores[target]
                    scores[target] = (old_score * freshness, tt, reason)
        except Exception:
            pass
    return scores


# ── Post-processing: Directory expansion ────────────────────────────

def _expand_directory_targets(
    scores: dict[str, tuple[float, str, str]],
) -> dict[str, tuple[float, str, str]]:
    """Replace directory targets with their entry-point files."""
    workspace = os.getcwd()
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
                    scores[rel] = (score * 0.8, "file", f"index of {os.path.basename(target)}/")
                break
    return scores


# ── Path normalization ───────────────────────────────────────────────

def _normalize_path(path: str) -> str:
    """Normalize a file path to relative (strips workspace prefix)."""
    if not path:
        return path
    if os.path.isabs(path):
        workspace = os.getcwd()
        if path.startswith(workspace + "/"):
            return path[len(workspace) + 1:]
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
    """Adaptive blend factor: 0 = pure prediction, 1 = pure reactive."""
    base_alpha = min(0.85, iteration / 8)
    if reactive_signal_count < 3:
        base_alpha *= 0.5
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
