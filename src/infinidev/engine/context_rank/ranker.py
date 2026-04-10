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

def _extract_identifiers(text: str) -> list[str]:
    """Extract potential code identifiers from natural language text.

    Returns identifiers in priority order:
    1. Backtick-quoted (`fromPlugin`)
    2. Slash-paths (src/agent/, tool/registry.ts) — split into segments
    3. CamelCase (ToolRegistry, AgentLoop)
    4. dot.notation (Agent.handleEvent)
    5. snake_case (tool_registry)
    """
    found: list[str] = []
    # 1. Backtick-quoted identifiers
    found.extend(re.findall(r'`([^`]+)`', text))
    # 2. Slash-paths — extract the interesting segments (filename or last dir)
    _EXT = (
        # Web / JS ecosystem
        r"ts|tsx|js|jsx|mjs|cjs|vue|svelte|astro|"
        # Python
        r"py|pyi|pyx|"
        # Systems
        r"rs|go|c|h|cpp|cc|cxx|hpp|hxx|"
        # JVM
        r"java|kt|kts|scala|groovy|clj|cljs|"
        # Ruby / Elixir / Erlang
        r"rb|erb|ex|exs|erl|"
        # .NET
        r"cs|fs|fsx|vb|"
        # Scripts / shell
        r"sh|bash|zsh|fish|ps1|"
        # Data / config / markup
        r"json|yaml|yml|toml|ini|cfg|xml|html|htm|css|scss|sass|less|md|mdx|rst|"
        # Swift / ObjC
        r"swift|m|mm|"
        # Other
        r"php|lua|r|dart|zig|nim|hs|ml|elm|sql"
    )
    for path_match in re.finditer(rf'\b([a-z_][\w./-]*\.(?:{_EXT}))\b', text):
        # Full path
        found.append(path_match.group(1))
    # Also extract meaningful directory names from paths like "src/agent/"
    for dir_match in re.finditer(r'(?:^|[\s/])([a-z_][\w-]{3,})(?=/)', text):
        found.append(dir_match.group(1))
    # 3. CamelCase: ToolRegistry, AgentLoop (at least 2 humps)
    found.extend(re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\b', text))
    # 3b. Single-word capitalized tech terms: Agent, Session, Provider (4+ chars)
    found.extend(re.findall(r'\b([A-Z][a-z]{3,})\b', text))
    # 4. dot.notation: Agent.handleEvent, tool.registry
    found.extend(re.findall(r'\b(\w+\.\w+(?:\.\w+)*)\b', text))
    # 4b. camelCase (starts lowercase, has 1+ humps): fromPlugin, handleEvent
    found.extend(re.findall(r'\b([a-z][a-z0-9]+[A-Z][a-zA-Z0-9]+)\b', text))
    # 5. snake_case: at least 2 parts
    found.extend(re.findall(r'\b([a-z]+_[a-z]+(?:_[a-z]+)*)\b', text))

    # Deduplicate preserving order, filter short, skip common English words
    _STOP = {"the", "this", "that", "with", "from", "read", "show", "work",
             "find", "what", "how", "does", "explain", "main", "file", "test",
             "tests", "code", "used", "make", "write", "edit", "name", "line",
             "function", "class", "method", "type", "interface", "return",
             "list", "each", "directory", "structure", "system", "part",
             # Imperative verbs often capitalized at sentence start
             "check", "look", "give", "tell", "call", "open", "save",
             # Monorepo / packaging boilerplate — too generic to be useful
             "packages", "package", "node_modules", "dist", "build", "lib",
             "source", "target", "project", "module", "modules"}
    seen: set[str] = set()
    result: list[str] = []
    for ident in found:
        ident_lc = ident.lower()
        if ident_lc in _STOP:
            continue
        if ident not in seen and len(ident) >= _MIN_IDENT_LEN:
            seen.add(ident)
            result.append(ident)
    return result[:15]


def _compute_mention_scores(
    current_input: str, project_id: int,
) -> dict[str, tuple[float, str, str]]:
    """Boost files/symbols explicitly mentioned in the input."""
    identifiers = _extract_identifiers(current_input)
    if not identifiers:
        return {}

    try:
        from infinidev.code_intel.query import search_symbols
    except ImportError:
        return {}

    result: dict[str, tuple[float, str, str]] = {}
    for ident in identifiers:
        # Split dot notation for sub-searches
        parts = ident.split(".")
        for part in parts:
            if len(part) < _MIN_IDENT_LEN:
                continue
            try:
                symbols = search_symbols(project_id, part, limit=5)
            except Exception:
                continue
            for sym in symbols:
                # Boost the file
                key_file = _normalize_path(sym.file_path)
                if key_file and (key_file not in result or result[key_file][0] < 5.0):
                    result[key_file] = (5.0, "file", f"contains '{ident}' mentioned in input")
                # Boost the symbol
                key_sym = sym.qualified_name or sym.name
                if key_sym and (key_sym not in result or result[key_sym][0] < 5.0):
                    result[key_sym] = (5.0, "symbol", f"'{ident}' mentioned in input")
    return result


# ── Channel 4: Semantic findings match ──────────────────────────────

def _compute_finding_scores(
    cached_embedding: bytes | None,
    current_input: str,
    project_id: int,
) -> dict[str, tuple[float, str, str]]:
    """Score findings by direct semantic similarity to the query."""
    from infinidev.tools.base.embeddings import compute_embedding, embedding_from_blob
    from infinidev.tools.base.dedup import _cosine_similarity

    query_emb = cached_embedding or compute_embedding(current_input)
    if query_emb is None:
        return {}
    query_vec = np.frombuffer(query_emb, dtype=np.float32)

    try:
        def _fetch(conn):
            return conn.execute(
                "SELECT topic, content, embedding FROM findings "
                "WHERE project_id = ? AND embedding IS NOT NULL AND status = 'active' "
                "LIMIT 200",
                (project_id,),
            ).fetchall()
        rows = execute_with_retry(_fetch)
    except Exception:
        return {}

    result: dict[str, tuple[float, str, str]] = {}
    for row in rows:
        try:
            f_vec = embedding_from_blob(row["embedding"])
            sim = float(_cosine_similarity(query_vec, f_vec))
        except Exception:
            continue
        if sim < 0.5:
            continue
        topic = row["topic"]
        score = sim * 3.0
        preview = (row["content"] or "")[:80]
        result[topic] = (score, "finding", f"semantic match (sim={sim:.2f}): {preview}")
    return result


# ── Channel 5: Semantic docstring match (BM25) ─────────────────────

def _compute_docstring_scores(
    current_input: str, project_id: int,
) -> dict[str, tuple[float, str, str]]:
    """Score symbols by docstring/signature relevance via BM25.

    Only considers functions, methods, and classes — variables and
    constants produce too many false positives from BM25 token matching.
    """
    try:
        from infinidev.code_intel.query import search_by_docstring
        matches = search_by_docstring(project_id, current_input, limit=20)
    except Exception:
        return {}

    _RELEVANT_KINDS = {"function", "method", "class", "interface"}
    result: dict[str, tuple[float, str, str]] = {}
    for sym, _bm25_rank in matches:
        kind = sym.kind.value if hasattr(sym.kind, "value") else str(sym.kind)
        if kind not in _RELEVANT_KINDS:
            continue
        # Boost the file
        fp = _normalize_path(sym.file_path)
        if fp:
            existing = result.get(fp, (0.0, "", ""))
            if 3.0 > existing[0]:
                result[fp] = (3.0, "file", f"contains '{sym.name}' with matching docstring")
        # Boost the symbol
        key = sym.qualified_name or sym.name
        if key:
            result[key] = (3.5, "symbol", f"docstring matches query")
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
