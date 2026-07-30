"""Recoverable working memory for the agent loop.

The loop engine rebuilds its prompt from scratch every iteration and keeps
only compact step summaries — raw tool output is dropped so the context
window stays small. That keeps the model fast, but it also means anything
not captured in a ~50-token summary is *gone*: a file listing read three
steps ago, the exact error message from a failed test, the signature the
model looked up and then forgot.

This module makes that eviction **recoverable instead of destructive**.
When a step closes, everything leaving the model's context is archived
here — indexed by embedding — and the model gets a ``recall_context`` tool
to search it back. The public chat transcript is never touched: this is
the model's memory, not the user's.

    step closes
      → summary stays in the prompt   (small, always visible)
      → raw tool output is archived   (searchable, out of the prompt)
      → recall_context(query)         (pulls back exactly what's needed)

Storage is the existing SQLite database (``working_memory`` table).
Embeddings are computed on a background worker so archiving never adds
latency to the loop; a search flushes the queue first so results are
never stale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from infinidev.tools.base.db import execute_with_retry

logger = logging.getLogger(__name__)

# Content shorter than this carries no recall value ("OK", "[]", "done").
MIN_ARCHIVE_CHARS = 60
# Hard cap per entry — recall returns excerpts, not whole files.
MAX_ARCHIVE_CHARS = 8000
# Embedding input is capped separately: MiniLM truncates at 256 tokens
# anyway, so feeding it more is pure cost.
MAX_EMBED_CHARS = 1200
# Relevance floor for recall. Deliberately low: MiniLM scores a natural
# language question against a stack trace or a code listing around
# 0.15–0.40 even when they are about exactly the same thing (measured on
# this repo's own archives), so the 0.82 threshold used for *dedup* would
# reject almost every genuine recall. The asymmetry is intentional — a
# false positive costs a few lines of context, a false negative makes the
# model re-run an expensive command it already ran. Ranking plus a small
# ``limit`` does the real filtering.
MIN_RECALL_SCORE = 0.12


@dataclass(slots=True)
class MemoryRecord:
    """One archived unit of context the model can pull back."""

    id: str
    session_id: str
    step_index: int
    kind: str
    title: str
    content: str
    created_at: str = ""
    score: float = 0.0

    def render(self, max_chars: int = 1500) -> str:
        body = self.content
        if len(body) > max_chars:
            body = body[:max_chars] + f"\n…[{len(self.content) - max_chars} more chars]"
        return f"[step {self.step_index} · {self.kind}] {self.title}\n{body}"


@dataclass(slots=True)
class _PendingEmbed:
    record_id: str
    text: str
    # Bound at enqueue time: the worker is a process-wide singleton, so it
    # must write back to the database the row was inserted into rather than
    # to whatever ``settings.DB_PATH`` happens to say when the batch runs.
    db_path: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_table(db_path: str | None = None) -> None:
    # NOTE: ``execute_with_retry`` does not commit — every writer commits
    # inside its own callback. Without it the rows stay in an open
    # transaction, invisible to the embedding worker's connection (which is
    # a different thread, hence a different SQLite connection).
    def _create(conn):
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS working_memory ("
            " id TEXT PRIMARY KEY,"
            " session_id TEXT NOT NULL,"
            " step_index INTEGER NOT NULL DEFAULT 0,"
            " kind TEXT NOT NULL,"
            " title TEXT NOT NULL,"
            " content TEXT NOT NULL,"
            " content_hash TEXT NOT NULL,"
            " embedding BLOB,"
            " created_at TEXT NOT NULL"
            ");"
            "CREATE INDEX IF NOT EXISTS working_memory_session_idx"
            " ON working_memory(session_id, step_index);"
            "CREATE UNIQUE INDEX IF NOT EXISTS working_memory_dedup_idx"
            " ON working_memory(session_id, content_hash);"
        )
        conn.commit()

    execute_with_retry(_create, db_path=db_path)


class WorkingMemory:
    """Archive of context evicted from the model's prompt, searchable by meaning.

    One instance per session. Thread-safe: the loop archives from the
    worker thread while the UI may read stats from another.
    """

    _embed_queue: "queue.Queue[_PendingEmbed | None]" = queue.Queue()
    _embed_thread: threading.Thread | None = None
    _embed_lock = threading.Lock()
    # Tracks work that is *enqueued or in flight*. The queue going empty is
    # not the same as the embeddings being written: the worker pops a batch
    # (queue now empty) and only then runs the model and the UPDATE. Waiting
    # on the queue alone would let a search run against NULL vectors and
    # silently degrade to keyword scoring.
    _inflight = 0
    _inflight_cv = threading.Condition()

    def __init__(
        self, session_id: str, *, embed: bool = True, db_path: str | None = None
    ) -> None:
        from infinidev.config.settings import settings

        self.session_id = session_id or "default"
        self._embed_enabled = embed
        self._db_path = db_path or settings.DB_PATH
        self._seen_hashes: set[str] = set()
        self._archived = 0
        self._recalled = 0
        try:
            _ensure_table(self._db_path)
            self._ready = True
        except Exception:
            logger.debug("working_memory table unavailable", exc_info=True)
            self._ready = False

    # ── archiving ────────────────────────────────────────────────────

    def archive_step(
        self,
        step_index: int,
        messages: list[dict[str, Any]],
        summary: str = "",
    ) -> int:
        """Archive a finished step's raw exchanges. Returns entries stored.

        Called at the exact moment the loop discards ``messages`` — every
        tool result that mattered goes to disk before the prompt is rebuilt
        without it.
        """
        if not self._ready:
            return 0
        records = list(self._extract(step_index, messages, summary))
        stored = 0
        for record in records:
            if self._store(record):
                stored += 1
        if stored:
            self._archived += stored
            logger.debug(
                "archived %d entries from step %d (session %s)",
                stored,
                step_index,
                self.session_id,
            )
        return stored

    def remember(
        self, title: str, content: str, *, kind: str = "note", step_index: int = 0
    ) -> bool:
        """Archive one explicit entry (used by notes and findings)."""
        if not self._ready:
            return False
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            session_id=self.session_id,
            step_index=step_index,
            kind=kind,
            title=title.strip()[:200],
            content=content[:MAX_ARCHIVE_CHARS],
            created_at=_now(),
        )
        return self._store(record)

    def _extract(
        self, step_index: int, messages: list[dict[str, Any]], summary: str
    ):
        """Turn a step's message list into archivable records.

        Pairs each assistant tool call with its result so a recalled entry
        reads as "what I asked" + "what came back", which is what makes it
        useful three steps later.
        """
        pending: dict[str, str] = {}  # tool_call_id → "tool(args)"
        for message in messages:
            role = message.get("role")
            if role == "assistant":
                for call in message.get("tool_calls", []) or []:
                    function = call.get("function", {}) or {}
                    name = function.get("name", "tool")
                    args = function.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args else {}
                        except (json.JSONDecodeError, TypeError):
                            args = {"raw": args[:120]}
                    label = _format_call(name, args if isinstance(args, dict) else {})
                    call_id = call.get("id") or name
                    pending[call_id] = label
            elif role == "tool":
                content = str(message.get("content", "") or "")
                if len(content) < MIN_ARCHIVE_CHARS:
                    continue
                call_id = message.get("tool_call_id", "")
                title = pending.get(call_id, message.get("name", "tool result"))
                yield MemoryRecord(
                    id=str(uuid.uuid4()),
                    session_id=self.session_id,
                    step_index=step_index,
                    kind="tool_output",
                    title=title[:200],
                    content=content[:MAX_ARCHIVE_CHARS],
                    created_at=_now(),
                )
        if summary and len(summary) >= MIN_ARCHIVE_CHARS:
            yield MemoryRecord(
                id=str(uuid.uuid4()),
                session_id=self.session_id,
                step_index=step_index,
                kind="step_summary",
                title=f"Summary of step {step_index}",
                content=summary[:MAX_ARCHIVE_CHARS],
                created_at=_now(),
            )

    def _store(self, record: MemoryRecord) -> bool:
        digest = hashlib.sha256(record.content.encode("utf-8")).hexdigest()
        if digest in self._seen_hashes:
            return False
        self._seen_hashes.add(digest)

        def _insert(conn):
            conn.execute(
                "INSERT OR IGNORE INTO working_memory ("
                " id, session_id, step_index, kind, title, content,"
                " content_hash, embedding, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                (
                    record.id,
                    record.session_id,
                    record.step_index,
                    record.kind,
                    record.title,
                    record.content,
                    digest,
                    record.created_at or _now(),
                ),
            )
            conn.commit()
            return conn.total_changes

        try:
            execute_with_retry(_insert, db_path=self._db_path)
        except Exception:
            logger.debug("working_memory insert failed", exc_info=True)
            return False
        if self._embed_enabled:
            self._enqueue_embed(record)
        return True

    # ── embedding worker ─────────────────────────────────────────────

    def _enqueue_embed(self, record: MemoryRecord) -> None:
        text = f"{record.title}\n{record.content[:MAX_EMBED_CHARS]}"
        with WorkingMemory._inflight_cv:
            WorkingMemory._inflight += 1
        WorkingMemory._embed_queue.put(
            _PendingEmbed(record.id, text, self._db_path)
        )
        self._ensure_embed_worker()

    @classmethod
    def _settle(cls, count: int) -> None:
        """Mark *count* embed jobs as finished and wake any waiting search."""
        if count <= 0:
            return
        with cls._inflight_cv:
            cls._inflight = max(0, cls._inflight - count)
            if cls._inflight == 0:
                cls._inflight_cv.notify_all()

    @classmethod
    def _ensure_embed_worker(cls) -> None:
        with cls._embed_lock:
            if cls._embed_thread is not None and cls._embed_thread.is_alive():
                return
            cls._embed_thread = threading.Thread(
                target=cls._embed_loop, name="working-memory-embed", daemon=True
            )
            cls._embed_thread.start()

    @classmethod
    def _embed_loop(cls) -> None:
        """Drain the queue in batches — embedding is far cheaper batched."""
        while True:
            try:
                first = cls._embed_queue.get(timeout=30)
            except queue.Empty:
                return  # idle: let the thread die, it respawns on demand
            if first is None:
                cls._embed_queue.task_done()
                return
            batch = [first]
            while len(batch) < 16:
                try:
                    item = cls._embed_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    cls._embed_queue.task_done()
                    break
                batch.append(item)
            try:
                cls._embed_batch(batch)
            except Exception:
                logger.debug("embedding batch failed", exc_info=True)
            finally:
                for _ in batch:
                    cls._embed_queue.task_done()
                cls._settle(len(batch))

    @staticmethod
    def _embed_batch(batch: list[_PendingEmbed]) -> None:
        from infinidev.tools.base.dedup import _get_embed_fn

        vectors = _get_embed_fn()([item.text for item in batch])
        # One UPDATE per database: a batch can mix sessions, and (in tests
        # or after a workspace switch) those sessions can live in different
        # database files.
        by_db: dict[str, list[tuple[str, Any]]] = {}
        for item, vector in zip(batch, vectors):
            blob = np.asarray(vector, dtype=np.float32).tobytes()
            by_db.setdefault(item.db_path, []).append((blob, item.record_id))

        for db_path, updates in by_db.items():

            def _update(conn, _updates=updates):
                conn.executemany(
                    "UPDATE working_memory SET embedding = ? WHERE id = ?", _updates
                )
                conn.commit()

            execute_with_retry(_update, db_path=db_path)

    @classmethod
    def flush(cls, timeout: float = 5.0) -> bool:
        """Block until every queued embedding is written, or *timeout* elapses.

        Returns whether the archive is fully indexed. A ``False`` here is not
        an error — the search simply falls back to keyword scoring for the
        entries that have not been vectorised yet.
        """
        deadline = time.monotonic() + timeout
        with cls._inflight_cv:
            while cls._inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                cls._inflight_cv.wait(remaining)
        return True

    # ── recall ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 5,
        *,
        all_sessions: bool = False,
        min_score: float = MIN_RECALL_SCORE,
    ) -> list[MemoryRecord]:
        """Return archived entries most relevant to *query*.

        Semantic when embeddings are available, keyword-scored otherwise;
        either way the caller gets records, never an exception.
        """
        if not self._ready or not query.strip():
            return []
        # Generous: the first call in a process pays for loading MiniLM.
        # Missing the window is survivable (keyword scoring), so this is a
        # ceiling, not a typical cost.
        self.flush(timeout=20.0)
        rows = self._load_rows(all_sessions=all_sessions)
        if not rows:
            return []
        scored = self._score_semantic(query, rows)
        if scored is None:
            scored = _score_keyword(query, rows)
        results = [record for record in scored if record.score >= min_score]
        results.sort(key=lambda record: record.score, reverse=True)
        self._recalled += len(results[:limit])
        return results[:limit]

    def _load_rows(self, *, all_sessions: bool) -> list[tuple]:
        def _select(conn):
            if all_sessions:
                return conn.execute(
                    "SELECT id, session_id, step_index, kind, title, content,"
                    " embedding, created_at FROM working_memory"
                    " ORDER BY created_at DESC LIMIT 2000"
                ).fetchall()
            return conn.execute(
                "SELECT id, session_id, step_index, kind, title, content,"
                " embedding, created_at FROM working_memory"
                " WHERE session_id = ? ORDER BY step_index DESC LIMIT 2000",
                (self.session_id,),
            ).fetchall()

        try:
            return execute_with_retry(_select, db_path=self._db_path) or []
        except Exception:
            logger.debug("working_memory select failed", exc_info=True)
            return []

    def _score_semantic(self, query: str, rows: list[tuple]) -> list[MemoryRecord] | None:
        """Cosine-rank rows that have embeddings. ``None`` = not possible.

        Rows still waiting on the embedder are keyword-scored and damped
        rather than dropped: an entry archived seconds ago is usually the
        most relevant one, and silently omitting it would look like the
        archive lost it.
        """
        embedded = [row for row in rows if row[6]]
        if not embedded:
            return None
        try:
            from infinidev.tools.base.dedup import _get_embed_fn

            query_vec = np.asarray(_get_embed_fn()([query])[0], dtype=np.float32)
        except Exception:
            logger.debug("query embedding failed; falling back to keywords")
            return None
        norm = float(np.linalg.norm(query_vec))
        if norm == 0:
            return None
        records: list[MemoryRecord] = []
        for row in embedded:
            vector = np.frombuffer(row[6], dtype=np.float32)
            if vector.size != query_vec.size:
                continue
            denominator = norm * float(np.linalg.norm(vector))
            score = 0.0 if denominator == 0 else float(query_vec @ vector) / denominator
            records.append(_row_to_record(row, score))
        unembedded = [row for row in rows if not row[6]]
        if unembedded:
            for record in _score_keyword(query, unembedded):
                record.score *= 0.5
                records.append(record)
        return records or None

    # ── introspection ────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        def _count(conn):
            return conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(content)), 0)"
                " FROM working_memory WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()

        try:
            total, chars = execute_with_retry(_count, db_path=self._db_path) or (0, 0)
        except Exception:
            total, chars = 0, 0
        return {
            "session_id": self.session_id,
            "entries": int(total or 0),
            "archived_this_run": self._archived,
            "recalled_this_run": self._recalled,
            "approx_tokens_offloaded": int((chars or 0) / 4),
        }

    def clear(self) -> int:
        """Drop this session's archive (used when a session is reset)."""

        def _delete(conn):
            cursor = conn.execute(
                "DELETE FROM working_memory WHERE session_id = ?", (self.session_id,)
            )
            conn.commit()
            return cursor.rowcount

        try:
            removed = execute_with_retry(_delete, db_path=self._db_path) or 0
        except Exception:
            return 0
        self._seen_hashes.clear()
        return int(removed)


def _row_to_record(row: tuple, score: float) -> MemoryRecord:
    return MemoryRecord(
        id=row[0],
        session_id=row[1],
        step_index=int(row[2] or 0),
        kind=row[3],
        title=row[4],
        content=row[5],
        created_at=row[7],
        score=score,
    )


def _score_keyword(query: str, rows: list[tuple]) -> list[MemoryRecord]:
    """Token-overlap ranking, normalised to roughly the cosine range."""
    tokens = {token for token in query.lower().split() if len(token) > 2}
    if not tokens:
        return []
    records: list[MemoryRecord] = []
    for row in rows:
        haystack = f"{row[4]}\n{row[5]}".lower()
        hits = sum(1 for token in tokens if token in haystack)
        if not hits:
            continue
        records.append(_row_to_record(row, hits / len(tokens)))
    return records


# ── per-session registry ──────────────────────────────────────────────────

_memories: dict[str, WorkingMemory] = {}
_registry_lock = threading.Lock()


def get_working_memory(session_id: str | None) -> WorkingMemory:
    """Return (creating if needed) the archive for *session_id*."""
    key = session_id or "default"
    with _registry_lock:
        memory = _memories.get(key)
        if memory is None:
            memory = WorkingMemory(key)
            _memories[key] = memory
        return memory


def reset_working_memory(session_id: str | None = None) -> None:
    """Forget cached instances — used by tests and session switches."""
    with _registry_lock:
        if session_id is None:
            _memories.clear()
        else:
            _memories.pop(session_id or "default", None)


def _format_call(name: str, args: dict[str, Any]) -> str:
    """Render a tool call as a short, searchable label."""
    interesting = ("file_path", "path", "pattern", "query", "command", "name")
    parts = []
    for key in interesting:
        value = args.get(key)
        if value:
            parts.append(f"{key}={str(value)[:80]}")
    return f"{name}({', '.join(parts)})" if parts else f"{name}()"
