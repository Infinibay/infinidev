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
import sqlite3
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
# Embedding input is capped separately so long command output does not dilute
# its title and leading evidence in the additive static representation.
MAX_EMBED_CHARS = 1200
# Relevance floor for recall. Deliberately low: dense embeddings score a natural
# language question against a stack trace or a code listing around
# 0.15–0.40 even when they are about exactly the same thing (measured on
# this repo's own archives), so the 0.82 threshold used for *dedup* would
# reject almost every genuine recall. The asymmetry is intentional — a
# false positive costs a few lines of context, a false negative makes the
# model re-run an expensive command it already ran. Ranking plus a small
# ``limit`` does the real filtering.
MIN_RECALL_SCORE = 0.12

# Traceable notes live inside ``working_memory.content`` as versioned JSON.
# Keeping the envelope in the existing private archive avoids a canonical DB
# migration while still giving every occurrence a stable identity.
TRACEABLE_NOTE_SCHEMA = "infinidev.traceable_note"
TRACEABLE_NOTE_VERSION = 2
_SUPPORTED_TRACEABLE_NOTE_VERSIONS = frozenset({1, TRACEABLE_NOTE_VERSION})
TRACEABLE_NOTE_TYPES = frozenset({"auto_note", "artifact_analysis"})
CLAIM_CLASSIFICATIONS = frozenset({
    "observation", "inference", "recommendation", "requirement", "analysis",
})
MAX_NOTE_PARENTS = 16
MAX_NOTE_CITATIONS = 64
MAX_NOTE_GENERATION = 4
MAX_NOTE_SUMMARY_CHARS = 4000
MAX_NOTE_ID_CHARS = 200


class TraceableNoteError(ValueError):
    """A traceable note is malformed or exceeds a provenance limit."""


@dataclass(frozen=True, slots=True)
class NoteCitation:
    """Structured pointer to one source occurrence, never copied source text."""

    occurrence_id: str
    source_artifact_id: int | None
    step_index: int
    tool_call_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence_id": self.occurrence_id,
            "source_artifact_id": self.source_artifact_id,
            "step_index": self.step_index,
            "tool_call_id": self.tool_call_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NoteCitation":
        return cls(
            occurrence_id=_validate_note_id(value.get("occurrence_id")),
            source_artifact_id=_validate_artifact_id(value.get("source_artifact_id")),
            step_index=_validate_step_index(value.get("step_index")),
            tool_call_id=_validate_tool_call_id(value.get("tool_call_id")),
        )


@dataclass(frozen=True, slots=True)
class TraceableNoteEnvelope:
    """Immutable claim with source, evidence, confidence, and validity state."""

    note_type: str
    occurrence_id: str
    source_artifact_id: int | None
    step_index: int
    tool_call_id: str | None
    generation: int
    parent_ids: tuple[str, ...]
    summary: str
    citations: tuple[NoteCitation, ...]
    claim_classification: str = "observation"
    source: str = "working_memory"
    confidence: float | None = None
    still_valid: bool | None = None
    schema: str = TRACEABLE_NOTE_SCHEMA
    version: int = TRACEABLE_NOTE_VERSION

    def to_dict(self) -> dict[str, Any]:
        citations = [citation.to_dict() for citation in self.citations]
        provenance = {
            "occurrence_id": self.occurrence_id,
            "source_artifact_id": self.source_artifact_id,
            "step_index": self.step_index,
            "tool_call_id": self.tool_call_id,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
        }
        return {
            "schema": self.schema,
            "version": self.version,
            "type": self.note_type,
            "occurrence_id": self.occurrence_id,
            "source_artifact_id": self.source_artifact_id,
            "step_index": self.step_index,
            "tool_call_id": self.tool_call_id,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "summary": self.summary,
            "citations": citations,
            "claim": {
                "text": self.summary,
                "classification": self.claim_classification,
                "source": self.source,
                "evidence": citations,
                "confidence": self.confidence,
                "provenance": provenance,
                "still_valid": self.still_valid,
            },
        }

    def to_json(self) -> str:
        """Return a canonical representation while retaining list order."""
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, content: str) -> "TraceableNoteEnvelope":
        try:
            value = json.loads(content)
        except (json.JSONDecodeError, TypeError) as err:
            raise TraceableNoteError("traceable note is not valid JSON") from err
        if not isinstance(value, dict):
            raise TraceableNoteError("traceable note must be a JSON object")
        if value.get("schema") != TRACEABLE_NOTE_SCHEMA:
            raise TraceableNoteError("unknown traceable note schema")
        version = value.get("version")
        if version not in _SUPPORTED_TRACEABLE_NOTE_VERSIONS:
            raise TraceableNoteError("unsupported traceable note version")
        note_type = value.get("type")
        if note_type not in TRACEABLE_NOTE_TYPES:
            raise TraceableNoteError("unsupported traceable note type")
        parent_values = value.get("parent_ids", [])
        citation_values = value.get("citations", [])
        if not isinstance(parent_values, list) or not isinstance(citation_values, list):
            raise TraceableNoteError("parents and citations must be arrays")
        if len(parent_values) > MAX_NOTE_PARENTS:
            raise TraceableNoteError("traceable note has too many parents")
        if len(citation_values) > MAX_NOTE_CITATIONS:
            raise TraceableNoteError("traceable note has too many citations")
        try:
            citations = tuple(
                NoteCitation.from_dict(item) for item in citation_values
                if isinstance(item, dict)
            )
        except (TypeError, ValueError) as err:
            raise TraceableNoteError("invalid traceable note citation") from err
        if len(citations) != len(citation_values):
            raise TraceableNoteError("traceable note citation must be an object")
        generation = _validate_generation(value.get("generation"))
        summary = _redact_note_summary(value.get("summary"))
        claim = value.get("claim") if version >= 2 else {}
        if not isinstance(claim, dict):
            raise TraceableNoteError("traceable note claim must be an object")
        classification = _validate_claim_classification(
            claim.get("classification", "observation")
        )
        source = _validate_claim_source(claim.get("source", "working_memory"))
        confidence = _validate_claim_confidence(claim.get("confidence"))
        still_valid = _validate_still_valid(claim.get("still_valid"))
        return cls(
            note_type=note_type,
            occurrence_id=_validate_note_id(value.get("occurrence_id")),
            source_artifact_id=_validate_artifact_id(value.get("source_artifact_id")),
            step_index=_validate_step_index(value.get("step_index")),
            tool_call_id=_validate_tool_call_id(value.get("tool_call_id")),
            generation=generation,
            parent_ids=tuple(_validate_note_id(item) for item in parent_values),
            summary=summary,
            citations=citations,
            claim_classification=classification,
            source=source,
            confidence=confidence,
            still_valid=still_valid,
        )


def _validate_note_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TraceableNoteError("traceable note identity must be non-empty")
    value = value.strip()
    if len(value) > MAX_NOTE_ID_CHARS:
        raise TraceableNoteError("traceable note identity is too long")
    return value


def _validate_artifact_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TraceableNoteError("source_artifact_id must be a positive integer")
    return value


def _validate_step_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TraceableNoteError("step_index must be a non-negative integer")
    return value


def _validate_tool_call_id(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TraceableNoteError("tool_call_id must be a non-empty string")
    value = value.strip()
    if len(value) > MAX_NOTE_ID_CHARS:
        raise TraceableNoteError("tool_call_id is too long")
    return value


def _validate_generation(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceableNoteError("generation must be an integer")
    if value < 0 or value > MAX_NOTE_GENERATION:
        raise TraceableNoteError("traceable note generation is out of range")
    return value


def _validate_claim_classification(value: Any) -> str:
    if value not in CLAIM_CLASSIFICATIONS:
        raise TraceableNoteError("unsupported claim classification")
    return str(value)


def _validate_claim_source(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TraceableNoteError("claim source must be non-empty")
    return value.strip()[:200]


def _validate_claim_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceableNoteError("claim confidence must be numeric or null")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise TraceableNoteError("claim confidence must be between 0 and 1")
    return confidence


def _validate_still_valid(value: Any) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise TraceableNoteError("still_valid must be boolean or null")
    return value


def _redact_note_summary(summary: str) -> str:
    if not isinstance(summary, str) or not summary.strip():
        raise TraceableNoteError("traceable note summary must be non-empty")
    from infinidev.config.secrets import redact

    redacted = redact(summary.strip())
    if len(redacted) > MAX_NOTE_SUMMARY_CHARS:
        raise TraceableNoteError("traceable note summary is too long")
    return redacted


def create_traceable_note(
    note_type: str,
    summary: str,
    *,
    source_artifact_id: int | None = None,
    step_index: int = 0,
    tool_call_id: str | None = None,
    occurrence_id: str | None = None,
    citations: tuple[NoteCitation, ...] | list[NoteCitation] | None = None,
    claim_classification: str = "observation",
    source: str = "working_memory",
    confidence: float | None = None,
    still_valid: bool | None = None,
) -> TraceableNoteEnvelope:
    """Create one source occurrence without conflating equal summaries."""
    if note_type not in TRACEABLE_NOTE_TYPES:
        raise TraceableNoteError("unsupported traceable note type")
    note_id = _validate_note_id(occurrence_id or f"note:{uuid.uuid4()}")
    artifact_id = _validate_artifact_id(source_artifact_id)
    step = _validate_step_index(step_index)
    call_id = _validate_tool_call_id(tool_call_id)
    note_citations = tuple(citations or ())
    if len(note_citations) > MAX_NOTE_CITATIONS:
        raise TraceableNoteError("traceable note has too many citations")
    if not all(isinstance(item, NoteCitation) for item in note_citations):
        raise TraceableNoteError("citations must contain NoteCitation values")
    if not note_citations:
        note_citations = (
            NoteCitation(note_id, artifact_id, step, call_id),
        )
    envelope = TraceableNoteEnvelope(
        note_type=note_type,
        occurrence_id=note_id,
        source_artifact_id=artifact_id,
        step_index=step,
        tool_call_id=call_id,
        generation=0,
        parent_ids=(),
        summary=_redact_note_summary(summary),
        citations=note_citations,
        claim_classification=_validate_claim_classification(claim_classification),
        source=_validate_claim_source(source),
        confidence=_validate_claim_confidence(confidence),
        still_valid=_validate_still_valid(still_valid),
    )
    if len(envelope.to_json()) > MAX_ARCHIVE_CHARS:
        raise TraceableNoteError("traceable note envelope is too large")
    return envelope


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
        """Render content with enough provenance to judge its authority and age."""

        body = self.content
        if len(body) > max_chars:
            body = body[:max_chars] + f"\n…[{len(self.content) - max_chars} more chars]"
        provenance = (
            f"source=working-memory record={self.id} session={self.session_id} "
            f"step={self.step_index} kind={self.kind} "
            f"created={self.created_at or 'unknown'} authority=advisory"
        )
        return f"[{provenance}]\n{self.title}\n{body}"


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
            " embedding_space TEXT,"
            " created_at TEXT NOT NULL"
            ");"
            "CREATE INDEX IF NOT EXISTS working_memory_session_idx"
            " ON working_memory(session_id, step_index);"
            "CREATE UNIQUE INDEX IF NOT EXISTS working_memory_dedup_idx"
            " ON working_memory(session_id, content_hash);"
        )
        try:
            conn.execute("ALTER TABLE working_memory ADD COLUMN embedding_space TEXT")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
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
        self._store_lock = threading.Lock()
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
    ) -> list[str]:
        """Archive a finished step's raw exchanges. Returns the titles stored.

        Called at the exact moment the loop discards ``messages`` — every
        tool result that mattered goes to disk before the prompt is rebuilt
        without it.

        The titles, not a count, because they are what a caller can *use*: each
        one is the label a row was filed under, which makes it the query that
        pulls that row back through ``recall_context``. The plan block renders
        them so a closed step points at its own evidence.
        """
        if not self._ready:
            return []
        records = list(self._extract(step_index, messages, summary))
        titles = [record.title for record in records if self._store(record)]
        if titles:
            self._archived += len(titles)
            logger.debug(
                "archived %d entries from step %d (session %s)",
                len(titles),
                step_index,
                self.session_id,
            )
        return titles

    def archive_calls(
        self, step_index: int, calls: list[tuple[str, str, str]]
    ) -> list[str]:
        """Archive ``(name, arguments, result)`` triples caught at the source.

        ``archive_step`` reconstructs exchanges by pairing assistant tool
        calls with their ``role: "tool"`` results, which only exists in
        function-calling mode and only until ``compact_for_small`` rewrites
        it. This path takes the body the tool actually returned, so what
        gets archived does not depend on which tool-calling mode the model
        supports or how aggressively its context was compacted.

        Deduplication happens in ``_store`` by content hash, so calling
        this alongside ``archive_step`` stores each exchange once. Returns the
        titles stored, for the same reason ``archive_step`` does.
        """
        if not self._ready:
            return []
        titles: list[str] = []
        for name, arguments, body in calls:
            if len(body) < MIN_ARCHIVE_CHARS:
                continue
            try:
                args = json.loads(arguments) if arguments else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            title = _format_call(name, args if isinstance(args, dict) else {})
            if self.remember(
                title, body, kind="tool_output", step_index=step_index,
            ):
                titles.append(title)
        return titles

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

    def remember_traceable(self, note: TraceableNoteEnvelope) -> bool:
        """Store a validated note envelope without changing legacy ``remember``."""
        if not self._ready:
            return False
        # Round-trip validation prevents callers from constructing an invalid
        # dataclass directly and bypassing the version/limit checks.
        validated = TraceableNoteEnvelope.from_json(note.to_json())
        content = validated.to_json()
        if len(content) > MAX_ARCHIVE_CHARS:
            raise TraceableNoteError("traceable note envelope is too large")
        return self._store(MemoryRecord(
            id=validated.occurrence_id,
            session_id=self.session_id,
            step_index=validated.step_index,
            kind=validated.note_type,
            title=f"{validated.note_type}:{validated.occurrence_id}"[:200],
            content=content,
            created_at=_now(),
        ))

    def load_traceable_notes(
        self, *, kinds: tuple[str, ...] = ("auto_note", "artifact_analysis")
    ) -> list[TraceableNoteEnvelope]:
        """Load source and compacted notes in immutable creation order."""
        selected = tuple(kind for kind in kinds if kind in TRACEABLE_NOTE_TYPES)
        if not self._ready or not selected:
            return []
        placeholders = ",".join("?" for _ in selected)

        def _select(conn):
            return conn.execute(
                f"SELECT content FROM working_memory WHERE session_id = ? "
                f"AND kind IN ({placeholders}) ORDER BY created_at, rowid",
                (self.session_id, *selected),
            ).fetchall()

        try:
            rows = execute_with_retry(_select, db_path=self._db_path) or []
        except Exception:
            logger.debug("traceable note select failed", exc_info=True)
            return []
        notes: list[TraceableNoteEnvelope] = []
        for (content,) in rows:
            try:
                notes.append(TraceableNoteEnvelope.from_json(content))
            except TraceableNoteError:
                logger.debug("ignored invalid traceable note %s", content[:80])
        return notes

    def compact_traceable_notes(
        self,
        sources: list[TraceableNoteEnvelope] | tuple[TraceableNoteEnvelope, ...],
        summary: str,
        *,
        step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> TraceableNoteEnvelope:
        """Create or return one deterministic immutable analysis.

        ``sources`` is intentionally ordered: parent order and citation order
        survive compaction. Source rows are never updated or deleted. Repeating
        the same ordered compaction derives the same occurrence id, so the DB
        unique constraint makes the operation idempotent.
        """
        if not sources:
            raise TraceableNoteError("compaction requires at least one source")
        if len(sources) > MAX_NOTE_PARENTS:
            raise TraceableNoteError("compaction has too many parents")
        validated = tuple(
            TraceableNoteEnvelope.from_json(source.to_json()) for source in sources
        )
        parent_ids = tuple(source.occurrence_id for source in validated)
        if len(set(parent_ids)) != len(parent_ids):
            raise TraceableNoteError("compaction sources must be distinct occurrences")
        generation = max(source.generation for source in validated) + 1
        _validate_generation(generation)
        citations = _merge_citations(validated)
        compacted_step = (
            max(source.step_index for source in validated)
            if step_index is None else _validate_step_index(step_index)
        )
        compacted_call = _validate_tool_call_id(tool_call_id)
        redacted_summary = _redact_note_summary(summary)
        identity_payload = json.dumps(
            {
                "schema": TRACEABLE_NOTE_SCHEMA,
                "version": TRACEABLE_NOTE_VERSION,
                "session_id": self.session_id,
                "type": "artifact_analysis",
                "parents": parent_ids,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        occurrence_id = "analysis:" + hashlib.sha256(
            identity_payload.encode("utf-8")
        ).hexdigest()
        # Same ordered sources mean the same occurrence, independent of a
        # caller retry producing slightly different prose or metadata. The
        # first immutable row wins; subsequent calls return that exact row.
        existing = next(
            (
                item for item in self.load_traceable_notes(
                    kinds=("artifact_analysis",)
                )
                if item.occurrence_id == occurrence_id
            ),
            None,
        )
        if existing is not None:
            return existing
        note = TraceableNoteEnvelope(
            note_type="artifact_analysis",
            occurrence_id=occurrence_id,
            source_artifact_id=None,
            step_index=compacted_step,
            tool_call_id=compacted_call,
            generation=generation,
            parent_ids=parent_ids,
            summary=redacted_summary,
            citations=citations,
            claim_classification="analysis",
            source="derived_compaction",
            confidence=(
                min(item.confidence for item in validated if item.confidence is not None)
                if any(item.confidence is not None for item in validated)
                else None
            ),
            still_valid=(
                False if any(item.still_valid is False for item in validated)
                else True if all(item.still_valid is True for item in validated)
                else None
            ),
        )
        if len(note.to_json()) > MAX_ARCHIVE_CHARS:
            raise TraceableNoteError("compacted note envelope is too large")
        if self.remember_traceable(note):
            return note
        # A concurrent compactor may have inserted this deterministic id after
        # the lookup above. Return the immutable winner, not an unstored local
        # variant; an actual storage failure stays explicit to the caller.
        existing = next(
            (
                item for item in self.load_traceable_notes(
                    kinds=("artifact_analysis",)
                )
                if item.occurrence_id == occurrence_id
            ),
            None,
        )
        if existing is None:
            raise TraceableNoteError("compacted note could not be persisted")
        return existing

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

        def _insert(conn):
            cursor = conn.execute(
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
            return cursor.rowcount == 1

        # Serialise the in-process cache check with the DB insert. Crucially,
        # a failed INSERT does not poison the cache: a later retry of the same
        # content still reaches SQLite.
        with self._store_lock:
            if digest in self._seen_hashes:
                return False
            try:
                inserted = bool(execute_with_retry(_insert, db_path=self._db_path))
            except Exception:
                logger.debug("working_memory insert failed", exc_info=True)
                return False
            if not inserted:
                return False
            self._seen_hashes.add(digest)
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
        from infinidev.tools.base.embeddings import (
            current_embedding_space,
            embed_passages,
        )

        vectors = embed_passages([item.text for item in batch])
        embedding_space = current_embedding_space()
        # One UPDATE per database: a batch can mix sessions, and (in tests
        # or after a workspace switch) those sessions can live in different
        # database files.
        by_db: dict[str, list[tuple[bytes, str, str]]] = {}
        for item, vector in zip(batch, vectors):
            blob = np.asarray(vector, dtype=np.float32).tobytes()
            by_db.setdefault(item.db_path, []).append(
                (blob, embedding_space, item.record_id)
            )

        for db_path, updates in by_db.items():

            def _update(conn, _updates=updates):
                conn.executemany(
                    "UPDATE working_memory "
                    "SET embedding = ?, embedding_space = ? WHERE id = ?",
                    _updates,
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
        # Generous: the first call in a process pays for loading the embedding table.
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
                    " embedding, embedding_space, created_at FROM working_memory"
                    " ORDER BY created_at DESC LIMIT 2000"
                ).fetchall()
            return conn.execute(
                "SELECT id, session_id, step_index, kind, title, content,"
                " embedding, embedding_space, created_at FROM working_memory"
                " WHERE session_id = ? ORDER BY step_index DESC LIMIT 2000",
                (self.session_id,),
            ).fetchall()

        try:
            return execute_with_retry(_select, db_path=self._db_path) or []
        except Exception:
            logger.debug("working_memory select failed", exc_info=True)
            return []

    def recent_records(
        self,
        *,
        limit: int = 20,
        kinds: set[str] | None = None,
    ) -> list[MemoryRecord]:
        """Return recent records with archive provenance, newest first."""

        bounded = max(1, min(int(limit), 200))
        records = [_row_to_record(row, 0.0) for row in self._load_rows(all_sessions=False)]
        if kinds is not None:
            records = [record for record in records if record.kind in kinds]
        return records[:bounded]

    def _score_semantic(self, query: str, rows: list[tuple]) -> list[MemoryRecord] | None:
        """Cosine-rank rows that have embeddings. ``None`` = not possible.

        Rows still waiting on the embedder are keyword-scored and damped
        rather than dropped: an entry archived seconds ago is usually the
        most relevant one, and silently omitting it would look like the
        archive lost it.
        """
        try:
            from infinidev.tools.base.embeddings import current_embedding_space

            embedding_space = current_embedding_space()
        except Exception:
            logger.debug("embedding-space identity failed; falling back to keywords")
            return None
        embedded = [row for row in rows if row[6] and row[7] == embedding_space]
        if not embedded:
            return None
        try:
            from infinidev.tools.base.embeddings import embed_queries

            query_vec = np.asarray(embed_queries([query])[0], dtype=np.float32)
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
        embedded_ids = {row[0] for row in embedded}
        unembedded = [row for row in rows if row[0] not in embedded_ids]
        if unembedded:
            for record in _score_keyword(query, unembedded):
                record.score *= 0.5
                records.append(record)
        # Exact lexical evidence remains valuable even when an embedding is
        # present. Use the stronger independent signal instead of allowing a
        # weak semantic cosine to hide a literal error, command, or symbol.
        by_id = {record.id: record for record in records}
        for lexical in _score_keyword(query, rows):
            existing = by_id.get(lexical.id)
            if existing is None:
                records.append(lexical)
                by_id[lexical.id] = lexical
            else:
                existing.score = max(existing.score, lexical.score)
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


def _merge_citations(
    sources: tuple[TraceableNoteEnvelope, ...],
) -> tuple[NoteCitation, ...]:
    """Flatten citations in source order, keeping first occurrence identity."""
    merged: list[NoteCitation] = []
    seen: set[tuple[str, int | None, int, str | None]] = set()
    for source in sources:
        for citation in source.citations:
            key = (
                citation.occurrence_id,
                citation.source_artifact_id,
                citation.step_index,
                citation.tool_call_id,
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(citation)
            if len(merged) > MAX_NOTE_CITATIONS:
                raise TraceableNoteError("compaction has too many citations")
    return tuple(merged)


def _row_to_record(row: tuple, score: float) -> MemoryRecord:
    return MemoryRecord(
        id=row[0],
        session_id=row[1],
        step_index=int(row[2] or 0),
        kind=row[3],
        title=row[4],
        content=row[5],
        created_at=row[8],
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
