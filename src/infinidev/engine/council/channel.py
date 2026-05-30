"""The council's shared channel — a thread-safe blackboard of threads.

Subagents never message each other directly. They post to **threads**
on a single shared :class:`Channel`. Each round, every member reads a
rendered *digest* of the channel (the frozen snapshot from the previous
round) and contributes at most one post. The runner collects those
posts and commits them atomically at the round boundary — so members
running concurrently within a round never see each other's in-flight
writes, which keeps the debate deterministic and race-free.

The channel is the single source of truth for the debate. Members are
stateless across rounds: everything they "remember" is re-derived from
:meth:`Channel.render_digest`. That is what makes the barrier-per-round
model trivial — there is no per-member history to reconcile.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Message:
    """A single post on the channel.

    Frozen because, once committed, a post is immutable history. Edits
    happen by posting a new message (a reply), never by mutation.
    """

    id: str
    author: str            # member_id, or "moderator"
    thread_id: str
    round: int
    text: str
    parent_id: str | None = None       # reply-to within the thread
    refs: list[str] = field(default_factory=list)  # files/symbols cited


@dataclass
class Thread:
    """A topic on the channel. Ordered list of messages."""

    id: str
    title: str
    opened_by: str
    messages: list[Message] = field(default_factory=list)


class Channel:
    """Thread-safe blackboard of deliberation threads.

    Mutation (``open_thread`` / ``post``) is guarded by a lock so the
    runner can commit a round's collected posts without racing. Reads
    (``render_digest``) take a consistent snapshot under the same lock.
    """

    def __init__(self, question: str) -> None:
        self.question = question
        self._threads: dict[str, Thread] = {}
        self._thread_order: list[str] = []
        self._seq = 0  # monotonic id counter (no wall-clock — keeps runs reproducible)
        self._lock = threading.Lock()

    # ── Mutation ─────────────────────────────────────────────────────────

    def open_thread(
        self, *, author: str, title: str, opening_text: str, round: int,
        refs: list[str] | None = None,
    ) -> Thread:
        """Open a new thread and seed it with one opening message."""
        with self._lock:
            tid = self._next_id("t")
            thread = Thread(id=tid, title=title.strip() or "(untitled)", opened_by=author)
            mid = self._next_id("m")
            thread.messages.append(Message(
                id=mid, author=author, thread_id=tid, round=round,
                text=opening_text.strip(), refs=list(refs or []),
            ))
            self._threads[tid] = thread
            self._thread_order.append(tid)
            return thread

    def post(
        self, *, author: str, thread_id: str, text: str, round: int,
        parent_id: str | None = None, refs: list[str] | None = None,
    ) -> Message | None:
        """Append a message to an existing thread.

        Returns the committed :class:`Message`, or ``None`` if the
        thread id does not exist (a member referenced a stale/invented
        thread — we drop the post rather than crash the round).
        """
        with self._lock:
            thread = self._threads.get(thread_id)
            if thread is None:
                return None
            mid = self._next_id("m")
            msg = Message(
                id=mid, author=author, thread_id=thread_id, round=round,
                text=text.strip(), parent_id=parent_id, refs=list(refs or []),
            )
            thread.messages.append(msg)
            return msg

    def _next_id(self, prefix: str) -> str:
        # Caller already holds the lock.
        self._seq += 1
        return f"{prefix}-{self._seq}"

    # ── Inspection ───────────────────────────────────────────────────────

    @property
    def thread_titles(self) -> list[tuple[str, str]]:
        with self._lock:
            return [(tid, self._threads[tid].title) for tid in self._thread_order]

    def all_messages(self) -> list[Message]:
        with self._lock:
            out: list[Message] = []
            for tid in self._thread_order:
                out.extend(self._threads[tid].messages)
            return out

    def snapshot(self) -> "Channel":
        """Return a deep-ish frozen copy for safe concurrent reads.

        Messages are already frozen; we only need to copy the thread
        containers so a concurrent commit can't append under a reader's
        feet. Used to hand every member in a round the *same* immutable
        view of the previous round.
        """
        with self._lock:
            clone = Channel(self.question)
            clone._seq = self._seq
            clone._thread_order = list(self._thread_order)
            for tid in self._thread_order:
                t = self._threads[tid]
                clone._threads[tid] = Thread(
                    id=t.id, title=t.title, opened_by=t.opened_by,
                    messages=list(t.messages),
                )
            return clone

    # ── Rendering ────────────────────────────────────────────────────────

    def render_digest(
        self, *, for_author: str | None = None, recent_rounds: int = 2,
        current_round: int = 0,
    ) -> str:
        """Render the channel as text for an agent's prompt.

        * ``for_author`` — when given, that author's own posts are
          tagged ``[YOU said]`` so the model can tell its contributions
          apart from others'.
        * ``recent_rounds`` — messages from rounds older than
          ``current_round - recent_rounds`` are summarised to one line
          per thread instead of quoted in full, keeping the prompt
          bounded as the debate grows.

        The format is plain, thread-grouped text — deliberately simple
        so small local models parse it reliably.
        """
        with self._lock:
            if not self._thread_order:
                return "(the channel is empty — no threads opened yet)"
            cutoff = current_round - recent_rounds
            lines: list[str] = [f"CHANNEL — question under debate:\n  {self.question}", ""]
            for tid in self._thread_order:
                t = self._threads[tid]
                lines.append(f"### Thread {t.id}: {t.title}")
                old = [m for m in t.messages if m.round < cutoff]
                recent = [m for m in t.messages if m.round >= cutoff]
                if old:
                    authors = sorted({m.author for m in old})
                    lines.append(
                        f"  …[{len(old)} earlier message(s) from "
                        f"{', '.join(authors)} omitted for brevity]"
                    )
                for m in recent:
                    who = "YOU said" if (for_author and m.author == for_author) else m.author
                    ref = f"  (refs: {', '.join(m.refs)})" if m.refs else ""
                    reply = f" ↪{m.parent_id}" if m.parent_id else ""
                    lines.append(f"  [{m.id}{reply}] {who}: {m.text}{ref}")
                lines.append("")
            return "\n".join(lines).rstrip()


__all__ = ["Channel", "Thread", "Message"]
