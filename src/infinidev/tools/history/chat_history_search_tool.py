"""Search the durable chat transcript and read the messages around a hit.

``history_search`` covers the engine event log; this tool covers what the user
and the agent actually said, which lives in ``session_messages``. Two modes
share one contract: ``text`` greps the searchable projection of every stored
message, ``semantic`` ranks by embedding similarity. Either way a hit is an
anchor id, and ``before``/``after`` return the neighbouring transcript rows so
the agent can see what led to the hit and what followed it.
"""

from __future__ import annotations

import logging
from typing import Literal, Type

import numpy as np
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)

#: Cap on the text returned per transcript row so a window of neighbours cannot
#: blow the caller's context budget.
_RETURNED_TEXT_LIMIT = 1200


class ChatHistorySearchInput(BaseModel):
    query: str = Field(
        default="",
        description=(
            "Words to look for in past chat messages. In text mode every token "
            "must appear (AND, case-insensitive); in semantic mode the whole "
            "query is embedded and ranked by similarity."
        ),
    )
    mode: Literal["text", "semantic"] = Field(
        default="text",
        description=(
            "text = exact word matching over the transcript; semantic = "
            "conceptual similarity. Semantic mode requires a query."
        ),
    )
    limit: int = Field(default=5, ge=1, le=25, description="Max anchor messages.")
    threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity in semantic mode.",
    )
    before: int = Field(
        default=2,
        ge=0,
        le=20,
        description="Transcript messages to return before each anchor.",
    )
    after: int = Field(
        default=2,
        ge=0,
        le=20,
        description="Transcript messages to return after each anchor.",
    )
    message_id: int | None = Field(
        default=None,
        description=(
            "Skip searching and return exactly this message plus its "
            "before/after neighbours (use the id from a previous hit)."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description="Transcript of another session. Defaults to the current one.",
    )


class ChatHistorySearchTool(InfinibayBaseTool):
    name: str = "chat_history_search"
    is_read_only: bool = True
    description: str = (
        "Search this session's full chat history (user messages, assistant "
        "replies, tool calls and results) and read the messages around a hit. "
        "mode='text' requires every query word to appear; mode='semantic' "
        "ranks by embedding similarity. Each hit carries its transcript id plus "
        "the `before`/`after` neighbouring messages, so you can recover what "
        "was decided earlier in a conversation that is no longer in context. "
        "Pass message_id to re-open one known message with its neighbours."
    )
    args_schema: Type[BaseModel] = ChatHistorySearchInput

    def _run(
        self,
        query: str = "",
        mode: str = "text",
        limit: int = 5,
        threshold: float = 0.45,
        before: int = 2,
        after: int = 2,
        message_id: int | None = None,
        session_id: str | None = None,
    ) -> str:
        from infinidev.db.service import (
            get_session_message_window,
            search_session_messages,
        )

        effective_session_id = session_id or self.session_id
        if not effective_session_id:
            return self._error("no session id available to search chat history")

        query = (query or "").strip()

        try:
            if message_id is not None:
                window = get_session_message_window(
                    effective_session_id, message_id, before=before, after=after
                )
                if not window:
                    return self._error(f"no transcript message with id={message_id}")
                hits = [self._anchor_from_window(window, message_id)]
            elif mode == "semantic":
                if not query:
                    return self._error("semantic mode requires a non-empty query")
                hits = self._semantic_hits(
                    session_id=effective_session_id,
                    query=query,
                    threshold=threshold,
                    limit=limit,
                )
            else:
                if not query:
                    return self._error("text mode requires a non-empty query")
                hits = [
                    {
                        "message_id": row["id"],
                        "created_at": row["created_at"],
                        "snippet": row["snippet"],
                    }
                    for row in search_session_messages(
                        effective_session_id, query, limit=limit
                    )
                ]

            for hit in hits:
                if before or after:
                    hit["context"] = self._project_window(
                        get_session_message_window(
                            effective_session_id,
                            hit["message_id"],
                            before=before,
                            after=after,
                        ), query=query,
                    )
        except Exception as exc:
            logger.debug("chat_history_search failed", exc_info=True)
            return self._error(f"Chat history search failed: {exc}")

        self._log_tool_usage(
            f"Searched chat history ({mode}) for {query!r} — {len(hits)} hit(s)"
        )
        payload = {
            "mode": "window" if message_id is not None else mode,
            "session_id": effective_session_id,
            "query": query,
            "count": len(hits),
            "results": hits,
        }
        if not hits and mode == "text":
            payload["hint"] = (
                "Text mode requires ALL query words in ONE message. Retry with one "
                "distinctive word (for example 'SQLite'), or use mode='semantic' "
                "for a natural-language question. No matches does not prove no decision exists."
            )
        return self._success(payload)

    @staticmethod
    def _anchor_from_window(window: list[dict], message_id: int) -> dict:
        """Build a hit for an explicitly requested message id."""
        anchor = next((row for row in window if row["id"] == message_id), None)
        if anchor is None:
            return {"message_id": message_id, "snippet": ""}
        return {
            "message_id": anchor["id"],
            "created_at": anchor["created_at"],
            "snippet": (anchor["search_text"] or "")[:_RETURNED_TEXT_LIMIT],
        }

    @staticmethod
    def _project_window(window: list[dict], *, query: str = "") -> list[dict]:
        """Trim neighbour rows to the fields the agent needs to read them."""
        from infinidev.db.service import _history_snippet

        needle = query.split()[0] if query.split() else ""
        return [
            {
                "message_id": row["id"],
                "created_at": row["created_at"],
                "is_anchor": bool(row.get("is_anchor")),
                "text": _history_snippet(
                    row["search_text"] or "", needle, width=_RETURNED_TEXT_LIMIT,
                ) if needle and row.get("is_anchor") else (
                    row["search_text"] or ""
                )[:_RETURNED_TEXT_LIMIT],
                "truncated": len(row["search_text"] or "") > _RETURNED_TEXT_LIMIT,
            }
            for row in window
        ]

    def _semantic_hits(
        self, *, session_id: str, query: str, threshold: float, limit: int
    ) -> list[dict]:
        """Rank transcript rows by cosine similarity, backfilling vectors."""
        from infinidev.db.service import (
            get_session_message_embedding_rows,
            store_session_message_embeddings,
        )
        from infinidev.tools.base.dedup import _cosine_similarity
        from infinidev.tools.base.embeddings import (
            current_embedding_space,
            embed_passages,
            embed_queries,
            embedding_from_blob,
            embedding_is_current,
        )

        rows = get_session_message_embedding_rows(session_id, limit=500)
        if not rows:
            return []

        query_vector = np.asarray(embed_queries([query])[0], dtype=np.float32)
        dim = int(query_vector.shape[0])
        live_space = current_embedding_space()

        stale = [
            row
            for row in rows
            if not embedding_is_current(
                row.get("embedding"),
                row.get("embedding_space"),
                live_space=live_space,
                dim=dim,
            )
        ]
        if stale:
            vectors = embed_passages([row["search_text"] for row in stale])
            refreshed: list[tuple[int, bytes]] = []
            for row, vector in zip(stale, vectors, strict=True):
                array = np.asarray(vector, dtype=np.float32)
                row["embedding"] = array.tobytes()
                row["embedding_space"] = live_space
                refreshed.append((int(row["id"]), array.tobytes()))
            store_session_message_embeddings(refreshed, live_space)

        scored: list[dict] = []
        for row in rows:
            blob = row.get("embedding")
            if not embedding_is_current(
                blob, row.get("embedding_space"), live_space=live_space, dim=dim
            ):
                continue
            similarity = _cosine_similarity(query_vector, embedding_from_blob(blob))
            if similarity < threshold:
                continue
            scored.append({
                "message_id": row["id"],
                "created_at": row["created_at"],
                "similarity": round(float(similarity), 4),
                "snippet": (row["search_text"] or "")[:_RETURNED_TEXT_LIMIT],
            })

        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:limit]
