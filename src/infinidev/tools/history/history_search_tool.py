"""Hybrid full-text + structured search over the execution event log."""

from __future__ import annotations

import logging
from typing import Type

from pydantic import BaseModel, Field

from infinidev.engine.history import events as ev
from infinidev.engine.history import store
from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)


class HistorySearchInput(BaseModel):
    query: str = Field(
        default="",
        description=(
            "Full-text query matched against event payloads (supports FTS "
            "operators like | and quoted phrases). Leave empty to browse the "
            "most recent matching events."
        ),
    )
    run_id: str | None = Field(
        default=None, description="Restrict results to one engine run."
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Session to search. Omitted = the current session; '0' = every "
            "session in this workspace."
        ),
    )
    event_type: str | None = Field(
        default=None,
        description=(
            "Restrict to one event type, e.g. run_started, engine_selected, "
            "task_closed, digest_created."
        ),
    )
    node_id: str | None = Field(
        default=None, description="Restrict to events attached to one node/task id."
    )
    after: float | None = Field(
        default=None, description="Only events at/after this Unix timestamp."
    )
    before: float | None = Field(
        default=None, description="Only events at/before this Unix timestamp."
    )
    include_archive: bool = Field(
        default=False,
        description="Also return archive_only events (hidden by default).",
    )
    limit: int = Field(default=10, ge=1, le=50)
    context_window: int = Field(
        default=0,
        ge=0,
        le=5,
        description="How many surrounding events to include around each hit.",
    )


class HistorySearchTool(InfinibayBaseTool):
    name: str = "history_search"
    is_read_only: bool = True
    description: str = (
        "Search the execution event log — the append-only record of what each "
        "task engine did (engine selection, stages, tasks, digests). Combines "
        "full-text matching on event payloads with structured filters "
        "(run_id, event_type, node_id, time range). Returns stable event ids, "
        "snippets and a match reason. Use history_read to open specific events "
        "and history_trace to follow a causal chain."
    )
    args_schema: Type[BaseModel] = HistorySearchInput

    def _run(
        self,
        query: str = "",
        run_id: str | None = None,
        session_id: str | None = None,
        event_type: str | None = None,
        node_id: str | None = None,
        after: float | None = None,
        before: float | None = None,
        include_archive: bool = False,
        limit: int = 10,
        context_window: int = 0,
    ) -> str:
        from infinidev.tools.history._common import project_event

        if session_id is None:
            effective_session_id = self.session_id
        elif session_id == "0":
            effective_session_id = None
        else:
            effective_session_id = session_id

        if event_type and event_type not in ev.KNOWN_EVENT_TYPES:
            logger.debug("history_search: unknown event_type %r", event_type)

        try:
            hits = store.search_events(
                query=query,
                session_id=effective_session_id,
                run_id=run_id,
                event_type=event_type,
                node_id=node_id,
                after=after,
                before=before,
                include_archive=include_archive,
                limit=limit,
            )
        except Exception as exc:
            return self._error(f"History search failed: {exc}")

        results = []
        for event in hits:
            item = project_event(event, full_payload=False)
            item["score"] = event.get("score")
            item["match_reason"] = event.get("match_reason", "")
            if context_window:
                item["context"] = [
                    project_event(neighbor, full_payload=False)
                    for neighbor in store.events_around(
                        event["run_id"],
                        event["sequence"],
                        window_before=context_window,
                        window_after=context_window,
                    )
                    if neighbor.get("event_id") != event.get("event_id")
                ]
            results.append(item)

        self._log_tool_usage(
            f"Searched history for {query!r} — {len(results)} hit(s)"
        )
        return self._success({
            "query": query,
            "count": len(results),
            "results": results,
        })
