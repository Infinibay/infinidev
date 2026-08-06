"""Read specific execution events by id, with an optional surrounding window."""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field

from infinidev.engine.history import store
from infinidev.tools.base.base_tool import InfinibayBaseTool


class HistoryReadInput(BaseModel):
    event_ids: list[str] = Field(
        default_factory=list,
        description="Stable event ids (evt_…) returned by history_search.",
    )
    run_id: str | None = Field(
        default=None,
        description=(
            "Read the full timeline of one run instead of specific event ids. "
            "Used when event_ids is empty."
        ),
    )
    window_before: int = Field(
        default=0, ge=0, le=10,
        description="Events to include before each requested event (same run).",
    )
    window_after: int = Field(
        default=0, ge=0, le=10,
        description="Events to include after each requested event (same run).",
    )
    limit: int = Field(default=50, ge=1, le=200)
    include_archive: bool = Field(
        default=False,
        description="Also return archive_only events (hidden by default).",
    )


class HistoryReadTool(InfinibayBaseTool):
    name: str = "history_read"
    is_read_only: bool = True
    description: str = (
        "Read execution events by id, decoding their full payload. Pass "
        "event_ids from history_search, or pass run_id to read that run's "
        "whole timeline. archive_only events expose metadata but withhold "
        "their payload. window_before/window_after add surrounding events "
        "from the same run for context."
    )
    args_schema: Type[BaseModel] = HistoryReadInput

    def _run(
        self,
        event_ids: list[str] | None = None,
        run_id: str | None = None,
        window_before: int = 0,
        window_after: int = 0,
        limit: int = 50,
        include_archive: bool = False,
    ) -> str:
        from infinidev.tools.history._common import project_event

        ids = [eid for eid in (event_ids or []) if eid]

        if ids:
            try:
                events = store.read_events(
                    ids,
                    window_before=window_before,
                    window_after=window_after,
                )
            except Exception as exc:
                return self._error(f"History read failed: {exc}")
            if not include_archive:
                events = [
                    e for e in events if e.get("visibility") != "archive_only"
                ]
        elif run_id:
            events = store.list_run_events(
                run_id, include_archive=include_archive, limit=limit
            )
        else:
            return self._error(
                "Provide event_ids or run_id to read history."
            )

        events = events[:limit]
        payload = {
            "count": len(events),
            "events": [project_event(e, full_payload=True) for e in events],
        }
        self._log_tool_usage(f"Read {len(events)} history event(s)")
        return self._success(payload)
