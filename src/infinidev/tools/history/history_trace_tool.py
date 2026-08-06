"""Reconstruct a causal chain from the execution event log."""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field

from infinidev.engine.history import store
from infinidev.tools.base.base_tool import InfinibayBaseTool


class HistoryTraceInput(BaseModel):
    event_id: str | None = Field(
        default=None,
        description=(
            "Starting event id (evt_…). The tool walks parent_event_id links "
            "back to the root and returns the chain root-first."
        ),
    )
    run_id: str | None = Field(
        default=None,
        description=(
            "Return the full ordered timeline of one run instead of a single "
            "ancestor chain. Used when event_id is empty."
        ),
    )
    max_depth: int = Field(default=50, ge=1, le=200)
    include_archive: bool = Field(
        default=False,
        description="Also return archive_only events (hidden by default).",
    )


class HistoryTraceTool(InfinibayBaseTool):
    name: str = "history_trace"
    is_read_only: bool = True
    description: str = (
        "Reconstruct causal history from the event log. Give an event_id to "
        "walk its parent links back to the run's root (answers 'what led to "
        "this?'), or a run_id to get that run's full ordered timeline. Use it "
        "to explain why a decision was made, what evidence a verification "
        "relied on, or what changed after a user message."
    )
    args_schema: Type[BaseModel] = HistoryTraceInput

    def _run(
        self,
        event_id: str | None = None,
        run_id: str | None = None,
        max_depth: int = 50,
        include_archive: bool = False,
    ) -> str:
        from infinidev.tools.history._common import project_event

        if event_id:
            try:
                chain = store.trace_chain(event_id, max_depth=max_depth)
            except Exception as exc:
                return self._error(f"History trace failed: {exc}")
            if not include_archive:
                chain = [e for e in chain if e.get("visibility") != "archive_only"]
            if not chain:
                return self._success({
                    "start_event_id": event_id,
                    "count": 0,
                    "chain": [],
                    "note": "No event found for that id.",
                })
            self._log_tool_usage(
                f"Traced {len(chain)}-event chain from {event_id}"
            )
            return self._success({
                "start_event_id": event_id,
                "count": len(chain),
                "chain": [project_event(e, full_payload=True) for e in chain],
            })

        if run_id:
            events = store.list_run_events(
                run_id, include_archive=include_archive, limit=max_depth
            )
            run = store.get_run(run_id) or {}
            self._log_tool_usage(
                f"Traced run {run_id} timeline — {len(events)} event(s)"
            )
            return self._success({
                "run_id": run_id,
                "engine": run.get("engine"),
                "status": run.get("status"),
                "count": len(events),
                "chain": [project_event(e, full_payload=True) for e in events],
            })

        return self._error("Provide event_id or run_id to trace history.")
