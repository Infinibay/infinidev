"""Tool call classification and pseudo-tool processing."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from infinidev.engine.hooks.hooks import hook_manager as _hook_manager, HookContext as _HookContext, HookEvent as _HookEvent
from infinidev.engine.formats.tool_call_parser import safe_json_loads as _safe_json_loads

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext
    from infinidev.engine.loop.llm_caller import ClassifiedCalls


class ToolProcessor:
    """Classifies tool calls and orchestrates execution + message building."""

    @staticmethod
    def classify(tool_calls: list[Any]) -> ClassifiedCalls:
        """Separate tool calls into categories."""
        result = ClassifiedCalls()
        for tc in tool_calls:
            name = tc.function.name
            if name == "step_complete":
                result.step_complete = tc
            elif name == "add_note":
                result.notes.append(tc)
            elif name == "add_session_note":
                result.session_notes.append(tc)
            elif name == "think":
                result.thinks.append(tc)
            else:
                result.regular.append(tc)
        return result

    @staticmethod
    def process_pseudo_tools(
        ctx: ExecutionContext, classified: ClassifiedCalls,
        engine: "LoopEngine",
    ) -> None:
        """Handle think, add_note, add_session_note calls."""
        _MAX_NOTES = 20
        _MAX_SESSION_NOTES = 10

        for tk in classified.thinks:
            try:
                tk_args = _safe_json_loads(tk.function.arguments) if isinstance(tk.function.arguments, str) else (tk.function.arguments or {})
                reasoning = tk_args.get("reasoning", "").strip()
                if reasoning:
                    _hook_manager.dispatch(_HookContext(
                        event=_HookEvent.POST_TOOL,
                        tool_name="think",
                        arguments=tk_args,
                        result=reasoning,
                        project_id=ctx.project_id,
                        agent_id=ctx.agent_id,
                    ))
            except (json.JSONDecodeError, AttributeError):
                pass

        for nc in classified.notes:
            try:
                nc_args = _safe_json_loads(nc.function.arguments) if isinstance(nc.function.arguments, str) else (nc.function.arguments or {})
                note_text = nc_args.get("note", "").strip()
                if note_text and len(ctx.state.notes) < _MAX_NOTES:
                    ctx.state.notes.append(note_text)
                    ctx.state.tool_calls_since_last_note = 0
            except (json.JSONDecodeError, AttributeError):
                pass

        for snc in classified.session_notes:
            try:
                snc_args = _safe_json_loads(snc.function.arguments) if isinstance(snc.function.arguments, str) else (snc.function.arguments or {})
                note_text = snc_args.get("note", "").strip()
                if note_text and len(engine.session_notes) < _MAX_SESSION_NOTES:
                    engine.session_notes.append(note_text)
            except (json.JSONDecodeError, AttributeError):
                pass
