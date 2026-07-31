"""Tool call classification and pseudo-tool processing."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from infinidev.engine._best_effort import best_effort
from infinidev.engine.hooks.hooks import hook_manager as _hook_manager, HookContext as _HookContext, HookEvent as _HookEvent
from infinidev.engine.formats.tool_call_parser import safe_json_loads as _safe_json_loads

from infinidev.engine.loop.classified_calls import ClassifiedCalls

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext

# Auto-notes are written by the engine, not the model, and they record
# something the model can always get back (it read the file once, it can
# read it again). That makes them the ones to evict when the note budget
# is full and a real finding needs the slot. The prefix is how the two
# are told apart, so both writer and evictor read it from here.
AUTO_NOTE_PREFIX = "Read "


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
                if note_text:
                    classified.note_results[nc.id] = ToolProcessor._save_note(
                        ctx, note_text, _MAX_NOTES,
                    )
            except (json.JSONDecodeError, AttributeError):
                pass

        for snc in classified.session_notes:
            try:
                snc_args = _safe_json_loads(snc.function.arguments) if isinstance(snc.function.arguments, str) else (snc.function.arguments or {})
                note_text = snc_args.get("note", "").strip()
                if note_text and len(engine.session_notes) < _MAX_SESSION_NOTES:
                    engine.session_notes.append(note_text)
                    # Persist so a resumed session (`-c`) can re-load it.
                    # Soft-fails: an in-memory note is still useful this run.
                    with best_effort("session note persist failed"):
                        from infinidev.tools.base.context import get_current_session_id
                        from infinidev.db.service import persist_session_note
                        persist_session_note(get_current_session_id(), note_text)
            except (json.JSONDecodeError, AttributeError):
                pass

        # Plan tools (add_step, modify_step, remove_step) are now real tools
        # handled via execute_tool_call — no pseudo-tool processing needed.

    @staticmethod
    def _save_note(ctx: ExecutionContext, note_text: str, max_notes: int) -> str:
        """Store one model-written note, and say honestly what happened.

        The cap is task-wide and nothing rotates it, so a long task used to
        hit it and then silently drop every note that followed while still
        answering ``{"status": "noted"}``. Three things changed:

        - An auto-note is evicted to make room. ``Read <path>`` is
          reconstructible — the model can read the file again — whereas a
          conclusion it reached is not. When both cannot fit, the cheap one
          goes.
        - A note that still cannot fit is archived to working memory, so
          ``recall_context`` can find it even though the prompt cannot
          carry it.
        - The counter is reset either way. It gates the "SAVE NOTES NOW"
          nudge, and leaving it high after a drop meant the nudge fired on
          every remaining iteration, demanding a call that had become a
          permanent no-op.
        """
        ctx.state.tool_calls_since_last_note = 0

        if len(ctx.state.notes) < max_notes:
            ctx.state.notes.append(note_text)
            return '{"status": "noted"}'

        for i, existing in enumerate(ctx.state.notes):
            if existing.startswith(AUTO_NOTE_PREFIX):
                ctx.state.notes.pop(i)
                ctx.state.notes.append(note_text)
                return (
                    '{"status": "noted", "evicted": "an auto-generated '
                    f'\\"{AUTO_NOTE_PREFIX.strip()}\\" note was dropped to make room"}}'
                )

        archived = False
        with best_effort("archiving a dropped note failed"):
            from infinidev.engine.working_memory import get_working_memory

            archived = get_working_memory(ctx.session_id).remember(
                "dropped note", note_text, kind="note",
                step_index=len(ctx.state.history),
            )
        where = (
            "it was archived to working memory — recall_context can find it"
            if archived else "it was NOT saved anywhere"
        )
        return json.dumps({
            "status": "dropped",
            "reason": f"the {max_notes}-note limit is full and every note is "
                      f"model-written, so none could be evicted; {where}",
        })

    @staticmethod
    def auto_note_for_small(
        ctx: "ExecutionContext",
        tool_name: str,
        tool_args: dict | str,
        tool_result: str,
    ) -> None:
        """For small models: auto-save a note when reading files.

        Small models often forget to call add_note after reading a file,
        losing critical context between steps.  This automatically records
        the file path (and optionally key symbols) as a note.
        """
        _MAX_NOTES = 20
        if not ctx.is_small:
            return

        if isinstance(tool_args, str):
            import json as _json
            try:
                tool_args = _json.loads(tool_args) if tool_args.strip() else {}
            except Exception:
                tool_args = {}

        if tool_name in ("read_file", "partial_read"):
            path = tool_args.get("file_path", tool_args.get("path", ""))
            if path and len(ctx.state.notes) < _MAX_NOTES:
                # Check if we already have a note about this file
                path_short = path.split("/")[-1] if "/" in path else path
                already_noted = any(path_short in n for n in ctx.state.notes)
                if not already_noted:
                    ctx.state.notes.append(f"{AUTO_NOTE_PREFIX}{path}")
                    ctx.state.tool_calls_since_last_note = 0
