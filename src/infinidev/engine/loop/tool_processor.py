"""Tool call classification and pseudo-tool processing."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from infinidev.engine.hooks.hooks import hook_manager as _hook_manager, HookContext as _HookContext, HookEvent as _HookEvent
from infinidev.engine.formats.tool_call_parser import safe_json_loads as _safe_json_loads

from infinidev.engine.loop.classified_calls import ClassifiedCalls

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext


class ToolProcessor:
    """Classifies tool calls and orchestrates execution + message building."""

    _PLAN_OP_NAMES = frozenset({"add_step", "modify_step", "remove_step"})

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
            elif name in ToolProcessor._PLAN_OP_NAMES:
                result.plan_ops.append(tc)
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

        # Plan operation pseudo-tools: add_step, modify_step, remove_step
        # Batch all ops and apply once so the bulk-removal guard works correctly.
        from infinidev.engine.loop.step_operation import StepOperation
        plan_op_batch: list[StepOperation] = []
        for po in classified.plan_ops:
            try:
                po_args = _safe_json_loads(po.function.arguments) if isinstance(po.function.arguments, str) else (po.function.arguments or {})
                if not isinstance(po_args, dict):
                    continue
                name = po.function.name
                raw_index = po_args.get("index")
                if name == "add_step":
                    title = po_args.get("title", "")
                    desc = po_args.get("description", "")
                    if title:
                        if raw_index is not None:
                            index = int(raw_index)
                        else:
                            # Auto-assign: max existing index + 1 (accounting for batch)
                            existing_max = max((s.index for s in ctx.state.plan.steps), default=0)
                            batch_max = max((op.index for op in plan_op_batch if op.op == "add"), default=0)
                            index = max(existing_max, batch_max) + 1
                        plan_op_batch.append(StepOperation(op="add", index=index, title=title, description=desc))
                elif raw_index is not None:
                    index = int(raw_index)
                    if name == "modify_step":
                        title = po_args.get("title", "")
                        desc = po_args.get("description", "")
                        plan_op_batch.append(StepOperation(op="modify", index=index, title=title, description=desc))
                    elif name == "remove_step":
                        plan_op_batch.append(StepOperation(op="remove", index=index))
            except (json.JSONDecodeError, AttributeError, ValueError, TypeError):
                pass
        if plan_op_batch:
            ctx.state.plan.apply_operations(plan_op_batch)
