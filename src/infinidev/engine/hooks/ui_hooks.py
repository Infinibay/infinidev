"""Default UI hooks — translate engine hooks into EventBus events for TUI display.

This module is the bridge between the hook system (inline, modifiable)
and the existing EventBus (fire-and-forget, consumed by TUI/classic CLI).
Call ``register_ui_hooks()`` once at startup to activate.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from infinidev.engine.hooks.hooks import hook_manager, HookEvent, HookContext
from infinidev.flows.event_listeners import event_bus

logger = logging.getLogger(__name__)

_registered = False


# ── Helpers (imported from canonical engine_logging module) ───────────────

def _extract_detail(tool_name: str, arguments: dict[str, Any] | str) -> str:
    from infinidev.engine.engine_logging import extract_tool_detail
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return extract_tool_detail(tool_name, arguments)


def _extract_preview(tool_name: str, result: str) -> str:
    from infinidev.engine.engine_logging import extract_tool_output_preview
    return extract_tool_output_preview(tool_name, result)


def _extract_error(result: str) -> str:
    from infinidev.engine.engine_logging import extract_tool_error
    return extract_tool_error(result)


# ── POST_TOOL: emit loop_tool_call + special handling ────────────────────

def _on_post_tool(ctx: HookContext) -> None:
    """Emit ``loop_tool_call`` event after each real tool execution."""
    # Skip pseudo-tools handled separately
    if ctx.tool_name in ("think", "step_complete", "add_note", "add_session_note"):
        return

    # Handle send_message specially
    if ctx.tool_name == "send_message":
        _on_send_message(ctx)
        return

    result = ctx.result or ""
    tool_detail = _extract_detail(ctx.tool_name, ctx.arguments)
    tool_error = _extract_error(result)
    tool_preview = _extract_preview(ctx.tool_name, result) if result else ""

    # Classic CLI logging (no-op when TUI is active)
    _cli_log_tool(ctx, tool_detail, tool_error, tool_preview)

    event_bus.emit("loop_tool_call", ctx.project_id, ctx.agent_id, {
        "agent_id": ctx.agent_id,
        "agent_name": ctx.metadata.get("agent_name", ctx.agent_id),
        "tool_name": ctx.tool_name,
        "tool_detail": tool_detail,
        "tool_error": tool_error,
        "tool_output_preview": tool_preview,
        "call_num": ctx.metadata.get("call_num", 0),
        "total_calls": ctx.metadata.get("total_calls", 0),
        "iteration": ctx.metadata.get("iteration", 0),
        "tokens_total": ctx.metadata.get("tokens_total", 0),
        "prompt_tokens": ctx.metadata.get("prompt_tokens", 0),
        "completion_tokens": ctx.metadata.get("completion_tokens", 0),
    })


def _on_send_message(ctx: HookContext) -> None:
    """Emit ``loop_user_message`` for the send_message tool."""
    message = ctx.arguments.get("message", "")
    if message:
        event_bus.emit("loop_user_message", ctx.project_id, ctx.agent_id, {
            "message": message,
        })


# ── POST_TOOL: think pseudo-tool ────────────────────────────────────────

def _on_think(ctx: HookContext) -> None:
    """Emit ``loop_think`` when the think pseudo-tool is dispatched."""
    if ctx.tool_name != "think":
        return
    reasoning = (ctx.result or "").strip()
    if not reasoning:
        return

    # Classic CLI
    if not event_bus.has_subscribers:
        from infinidev.engine.engine_logging import log as _log, DIM as _DIM, RESET as _RESET
        _log(f"  {_DIM}💭 {reasoning[:200]}{_RESET}")

    event_bus.emit("loop_think", ctx.project_id, ctx.agent_id, {
        "reasoning": reasoning,
    })


# ── PRE_STEP: emit step-start update ────────────────────────────────────

def _on_pre_step(ctx: HookContext) -> None:
    """Emit ``loop_step_update`` with ``status=active`` at the start of each step."""
    meta = ctx.metadata
    state = meta.get("state")
    if state is None:
        return

    plan = meta.get("plan", state.plan)
    iteration = meta.get("iteration", 0)

    # Determine active step description (mirrors original logic)
    active = plan.active_step if plan else None
    if active:
        active_desc = active.description
    elif not plan or not plan.steps:
        active_desc = "Planning..."
    else:
        done_steps = [s for s in plan.steps if s.status == "done"]
        active_desc = f"Continuing ({done_steps[-1].description})" if done_steps else "Working..."

    event_bus.emit("loop_step_update", ctx.project_id, ctx.agent_id, {
        "agent_id": ctx.agent_id,
        "agent_name": meta.get("agent_name", ctx.agent_id),
        "iteration": iteration + 1,
        "step_description": active_desc,
        "status": "active",
        "summary": "",
        "plan_steps": [
            {"index": s.index, "description": s.description, "status": s.status}
            for s in (plan.steps if plan else [])
        ],
        "tool_calls_step": 0,
        "tool_calls_total": state.total_tool_calls,
        "tokens_total": state.total_tokens,
        "prompt_tokens": state.last_prompt_tokens,
        "completion_tokens": state.last_completion_tokens,
    })


# ── POST_STEP: emit step-done update ────────────────────────────────────

def _on_post_step(ctx: HookContext) -> None:
    """Emit ``loop_step_update`` with completion data after each step."""
    meta = ctx.metadata
    state = meta.get("state")
    step_result = meta.get("step_result")
    if state is None or step_result is None:
        return

    iteration = meta.get("iteration", 0)
    action_tool_calls = meta.get("action_tool_calls", 0)

    done_steps = [s for s in state.plan.steps if s.status == "done"]
    if done_steps:
        done_desc = done_steps[-1].description
    elif step_result.summary:
        done_desc = step_result.summary[:120]
    else:
        done_desc = ""

    event_bus.emit("loop_step_update", ctx.project_id, ctx.agent_id, {
        "agent_id": ctx.agent_id,
        "agent_name": meta.get("agent_name", ctx.agent_id),
        "iteration": iteration + 1,
        "step_description": done_desc,
        "status": step_result.status,
        "summary": step_result.summary[:200],
        "plan_steps": [
            {"index": s.index, "description": s.description, "status": s.status}
            for s in state.plan.steps
        ],
        "tool_calls_step": action_tool_calls,
        "tool_calls_total": state.total_tool_calls,
        "tokens_total": state.total_tokens,
        "prompt_tokens": state.last_prompt_tokens,
        "completion_tokens": state.last_completion_tokens,
    })


# ── Classic CLI logging helper ───────────────────────────────────────────

def _cli_log_tool(
    ctx: HookContext,
    tool_detail: str,
    tool_error: str,
    tool_preview: str,
) -> None:
    """Log tool call to stderr in classic CLI mode (no-op when TUI is active)."""
    if event_bus.has_subscribers:
        return
    from infinidev.engine.engine_logging import (
        log as _log, log_tool as _log_tool,
        BLUE as _BLUE, DIM as _DIM, RED as _RED, RESET as _RESET,
    )
    meta = ctx.metadata
    agent_name = meta.get("agent_name", ctx.agent_id)
    iteration = meta.get("iteration", 0)
    call_num = meta.get("call_num", 0)
    total_calls = meta.get("total_calls", 0)

    if meta.get("verbose", True):
        _log_tool(agent_name, iteration + 1, ctx.tool_name, call_num, total_calls)
        if tool_detail:
            _log(f"{_BLUE}│{_RESET}     {_DIM}{tool_detail}{_RESET}")
        if tool_error:
            _log(f"{_BLUE}│{_RESET}     {_RED}✗ {tool_error}{_RESET}")
        elif tool_preview:
            for line in tool_preview.splitlines():
                _log(f"{_BLUE}│{_RESET}     {_DIM}{line}{_RESET}")


# ── Registration ─────────────────────────────────────────────────────────

def register_ui_hooks() -> None:
    """Register all default UI hooks. Safe to call multiple times."""
    global _registered
    if _registered:
        return
    _registered = True

    # Priority 200: run after user hooks (default 100)
    hook_manager.register(HookEvent.POST_TOOL, _on_post_tool, priority=200, name="ui:tool_call")
    hook_manager.register(HookEvent.POST_TOOL, _on_think, priority=200, name="ui:think")
    hook_manager.register(HookEvent.PRE_STEP, _on_pre_step, priority=200, name="ui:step_start")
    hook_manager.register(HookEvent.POST_STEP, _on_post_step, priority=200, name="ui:step_done")


def unregister_ui_hooks() -> None:
    """Remove all default UI hooks."""
    global _registered
    if not _registered:
        return
    _registered = False

    hook_manager.unregister(HookEvent.POST_TOOL, _on_post_tool)
    hook_manager.unregister(HookEvent.POST_TOOL, _on_think)
    hook_manager.unregister(HookEvent.PRE_STEP, _on_pre_step)
    hook_manager.unregister(HookEvent.POST_STEP, _on_post_step)
