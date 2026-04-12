"""EventBus -> UI state dispatcher.

Translates engine events into state mutations on the InfinidevApp. This is
a pure function of (app_state, event_type, data) with no framework
dependencies — making it testable without prompt_toolkit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from infinidev.ui.controls.chat_history import format_tool_chat_message
from infinidev.ui.theme import BAR_FILLED, BAR_EMPTY

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp

logger = logging.getLogger(__name__)


def process_event(app: InfinidevApp, event_type: str, data: dict[str, Any]) -> None:
    """Dispatch a single event to the appropriate handler.

    Called from the event bus subscriber. After mutating app state,
    the caller should call app.invalidate() to trigger a redraw.
    """
    try:
        _dispatch(app, event_type, data)
    except Exception:
        logger.debug("process_event(%s) failed", event_type, exc_info=True)
        app.add_log(f"x UI event error: {event_type}")
        # deque(maxlen=15) handles the cap automatically


def _dispatch(app: InfinidevApp, event_type: str, data: dict[str, Any]) -> None:
    """Inner dispatch — all state mutations happen here."""

    # ── Loop engine events ───────────────────────────────────────────

    if event_type == "loop_step_update":
        # Clear all transient state on step transition
        app._thinking_text = ""
        app._streaming_tool_name = None
        app._streaming_token_count = 0
        app._actions_text = ""  # Reset so "waiting for LLM..." animation shows
        steps = data.get("plan_steps", [])
        if steps:
            lines = []
            for s in steps:
                icon = {"done": "v", "active": ">", }.get(s["status"], "o")
                lines.append(f"{icon} {s['title']}")
            app._steps_text = "\n".join(lines)
        else:
            app._steps_text = "Waiting for plan..."

        desc = data.get("step_title", "")
        summary = data.get("summary", "")
        iteration = data.get("iteration", 0)
        status = data.get("status", "")

        # Step transition message in chat
        if desc and status == "active":
            app.add_message("Step", f"--- Step {iteration}: {desc} ---", "system")

        plan_text = f"Step {iteration}: {desc}" if iteration else desc
        if summary:
            plan_text += f"\n{summary}"
        if status and status != "active":
            plan_text += f"\n[{status}]"
        app._plan_text = plan_text

        # Update context tokens
        app.update_context_tokens(
            task_tokens=data.get("tokens_total", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
        )

    elif event_type == "loop_tool_call":
        # Clear streaming state — tool is now executing
        app._streaming_tool_name = None
        app._streaming_token_count = 0

        tool_name = data.get("tool_name", "")
        tool_detail = data.get("tool_detail", "")
        tool_error = data.get("tool_error", "")
        tool_output = data.get("tool_output_preview", "")

        action_text = f">> {tool_name}\n"
        if tool_detail:
            action_text += f"   {tool_detail}\n"
        if tool_error:
            action_text += f"   x {tool_error}\n"
        elif tool_output:
            for line in tool_output.splitlines()[:4]:
                action_text += f"   {line}\n"
        app._actions_text = action_text.rstrip()

        chat_msg = format_tool_chat_message(tool_name, tool_detail, tool_error, tool_output)
        if chat_msg:
            app.add_message("Tool", chat_msg, "system")

        app.update_context_tokens(
            task_tokens=data.get("tokens_total", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
        )

    elif event_type == "loop_user_message":
        msg = data.get("message", "")
        if msg:
            app.add_message("Infinidev", msg, "agent")

    elif event_type == "loop_file_changed":
        path = data.get("path", "")
        diff = data.get("diff", "")
        action = data.get("action", "modified")
        num_changes = data.get("num_changes", 1)
        # Each file change gets its own chat message with the diff
        import os
        icon = "+" if action == "created" else "~"
        basename = os.path.basename(path)
        count_str = f" ({num_changes} edits)" if num_changes > 1 else ""
        header = f"{icon} {basename}{count_str}"
        app.chat_messages.append({
            "sender": "File",
            "text": header,
            "type": "diff",
            "diff_text": diff,
            "diff_path": path,
            "diff_action": action,
            "collapsed": True,
        })
        app._chat_history_control.invalidate_cache()
        app.invalidate()

    elif event_type == "loop_llm_call_start":
        # LLM call starting — clear streaming state but keep _actions_text
        # so the previous tool result remains visible while waiting.
        # _actions_text is cleared by loop_step_update on step transitions.
        app._streaming_tool_name = None
        app._streaming_token_count = 0

    elif event_type == "loop_stream_status":
        # Streaming progress — show in ACTIONS with token count + tool detection
        phase = data.get("phase", "")
        token_count = data.get("token_count", 0)
        tool_name = data.get("tool_name")

        if phase == "done":
            # Stream finished (or failed) — clear streaming UI state
            app._streaming_tool_name = None
            app._streaming_token_count = 0
        elif phase == "tool_detected" and tool_name:
            app._streaming_tool_name = tool_name
            app._streaming_token_count = token_count
            app._actions_text = ""  # Clear — fragments handle it now
        else:
            app._streaming_token_count = token_count

    elif event_type == "loop_thinking_chunk":
        # Streaming thinking — append to the THINKING sidebar panel
        chunk = data.get("text", "")
        if chunk:
            app._thinking_text += chunk
            # Keep only last ~500 chars to prevent sidebar overflow
            if len(app._thinking_text) > 500:
                app._thinking_text = "..." + app._thinking_text[-450:]
            # Throttle redraws to ~10 FPS to avoid excessive invalidation
            import time
            now = time.monotonic()
            last = getattr(app, '_last_thinking_invalidate', 0.0)
            if now - last > 0.1:
                app._last_thinking_invalidate = now
                app.invalidate()

    elif event_type == "loop_think":
        reasoning = data.get("reasoning", "").strip()
        if reasoning:
            agent_id = data.get("_agent_id", "")
            sender = "Analyst" if agent_id == "analyst" else "Thinking"
            app.add_message(sender, reasoning, "think")
            # Show in THINKING panel (truncated). Cleared on step transition.
            if len(reasoning) > 500:
                app._thinking_text = "..." + reasoning[-450:]
            else:
                app._thinking_text = reasoning

    elif event_type == "loop_behavior_update":
        # Intentionally silent — verdicts are inspected via /debug → Behavior.
        # No chat message, no log line. The BehaviorScorer already stored
        # the event in its history when this fired.
        pass

    elif event_type == "loop_log":
        level = data.get("level", "warning")
        msg = data.get("message", "")
        icon = "!" if level == "warning" else "x"
        line = f"{icon} {msg}"
        app.add_log(line)
        # deque(maxlen=15) handles the cap automatically

    # ── Tree engine events ───────────────────────────────────────────

    elif event_type == "tree_init":
        root = data.get("root_problem", "")
        n = data.get("num_children", 0)
        logic = data.get("logic", "AND")
        app._plan_text = f"Tree: {root[:80]}\n   {logic} -> {n} sub-problems"
        app._steps_text = "Initializing tree..."
        app._actions_text = "Tree decomposed"

    elif event_type == "tree_node_exploring":
        node_id = data.get("node_id", "")
        problem = data.get("problem", "")
        depth = data.get("depth", 0)
        indent = "  " * depth
        app._steps_text = f"[{node_id}]\n{indent}{problem[:100]}"
        app._actions_text = f"Exploring [{node_id}]..."
        app.update_context_tokens(prompt_tokens=data.get("prompt_tokens", 0))

    elif event_type == "tree_tool_call":
        node_id = data.get("node_id", "")
        tool = data.get("tool_name", "")
        args = data.get("args_preview", "")
        action_text = f">> [{node_id}] {tool}"
        if args:
            action_text += f"\n   {args[:60]}"
        app._actions_text = action_text
        app.update_context_tokens(prompt_tokens=data.get("prompt_tokens", 0))

    elif event_type == "tree_node_resolved":
        node_id = data.get("node_id", "")
        state = data.get("state", "")
        conf = data.get("confidence", "")
        summary = data.get("summary", "")
        state_icon = {
            "solvable": "OK", "unsolvable": "NO", "mitigable": "!",
            "needs_decision": "?", "needs_experiment": "EX",
        }.get(state, "*")
        short_line = f"[{state_icon}] [{node_id}] {state} ({conf})"
        app._tree_resolved_lines.append(short_line)

        step_text = short_line
        if summary:
            step_text += f"\n   {summary[:80]}"
        app._steps_text = step_text

        log_line = short_line
        if summary:
            log_line += f" - {summary[:60]}"
        app.add_log(log_line)
        # deque(maxlen=15) handles the cap automatically
        app.update_context_tokens(prompt_tokens=data.get("prompt_tokens", 0))

    elif event_type == "tree_propagation":
        root_state = data.get("root_state", "?")
        total = data.get("total_nodes", 0)
        resolved = data.get("resolved_nodes", 0)
        pct = (resolved / total * 100) if total > 0 else 0
        bar_len = 20
        filled = int(bar_len * resolved / total) if total > 0 else 0
        bar = BAR_FILLED * filled + BAR_EMPTY * (bar_len - filled)
        tree_text = f"Tree Root: {root_state}\n   {bar} {resolved}/{total} ({pct:.0f}%)"
        if app._tree_resolved_lines:
            tree_text += "\n" + "\n".join(app._tree_resolved_lines[-8:])
        app._plan_text = tree_text

    elif event_type == "tree_fact_discovered":
        node_id = data.get("node_id", "")
        fact = data.get("fact_content", "")
        tool = data.get("source_tool", "")
        line = f"! [{node_id}] {fact[:60]}"
        if tool:
            line += f" (via {tool})"
        app.add_log(line)
        # deque(maxlen=15) handles the cap automatically

    elif event_type == "tree_synthesizing":
        total = data.get("total_nodes", 0)
        app._steps_text = f"Synthesizing {total} nodes..."
        app._actions_text = "Generating synthesis..."

    elif event_type == "tree_budget_warning":
        used = data.get("used", 0)
        limit = data.get("limit", 0)
        btype = data.get("type", "")
        line = f"! Budget {btype}: {used}/{limit}"
        app.add_log(line)
        # deque(maxlen=15) handles the cap automatically

    elif event_type == "tree_finished":
        status = data.get("status", "?")
        total = data.get("total_nodes", 0)
        app._steps_text = f"Complete: {total} nodes\n   Root: {status}"
        app._actions_text = "Idle"
        app._tree_resolved_lines.clear()

    # ── Analysis events ──────────────────────────────────────────────

    elif event_type == "analysis_start":
        round_num = data.get("round", 1)
        app._actions_text = f"Analyzing request... (round {round_num})"

    elif event_type == "analysis_research":
        queries = data.get("queries", [])
        preview = ", ".join(q[:30] for q in queries[:2])
        app._actions_text = f"Researching: {preview}"

    elif event_type == "analysis_complete":
        action = data.get("action", "")
        app._actions_text = f"Analysis: {action}"

    # ── Review events ────────────────────────────────────────────────

    elif event_type == "review_start":
        app._actions_text = "Code review..."

    elif event_type == "review_complete":
        verdict = data.get("verdict", "")
        issues = data.get("issue_count", 0)
        text = f"Review: {verdict}"
        if issues:
            text += f" ({issues} issues)"
        app._actions_text = text

    # ── Gather events ────────────────────────────────────────────────

    elif event_type == "gather_status":
        app._actions_text = data.get("text", "")

    elif event_type == "gather_error":
        msg = data.get("message", "")
        app._actions_text = f"Gather skipped: {msg}"
