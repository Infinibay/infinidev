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

LIVE_TOOL_OUTPUT_LINES = 20
_LIVE_TOOL_LINE_CHARS = 1000
_MAX_THINKING_TRANSCRIPT_CHARS = 2 * 1024 * 1024
_THINKING_TRUNCATED = "[Earlier streamed reasoning truncated for memory safety]\n"
_RUNTIME_STATE_EVENTS = {
    "loop_step_update",
    "loop_tool_start",
    "loop_tool_call",
    "loop_file_changed",
    "tree_init",
    "tree_node_exploring",
    "tree_node_resolved",
    "tree_propagation",
    "tree_synthesizing",
    "tree_finished",
}


def _tool_message(app: InfinidevApp, run_id: str) -> dict[str, Any] | None:
    if not run_id:
        return None
    for message in reversed(app.chat_messages):
        if message.get("tool_run_id") == run_id:
            return message
    return None


def _append_live_output(message: dict[str, Any], chunk: str) -> None:
    """Append output while retaining only a small, renderable line tail."""
    partial = str(message.get("_live_output_partial") or "")
    text = partial + str(chunk).replace("\r\n", "\n").replace("\r", "\n")
    parts = text.split("\n")
    partial = parts.pop() if parts else ""

    lines = list(message.get("live_output_tail") or [])
    lines.extend(line[-_LIVE_TOOL_LINE_CHARS:] for line in parts)
    message["live_output_tail"] = lines[-LIVE_TOOL_OUTPUT_LINES:]
    message["_live_output_partial"] = partial[-_LIVE_TOOL_LINE_CHARS:]


def process_event(app: InfinidevApp, event_type: str, data: dict[str, Any]) -> None:
    """Dispatch a single event to the appropriate handler.

    Called from the event bus subscriber. After mutating app state,
    the caller should call app.invalidate() to trigger a redraw.
    """
    try:
        _dispatch(app, event_type, data)
        if (
            event_type in _RUNTIME_STATE_EVENTS
            or (event_type == "loop_stream_status" and data.get("phase") == "done")
        ):
            persist = getattr(app, "_persist_runtime_state", None)
            if callable(persist):
                persist()
    except Exception:
        logger.debug("process_event(%s) failed", event_type, exc_info=True)
        app.add_log(f"x UI event error: {event_type}")
        # deque(maxlen=15) handles the cap automatically


def _dispatch(app: InfinidevApp, event_type: str, data: dict[str, Any]) -> None:
    """Inner dispatch — all state mutations happen here."""

    if event_type.startswith("council_"):
        council_id = data.get("council_id", "")
        if event_type == "council_started" and council_id:
            app.open_agent_tab(council_id)
            app.active_tab = "chat"
            app.focus_chat()
        elif council_id and event_type in {"council_agent_message", "council_finished"}:
            app.refresh_agent_tabs(council_id)
        return

    # ── Loop engine events ───────────────────────────────────────────

    if event_type == "loop_step_update":
        # Clear all transient state on step transition
        app._thinking_text = ""
        app._thinking_full = ""
        app._streaming_tool_name = None
        app._streaming_token_count = 0
        app._actions_text = ""  # Reset so "waiting for LLM..." animation shows
        steps = data.get("plan_steps", [])
        app._session_plan_steps = [
            dict(step) for step in steps if isinstance(step, dict)
        ]
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

        # The title belongs in the STEPS/plan panels. The developer's
        # plain-language send_message at step start supplies the useful chat
        # context without duplicating an internal plan label here.
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

    elif event_type == "loop_tool_start":
        app._streaming_tool_name = None
        app._streaming_token_count = 0

        tool_name = data.get("tool_name", "")
        tool_detail = data.get("tool_detail", "")
        tool_args = data.get("tool_arguments") or {}
        if not isinstance(tool_args, dict):
            tool_args = {}
        live_detail = str(tool_args.get("command") or tool_detail or "")
        app._actions_text = f">> {tool_name}"
        if live_detail:
            app._actions_text += f"\n   {live_detail}"

        message = {
            "sender": "Tool",
            "text": live_detail or tool_name,
            "type": "tool_call",
            "tool_run_id": data.get("tool_run_id", ""),
            "tool_name": tool_name,
            "args": tool_args,
            "result": "",
            "error": "",
            "running": True,
            "live_output_tail": [],
            "_live_output_partial": "",
            "visible": True,
        }
        app.chat_messages.append(message)
        persist = getattr(app, "_persist_session_message", None)
        if callable(persist):
            persist(message)
        label_detail = live_detail[:120]
        app._chat_history_control.work_label = (
            f"Running {tool_name}: {label_detail}".rstrip(": ")
        )
        app._chat_history_control.invalidate_cache()
        app.invalidate()

    elif event_type == "loop_tool_output":
        message = _tool_message(app, data.get("tool_run_id", ""))
        if message is not None and message.get("running"):
            _append_live_output(message, data.get("chunk", ""))
            app._chat_history_control.invalidate_cache()
            app.invalidate()

    elif event_type == "loop_tool_call":
        # The tool completed; replace its running row instead of duplicating it.
        app._streaming_tool_name = None
        app._streaming_token_count = 0

        tool_name = data.get("tool_name", "")
        tool_detail = data.get("tool_detail", "")
        tool_error = data.get("tool_error", "")
        tool_output = data.get("tool_output_preview", "")
        tool_args = data.get("tool_arguments") or {}
        tool_result_full = data.get("tool_result_full") or ""

        # Sidebar action text — unchanged. The chat row carries the
        # full payload now.
        action_text = f">> {tool_name}\n"
        if tool_detail:
            action_text += f"   {tool_detail}\n"
        if tool_error:
            action_text += f"   x {tool_error}\n"
        elif tool_output:
            for line in tool_output.splitlines()[:4]:
                action_text += f"   {line}\n"
        app._actions_text = action_text.rstrip()

        message = _tool_message(app, data.get("tool_run_id", ""))
        display_text = (message or {}).get("text") or tool_detail or tool_name
        completed = {
            "sender": "Tool",
            "text": display_text,
            "type": "tool_call",
            "tool_name": tool_name,
            "args": tool_args if isinstance(tool_args, dict) else {},
            "result": tool_result_full,
            "error": tool_error,
            "exec_data": data.get("exec_data"),
            "running": False,
            "visible": True,
        }
        if message is None:
            completed["tool_run_id"] = data.get("tool_run_id", "")
            app.chat_messages.append(completed)
            message = completed
        else:
            resume_message_id = message.get("_resume_message_id")
            message.clear()
            message.update(completed)
            message["tool_run_id"] = data.get("tool_run_id", "")
            if resume_message_id is not None:
                message["_resume_message_id"] = resume_message_id
        persist = getattr(app, "_persist_session_message", None)
        if callable(persist):
            persist(message)
        app._chat_history_control.work_label = "Working"
        app._chat_history_control.invalidate_cache()
        app.invalidate()

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
        message = {
            "sender": "File",
            "text": header,
            "type": "diff",
            "diff_text": diff,
            "diff_path": path,
            "diff_action": action,
            "collapsed": False,  # always expanded — user wants no accordion
        }
        app.chat_messages.append(message)
        persist = getattr(app, "_persist_session_message", None)
        if callable(persist):
            persist(message)
        app._chat_history_control.invalidate_cache()
        app.invalidate()

    elif event_type == "loop_llm_call_start":
        # Keep the previous action visible, but make the transcript's live
        # indicator say what is happening now. Otherwise a completed tool's
        # stale "Working" label makes a long model turn look hung.
        app._streaming_tool_name = None
        app._streaming_token_count = 0
        phase = data.get("phase", "deciding")
        label = "Model is planning" if phase == "planning" else "Model is deciding next action"
        app._actions_text = label
        app._chat_history_control.work_label = label
        app._chat_history_control.invalidate_cache()
        app.invalidate()

    elif event_type == "loop_stream_status":
        # Streaming progress — show in ACTIONS with token count + tool detection
        phase = data.get("phase", "")
        token_count = data.get("token_count", 0)
        tool_name = data.get("tool_name")

        if phase == "done":
            # Stream finished (or failed) — clear streaming UI state.
            # Flush the FULL accumulated native thinking buffer into the
            # chat as a permanent `think` message so extended-thinking
            # models leave a complete, untruncated trail. Reset BOTH the
            # full accumulator and the sidebar view afterwards: there are
            # several LLM calls per step, each ending in a "done" event, so
            # without the reset every subsequent "done" would re-emit the
            # same buffer (duplicate) and the sidebar's truncated view
            # would leak into the chat (cut off).
            full = (getattr(app, "_thinking_full", "") or "").strip()
            if full:
                app.add_message("Thinking", full, "think")
            app._thinking_full = ""
            app._thinking_text = ""
            app._streaming_tool_name = None
            app._streaming_token_count = 0
        elif phase == "tool_detected" and tool_name:
            app._streaming_tool_name = tool_name
            app._streaming_token_count = token_count
            app._actions_text = ""  # Clear — fragments handle it now
        else:
            app._streaming_token_count = token_count

    elif event_type == "loop_thinking_chunk":
        # Streaming thinking — accumulate the FULL reasoning (for the
        # eventual permanent chat flush on stream-done) and render a
        # truncated VIEW of it in the THINKING sidebar panel.
        chunk = data.get("text", "")
        if chunk:
            combined = app._thinking_full + chunk
            if len(combined) > _MAX_THINKING_TRANSCRIPT_CHARS:
                tail_size = _MAX_THINKING_TRANSCRIPT_CHARS - len(_THINKING_TRUNCATED)
                app._thinking_full = _THINKING_TRUNCATED + combined[-tail_size:]
            else:
                app._thinking_full = combined
            # Sidebar view: keep only last ~500 chars to prevent overflow.
            if len(app._thinking_full) > 500:
                app._thinking_text = "..." + app._thinking_full[-450:]
            else:
                app._thinking_text = app._thinking_full
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
            # Deliberately do NOT touch _thinking_full — this reasoning has
            # already been written to chat above, so the next stream-done
            # flush must not re-emit it as a duplicate.
            if len(reasoning) > 500:
                app._thinking_text = "..." + reasoning[-450:]
            else:
                app._thinking_text = reasoning

    elif event_type == "loop_assistant_message":
        # Pair-programming critic spoke up. Rendered as its own message
        # type so consecutive verdicts fold into one collapsible line
        # (see controls/critic_widget.py) instead of pushing the
        # assistant's reply off screen — the critic talks on most steps.
        #
        # Severity, model and source travel as fields rather than being
        # baked into the sender string and the body text: the compact
        # renderer needs them separately (it counts rejects, and shows the
        # model only once a verdict is expanded), and a body that starts
        # with "(model)" would put that noise into the preview line.
        action = data.get("action", "information")
        message = (data.get("message") or "").strip()
        if message:
            tag = {
                "reject": "REJECT",
                "recommendation": "RECOMMEND",
                "information": "INFO",
            }.get(action, "INFO")
            app.add_message(
                f"Assistant · {tag}", message, "critic",
                critic_action=action,
                critic_model=data.get("model") or "assistant",
                critic_source=data.get("source") or "tools",
            )

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
