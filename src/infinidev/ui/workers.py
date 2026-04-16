"""Background worker helpers for the Infinidev TUI.

Wraps concurrent.futures for running engine tasks off the UI thread.
The prompt_toolkit Application event loop stays responsive while workers
execute blocking engine calls.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp

logger = logging.getLogger(__name__)

# Single shared executor — exclusive=True tasks use the lock below
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="infinidev-worker")
_exclusive_lock = threading.Lock()


def run_in_background(app: InfinidevApp, fn: Callable, *args, exclusive: bool = False) -> Future:
    """Submit a function to run in a background thread.

    If exclusive=True, only one exclusive task runs at a time (same semantics
    as Textual's @work(exclusive=True)). The lock prevents overlapping engine
    executions.
    """
    if exclusive:
        def _guarded():
            with _exclusive_lock:
                return fn(*args)
        future = _executor.submit(_guarded)
    else:
        future = _executor.submit(fn, *args)

    # When done, invalidate the UI so the result is visible
    def _on_done(f: Future):
        try:
            exc = f.exception()
            if exc:
                logger.error("Worker failed: %s", exc, exc_info=exc)
        except Exception:
            pass
        app.invalidate()

    future.add_done_callback(_on_done)
    return future


def run_engine_task(app: InfinidevApp, user_input: str) -> None:
    """Run the main engine pipeline in a background thread.

    Thin adapter over :func:`engine.orchestration.run_task`. Persists
    the user turn, refreshes the context calculator, runs the unified
    pipeline through a :class:`TUIHooks` adapter, then persists the
    result. Everything pipeline-shaped lives in
    ``engine/orchestration/pipeline.py``.
    """
    from infinidev.config.settings import reload_all
    from infinidev.db.service import store_conversation_turn, get_recent_summaries
    from infinidev.engine.orchestration import run_task
    from infinidev.ui.hooks_tui import TUIHooks

    app._file_diffs = {}
    hooks = TUIHooks(app)

    try:
        # Reload settings each turn so /settings changes apply.
        # The pipeline itself does NOT reload (see run_task docstring).
        reload_all()
        store_conversation_turn(app.session_id, "user", user_input)
        summaries = get_recent_summaries(app.session_id, limit=10)
        app.agent._session_summaries = summaries

        # Refresh the context-token meter before the LLM starts so the
        # status bar shows the right value during analysis.
        app.context_calculator.update_chat(user_input, summaries)
        app._context_status = app.context_calculator.get_context_status()
        app.invalidate()

        force_gather = bool(getattr(app, "_gather_next_task", False))
        app._gather_next_task = False

        result = run_task(
            agent=app.agent,
            user_input=user_input,
            session_id=app.session_id,
            engine=app.engine,
            reviewer=app.reviewer,
            hooks=hooks,
            force_gather=force_gather,
        )

        app._chat_history_control.show_thinking = False
        if result:
            app.add_message("Infinidev", result, "agent")
        store_conversation_turn(
            app.session_id, "assistant",
            result or "",
            (result or "")[:200],
        )
        app._actions_text = "Idle"
        app.invalidate()

    except Exception as e:
        app._analysis_waiting = False
        app._chat_history_control.show_thinking = False
        app.add_message("Error", str(e), "system")
    finally:
        app._engine_running = False
        app._context_flow = ""
        app.invalidate()
        _drain_pending(app)


def _run_flow_task(app: InfinidevApp, flow_name: str,
                   task_prompt: tuple[str, str],
                   use_tree_engine: bool = False,
                   empty_result_msg: str = "Done.") -> None:
    """Shared implementation for /init, /explore, /brainstorm.

    Thin adapter over :func:`engine.orchestration.run_flow_task`. The
    pipeline owns activate/deactivate context and engine selection;
    this wrapper just handles the TUI persistence + pending-input
    drain that's specific to the workers thread pool.
    """
    from infinidev.config.settings import reload_all
    from infinidev.db.service import store_conversation_turn
    from infinidev.engine.orchestration import run_flow_task
    from infinidev.ui.hooks_tui import TUIHooks

    hooks = TUIHooks(app)

    try:
        reload_all()
        app._context_flow = flow_name
        app.invalidate()

        try:
            result = run_flow_task(
                agent=app.agent,
                flow=flow_name,
                task_prompt=task_prompt,
                session_id=app.session_id,
                engine=app.engine,
                hooks=hooks,
                use_tree_engine=use_tree_engine,
            )
            if not result or not result.strip():
                result = empty_result_msg
        except Exception as e:
            result = f"{flow_name.title()} failed: {e}"

        app._chat_history_control.show_thinking = False
        app.add_message("Infinidev", result, "agent")
        store_conversation_turn(app.session_id, "assistant", result, result[:200])
        app._actions_text = "Idle"
        app.invalidate()
    except Exception as e:
        app._chat_history_control.show_thinking = False
        app.add_message("Error", str(e), "system")
    finally:
        app._engine_running = False
        app._context_flow = ""
        app.invalidate()
        _drain_pending(app)


def run_init_task(app: InfinidevApp) -> None:
    """Run /init in background thread."""
    from infinidev.prompts.init_project import INIT_TASK_DESCRIPTION, INIT_EXPECTED_OUTPUT
    _run_flow_task(app, "document",
                   task_prompt=(INIT_TASK_DESCRIPTION, INIT_EXPECTED_OUTPUT),
                   empty_result_msg="Project initialization complete.")


def run_explore_task(app: InfinidevApp, problem: str) -> None:
    """Run /explore in background thread."""
    from infinidev.engine.flows import get_flow_config
    flow_config = get_flow_config("explore")
    _run_flow_task(app, "explore",
                   task_prompt=(problem, flow_config.expected_output),
                   use_tree_engine=True,
                   empty_result_msg="Exploration complete (no synthesis produced).")


def run_brainstorm_task(app: InfinidevApp, problem: str) -> None:
    """Run /brainstorm in background thread."""
    from infinidev.engine.flows import get_flow_config
    flow_config = get_flow_config("brainstorm")
    _run_flow_task(app, "brainstorm",
                   task_prompt=(problem, flow_config.expected_output),
                   use_tree_engine=True,
                   empty_result_msg="Brainstorm complete (no synthesis produced).")


def run_plan_task(app: InfinidevApp, task_description: str) -> None:
    """Run /plan in background thread.

    Thin adapter over :meth:`PhaseEngine.execute_with_plan_review`.
    The phase engine owns the classify → investigate → plan → execute
    pipeline; this wrapper provides:

      * an ``on_plan_ready`` callback that displays the plan in the
        chat and blocks on ``_plan_review_event`` until the user
        approves, cancels, or supplies feedback;
      * an ``on_step_start`` callback (delegated to :class:`TUIHooks`);
      * the post-execution review-rework loop (kept here because the
        review pass is shared with the regular ``run_engine_task`` and
        will move into a hook in a follow-up).
    """
    import threading
    from infinidev.config.settings import reload_all, settings as _settings
    from infinidev.db.service import store_conversation_turn, get_recent_summaries
    from infinidev.engine.phases.phase_engine import PhaseEngine
    from infinidev.ui.hooks_tui import TUIHooks

    hooks = TUIHooks(app)

    try:
        reload_all()
        store_conversation_turn(app.session_id, "user", f"/plan {task_description}")
        summaries = get_recent_summaries(app.session_id, limit=10)
        app.agent._session_summaries = summaries

        app._context_flow = "plan"
        app._actions_text = "Planning..."
        app.invalidate()

        phase_engine = PhaseEngine()

        def _on_plan_ready(plan_steps: list[dict]) -> tuple[str, str]:
            """Show the plan in the chat and block until the user replies."""
            plan_display = _format_plan_for_display(plan_steps)
            app._chat_history_control.show_thinking = False
            app.add_message("Planner", plan_display, "agent")
            app._actions_text = "Waiting for plan review..."
            app.invalidate()

            app._plan_review_event = threading.Event()
            app._plan_review_waiting = True
            app._plan_review_answer = ""
            app._plan_review_event.wait()

            answer = (app._plan_review_answer or "").strip()
            app._plan_review_event = None
            app._plan_review_waiting = False

            low = answer.lower()
            if low in ("y", "yes", "approve", ""):
                return ("approve", "")
            if low in ("n", "no", "cancel"):
                app._chat_history_control.show_thinking = False
                app.add_message("System", "Plan cancelled.", "system")
                app._actions_text = "Idle"
                app.invalidate()
                return ("cancel", "")
            app.add_message("System", "Regenerating plan with feedback...", "system")
            return ("feedback", answer)

        app.agent.activate_context(session_id=app.session_id)
        try:
            result = phase_engine.execute_with_plan_review(
                agent=app.agent,
                task_description=task_description,
                on_plan_ready=_on_plan_ready,
                on_step_start=hooks.on_step_start,
                verbose=True,
            )
        finally:
            app.agent.deactivate()

        # Post-execution review-rework loop. Same shape as run_engine_task's
        # review phase — kept inline here for now because the review API
        # takes an `engine=` instance and PhaseEngine has its own. A future
        # cleanup will move this into hooks.on_review() once that exists.
        if _settings.REVIEW_ENABLED and phase_engine.has_file_changes():
            from infinidev.engine.analysis.review_engine import run_review_rework_loop

            app._actions_text = "Code review..."
            app.invalidate()

            def _review_status(level: str, msg: str) -> None:
                if level == "approved":
                    app.add_message("Reviewer", f"Code review: APPROVED. {msg}", "system")
                elif level == "rejected":
                    app.add_message("Reviewer", msg, "system")
                    app.add_message("System", "Re-running with review feedback...", "system")

            try:
                result, _ = run_review_rework_loop(
                    engine=phase_engine._last_engine,
                    agent=app.agent,
                    session_id=app.session_id,
                    task_prompt=(task_description, "Complete the task."),
                    initial_result=result or "",
                    reviewer=app.reviewer,
                    recent_messages=get_recent_summaries(app.session_id, limit=5),
                    on_status=_review_status,
                )
            except Exception as review_err:
                logger.error("Review phase failed: %s", review_err, exc_info=True)
                app.add_message("System", f"Review error: {review_err}", "system")

        app._chat_history_control.show_thinking = False
        app.add_message("Infinidev", result or "Plan executed.", "agent")
        store_conversation_turn(
            app.session_id, "assistant",
            result or "Plan executed.",
            (result or "")[:200],
        )
        app._actions_text = "Idle"
        app.invalidate()

    except Exception as e:
        app._plan_review_waiting = False
        app._chat_history_control.show_thinking = False
        app.add_message("Error", str(e), "system")
    finally:
        app._engine_running = False
        app._context_flow = ""
        app.invalidate()
        _drain_pending(app)


def _format_plan_for_display(plan_steps: list[dict]) -> str:
    """Format plan steps as a readable numbered list for user review."""
    lines = ["## Generated Plan\n"]
    for step in plan_steps:
        files = ", ".join(step.get("files", [])) or ""
        files_str = f"  [{files}]" if files else ""
        lines.append(f"  {step['step']}. {step.get('title', step.get('description', ''))}{files_str}")
    lines.append("")
    lines.append("Type **y** to approve and execute, **n** to cancel, or type feedback to revise.")
    return "\n".join(lines)


def _drain_pending(app: InfinidevApp) -> None:
    """Process the next queued input if any."""
    if app._pending_inputs:
        next_input = app._pending_inputs.pop(0)
        app._engine_running = True
        app._chat_history_control.show_thinking = True
        app.invalidate()
        run_in_background(app, run_engine_task, app, next_input, exclusive=True)
