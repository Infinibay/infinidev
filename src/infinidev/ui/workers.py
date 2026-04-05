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

    This is the prompt_toolkit equivalent of the Textual run_engine() @work method.
    """
    from infinidev.config.settings import reload_all, settings as _settings
    from infinidev.db.service import store_conversation_turn, get_recent_summaries
    from infinidev.agents.base import InfinidevAgent

    app._file_diffs = {}

    try:
        reload_all()
        store_conversation_turn(app.session_id, "user", user_input)
        summaries = get_recent_summaries(app.session_id, limit=10)
        app.agent._session_summaries = summaries

        # Update context
        app.context_calculator.update_chat(user_input, summaries)
        app._context_status = app.context_calculator.get_context_status()
        app.invalidate()

        # ── Analysis phase ───────────────────────────────────
        if _settings.ANALYSIS_ENABLED:
            app.analyst.reset()
            analysis_input = user_input
            analysis = app.analyst.analyze(analysis_input, session_summaries=summaries)

            # Question loop
            while analysis.action == "ask" and app.analyst.can_ask_more:
                questions_text = analysis.format_questions_for_user()
                app._chat_history_control.show_thinking = False
                app.add_message("Analyst", questions_text, "agent")

                # Block and wait for user answer
                app._analysis_event = threading.Event()
                app._analysis_waiting = True
                app._analysis_answer = ""
                app._analysis_original_input = user_input
                app._analysis_event.wait()

                answer = app._analysis_answer
                app._analysis_event = None

                if not answer.strip():
                    break

                app.analyst.add_answer(questions_text, answer)
                app._chat_history_control.show_thinking = True
                app.invalidate()
                analysis_input = user_input + "\n\nUser clarification: " + answer
                analysis = app.analyst.analyze(analysis_input, session_summaries=summaries)

            task_prompt = analysis.build_flow_prompt()

            # "done" pseudo-flow
            if analysis.flow == "done":
                app._context_flow = "done"
                app._chat_history_control.show_thinking = False
                app.add_message("Infinidev", analysis.reason or analysis.original_input, "agent")
                store_conversation_turn(
                    app.session_id, "assistant",
                    analysis.reason or analysis.original_input,
                    (analysis.reason or analysis.original_input)[:200],
                )
                app._actions_text = "Idle"
                app.invalidate()
                return

            if analysis.action == "proceed":
                # Only show spec + ask for confirmation for develop flow
                # Research, document, sysadmin flows don't need user approval
                _needs_confirmation = analysis.flow == "develop"

                if _needs_confirmation:
                    spec = analysis.specification
                    spec_parts = []
                    if spec.get("summary"):
                        spec_parts.append(f"Summary: {spec['summary']}")
                    for key in ("requirements", "hidden_requirements", "assumptions", "out_of_scope"):
                        items = spec.get(key, [])
                        if items:
                            spec_parts.append(f"\n{key.replace('_', ' ').title()}:")
                            for item in items:
                                spec_parts.append(f"  - {item}")
                    if spec.get("technical_notes"):
                        spec_parts.append(f"\nTechnical Notes: {spec['technical_notes']}")
                    spec_parts.append("\nProceed? Type y to proceed, n to cancel, or add feedback.")

                    app._chat_history_control.show_thinking = False
                    app.add_message("Analyst", "\n".join(spec_parts), "agent")

                    app._analysis_event = threading.Event()
                    app._analysis_waiting = True
                    app._analysis_answer = ""
                    app._analysis_original_input = user_input
                    app._analysis_event.wait()

                    confirm = app._analysis_answer.strip()
                    app._analysis_event = None

                    if confirm.lower() in ("n", "no", "cancel"):
                        app._chat_history_control.show_thinking = False
                        app.add_message("System", "Development skipped.", "system")
                        app._actions_text = "Idle"
                        app.invalidate()
                        return

                    if confirm and confirm.lower() not in ("y", "yes", ""):
                        desc, expected = task_prompt
                        desc += f"\n\n## Additional User Feedback\n{confirm}"
                        task_prompt = (desc, expected)

                    app._chat_history_control.show_thinking = True
                    app.invalidate()

            from infinidev.engine.flows import get_flow_config
            from infinidev.prompts.flows import get_flow_identity
            flow_config = get_flow_config(analysis.flow)
            app.agent._system_prompt_identity = get_flow_identity(analysis.flow)
            app.agent.backstory = flow_config.backstory
            desc, _ = task_prompt
            task_prompt = (desc, flow_config.expected_output)
        else:
            task_prompt = (user_input, "Complete the task and report findings.")
            flow_config = None

        # ── Gather phase ─────────────────────────────────────
        flow_label = analysis.flow if _settings.ANALYSIS_ENABLED else "develop"
        _do_gather = _settings.GATHER_ENABLED or app._gather_next_task
        app._gather_next_task = False
        if _do_gather and flow_label == "develop":
            try:
                from infinidev.gather import run_gather
                chat_history = [
                    {"role": "user" if "[user]" in s.lower() else "assistant", "content": s}
                    for s in get_recent_summaries(app.session_id, limit=10)
                ]
                brief = run_gather(user_input, chat_history, analysis, app.agent)
                desc, expected = task_prompt
                task_prompt = (brief.render() + "\n\n" + desc, expected)
                from infinidev.flows.event_listeners import event_bus as _eb
                _eb.emit("gather_status", 0, "", {"text": f"Gathered: {brief.summary()}"})
            except Exception as exc:
                from infinidev.flows.event_listeners import event_bus as _eb
                _eb.emit("gather_error", 0, "", {"message": str(exc)})

        # ── Development phase ────────────────────────────────
        app._context_flow = flow_label
        app._actions_text = f"Running [{flow_label}]..."
        app.invalidate()

        app.agent.activate_context(session_id=app.session_id)
        try:
            if flow_label == "explore":
                from infinidev.engine.tree import TreeEngine
                tree_engine = TreeEngine()
                result = tree_engine.execute(
                    agent=app.agent, task_prompt=task_prompt, verbose=True,
                )
            else:
                result = app.engine.execute(
                    agent=app.agent, task_prompt=task_prompt, verbose=True,
                )
            if not result or not result.strip():
                result = "Done. (no additional output)"
        finally:
            app.agent.deactivate()

        # ── Review phase ─────────────────────────────────────
        run_review = flow_config.run_review if flow_config else True
        if _settings.REVIEW_ENABLED and run_review and flow_label != "explore" and app.engine.has_file_changes():
            from infinidev.engine.analysis.review_engine import run_review_rework_loop

            app._actions_text = "Code review..."
            app.invalidate()

            def _review_status(level: str, msg: str) -> None:
                if level == "verification_pass":
                    app.add_message("Verifier", f"PASS. {msg}", "system")
                elif level == "verification_fail":
                    app.add_message("Verifier", f"FAIL. {msg}", "system")
                    app.add_message("System", "Re-running to fix test failures...", "system")
                elif level == "approved":
                    app.add_message("Reviewer", f"Code review: APPROVED. {msg}", "system")
                elif level == "rejected":
                    app.add_message("Reviewer", msg, "system")
                    app.add_message("System", "Re-running with review feedback...", "system")
                elif level == "max_reviews":
                    app.add_message("Reviewer", "Max review rounds reached.", "system")

            try:
                result, _ = run_review_rework_loop(
                    engine=app.engine,
                    agent=app.agent,
                    session_id=app.session_id,
                    task_prompt=task_prompt,
                    initial_result=result,
                    reviewer=app.reviewer,
                    recent_messages=get_recent_summaries(app.session_id, limit=5),
                    on_status=_review_status,
                )
            except Exception as review_err:
                logger.error("Review phase failed: %s", review_err, exc_info=True)
                app.add_message("System", f"Review error: {review_err}", "system")

        # ── Finish ───────────────────────────────────────────
        app._chat_history_control.show_thinking = False
        app.add_message("Infinidev", result, "agent")
        store_conversation_turn(app.session_id, "assistant", result, result[:200])
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
    """Shared implementation for /init, /explore, /brainstorm."""
    from infinidev.config.settings import reload_all
    from infinidev.db.service import store_conversation_turn
    from infinidev.engine.flows import get_flow_config

    try:
        reload_all()
        app._context_flow = flow_name
        app._actions_text = f"Running [{flow_name}]..."
        app.invalidate()

        flow_config = get_flow_config(flow_name)
        from infinidev.prompts.flows import get_flow_identity
        app.agent._system_prompt_identity = get_flow_identity(flow_name)
        app.agent.backstory = flow_config.backstory

        app.agent.activate_context(session_id=app.session_id)
        try:
            if use_tree_engine:
                from infinidev.engine.tree import TreeEngine
                engine = TreeEngine()
            else:
                engine = app.engine
            result = engine.execute(
                agent=app.agent, task_prompt=task_prompt, verbose=True,
            )
            if not result or not result.strip():
                result = empty_result_msg
        except Exception as e:
            result = f"{flow_name.title()} failed: {e}"
        finally:
            app.agent.deactivate()

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
    """Run /plan in background thread: investigate → plan → review → execute."""
    import threading
    from infinidev.config.settings import reload_all, settings as _settings
    from infinidev.db.service import store_conversation_turn, get_recent_summaries
    from infinidev.engine.phases.phase_engine import PhaseEngine
    from infinidev.engine.phases.investigator import _investigate_iteratively
    from infinidev.engine.phases.plan_generator import _generate_plan
    from infinidev.engine.phases.plan_executor import _execute_plan
    from infinidev.engine.test_checkpoint import TestCheckpoint
    from infinidev.prompts.phases import get_strategy
    from infinidev.tools.base.context import get_current_workspace_path

    try:
        reload_all()
        store_conversation_turn(app.session_id, "user", f"/plan {task_description}")
        summaries = get_recent_summaries(app.session_id, limit=10)
        app.agent._session_summaries = summaries

        app._context_flow = "plan"
        app._actions_text = "Planning..."
        app.invalidate()

        phase_engine = PhaseEngine()

        # Init test checkpoint
        workdir = get_current_workspace_path()
        phase_engine._test_checkpoint = TestCheckpoint(None, workdir)

        # ── Classify ─────────────────────────────────────
        classification = phase_engine._classify(app.agent, task_description, verbose=True)
        from infinidev.gather.models import DEPTH_CONFIGS
        depth_config = DEPTH_CONFIGS.get(classification.depth)
        task_type = classification.ticket_type.value
        strategy = get_strategy(task_type)

        app.agent.activate_context(session_id=app.session_id)
        try:
            # ── Phases 1+2: Iterative investigation ──────
            strategy.investigate_max_tool_calls = depth_config.investigate_max_tool_calls
            answers, all_notes = _investigate_iteratively(
                app.agent, task_description, strategy, None, verbose=True,
                max_questions=depth_config.questions_max,
                skip_investigate=depth_config.skip_investigate,
            )

            # ── Phase 3: Plan + Review Loop ──────────────
            feedback_context = ""
            strategy.plan_min_steps = depth_config.plan_min_steps

            while True:
                plan_desc = task_description
                if feedback_context:
                    plan_desc += f"\n\n## USER FEEDBACK ON PREVIOUS PLAN\n{feedback_context}"

                app._actions_text = "Generating plan..."
                app._chat_history_control.show_thinking = True
                app.invalidate()

                plan_steps = _generate_plan(
                    app.agent, plan_desc, answers, all_notes, strategy, None, verbose=True,
                )

                if not plan_steps:
                    app._chat_history_control.show_thinking = False
                    app.add_message("System", "Failed to generate a plan.", "system")
                    app._actions_text = "Idle"
                    app.invalidate()
                    return

                # Display plan
                plan_display = _format_plan_for_display(plan_steps)
                app._chat_history_control.show_thinking = False
                app.add_message("Planner", plan_display, "agent")
                app._actions_text = "Waiting for plan review..."
                app.invalidate()

                # Wait for user response
                app._plan_review_event = threading.Event()
                app._plan_review_waiting = True
                app._plan_review_answer = ""
                app._plan_review_event.wait()

                answer = app._plan_review_answer.strip()
                app._plan_review_event = None

                if answer.lower() in ("y", "yes", "approve", ""):
                    break
                elif answer.lower() in ("n", "no", "cancel"):
                    app._chat_history_control.show_thinking = False
                    app.add_message("System", "Plan cancelled.", "system")
                    app._actions_text = "Idle"
                    app.invalidate()
                    return
                else:
                    feedback_context = answer
                    app.add_message("System", "Regenerating plan with feedback...", "system")
                    continue

            # ── Phase 4: Execute ─────────────────────────
            app._chat_history_control.show_thinking = True
            app._actions_text = "Executing plan..."
            app.add_message("System", "Plan approved. Executing...", "system")
            app.invalidate()

            _completed_step_nums: set[int] = set()

            def _on_step_start(step_num, total, all_steps, completed_list):
                """Update the TUI STEPS panel with phase plan progress."""
                # Track completed steps by number (previous step is done)
                for s in all_steps:
                    if s["step"] < step_num:
                        _completed_step_nums.add(s["step"])

                lines = []
                for s in all_steps:
                    s_num = s["step"]
                    s_title = s.get("title", "")
                    if s_num in _completed_step_nums:
                        lines.append(f"v {s_title}")
                    elif s_num == step_num:
                        lines.append(f"> {s_title}")
                    else:
                        lines.append(f"o {s_title}")
                app._steps_text = "\n".join(lines)
                app._actions_text = f"Executing step {step_num}/{total}..."
                app.invalidate()

            result, _last_engine = _execute_plan(
                app.agent, task_description, "Complete the task.",
                answers, all_notes, plan_steps, strategy, None, depth_config, verbose=True,
                on_step_start=_on_step_start,
            )
            phase_engine._last_engine = _last_engine

        finally:
            app.agent.deactivate()

        # ── Review phase (if enabled) ────────────────────
        if (_settings.REVIEW_ENABLED
                and phase_engine.has_file_changes()):
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

        # ── Finish ───────────────────────────────────────
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
