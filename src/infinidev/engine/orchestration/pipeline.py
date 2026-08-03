"""Unified, UI-agnostic task pipeline — chat-agent-first edition.

This module owns the entire task lifecycle as a single function
:func:`run_task`:

  user turn  →  ChatAgent  →  (respond?)  done.
                    │
                    └ (escalate)  AnalystPlanner  →  Gather  →  LoopEngine.execute(initial_plan=plan)  →  Review  →  done.

Every side effect that needs to reach a human (showing the chat reply,
showing the plan, status updates) goes through the
:class:`OrchestrationHooks` Protocol. The function imports nothing
from ``click``, ``prompt_toolkit``, ``threading``, or any UI module.

Three entry points wrap this function:

  * ``cli/main.py::_run_main``          → uses :class:`ClickHooks`
  * ``cli/main.py::_run_single_prompt`` → uses :class:`NonInteractiveHooks`
  * ``ui/workers.py::run_engine_task``  → uses TUI-specific hooks

If something is missing from the hooks Protocol, ADD IT HERE FIRST and
then update each adapter — never reach back into the UI from inside
this file.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Literal, Protocol, runtime_checkable

from infinidev.engine._best_effort import best_effort

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OrchestrationHooks — the only contract between pipeline and UI
# ─────────────────────────────────────────────────────────────────────────────


Phase = Literal[
    "chat",
    "council",
    "analysis",
    "gather",
    "execute",
    "review",
    "idle",
]


@runtime_checkable
class OrchestrationHooks(Protocol):
    """Side-effect Protocol the pipeline uses to talk to a UI.

    All methods MUST be safe to call from a worker thread. Implementations
    that drive a UI event loop (e.g. the TUI) are responsible for marshalling
    back to the UI thread internally.

    Methods are deliberately granular — the pipeline never assumes anything
    about how a UI presents data. A headless test can use :class:`NoOpHooks`
    to drop everything on the floor.
    """

    # ── Status / phase tracking ──────────────────────────────────────────
    def on_phase(self, phase: Phase) -> None:
        """Pipeline entered a new phase. *phase* is one of:
        ``"chat"``, ``"analysis"``, ``"gather"``, ``"execute"``,
        ``"review"``, ``"idle"``.
        UIs use this to update an "Actions" indicator."""

    def notify_token_usage(self, prompt_tokens: int, lane: str = "chat") -> None:
        """Report the real prompt size of one LLM call.

        *prompt_tokens* comes off ``response.usage`` — the only number that
        knows what the model actually received. *lane* is ``"chat"`` for the
        conversational loops and ``"task"`` for the developer's, which the UI
        meters separately because they hold different prompts."""

    def on_status(self, level: str, msg: str) -> None:
        """Status line for ad-hoc updates. *level* is informational
        (``"info"``, ``"warn"``, ``"error"``, ``"verification_pass"``,
        ``"verification_fail"``, ``"objectives_pass"``, ``"objectives_fail"``,
        ``"objectives_unverified"``, ``"objectives_regressed"``,
        ``"objectives_summary"``, ``"approved"``, ``"rejected"``,
        ``"max_reviews"``). UIs may colourise based on level."""

    def notify(self, speaker: str, msg: str, kind: str = "agent") -> None:
        """A speaker is producing a chat-style message. *kind* is one of
        ``"agent"`` (a model output), ``"system"`` (process feedback),
        ``"error"`` (failure)."""

    def notify_stream_chunk(
        self,
        speaker: str,
        chunk: str,
        kind: str = "agent",
    ) -> None:
        """Append a streaming text chunk. The FIRST chunk for a new
        ``(speaker, kind)`` pair creates a message; subsequent chunks
        append to that same message. A later :meth:`notify` call
        implicitly ends the stream — the next chunk opens a new
        message. UIs that render streaming incrementally (TUI chat
        panel, terminal with cursor control) override this; stateless
        adapters may concatenate and defer until a sentinel arrives."""

    def notify_stream_end(
        self,
        speaker: str,
        kind: str = "agent",
    ) -> None:
        """Mark the in-progress streaming message as complete.

        The distinction from ``notify`` is important for rich UIs:
        during streaming, markdown / syntax highlighting is typically
        skipped (partial ``**bold`` etc. renders ugly). Once the stream
        ends, the UI re-renders the message with full styling applied.
        Stateless adapters (e.g. plain terminal echo) can no-op this —
        the final newline they add on ``notify`` or session end is
        enough."""

    def mark_reply_shown(self) -> None:
        """Signal that the chat reply has already been displayed.

        The pipeline calls this in the ``respond`` branch after it has
        streamed / notified the reply, so UI adapters that ALSO render
        ``run_task``'s return value (the TUI worker) can skip the
        duplicate. Default no-op — classic CLI adapters print the return
        value themselves, so they're unaffected. The pipeline invokes it
        defensively (``getattr``), so adapters that don't define it are
        fine too."""

    # ── User interaction ─────────────────────────────────────────────────
    def ask_user(self, prompt: str, kind: str = "text") -> str | None:
        """Block until the user replies. Return the user's text answer
        (possibly empty), or ``None`` to indicate the caller cannot be
        interactive at all (single-shot mode). Pipeline branches that
        receive ``None`` MUST proceed with sensible defaults instead of
        failing."""

    # ── Progress / structured updates ────────────────────────────────────
    def on_step_start(
        self,
        step_num: int,
        total: int,
        all_steps: list[dict],
        completed: list[int],
    ) -> None:
        """A new step is about to begin. UIs that show a step list
        (e.g. the TUI STEPS panel) refresh here. Default impls may
        ignore this."""

    def on_file_change(self, path: str) -> None:
        """A file was modified by a tool. UIs that show diffs refresh
        here. Default impls may ignore this."""


def _runtime_event_bridge(hooks: OrchestrationHooks):
    """Surface runtime task state on the status line.

    Only events that tell the user something they cannot already see get
    through. "Task queued" / "Task started" / "Step completed" fired on
    every transition and said nothing the transcript wasn't already
    showing — a status line that repeats the obvious trains people to stop
    reading it.
    """

    def emit(event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "task_started":
            task = event.get("task")
            title = getattr(task, "title", "") if task is not None else ""
            if title:
                hooks.on_status("info", title)
        elif kind == "runtime_cancelled":
            hooks.on_status("warn", "Cancelled")

    return emit


# ─────────────────────────────────────────────────────────────────────────────
# Phase implementations — pure functions, take hooks as parameter
# ─────────────────────────────────────────────────────────────────────────────


def _run_gather_phase(
    *,
    user_input: str,
    agent: Any,
    task_prompt: tuple[str, str],
    session_id: str,
    force_gather: bool,
    hooks: OrchestrationHooks,
) -> tuple[str, str]:
    """Gather: collect codebase context before execution. Soft-fails.

    With the chat-agent redesign, no ``AnalysisResult`` / specification
    is fed to gather — the chat agent already explored relevant files
    and handed them to the planner. ``analyst_result=None`` tells
    run_gather to skip the ticket-synthesis step that depended on the
    old spec shape.
    """
    from infinidev.config.settings import settings as _settings
    from infinidev.db.service import get_recent_summaries

    if not (_settings.GATHER_ENABLED or force_gather):
        return task_prompt

    hooks.on_phase("gather")

    try:
        from infinidev.gather import run_gather

        agent.activate_context(session_id=session_id)
        hooks.on_status("info", "Gathering context...")
        chat_history = [
            {"role": "user" if "[user]" in s.lower() else "assistant", "content": s}
            for s in get_recent_summaries(session_id, limit=10)
        ]
        brief = run_gather(user_input, chat_history, None, agent)
        desc, expected = task_prompt
        desc = brief.render() + "\n\n" + desc
        hooks.on_status("info", f"Gathered: {brief.summary()}")
        return (desc, expected)
    except Exception as exc:
        hooks.on_status("warn", f"Gather failed (proceeding without): {exc}")
        return task_prompt


def _run_elaboration_phase(
    *,
    escalation: Any,
    session_id: str,
    project_id: int | None,
    workspace_path: str | None,
    hooks: OrchestrationHooks,
) -> Any:
    """Turn the vague request into a GroundedSpec before planning.

    Runs once per task on the single configured model. Returns a
    possibly-updated EscalationPacket carrying ``grounded_spec``. Soft-fails:
    any problem (or the complexity gate skipping it) returns the original
    escalation unchanged — elaboration enriches the handoff, it is never
    load-bearing for correctness.
    """
    from dataclasses import replace as _dc_replace
    from infinidev.config.settings import settings as _settings

    if not _settings.SPEC_ELABORATION_ENABLED:
        return escalation

    try:
        from infinidev.engine.analysis.spec_elaborator import (
            elaborate,
            should_elaborate,
        )

        if not should_elaborate(escalation):
            return escalation

        hooks.on_phase("analysis")
        hooks.on_status("info", "Elaborating the spec...")
        spec = elaborate(
            escalation,
            session_id=session_id,
            project_id=project_id,
            workspace_path=workspace_path,
        )
        if spec is None:
            return escalation

        # Surface the product decisions to the user — non-blocking, and each
        # one states the default that IS being built, so this reads as "here
        # is what I chose, correct me" rather than as a questionnaire the
        # user has to clear before any work starts. The gate in
        # ``_admissible_clarifications`` already capped and filtered these.
        if spec.clarifications_needed:
            lines = []
            for c in spec.clarifications_needed:
                others = [o for o in c.options if o.strip() and o != c.default]
                line = f"  - {c.question} - using: {c.default}"
                if others:
                    line += f" (alternatives: {'; '.join(others)})"
                lines.append(line)
            hooks.notify(
                "Infinidev",
                "Product decisions selected for this run. Execution will use "
                "these defaults; correct any choice that does not match your intent:\n"
                + "\n".join(lines),
                "agent",
            )
        return _dc_replace(escalation, grounded_spec=spec)
    except Exception:
        logger.debug("Spec elaboration phase failed; proceeding raw", exc_info=True)
        return escalation


def _run_council_phase(
    *,
    escalation: Any,
    session_id: str,
    project_id: int | None,
    workspace_path: str | None,
    hooks: OrchestrationHooks,
) -> Any:
    """Multi-agent deliberation between escalate and the planner.

    Runs ONLY when ``escalation.council_requested`` is set and the
    feature is enabled. Returns a possibly-updated EscalationPacket
    carrying the synthesised ``design_brief`` (and, if the council hit a
    genuine product fork, the user's answer folded into the request).

    Soft-fails: any problem returns the original escalation unchanged, so
    the pipeline always proceeds to the planner. The council enriches the
    handoff; it is never load-bearing for correctness.
    """
    from dataclasses import replace as _dc_replace
    from infinidev.config.settings import settings as _settings

    if not getattr(escalation, "council_requested", False):
        return escalation
    if not _settings.COUNCIL_ENABLED:
        return escalation

    hooks.on_phase("council")
    hooks.on_status("info", "Convening multi-agent council...")

    try:
        from infinidev.engine.council import run_council

        # Build the deliberation handoff from the escalation packet —
        # the council debates around the user's request and the chat
        # agent's understanding.
        handoff = (
            f"User request (verbatim):\n  {escalation.user_request}\n\n"
            f"Chat agent's understanding:\n  {escalation.understanding}\n\n"
            f"Council focus: {escalation.council_focus}"
        )
        if escalation.opened_files:
            handoff += "\n\nFiles already inspected upstream:\n" + "\n".join(
                f"  - {p}" for p in escalation.opened_files
            )

        brief = run_council(
            handoff,
            session_id=session_id,
            project_id=project_id,
            workspace_path=workspace_path,
            hooks=hooks,
        )
    except Exception as exc:
        logger.error("Council phase failed: %s", exc, exc_info=True)
        hooks.on_status("warn", f"Council failed (proceeding): {exc}")
        return escalation

    if brief is None:
        return escalation

    # Conditional user approval: only interrupt when the council flagged
    # a genuine product fork it must not decide alone. Otherwise flow
    # straight through. (See DesignBrief.user_decision_required.)
    enriched_request = escalation.user_request
    if brief.user_decision_required and brief.open_questions_for_user:
        answer = hooks.ask_user(brief.render_questions_for_user(), "text")
        if answer and answer.strip():
            enriched_request = (
                f"{escalation.user_request}\n\n"
                f"[User decision on the council's open question(s)]: "
                f"{answer.strip()}"
            )
            hooks.on_status("approved", "Incorporating your decision.")
        else:
            # Non-interactive or skipped — proceed with the council's
            # recommendation and note the unanswered questions as risks.
            hooks.on_status(
                "warn",
                "No decision provided — proceeding with the council's "
                "recommended approach.",
            )

    # Surface a short summary to the user (non-blocking).
    with best_effort("council preview notify failed"):
        hooks.notify("Council", brief.render_user_preview(), "agent")

    return _dc_replace(
        escalation,
        user_request=enriched_request,
        design_brief=brief,
    )


def _run_execution_phase(
    *,
    agent: Any,
    engine: Any,
    task_prompt: tuple[str, str],
    plan: Any,
    session_id: str,
    use_phase_engine: bool,
    hooks: OrchestrationHooks,
    initial_attachments: list[Any] | None = None,
    task: Any | None = None,
) -> tuple[str, Any]:
    """Execution: dispatch to LoopEngine (or PhaseEngine for ``--think``).

    ``plan`` is the :class:`infinidev.engine.analysis.plan.Plan` produced
    by the analyst planner. It is passed to LoopEngine via
    ``initial_plan=`` so the developer starts with a pre-approved plan
    (steps marked ``user_approved=True``).

    Tree-engine flows (``/explore``, ``/brainstorm``) no longer enter
    here — they go through :func:`run_flow_task` instead.
    """
    hooks.on_phase("execute")
    hooks.on_status("info", f"Working on: {task_prompt[0][:120]}")

    agent.activate_context(session_id=session_id)
    try:
        if use_phase_engine:
            from infinidev.engine.phases.phase_engine import PhaseEngine

            _depth_config = None
            if hasattr(agent, "_gather_brief") and agent._gather_brief:
                with best_effort("gather depth-config resolution failed"):
                    from infinidev.gather.models import DEPTH_CONFIGS

                    _depth_config = DEPTH_CONFIGS.get(
                        agent._gather_brief.classification.depth
                    )
            phase_eng = PhaseEngine()
            result = phase_eng.execute(
                agent=agent,
                task_prompt=task_prompt,
                task_type="feature",
                verbose=True,
                depth_config=_depth_config,
            )
            used_engine: Any = phase_eng
        else:
            result = engine.execute(
                agent=agent,
                task_prompt=task_prompt,
                verbose=True,
                initial_plan=plan,
                initial_attachments=initial_attachments,
                task=task,
            )
            used_engine = engine
        if not result or not result.strip():
            result = "Done. (no additional output)"
    finally:
        agent.deactivate()

    return result, used_engine


def _run_review_phase(
    *,
    engine: Any,
    agent: Any,
    session_id: str,
    task_prompt: tuple[str, str],
    result: str,
    reviewer: Any,
    hooks: OrchestrationHooks,
    acceptance_criteria: list[str] | None = None,
) -> str:
    """Review: run the review-rework loop if enabled and applicable.

    The chat-agent redesign always routes through the ``develop`` flow,
    whose :class:`FlowConfig` has ``run_review=True``. The review is
    still guarded by ``REVIEW_ENABLED`` and ``engine.has_file_changes()``
    so read-only developer runs (which shouldn't happen post-escalation,
    but might during development) silently skip it.
    """
    from infinidev.config.settings import settings as _settings
    from infinidev.db.service import get_recent_summaries

    if not (_settings.REVIEW_ENABLED and engine.has_file_changes()):
        return result

    hooks.on_phase("review")
    hooks.on_status("info", "Running code review...")

    from infinidev.engine.analysis.review_engine import run_review_rework_loop

    def _on_review_status(level: str, msg: str) -> None:
        hooks.on_status(level, msg)
        if level == "verification_fail":
            hooks.notify(
                "System",
                "Re-running developer to fix test failures...",
                "system",
            )
        elif level == "objectives_fail":
            hooks.notify(
                "System",
                "Objective check regressed — re-running developer to fix it...",
                "system",
            )
        elif level == "objectives_unverified":
            hooks.notify(
                "System",
                f"Note: {msg}",
                "system",
            )
        elif level == "objectives_regressed":
            hooks.notify("System", f"⚠ Regression: {msg}", "system")
        elif level == "objectives_summary":
            hooks.notify("System", msg, "system")
        elif level == "rejected":
            hooks.notify(
                "System",
                "Re-running developer to fix review issues...",
                "system",
            )

    try:
        result, _ = run_review_rework_loop(
            engine=engine,
            agent=agent,
            session_id=session_id,
            task_prompt=task_prompt,
            initial_result=result,
            reviewer=reviewer,
            recent_messages=get_recent_summaries(session_id, limit=5),
            on_status=_on_review_status,
            acceptance_criteria=acceptance_criteria,
        )
    except Exception as exc:
        logger.error("Review phase failed: %s", exc, exc_info=True)
        hooks.on_status("error", f"Review error: {exc}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────


def _ken_turn_context(user_input: str, session_id: str) -> str:
    """Open Ken's session for this turn and return what it hands back.

    This is the *only* place infinidev writes a ``user_prompt`` row, and it
    has to stay that way. ``similar_past_sessions`` reads the last fifty
    such rows across **every** agent sharing this index — there is no agent
    filter — so one row per plan step would flush the window within a couple
    of tasks and cost the user's other sessions their predictive channel.

    Both calls answer with prompt text, and the answer is the point.
    ``start`` is idempotent, so only the first turn of a conversation pays
    for it and only that turn gets the resume brief — Ken's "here is where
    you left off", already wrapped in ``<ken-session-brief>``. ``prompt``
    answers every turn with the freshly ranked ``<context-rank>`` block.
    Neither is re-wrapped here: they arrive tagged, which is precisely how
    Ken's own hooks feed them to Claude Code.
    """
    try:
        from infinidev.engine.ken_session import get_ken_session

        session = get_ken_session(session_id=session_id)
        if session is None:
            return ""
        return _join_blocks(session.start(), session.prompt(user_input))
    except Exception:
        logger.debug("ken turn context failed", exc_info=True)
        return ""


def _report_turn_end_to_ken(result: str, session_id: str) -> None:
    """Close the assistant turn with the reply Ken needs to read.

    Called from the two *terminal* returns of :func:`run_task` and from
    neither of the re-entering ones: a ``task_end_instruction`` hook makes
    the turn continue, and the pass that finally answers the user is the
    one that ends it. Firing here on the way into ``_reenter`` would
    advance Ken's per-turn decay clock twice for one exchange.
    """
    try:
        from infinidev.engine.ken_session import get_ken_session

        session = get_ken_session(session_id=session_id)
        if session is not None:
            session.turn_end(result or "")
    except Exception:
        logger.debug("ken turn-end report failed", exc_info=True)


def _join_blocks(*parts: str | None) -> str:
    """Blank-line-join the parts that actually have content."""
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def run_task(
    *,
    agent: Any,
    user_input: str,
    session_id: str,
    engine: Any,
    reviewer: Any,
    hooks: OrchestrationHooks,
    use_phase_engine: bool = False,
    force_gather: bool = False,
    attachments: list[Any] | None = None,
    _hook_reentry: bool = False,
) -> str:
    """Run a complete task through the chat-agent-first pipeline.

    Flow:

      1. ChatAgent receives the user's message. Returns ``respond`` or
         ``escalate``.
      2. On ``respond`` — the reply is shown via ``hooks.notify`` and
         ``run_task`` returns the reply text. No analyst, no developer.
      3. On ``escalate`` — the ``user_visible_preview`` is shown, then
         the planner produces a :class:`Plan`. The plan overview is
         shown via ``hooks.notify`` (non-blocking — approval already
         happened in chat).
      4. Gather runs (if enabled), then the developer executes with
         ``initial_plan=plan``.
      5. Review runs if files changed.

    The ``analyst`` parameter is GONE. The old :class:`AnalysisEngine`
    was deleted in the same commit that introduced this rewrite.

    Callers must still construct ``agent``, ``engine``, ``reviewer``,
    and ``hooks``. The project_id and workspace_path for the chat agent
    are resolved from the agent's bound context.

    A turn is also the unit the user's ``task_start`` / ``task_end_*``
    hooks are scoped to (see ``engine/user_hooks/``). ``_hook_reentry`` is
    set on the one extra pass a ``task_end_instruction`` hook can ask for:
    it suppresses ``task_start`` and the end-of-task instruction hook, so
    the re-entered turn cannot request a third, and keeps the hook's text
    out of the stored chat history — that instruction is scaffolding for
    this turn, not something the next turn should read back as if the user
    had typed it.

    NOTE: ``run_chat_agent`` and ``run_planner`` are imported lazily to
    avoid a circular import: ``engine.orchestration.__init__`` eagerly
    imports this module (to expose ``run_task``), and the chat agent's
    module transitively triggers that ``__init__`` through
    ``escalation_packet``. Top-level imports here would deadlock at
    package initialisation time. A previous cleanup attempt hoisted
    them and crashed the CLI on cold start; see the revert commit.
    """
    from infinidev.engine.orchestration.chat_agent import run_chat_agent
    from infinidev.engine.analysis.planner import run_planner
    from infinidev.tools.base.context import (
        get_context_for_agent,
        get_current_project_id,
        get_current_workspace_path,
    )

    # Reset the per-turn "reply already shown" flag — hooks can be reused
    # across turns (the classic REPL keeps a single ClickHooks). The
    # respond branch sets it via mark_reply_shown() so callers don't
    # re-render the reply from this function's return value.
    if hasattr(hooks, "reply_already_shown"):
        hooks.reply_already_shown = False

    # Plumb the orchestration hooks into the engine so the inner loop
    # can forward on_file_change / on_step_start as the worker
    # advances — including during the rework loop, which calls
    # engine.execute() again. The attribute is the only place hooks
    # need to live; engine.execute() does not take them as a kwarg
    # to keep its public signature stable.
    try:
        setattr(engine, "_hooks", hooks)
    except AttributeError:
        pass

    # A turn is the scope of a cancellation. The engine outlives it (the
    # TUI keeps one per session) and execute() no longer clears the flag
    # itself, because a turn enters execute() several times — the review's
    # rework loop re-enters the same instance — and clearing it there
    # resurrected a run the user had stopped. The hook re-entry is the
    # same turn continuing, so it must not clear anything.
    if not _hook_reentry:
        with contextlib.suppress(AttributeError):
            engine.begin_turn()

    from infinidev.engine.task_runtime import TaskRuntime

    runtime = TaskRuntime(task_id=session_id, on_event=_runtime_event_bridge(hooks))
    ken_context = ""
    if not _hook_reentry:
        # A re-entered turn is the same user turn continuing, so it neither
        # appends a second user message to the transcript nor reports a
        # second prompt to Ken — the text is a hook's, not the user's. It
        # also needs no fresh ranking: the blocks the first pass injected
        # described this same request.
        runtime.append_chat("user", user_input)
        ken_context = _ken_turn_context(user_input, session_id)
    root_task = runtime.add_task(
        "Follow-up requested by hook" if _hook_reentry else "Working on your request"
    )
    runtime.start_next_task()

    # ── task_start hook ─────────────────────────────────────────────────
    # Its output rides along to both consumers of the turn: the chat agent
    # (which may answer without escalating) and, further down, the
    # developer's task prompt. Injected into the copies handed to those
    # two, never into ``user_input`` itself, which is already recorded.
    # Ken's blocks ride the same rail as the hook's output — both are
    # context about this turn that neither consumer should have to fetch for
    # itself, and both must stay out of ``user_input``, which is already
    # recorded as what the user typed.
    turn_context = _join_blocks(
        ken_context,
        _run_task_start_hook(
            user_input=user_input,
            session_id=session_id,
            skip=_hook_reentry,
        ),
    )
    chat_input = f"{user_input}\n\n{turn_context}" if turn_context else user_input

    def _reenter(instruction: str) -> str:
        """Run the turn again with a hook's instruction as the input.

        Going back through ``run_task`` rather than re-driving the engine
        directly is what makes the follow-up a real turn: it is classified,
        planned, executed and reviewed like any other. A hook that asks for
        a deep review gets the whole pipeline, not a bare loop iteration.
        Attachments are not carried over — they belonged to the user's
        message, and re-sending them would pay for the same images twice.
        """
        from infinidev.engine.user_hooks import task_instruction

        return run_task(
            agent=agent,
            user_input=task_instruction(instruction),
            session_id=session_id,
            engine=engine,
            reviewer=reviewer,
            hooks=hooks,
            use_phase_engine=use_phase_engine,
            force_gather=force_gather,
            attachments=None,
            _hook_reentry=True,
        )

    # ── Chat agent ──────────────────────────────────────────────────────
    hooks.on_phase("chat")
    agent_id = getattr(agent, "agent_id", None) or getattr(agent, "id", None)
    ctx = get_context_for_agent(agent_id) if agent_id else None
    # Fall back to the current process context (thread-local / ContextVar /
    # env) whenever the per-agent context is missing OR has a None field.
    # An empty ToolContext() is returned when the agent was never
    # activated — catching `ctx is not None` alone would silently pass
    # None through, which breaks every code-intel tool with
    # "No project context". Falling back per-field keeps partial contexts
    # usable too (e.g. agent has project_id but not workspace_path).
    agent_project_id = (
        ctx.project_id
        if ctx and ctx.project_id is not None
        else get_current_project_id()
    )
    agent_workspace = (
        ctx.workspace_path
        if ctx and ctx.workspace_path is not None
        else get_current_workspace_path()
    )
    # Last-resort fallback to the agent's own project_id attribute so
    # tools don't crash when nothing else has been set — matches what
    # activate_context would have written.
    if agent_project_id is None:
        agent_project_id = getattr(agent, "project_id", None)
    chat_result = run_chat_agent(
        chat_input,
        session_id=session_id,
        project_id=agent_project_id,
        workspace_path=agent_workspace,
        hooks=hooks,
        attachments=attachments,
    )

    if chat_result.kind == "respond":
        if chat_result.error_traceback:
            # Exception-fallback path: the chat loop crashed and the
            # reply is a generic apology. Route through notify_error so
            # the UI can show the traceback in a collapsible widget.
            # Streaming is also cleanly terminated by the caller in
            # chat_agent.run_chat_agent's except block.
            hooks.notify_error(
                "Infinidev",
                chat_result.reply,
                chat_result.error_traceback,
            )
        elif chat_result.streamed:
            # Streaming already showed the text to the user chunk-by-chunk.
            # Signal end-of-stream so the UI can flip the `streaming`
            # flag on the message and re-render with markdown styling
            # (otherwise the final message stays in plain-text mode).
            hooks.notify_stream_end("Infinidev", "agent")
        else:
            hooks.notify("Infinidev", chat_result.reply, "agent")
        # The reply has now been shown (streamed, notified, or surfaced as
        # an error). Tell UI adapters so they don't re-render it from
        # run_task's return value. Defensive: classic CLI hooks don't
        # define this — they print the return value from the caller.
        _mark_shown = getattr(hooks, "mark_reply_shown", None)
        if callable(_mark_shown):
            _mark_shown()
        # End-of-task hooks fire here too: the turn is over, even though it
        # never reached the developer. ``files_changed=False`` is what lets
        # a hook written for the develop path opt out of the chat path
        # without the pipeline second-guessing which hooks belong where.
        from infinidev.engine.user_hooks import UserHookEvent

        followup = _task_end_hook(
            UserHookEvent.TASK_END_INSTRUCTION,
            user_input=user_input, session_id=session_id,
            result=chat_result.reply, files_changed=False,
            status="responded", skip=_hook_reentry,
        )
        if followup:
            runtime.complete_current_task(chat_result.reply)
            runtime.append_chat("assistant", chat_result.reply)
            return _reenter(followup)
        _store_task_hook_note(session_id, _task_end_hook(
            UserHookEvent.TASK_END_SUMMARY,
            user_input=user_input, session_id=session_id,
            result=chat_result.reply, files_changed=False,
            status="responded",
        ))
        hooks.on_phase("idle")
        runtime.complete_current_task(chat_result.reply)
        runtime.append_chat("assistant", chat_result.reply)
        # A turn the chat agent answered alone is still a turn: it advances
        # Ken's decay clock, and its reply is scanned for cited paths just
        # like the developer's. Reporting only the develop path left every
        # conversational exchange invisible to the ranker.
        _report_turn_end_to_ken(chat_result.reply, session_id)
        return chat_result.reply

    # ── Planner (escalate path) ─────────────────────────────────────────
    escalation = chat_result.escalation
    assert escalation is not None  # enforced by ChatAgentResult invariants
    # If the chat agent streamed plain text before deciding to escalate,
    # the partial streaming bubble is still flagged streaming=True. Finalize
    # it now — otherwise the upcoming preview/plan messages get appended
    # after it and finalize can never match it again (it only flips the
    # LAST message), leaving the bubble stuck in raw-markdown mode forever.
    # No-op when nothing was streamed (finalize_streaming_message guards).
    if chat_result.streamed:
        hooks.notify_stream_end("Infinidev", "agent")
    if escalation.user_visible_preview:
        hooks.notify("Infinidev", escalation.user_visible_preview, "agent")

    # ── Spec elaboration (vague request → grounded spec) ────────────────
    # Runs before the council/planner so both build on a grounded spec
    # instead of the raw request. Single configured model; soft-fails to
    # the original escalation (returns None → no grounded_spec attached).
    escalation = _run_elaboration_phase(
        escalation=escalation,
        session_id=session_id,
        project_id=(ctx.project_id if ctx else get_current_project_id()),
        workspace_path=(ctx.workspace_path if ctx else get_current_workspace_path()),
        hooks=hooks,
    )

    # ── Council (optional multi-agent deliberation) ─────────────────────
    # Runs only when the chat agent flagged council_requested. Enriches
    # the escalation with a synthesised design_brief that the planner
    # then reads. Soft-fails to the original escalation.
    escalation = _run_council_phase(
        escalation=escalation,
        session_id=session_id,
        project_id=(ctx.project_id if ctx else get_current_project_id()),
        workspace_path=(ctx.workspace_path if ctx else get_current_workspace_path()),
        hooks=hooks,
    )

    hooks.on_phase("analysis")
    hooks.on_status("info", "Planning...")
    plan = run_planner(
        escalation,
        session_id=session_id,
        project_id=(ctx.project_id if ctx else get_current_project_id()),
        workspace_path=(ctx.workspace_path if ctx else get_current_workspace_path()),
        hooks=hooks,
    )
    hooks.notify("Planner", plan.overview, "agent")

    # Configure agent identity for the develop flow before gather/execute.
    from infinidev.engine.flows import get_flow_config
    from infinidev.prompts.flows import get_flow_identity

    flow_config = get_flow_config("develop")
    if hasattr(agent, "_system_prompt_identity"):
        agent._system_prompt_identity = get_flow_identity("develop")
    if hasattr(agent, "backstory"):
        agent.backstory = flow_config.backstory

    # Build the developer's task prompt from the planner output. The
    # overview is the description; the flow config supplies the
    # canonical expected_output.
    # The turn context — Ken's ranked block plus the task_start hook's
    # output — rides on the description, not on the expected_output: the
    # latter is the flow's contract and the reviewer judges against it, so
    # anything added there would change what "done" means.
    task_prompt: tuple[str, str] = (
        f"{escalation.user_request}\n\n{turn_context}"
        if turn_context
        else escalation.user_request,
        flow_config.expected_output,
    )

    # Wrap the user free-text into a structured ``Task`` artefact so
    # the developer prompt and the assistant critic both see the same
    # XML-rendered spec. When the planner authored real, falsifiable
    # acceptance criteria, they become the Task's contract (replacing the
    # synthesised placeholder) and is_synthesised() flips to False so the
    # critic/reviewer treat them as ground truth. If construction fails
    # (e.g. user_request too short), we fall back to ``None`` and the legacy
    # plain ``<task>`` block is used — the pipeline never breaks because of
    # an enrichment failure.
    structured_task: Any | None = None
    try:
        from infinidev.engine.orchestration.task_schema import task_from_free_text

        structured_task = task_from_free_text(
            escalation.user_request,
            acceptance_criteria=list(getattr(plan, "acceptance_criteria", []) or []),
        )
    except Exception:
        logger.debug(
            "structured Task synthesis failed; using legacy <task>", exc_info=True
        )

    # ── Gather ──────────────────────────────────────────────────────────
    task_prompt = _run_gather_phase(
        user_input=user_input,
        agent=agent,
        task_prompt=task_prompt,
        session_id=session_id,
        force_gather=force_gather,
        hooks=hooks,
    )

    # ── Execute ─────────────────────────────────────────────────────────
    result, used_engine = _run_execution_phase(
        agent=agent,
        engine=engine,
        task_prompt=task_prompt,
        plan=plan,
        session_id=session_id,
        use_phase_engine=use_phase_engine,
        hooks=hooks,
        initial_attachments=list(escalation.attachments)
        if escalation.attachments
        else None,
        task=structured_task,
    )

    # ── Review ──────────────────────────────────────────────────────────
    # Feed the reviewer the planner's real acceptance criteria (not the
    # synthesised placeholder) so it judges against the actual contract.
    review_criteria: list[str] | None = None
    if structured_task is not None:
        from infinidev.engine.orchestration.task_schema import is_synthesised

        if not is_synthesised(structured_task):
            review_criteria = list(structured_task.acceptance_criteria)

    # Review re-enters engine.execute() up to three times through the
    # rework loop, and a cancelled run is exactly the one whose tests fail
    # — it stopped half-way. Reviewing it restarted the developer on the
    # user's repository minutes after they asked it to stop.
    if getattr(used_engine, "is_cancelled", False):
        logger.info("run_task: cancelled — skipping review and end-of-task hooks")
        runtime.record_step(result, step_id=root_task.id)
        runtime.complete_current_task(result)
        return result

    result = _run_review_phase(
        engine=used_engine,
        agent=agent,
        session_id=session_id,
        task_prompt=task_prompt,
        result=result,
        reviewer=reviewer,
        hooks=hooks,
        acceptance_criteria=review_criteria,
    )

    # ── task_end_instruction hook ───────────────────────────────────────
    # Last chance to add work, after the reviewer has had its say. If the
    # hook prints anything the turn is re-entered with it and *that* pass
    # owns the ending — its own end-of-task summary, its own stored work
    # summary — so neither closing step happens twice.
    from infinidev.engine.user_hooks import UserHookEvent

    _turn_status = getattr(used_engine, "_last_status", "") or "completed"
    followup = _task_end_hook(
        UserHookEvent.TASK_END_INSTRUCTION,
        user_input=user_input, session_id=session_id, result=result,
        files_changed=_turn_changed_files(used_engine), status=_turn_status,
        skip=_hook_reentry,
    )
    if followup:
        runtime.record_step(result, step_id=root_task.id)
        runtime.complete_current_task(result)
        return _reenter(followup)

    # ── Hidden work summary ─────────────────────────────────────────────
    # Record what the developer loop just did as a hidden conversation
    # turn so the NEXT turn's chat agent has continuity instead of starting
    # cold. Best-effort: a failure here must never sink a completed task.
    _store_work_summary(used_engine, session_id, result)
    _store_task_hook_note(session_id, _task_end_hook(
        UserHookEvent.TASK_END_SUMMARY,
        user_input=user_input, session_id=session_id, result=result,
        files_changed=_turn_changed_files(used_engine), status=_turn_status,
    ))

    hooks.on_phase("idle")
    runtime.record_step(result, step_id=root_task.id)
    runtime.complete_current_task(result)
    runtime.append_chat("assistant", result)
    _report_turn_end_to_ken(result, session_id)
    return result


def _turn_changed_files(engine: Any) -> bool:
    """Whether the turn touched anything on disk, for the hook payload.

    ``has_file_changes`` is a LoopEngine method; the legacy PhaseEngine
    path has no tracker, and a hook asking "did anything change?" is better
    told "no" than handed an AttributeError from the finish path.
    """
    probe = getattr(engine, "has_file_changes", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:
        return False


def _run_task_start_hook(
    *, user_input: str, session_id: str, skip: bool,
) -> str:
    """Output of the user's ``task_start`` hook, wrapped for the prompt.

    Empty string whenever there is nothing to add — no hook, no output, a
    failure, or a re-entered turn. Callers concatenate unconditionally.
    """
    if skip:
        return ""
    from infinidev.engine.user_hooks import (
        UserHookEvent, context_block, run_hooks, task_payload,
    )
    from infinidev.tools.base.context import (
        get_current_project_id, get_current_workspace_path,
    )

    workspace = get_current_workspace_path() or ""
    output = None
    with best_effort("task_start hook failed"):
        output = run_hooks(
            UserHookEvent.TASK_START,
            task_payload(
                session_id=session_id,
                user_input=user_input,
                workspace_path=workspace,
                project_id=get_current_project_id(),
            ),
            workspace_path=workspace or None,
        )
    if not output:
        return ""
    return context_block(UserHookEvent.TASK_START, output.text)


def _task_end_hook(
    event: Any, *, user_input: str, session_id: str, result: str,
    files_changed: bool, status: str, skip: bool = False,
) -> str:
    """Run one of the two end-of-task hooks and return its text.

    Shared by both: they take the same payload and differ only in what the
    caller does with the answer — one becomes another pass of work, the
    other becomes a note that outlives the turn.
    """
    if skip:
        return ""
    from infinidev.engine.user_hooks import run_hooks, task_payload
    from infinidev.tools.base.context import (
        get_current_project_id, get_current_workspace_path,
    )

    workspace = get_current_workspace_path() or ""
    output = None
    with best_effort("%s hook failed", getattr(event, "value", event)):
        output = run_hooks(
            event,
            task_payload(
                session_id=session_id,
                user_input=user_input,
                workspace_path=workspace,
                project_id=get_current_project_id(),
                result=result,
                files_changed=files_changed,
                status=status,
            ),
            workspace_path=workspace or None,
        )
    return output.text.strip() if output else ""


def _store_task_hook_note(session_id: str, note: str) -> None:
    """Persist a ``task_end_summary`` hook's output past the end of the turn.

    Filed under the same hidden ``work_summary`` role the developer's own
    end-of-task summary uses, which is what makes it survive: that role is
    excluded from the UI repaint but included in the history the next
    turn's chat agent reads. A hook that prints "deploy is frozen until
    Friday" is still saying it on the next turn.
    """
    if not session_id or not note:
        return
    with best_effort("task_end_summary note store failed"):
        from infinidev.db.service import store_conversation_turn

        store_conversation_turn(session_id, "work_summary", note)


def _store_work_summary(engine: Any, session_id: str, result: str) -> None:
    """Persist the hidden end-of-task work summary, if the engine offers one.

    Only the LoopEngine exposes ``build_work_summary``; the legacy
    PhaseEngine path is skipped via the ``hasattr`` guard. The summary is
    stored under ``role="work_summary"`` — excluded from the UI repaint
    (``get_all_turns``) but included in the model's history
    (``get_recent_turns_full``), so it is hidden from the user yet seen by
    the chat agent next turn.
    """
    if not session_id or not hasattr(engine, "build_work_summary"):
        return
    try:
        status = getattr(engine, "_last_status", "") or "completed"
        summary = engine.build_work_summary(result or "", status)
        if not summary:
            return
        from infinidev.db.service import store_conversation_turn

        store_conversation_turn(session_id, "work_summary", summary)
        # Also file it in working memory so a later turn can *search* it
        # ("what did we change in the auth module last time?") rather than
        # only receiving it as fixed prelude context.
        with best_effort("work summary archive failed"):
            from infinidev.engine.working_memory import get_working_memory

            get_working_memory(session_id).remember(
                "Task summary", summary, kind="task_summary"
            )
    except Exception:
        logger.warning("failed to store work summary", exc_info=True)


def run_flow_task(
    *,
    agent: Any,
    flow: str,
    task_prompt: tuple[str, str],
    session_id: str,
    engine: Any,
    hooks: OrchestrationHooks,
    use_tree_engine: bool = False,
) -> str:
    """Run a single flow directly, skipping the chat agent and planner.

    Used by terminal commands like ``/init``, ``/explore``,
    ``/brainstorm`` where the flow is already known and there is
    nothing to classify. Review is also skipped — these flows produce
    summary text, not code changes that need verifying.
    """
    from infinidev.engine.flows import get_flow_config
    from infinidev.prompts.flows import get_flow_identity

    hooks.on_phase("execute")
    flow_config = get_flow_config(flow)
    if hasattr(agent, "_system_prompt_identity"):
        agent._system_prompt_identity = get_flow_identity(flow)
    if hasattr(agent, "backstory"):
        agent.backstory = flow_config.backstory

    agent.activate_context(session_id=session_id)
    try:
        if use_tree_engine:
            from infinidev.engine.tree import TreeEngine

            engine_to_use: Any = TreeEngine()
            result = engine_to_use.execute(
                agent=agent,
                task_prompt=task_prompt,
                mode=flow,
            )
        else:
            engine_to_use = engine
            result = engine_to_use.execute(
                agent=agent,
                task_prompt=task_prompt,
                verbose=True,
            )
        if not result or not result.strip():
            result = "Done."
    finally:
        agent.deactivate()

    hooks.on_phase("idle")
    return result
