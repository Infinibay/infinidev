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

import logging
from typing import Any, Literal, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OrchestrationHooks — the only contract between pipeline and UI
# ─────────────────────────────────────────────────────────────────────────────


Phase = Literal["chat", "analysis", "gather", "execute", "review", "idle"]


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

    def on_status(self, level: str, msg: str) -> None:
        """Status line for ad-hoc updates. *level* is informational
        (``"info"``, ``"warn"``, ``"error"``, ``"verification_pass"``,
        ``"verification_fail"``, ``"approved"``, ``"rejected"``,
        ``"max_reviews"``). UIs may colourise based on level."""

    def notify(self, speaker: str, msg: str, kind: str = "agent") -> None:
        """A speaker is producing a chat-style message. *kind* is one of
        ``"agent"`` (a model output), ``"system"`` (process feedback),
        ``"error"`` (failure)."""

    def notify_stream_chunk(
        self, speaker: str, chunk: str, kind: str = "agent",
    ) -> None:
        """Append a streaming text chunk. The FIRST chunk for a new
        ``(speaker, kind)`` pair creates a message; subsequent chunks
        append to that same message. A later :meth:`notify` call
        implicitly ends the stream — the next chunk opens a new
        message. UIs that render streaming incrementally (TUI chat
        panel, terminal with cursor control) override this; stateless
        adapters may concatenate and defer until a sentinel arrives."""

    def notify_stream_end(
        self, speaker: str, kind: str = "agent",
    ) -> None:
        """Mark the in-progress streaming message as complete.

        The distinction from ``notify`` is important for rich UIs:
        during streaming, markdown / syntax highlighting is typically
        skipped (partial ``**bold`` etc. renders ugly). Once the stream
        ends, the UI re-renders the message with full styling applied.
        Stateless adapters (e.g. plain terminal echo) can no-op this —
        the final newline they add on ``notify`` or session end is
        enough."""

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
                try:
                    from infinidev.gather.models import DEPTH_CONFIGS
                    _depth_config = DEPTH_CONFIGS.get(
                        agent._gather_brief.classification.depth
                    )
                except Exception:
                    pass
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

    if not (
        _settings.REVIEW_ENABLED
        and engine.has_file_changes()
    ):
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
        )
    except Exception as exc:
        logger.error("Review phase failed: %s", exc, exc_info=True)
        hooks.on_status("error", f"Review error: {exc}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────


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
    agent_project_id = ctx.project_id if ctx and ctx.project_id is not None else get_current_project_id()
    agent_workspace = ctx.workspace_path if ctx and ctx.workspace_path is not None else get_current_workspace_path()
    # Last-resort fallback to the agent's own project_id attribute so
    # tools don't crash when nothing else has been set — matches what
    # activate_context would have written.
    if agent_project_id is None:
        agent_project_id = getattr(agent, "project_id", None)
    chat_result = run_chat_agent(
        user_input,
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
                "Infinidev", chat_result.reply, chat_result.error_traceback,
            )
        elif chat_result.streamed:
            # Streaming already showed the text to the user chunk-by-chunk.
            # Signal end-of-stream so the UI can flip the `streaming`
            # flag on the message and re-render with markdown styling
            # (otherwise the final message stays in plain-text mode).
            hooks.notify_stream_end("Infinidev", "agent")
        else:
            hooks.notify("Infinidev", chat_result.reply, "agent")
        hooks.on_phase("idle")
        return chat_result.reply

    # ── Planner (escalate path) ─────────────────────────────────────────
    escalation = chat_result.escalation
    assert escalation is not None  # enforced by ChatAgentResult invariants
    if escalation.user_visible_preview:
        hooks.notify("Infinidev", escalation.user_visible_preview, "agent")

    hooks.on_phase("analysis")
    hooks.on_status("info", "Planning...")
    plan = run_planner(
        escalation,
        session_id=session_id,
        project_id=(ctx.project_id if ctx else get_current_project_id()),
        workspace_path=(ctx.workspace_path if ctx else get_current_workspace_path()),
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
    task_prompt: tuple[str, str] = (
        escalation.user_request,
        flow_config.expected_output,
    )

    # Wrap the user free-text into a structured ``Task`` artefact so
    # the developer prompt and the assistant critic both see the same
    # XML-rendered spec. Today this is auto-synthesised from the user
    # request; in a follow-up the chat agent / planner can produce a
    # richer Task with explicit acceptance criteria, out_of_scope, etc.
    # If construction fails (e.g. user_request too short), we fall
    # back to ``None`` and the legacy plain ``<task>`` block is used —
    # the pipeline never breaks because of an enrichment failure.
    structured_task: Any | None = None
    try:
        from infinidev.engine.orchestration.task_schema import task_from_free_text
        structured_task = task_from_free_text(escalation.user_request)
    except Exception:
        logger.debug("structured Task synthesis failed; using legacy <task>", exc_info=True)

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
        initial_attachments=list(escalation.attachments) if escalation.attachments else None,
        task=structured_task,
    )

    # ── Review ──────────────────────────────────────────────────────────
    result = _run_review_phase(
        engine=used_engine,
        agent=agent,
        session_id=session_id,
        task_prompt=task_prompt,
        result=result,
        reviewer=reviewer,
        hooks=hooks,
    )

    hooks.on_phase("idle")
    return result


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
