"""Unified, UI-agnostic task pipeline.

This module owns the entire task lifecycle — analysis, gather, execute,
review — as a single function :func:`run_task`. Every side effect that
needs to reach a human (printing status, asking a question, showing a
spec, updating a STEPS panel) goes through the
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
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OrchestrationHooks — the only contract between pipeline and UI
# ─────────────────────────────────────────────────────────────────────────────

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
    def on_phase(self, phase: str) -> None:
        """Pipeline entered a new phase. *phase* is one of:
        ``"analysis"``, ``"gather"``, ``"execute"``, ``"review"``, ``"idle"``.
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

    # ── User interaction ─────────────────────────────────────────────────
    def ask_user(self, prompt: str, kind: str = "text") -> str | None:
        """Block until the user replies. Return the user's text answer
        (possibly empty), or ``None`` to indicate the caller cannot be
        interactive at all (single-shot mode). Pipeline branches that
        receive ``None`` MUST proceed with sensible defaults instead of
        failing.

        *kind* is a hint for the UI to render the prompt: ``"text"``
        (free-form), ``"confirm"`` (y/n), ``"clarification"`` (longer
        free-form, used during analysis question loop)."""

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
# Internal helpers (not exported)
# ─────────────────────────────────────────────────────────────────────────────

def _format_spec(spec: dict) -> str:
    """Render an analysis specification dict as a human-readable block.

    Used by both :func:`_run_analysis_phase` and the TUI's confirmation
    dialog. Pulled out so the formatting lives next to the code that
    consumes it, not duplicated across CLI and TUI implementations.
    """
    parts: list[str] = []
    if spec.get("summary"):
        parts.append(f"Summary: {spec['summary']}")
    for key in ("requirements", "hidden_requirements", "assumptions", "out_of_scope"):
        items = spec.get(key, [])
        if items:
            parts.append(f"\n{key.replace('_', ' ').title()}:")
            for item in items:
                parts.append(f"  - {item}")
    if spec.get("technical_notes"):
        parts.append(f"\nTechnical Notes: {spec['technical_notes']}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Phase implementations — pure functions, take hooks as parameter
# ─────────────────────────────────────────────────────────────────────────────

def _run_analysis_phase(
    *,
    user_input: str,
    analyst: Any,
    session_summaries: list[str],
    hooks: OrchestrationHooks,
    skip_analysis: bool = False,
) -> tuple[tuple[str, str] | None, Any, str]:
    """Analysis: classify, optionally ask questions, build a spec.

    Returns ``(task_prompt, analysis, flow)`` where *flow* may be:
      * the analysed flow name (``"develop"``, ``"document"``, ``"explore"``,
        ``"brainstorm"``, ``"sysadmin"``, ``"research"``)
      * ``"done"`` for trivial conversational replies (no execution)
      * ``"cancelled"`` if the user aborted at the spec confirmation step

    When analysis is disabled (globally via settings, or per-call via
    ``skip_analysis=True``), returns ``((user_input, default_expected),
    None, "develop")``. The per-call override exists for the imperative
    bypass in ``--prompt`` mode: action-verb tasks ("create", "fix",
    "refactor", ...) skip the analyst because the analyst was wrapping
    them in a "do NOT write files, only analyze" envelope.
    """
    from infinidev.config.settings import settings as _settings

    if skip_analysis or not _settings.ANALYSIS_ENABLED:
        return (user_input, "Complete the task and report findings."), None, "develop"

    # Pre-planning preamble: ALWAYS speak first. A single small LLM
    # call uses a forced ``respond`` tool to (1) write a short reply
    # to the user and (2) decide whether the heavy planning pipeline
    # should run. The reply is shown immediately so the user gets
    # feedback within ~1-2 seconds on every turn — for greetings the
    # pipeline short-circuits, for real tasks the analyst still runs
    # but with the user already informed that work is starting.
    from infinidev.engine.orchestration.conversational_fastpath import (
        try_conversational_fastpath,
    )
    preamble = try_conversational_fastpath(
        user_input, session_summaries=session_summaries,
    )
    if preamble is not None:
        preamble_result, preamble_reply, continue_planning = preamble
        # Always show the user-facing reply immediately.
        hooks.notify("Infinidev", preamble_reply, "agent")
        if not continue_planning:
            # Pure conversation — short-circuit the pipeline here.
            return (user_input, ""), preamble_result, "done"
        # Real work — fall through to the analyst with the reply
        # already on screen so the user knows the agent is working.

    analyst.reset()
    hooks.on_status("info", "Analyzing request...")
    analysis = analyst.analyze(user_input, session_summaries=session_summaries)

    # Question loop — only runs in interactive mode (ask_user returns
    # None for non-interactive callers, which short-circuits the loop).
    while analysis.action == "ask" and analyst.can_ask_more:
        questions_text = analysis.format_questions_for_user()
        hooks.notify("Analyst", questions_text, "agent")
        answer = hooks.ask_user(questions_text, kind="clarification")
        if answer is None or not answer.strip():
            break
        analyst.add_answer(questions_text, answer)
        analysis = analyst.analyze(
            user_input + "\n\nUser clarification: " + answer,
            session_summaries=session_summaries,
        )

    task_prompt = analysis.build_flow_prompt()

    # "done" pseudo-flow — analyst decided this was a greeting / trivial
    # question and produced its own reply directly. No execution needed.
    if analysis.flow == "done":
        hooks.notify(
            "Infinidev",
            analysis.reason or analysis.original_input,
            "agent",
        )
        return task_prompt, analysis, "done"

    # Spec confirmation — only required for the develop flow. Research,
    # document, sysadmin, etc. don't need user approval before running.
    if analysis.action == "proceed" and analysis.flow == "develop":
        spec_text = _format_spec(analysis.specification)
        if spec_text:
            hooks.notify("Analyst", spec_text, "agent")
        confirm = hooks.ask_user(
            "Proceed with implementation? (y to proceed, n to cancel, "
            "or feedback to revise)",
            kind="confirm",
        )
        # Non-interactive caller (confirm is None): proceed silently with
        # the analyst's spec as-is. This matches single-prompt behaviour.
        if confirm is not None:
            stripped = confirm.strip()
            if stripped.lower() in ("n", "no", "cancel"):
                return task_prompt, analysis, "cancelled"
            if stripped and stripped.lower() not in ("y", "yes"):
                desc, expected = task_prompt
                desc += f"\n\n## Additional User Feedback\n{stripped}"
                task_prompt = (desc, expected)

    # Apply flow config (sets the canonical expected_output for this flow).
    from infinidev.engine.flows import get_flow_config
    flow_config = get_flow_config(analysis.flow)
    desc, _ = task_prompt
    task_prompt = (desc, flow_config.expected_output)

    return task_prompt, analysis, analysis.flow


def _run_gather_phase(
    *,
    user_input: str,
    agent: Any,
    task_prompt: tuple[str, str],
    analysis: Any,
    session_id: str,
    force_gather: bool,
    hooks: OrchestrationHooks,
) -> tuple[str, str]:
    """Gather: collect codebase context before execution. Soft-fails: if
    the gather phase raises, the pipeline continues with the original
    task_prompt and reports the failure via ``on_status``."""
    from infinidev.config.settings import settings as _settings
    from infinidev.db.service import get_recent_summaries

    if not (_settings.GATHER_ENABLED or force_gather):
        return task_prompt

    try:
        from infinidev.gather import run_gather
        agent.activate_context(session_id=session_id)
        hooks.on_status("info", "Gathering context...")
        chat_history = [
            {"role": "user" if "[user]" in s.lower() else "assistant", "content": s}
            for s in get_recent_summaries(session_id, limit=10)
        ]
        brief = run_gather(user_input, chat_history, analysis, agent)
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
    flow: str,
    analysis: Any,
    session_id: str,
    use_phase_engine: bool,
    hooks: OrchestrationHooks,
) -> tuple[str, Any]:
    """Execution: dispatch to the appropriate engine.

    Returns ``(result_text, used_engine)``. The returned engine is the
    one that actually ran (LoopEngine, TreeEngine, or PhaseEngine) so
    the review phase can call ``has_file_changes()`` on the right
    instance — this matters for ``--think`` runs where PhaseEngine is
    used instead of the shared LoopEngine.
    """
    hooks.on_phase("execute")
    hooks.on_status("info", f"[{flow}] Working on: {task_prompt[0][:120]}")

    agent.activate_context(session_id=session_id)
    try:
        if flow in ("explore", "brainstorm"):
            from infinidev.engine.tree import TreeEngine
            tree_engine = TreeEngine()
            result = tree_engine.execute(
                agent=agent,
                task_prompt=task_prompt,
                mode=flow,
            )
            used_engine: Any = tree_engine
        elif use_phase_engine:
            from infinidev.engine.phases.phase_engine import PhaseEngine
            _task_type = "feature"
            if analysis is not None and hasattr(analysis, "specification"):
                _task_type = analysis.specification.get("task_type", "feature")
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
                task_type=_task_type,
                verbose=True,
                depth_config=_depth_config,
            )
            used_engine = phase_eng
        else:
            result = engine.execute(
                agent=agent,
                task_prompt=task_prompt,
                verbose=True,
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
    flow: str,
    flow_config: Any,
    hooks: OrchestrationHooks,
) -> str:
    """Review: run the review-rework loop if enabled and applicable."""
    from infinidev.config.settings import settings as _settings
    from infinidev.db.service import get_recent_summaries

    run_review = flow_config.run_review if flow_config else True
    if not (
        _settings.REVIEW_ENABLED
        and run_review
        and flow != "explore"
        and engine.has_file_changes()
    ):
        return result

    hooks.on_phase("review")
    hooks.on_status("info", "Running code review...")

    from infinidev.engine.analysis.review_engine import run_review_rework_loop

    def _on_review_status(level: str, msg: str) -> None:
        # Translate review-engine status callbacks into hook calls. The
        # review engine has its own callback shape (level + msg) which
        # we forward verbatim — UIs interpret known levels themselves.
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
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_task(
    *,
    agent: Any,
    user_input: str,
    session_id: str,
    engine: Any,
    analyst: Any,
    reviewer: Any,
    hooks: OrchestrationHooks,
    use_phase_engine: bool = False,
    force_gather: bool = False,
    skip_analysis: bool = False,
) -> str:
    """Run a complete task through the unified pipeline.

    This is the ONLY function that should be called from CLI / TUI / any
    future entry point. Each adapter is responsible for:

      1. Persisting the user turn (``store_conversation_turn``).
      2. Loading session summaries.
      3. Constructing an ``OrchestrationHooks`` implementation.
      4. Calling this function and persisting the result.

    Everything between those four steps lives here.

    Note on settings: this function does NOT call ``reload_all()``.
    Callers that want fresh-from-disk settings each turn (the classic
    interactive loop wants this so ``/settings`` changes apply between
    turns) should call it themselves before invoking ``run_task``.
    Callers that have applied in-memory overrides like ``--model``
    (single-prompt mode) MUST NOT call ``reload_all()`` or the
    overrides will be lost.
    """
    from infinidev.db.service import get_recent_summaries

    summaries = get_recent_summaries(session_id, limit=10)
    if hasattr(agent, "_session_summaries"):
        agent._session_summaries = summaries

    # ── Analysis ──────────────────────────────────────────────────────────
    hooks.on_phase("analysis")
    task_prompt, analysis, flow = _run_analysis_phase(
        user_input=user_input,
        analyst=analyst,
        session_summaries=summaries,
        hooks=hooks,
        skip_analysis=skip_analysis,
    )

    if flow in ("done", "cancelled"):
        hooks.on_phase("idle")
        # For "done" the analyst already produced the user-visible reply via
        # notify(); we still return it as the function result so the adapter
        # can persist it to conversation history.
        if flow == "done" and analysis is not None:
            return analysis.reason or analysis.original_input or ""
        return ""

    # Configure the agent identity/backstory for this flow before any
    # downstream phase touches it. Both gather and execute rely on this.
    from infinidev.engine.flows import get_flow_config
    from infinidev.prompts.flows import get_flow_identity
    flow_config = get_flow_config(flow)
    if hasattr(agent, "_system_prompt_identity"):
        agent._system_prompt_identity = get_flow_identity(flow)
    if hasattr(agent, "backstory"):
        agent.backstory = flow_config.backstory

    # ── Gather ────────────────────────────────────────────────────────────
    if flow == "develop":
        hooks.on_phase("gather")
        task_prompt = _run_gather_phase(
            user_input=user_input,
            agent=agent,
            task_prompt=task_prompt,
            analysis=analysis,
            session_id=session_id,
            force_gather=force_gather,
            hooks=hooks,
        )

    # ── Execute ───────────────────────────────────────────────────────────
    result, used_engine = _run_execution_phase(
        agent=agent,
        engine=engine,
        task_prompt=task_prompt,
        flow=flow,
        analysis=analysis,
        session_id=session_id,
        use_phase_engine=use_phase_engine,
        hooks=hooks,
    )

    # ── Review ────────────────────────────────────────────────────────────
    result = _run_review_phase(
        engine=used_engine,
        agent=agent,
        session_id=session_id,
        task_prompt=task_prompt,
        result=result,
        reviewer=reviewer,
        flow=flow,
        flow_config=flow_config,
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
    """Run a single flow directly, skipping analysis and review.

    Used by terminal commands like ``/init``, ``/explore``,
    ``/brainstorm`` where the flow is already known and there is
    nothing to classify. Review is also skipped — these flows produce
    summary text, not code changes that need verifying.

    *use_tree_engine* swaps the LoopEngine for a fresh TreeEngine
    instance (used by /explore and /brainstorm). Otherwise the shared
    *engine* is used.
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
