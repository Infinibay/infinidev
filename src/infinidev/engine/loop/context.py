"""Prompt construction for the plan-execute-summarize loop engine."""

from __future__ import annotations

import json
from typing import Any

from infinidev.engine._best_effort import best_effort
from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.prompt.tools_section import (
    build_tools_prompt_section,
    _build_tools_prompt_small,
)
from infinidev.engine.summarizer import SmartContextSummarizer

# Re-export so existing ``from infinidev.engine.loop.context import
# build_tools_prompt_section`` imports keep working after the extraction.
__all__ = [
    "build_system_prompt",
    "build_iteration_prompt",
    "build_tools_prompt_section",
    "CACHE_BREAKPOINT_MARKER",
]

# Sentinel inserted by ``build_system_prompt`` between the stable prefix
# (identity + tech + protocol) and the dynamic suffix (session context).
# ``prompt_cache.apply_prompt_caching`` detects this marker, splits the
# system message at that point and applies ``cache_control`` ONLY to the
# stable prefix — keeping the provider cache hot across iterations even
# as session_summaries grows. For providers without explicit cache
# breakpoints, the marker is stripped before the LLM call. Looks like a
# benign HTML comment if ever leaked.
CACHE_BREAKPOINT_MARKER = "<!--__INFINIDEV_CACHE_BREAK__-->"

# Static prompt text lives in loop/prompt/text.py; re-exported here so that
# existing `from ...loop.context import CLI_AGENT_IDENTITY, ...` keeps working.
from infinidev.engine.loop.prompt.text import (  # noqa: F401
    BEHAVIOR_GUIDELINES,
    BEHAVIOR_GUIDELINES_SMALL,
    CLI_AGENT_IDENTITY,
    CLI_AGENT_IDENTITY_SMALL,
    CRITIC_PROTOCOL_ADDENDUM,
    LOOP_PROTOCOL,
    LOOP_PROTOCOL_SMALL,
)



def build_system_prompt(
    backstory: str,
    *,
    tech_hints: list[str] | None = None,
    session_summaries: list[str] | None = None,
    identity_override: str | None = None,
    protocol_override: str | None = None,
    small_model: bool = False,
) -> str:
    """Combine CLI identity, tech guidelines, session context, and loop protocol.

    Args:
        identity_override: If provided, replaces CLI_AGENT_IDENTITY as the
            base identity section (used by analyst and other non-developer agents).
        protocol_override: If provided, replaces LOOP_PROTOCOL. Use for agents
            that don't need plan/step management (e.g. analyst).
        small_model: If True, use shortened prompts optimized for <25B models.
    """
    if small_model:
        identity = CLI_AGENT_IDENTITY_SMALL
        protocol = LOOP_PROTOCOL_SMALL
        behavior = BEHAVIOR_GUIDELINES_SMALL
    else:
        from infinidev.prompts.variants import get_variant

        identity = identity_override or get_variant("loop.identity") or CLI_AGENT_IDENTITY
        protocol = protocol_override or get_variant("loop.protocol") or LOOP_PROTOCOL
        behavior = BEHAVIOR_GUIDELINES

    # Stable prefix: identity + behavior + tech + protocol. Kept in this order
    # so the whole prefix can be cached as a single block — the dynamic session
    # context goes AFTER the cache breakpoint marker below. Behavior sits right
    # under the identity so the honesty/anti-laziness rules frame everything
    # the agent does, regardless of which prompt variant supplied the identity.
    parts: list[str] = [identity, behavior]

    # Tech-specific guidelines (skip for small models — too many tokens)
    if tech_hints and not small_model:
        from infinidev.prompts.tech import get_tech_prompt
        tech_sections = []
        for hint in tech_hints:
            prompt = get_tech_prompt(hint)
            if prompt:
                tech_sections.append(prompt)
        if tech_sections:
            parts.append("## Technology Guidelines\n\n" + "\n\n".join(tech_sections))

    parts.append(protocol)

    # When the pair-programming critic is enabled, teach the principal
    # that those `--- critic note ---` blocks at the end of tool
    # results are notes from a peer model and what to do with them.
    # Lazy import to avoid pulling settings at module import time.
    with best_effort("critic protocol addendum check failed"):
        from infinidev.config import settings as _settings
        if getattr(_settings, "ASSISTANT_LLM_ENABLED", False):
            parts.append(CRITIC_PROTOCOL_ADDENDUM)

    # Session context changes every iteration (step summaries grow over
    # time). Emit it AFTER the breakpoint marker so caching-capable
    # providers only mark the stable prefix as cacheable. Providers
    # without explicit cache control see the marker stripped by
    # ``apply_prompt_caching``.
    if session_summaries:
        numbered = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(session_summaries)
        )
        session_block = f"<session-context>\n{numbered}\n</session-context>"
        return (
            "\n\n".join(parts)
            + f"\n\n{CACHE_BREAKPOINT_MARKER}\n\n"
            + session_block
        )

    return "\n\n".join(parts)


def build_iteration_prompt(
    description: str,
    expected_output: str,
    state: LoopState,
    *,
    project_knowledge: list[dict] | None = None,
    context_rank_result: Any | None = None,
    max_context_tokens: int = 0,
    session_notes: list[str] | None = None,
    user_messages: list[str] | None = None,
    skip_plan: bool = False,
    small_model: bool = False,
    task: Any | None = None,  # infinidev.engine.orchestration.task_schema.Task
) -> str:
    """Build the user prompt for one iteration of the loop.

    Assembles <project-knowledge>, <task>, <notes>, <plan>,
    <previous-actions>, <current-action>, <next-actions>,
    <expected-output>, <user-message>, and <context-budget> XML blocks.
    """
    parts: list[str] = []

    _append_if(parts, _render_smart_summary(state, small_model))
    _append_if(parts, _render_project_knowledge(project_knowledge))
    _append_if(parts, _render_context_rank(context_rank_result))
    _append_if(parts, _render_workspace())
    _append_if(parts, _render_background_completions())
    _append_if(parts, _render_background_tasks())

    # Task: prefer the structured ``Task`` rendering when available
    # (set by the engine when the orchestration layer built one). Both
    # the principal AND the critic (via shared message history) see
    # the same block, so this is a single point of truth for "what
    # is the user asking for". Fall back to the legacy plain-text
    # block when no Task object was passed (legacy callers, tests).
    if task is not None:
        try:
            from infinidev.engine.orchestration.task_renderer import render_task_xml
            parts.append(render_task_xml(task))
        except Exception:
            # Defensive: if rendering somehow fails, never lose the
            # task — fall through to the plain block.
            parts.append(f"<task>\n{description}\n</task>")
    else:
        parts.append(f"<task>\n{description}\n</task>")

    # Reactive guidance — pre-baked how-to advice queued by the engine
    # at the end of the previous step when a stuck-pattern was detected.
    # Renders once and is consumed; the engine decides when to queue
    # the next one. Pure prompt overhead, no LLM call upstream.
    with best_effort("pending guidance render failed"):
        from infinidev.engine.guidance import drain_pending_guidance
        guidance_text = drain_pending_guidance(state)
        if guidance_text:
            parts.append(guidance_text)

    # File integrity notifications — the single source of truth for
    # "some file on disk is now syntactically broken". Populated by
    # the indexer on every successful parse, via one of three trigger
    # paths: (a) direct file tool writes, (b) the file watcher for
    # shell / external / IDE edits, (c) manual ``/reindex``. The
    # queue dedups by file path so a single breakage yields a single
    # warning; auto-heals when the file becomes valid again. See
    # ``code_intel.file_change_notifications`` for the full contract.
    with best_effort("file-integrity warning render failed"):
        from infinidev.code_intel.file_change_notifications import (
            drain_pending_notifications,
        )
        notifications = drain_pending_notifications()
        if notifications:
            lines = ["<file-integrity-warning>"]
            lines.append(
                "The following files on disk are currently in a "
                "syntactically broken state. Something (a shell "
                "command, an external editor, or one of your own "
                "edits) left them with tree-sitter parse errors. "
                "Read each file, understand what went wrong, and "
                "fix it with replace_lines or create_file before "
                "doing anything else — downstream tools that read "
                "these files will see garbage."
            )
            lines.append("")
            for n in notifications[:5]:
                lines.append(f"  {n.render()}")
            if len(notifications) > 5:
                lines.append(
                    f"  ... and {len(notifications) - 5} more broken file(s)."
                )
            lines.append("</file-integrity-warning>")
            parts.append("\n".join(lines))

    _append_if(parts, _render_opened_files(state))
    _append_if(parts, _render_session_notes(session_notes))
    _append_if(parts, _render_notes(state))
    # Note-taking nudges — two tiers: a gentle reminder when recent
    # tool calls haven't produced a note, and a stronger warning when a
    # step completed without ANY notes at all.
    note_nudge = _render_note_nudge(state, small_model)
    if note_nudge:
        parts.append(note_nudge)

    # Plan (if we have one) — skip for agents that don't use plans (e.g. analyst)
    if not skip_plan:
        # Plan overview: stable prose narrative from the planner. Rendered
        # every iteration so the agent keeps the big picture; the per-step
        # detail is shown only inside <current-action> for the active step.
        if state.plan.overview:
            parts.append(f"<plan-overview>\n{state.plan.overview}\n</plan-overview>")
        if state.plan.steps:
            parts.append(f"<plan>\n{state.plan.render()}\n</plan>")
        else:
            parts.append(
                "<plan>\nNo plan yet. Your FIRST action must be to call add_step(title=\"...\") "
                "2-3 times to create your initial steps, then call "
                "step_complete(summary=\"Plan created\", status=\"continue\").\n</plan>"
            )

    # Previous action summaries (rich format if available)
    if state.history:
        # Small models: only show last 2 records to save context
        history_slice = state.history[-2:] if small_model else state.history
        summaries = []
        for record in history_slice:
            lines = [f"### Step {record.step_index}: {record.summary}"]
            if record.changes_made:
                lines.append(f"  Changes: {record.changes_made}")
            if record.discovered_context:
                lines.append(f"  Context: {record.discovered_context}")
            if record.pending_items:
                lines.append(f"  Pending: {record.pending_items}")
            summaries.append("\n".join(lines))
        parts.append(f"<previous-actions>\n{chr(10).join(summaries)}\n</previous-actions>")

        # Consolidated anti-patterns from all steps — skip for small models
        if not small_model:
            all_anti = [r.anti_patterns for r in state.history if r.anti_patterns]
            if all_anti:
                avoid_lines = [f"- {ap}" for ap in all_anti]
                parts.append(
                    f"<avoid>\nDo NOT repeat these patterns from previous steps:\n"
                    f"{chr(10).join(avoid_lines)}\n</avoid>"
                )

        # Behavior summary from tracker — reinforces good patterns, warns about bad
        all_good = [msg for r in state.history for msg in r.behavior_good]
        all_bad = [msg for r in state.history for msg in r.behavior_bad]
        if all_good or all_bad:
            blines: list[str] = []
            if all_good:
                unique_good = list(dict.fromkeys(all_good))[-3:]
                blines.append("KEEP DOING: " + "; ".join(unique_good))
            if all_bad:
                unique_bad = list(dict.fromkeys(all_bad))[-3:]
                blines.append("STOP DOING: " + "; ".join(unique_bad))
            total_score = sum(r.behavior_score for r in state.history)
            blines.append(f"Behavior score: {total_score:+d}")
            parts.append(f"<behavior-summary>\n{chr(10).join(blines)}\n</behavior-summary>")

    # Current action — skip for agents that don't use plans
    active = state.plan.active_step if not skip_plan else None
    if active:
        parts.append(_render_current_action(active, state, small_model))
    elif state.plan.steps:
        # All planned steps are done — prompt to continue or finish
        parts.append(
            "<current-action>\n"
            "All planned steps are complete. Review what was accomplished against the task requirements.\n"
            "Either add new steps via add_step() if more work is needed,\n"
            "or call step_complete(status=\"done\", final_answer=\"...\") if the task is fully complete.\n"
            "</current-action>"
        )

    # Next actions (pending steps after current) — skip for non-plan agents
    if state.plan.steps and not skip_plan:
        next_steps = [
            s for s in state.plan.steps
            if s.status == "pending" and (active is None or s.index > active.index)
        ]
        if next_steps:
            lines = [f"{s.index}. {s.title}" for s in next_steps]
            parts.append(f"<next-actions>\n{chr(10).join(lines)}\n</next-actions>")

    # User messages injected mid-task (live guidance from the user).
    # The user is a human watching the agent work in a live session.
    # Silence is rude — they expect an acknowledgement BEFORE the
    # agent goes back to whatever it was doing. The wording below is
    # deliberately strong because models otherwise just fold the
    # message into their thinking and never tell the user they saw it.
    if user_messages:
        for msg in user_messages:
            parts.append(
                "<urgent-user-message>\n"
                "The user just sent this message WHILE you were working:\n\n"
                f"  \"{msg}\"\n\n"
                "YOUR VERY NEXT TOOL CALL MUST BE `send_message` with a brief "
                "(1-2 sentence) acknowledgement. Tell the user you saw their "
                "message and what you'll do about it. Examples:\n"
                "  send_message(message=\"Got it — I'll finish this edit and "
                "then look into your question.\")\n"
                "  send_message(message=\"Recibido. Pauso lo que estaba "
                "haciendo y voy a esto primero.\")\n\n"
                "ONLY AFTER sending the acknowledgement, continue your work. "
                "If the user is changing the task or asking you to stop, also "
                "call `modify_step` / `step_complete` with the new direction. "
                "Do NOT silently fold this message into your thinking — the "
                "user is waiting to hear from you.\n"
                "</urgent-user-message>"
            )

    # Expected output — prefer the active step's self-declared success criterion
    # over any global task-level expected output. The active step's
    # `expected_output` is something the model itself committed to when it
    # planned the step (via add_step), so it acts as a verification anchor.
    # The global `expected_output` (if set, e.g. by a flow that wraps the task)
    # is rendered only as a fallback when the active step has no criterion.
    active_step = state.plan.active_step if state.plan else None
    step_criterion = (active_step.expected_output.strip()
                      if active_step and active_step.expected_output else "")
    if step_criterion:
        parts.append(
            "<expected-output>\n"
            "This is the success criterion you set for the current step. "
            "Treat it as the test the step must pass before you move on.\n\n"
            f"  {step_criterion}\n\n"
            "Once the step's work is done, verify this criterion with a concrete "
            "check (read the file you edited, run the test, inspect the command "
            "output). Only then run step_complete. If the criterion turned out to "
            "be wrong or incomplete, use modify_step to refine it instead of "
            "lying to yourself about a green check.\n"
            "</expected-output>"
        )
    elif expected_output:
        parts.append(
            "<expected-output>\n"
            "Overall task expectation (set by the caller, not by you):\n\n"
            f"  {expected_output}\n\n"
            "When you create steps with add_step, give each one its own "
            "expected_output — a short, verifiable criterion you can check "
            "before completing that step. Run step_complete only after you have "
            "actually verified the step's outcome.\n"
            "</expected-output>"
        )

    # Context budget — inform the agent how much context remains
    # Use last_prompt_tokens (actual context window usage) not total_tokens (cumulative)
    if max_context_tokens > 0 and state.last_prompt_tokens > 0:
        used = state.last_prompt_tokens
        remaining = max(0, max_context_tokens - used)
        pct_used = min(100.0, (used / max_context_tokens) * 100)

        budget_lines = [
            f"Tokens used: {used} / {max_context_tokens} ({pct_used:.0f}%)",
            f"Tokens remaining: {remaining}",
        ]

        if pct_used >= 85:
            budget_lines.append(
                "⚠ CRITICAL: Context window almost full. You MUST wrap up immediately. "
                "Call step_complete with status=\"done\" and a final_answer summarizing "
                "what was accomplished and what remains unfinished."
            )
        elif pct_used >= 70:
            budget_lines.append(
                "⚠ WARNING: Context window running low. Finish the current step, then "
                "call step_complete with status=\"done\". In your final_answer, include "
                "a summary of what was done and list any remaining work as follow-up steps "
                "the user can request in a new conversation."
            )

        parts.append(
            "<context-budget>\n"
            + "\n".join(budget_lines)
            + "\n</context-budget>"
        )

    return "\n\n".join(parts)


# ── iteration prompt helpers ──────────────────────────────────────────────


def _append_if(parts: list[str], block: str) -> None:
    """Append ``block`` to ``parts`` only if it's non-empty.

    Lets the body of ``build_iteration_prompt`` be a flat list of
    ``_append_if(parts, _render_X(...))`` calls instead of an
    ``if X: parts.append(...)`` forest.
    """
    if block:
        parts.append(block)


def _render_smart_summary(state: LoopState, small_model: bool) -> str:
    """Smart context summary — skipped for small models (redundant with notes)."""
    if small_model:
        return ""
    summarizer = SmartContextSummarizer()
    smart_summary = summarizer.generate_summary(state)
    if not smart_summary:
        return ""
    return (
        "<smart-context-summary>\n"
        f"Loop progress summary:\n{smart_summary}\n</smart-context-summary>"
    )


def _render_project_knowledge(project_knowledge: list[dict] | None) -> str:
    """Auto-injected facts from the project knowledge DB."""
    if not project_knowledge:
        return ""
    kb_lines = [
        f"- [{f['finding_type']}] {f['topic']}: {f['content']}"
        for f in project_knowledge
    ]
    return (
        "<project-knowledge>\n"
        "Known facts about this project (from previous sessions):\n"
        + "\n".join(kb_lines)
        + "\n</project-knowledge>"
    )


def _render_context_rank(result: Any | None) -> str:
    """Render the ContextRank section with enriched resource previews.

    For files: includes symbol outlines (functions, classes, methods)
    from code intelligence so the model can act without extra tool calls.
    For findings: includes the finding content.
    """
    if result is None or result.empty:
        return ""
    lines = ["<context-rank>",
             "Based on your current task and past sessions, these resources are likely relevant.",
             "Symbol outlines are included so you can act on them directly."]
    if result.files:
        lines.append("\nFiles (by relevance):")
        for i, item in enumerate(result.files, 1):
            lines.append(f"  {i}. {item.target}  [score={item.score:.1f}] — {item.reason}")
            # Enrich with symbol outline from code intelligence
            outline = _get_file_symbol_outline(item.target)
            if outline:
                for sym_line in outline:
                    lines.append(f"       {sym_line}")
    if result.symbols:
        lines.append("\nSymbols:")
        for i, item in enumerate(result.symbols, 1):
            lines.append(f"  {i}. {item.target}  [score={item.score:.1f}] — {item.reason}")
            sig = _get_symbol_signature(item.target)
            if sig:
                lines.append(f"       {sig}")
    if result.findings:
        lines.append("\nFindings:")
        for i, item in enumerate(result.findings, 1):
            lines.append(f"  {i}. {item.target}  [score={item.score:.1f}] — {item.reason}")
            content = _get_finding_content(item.target)
            if content:
                lines.append(f"       {content}")
    lines.append("</context-rank>")
    return "\n".join(lines)


def _get_file_symbol_outline(file_path: str) -> list[str]:
    """Fetch symbol outline for a file from code intelligence."""
    try:
        from infinidev.tools.base.context import get_current_project_id, get_current_workspace_path
        from infinidev.code_intel.query import list_symbols
        import os
        project_id = get_current_project_id()
        if not project_id:
            return []
        # Resolve relative paths — ci_symbols stores absolute paths
        if not os.path.isabs(file_path):
            workspace = get_current_workspace_path() or os.getcwd()
            file_path = os.path.join(workspace, file_path)
        symbols = list_symbols(project_id, file_path, limit=30)
        if not symbols:
            return []
        result = []
        for s in symbols:
            kind = s.kind.value if hasattr(s.kind, 'value') else str(s.kind)
            if kind in ("function", "method", "class", "interface", "enum", "type_alias"):
                sig = s.signature or s.name
                line_info = f"L{s.line_start}"
                if s.line_end:
                    line_info += f"-{s.line_end}"
                parent = f" ({s.parent_symbol})" if s.parent_symbol else ""
                line = f"[{kind}] {sig}{parent}  {line_info}"
                # Append docstring (truncated) if available
                if s.docstring:
                    doc = s.docstring.replace("\n", " ").strip()
                    if len(doc) > 120:
                        doc = doc[:117] + "..."
                    line += f"\n         → {doc}"
                result.append(line)
        return result[:20]  # Cap to avoid prompt bloat
    except Exception:
        return []


def _get_symbol_signature(qualified_name: str) -> str:
    """Fetch signature for a symbol from code intelligence."""
    try:
        from infinidev.tools.base.context import get_current_project_id
        from infinidev.code_intel._db import execute_with_retry
        project_id = get_current_project_id()
        if not project_id:
            return ""
        def _query(conn):
            row = conn.execute(
                "SELECT signature, docstring, file_path, line_start FROM ci_symbols "
                "WHERE project_id = ? AND qualified_name = ? LIMIT 1",
                (project_id, qualified_name),
            ).fetchone()
            return row
        row = execute_with_retry(_query)
        if row:
            sig = row["signature"] or qualified_name
            loc = f"{row['file_path']}:{row['line_start']}"
            doc = f" — {row['docstring'][:100]}" if row["docstring"] else ""
            return f"{sig}  ({loc}){doc}"
        return ""
    except Exception:
        return ""


def _get_finding_content(topic: str) -> str:
    """Fetch finding content by topic."""
    try:
        from infinidev.tools.base.context import get_current_project_id
        from infinidev.code_intel._db import execute_with_retry
        project_id = get_current_project_id()
        if not project_id:
            return ""
        def _query(conn):
            row = conn.execute(
                "SELECT content FROM findings "
                "WHERE project_id = ? AND topic = ? ORDER BY updated_at DESC LIMIT 1",
                (project_id, topic),
            ).fetchone()
            return row
        row = execute_with_retry(_query)
        if row and row["content"]:
            content = row["content"]
            return content[:300] + ("..." if len(content) > 300 else "")
        return ""
    except Exception:
        return ""


def _render_workspace() -> str:
    """Tell the LLM which directory relative paths resolve against."""
    from infinidev.tools.base.context import get_current_workspace_path
    workspace = get_current_workspace_path() or ""
    if not workspace:
        import os
        workspace = os.getcwd()
    if not workspace:
        return ""
    return (
        f"<workspace>\nCurrent working directory: {workspace}\n"
        "All relative file paths are resolved against this directory.\n</workspace>"
    )


def _render_background_tasks() -> str:
    """Render the ``<background-tasks>`` block for in-flight background commands.

    Reads the process-global background task manager. Surfaces every task the
    agent launched with ``run_in_background`` so it remembers what is still
    running (and notices when a background command has died). Output is NOT
    inlined here — the agent calls ``background_status`` for that — we only
    show the label, status, runtime, and exit code so the section stays cheap.
    """
    try:
        from infinidev.tools.shell.background_manager import get_background_manager
    except Exception:
        return ""

    tasks = get_background_manager().list()
    if not tasks:
        return ""

    lines = [
        "<background-tasks>",
        "Commands you started with run_in_background. They are running (or "
        "have finished) independently of your turn. Use background_status to "
        "read their output and stop_background_task to stop one.",
    ]
    for t in tasks:
        lines.append(f"  [{t.id}] {t.description} — {t.status_line()}")
    lines.append("</background-tasks>")
    return "\n".join(lines)


def _render_background_completions() -> str:
    """Render a one-shot ``<background-task-finished>`` notice.

    Drains the background manager's completion queue — populated by a task's
    pump thread the moment it exits naturally. This is the signal that lets the
    agent notice a build/test/migration that finished WHILE it was busy on
    something else, without having to poll. Consuming: each finished task is
    announced exactly once, then cleared. Tasks the agent stopped itself, or
    already inspected via wait/background_status, never appear here.
    """
    try:
        from infinidev.tools.shell.background_manager import (
            drain_completed_notifications,
        )
    except Exception:
        return ""

    finished = drain_completed_notifications()
    if not finished:
        return ""

    lines = [
        "<background-task-finished>",
        "A background command you started has finished since your last step "
        "(it ran independently while you were working). If its result matters "
        "to what you're doing, read its output with background_status.",
    ]
    for t in finished:
        code = t.exit_code
        verdict = "ok" if code == 0 else f"FAILED (exit {code})"
        lines.append(
            f"  [{t.id}] {t.description} — exited {verdict} "
            f"after {t.runtime_seconds():.0f}s"
        )
    lines.append("</background-task-finished>")
    return "\n".join(lines)


def _render_opened_files(state: LoopState) -> str:
    """Files the agent has read or written recently, with TTL labels.

    The ``<opened-files>`` block is the mechanism that lets the model
    avoid redundant ``read_file`` calls between steps — the content IS
    the current file content (refreshed on every edit via the cache
    handlers in ``tool_executor.py``). Pinned entries are written-by-
    model files; the TTL number on unpinned entries counts down to
    eviction as more tool calls happen.
    """
    if not state.opened_files:
        return ""
    file_sections: list[str] = []
    for path, of in state.opened_files.items():
        if of.pinned:
            label = f"### {path} (written by you — pinned)\n```\n{of.content}\n```"
        else:
            label = f"### {path} (expires in {of.ttl} tool calls)\n```\n{of.content}\n```"
        file_sections.append(label)
    return (
        "<opened-files>\n"
        "IMPORTANT: These files are already loaded and up-to-date. "
        "Do NOT call read_file on them — the content below IS the current file content. "
        "After you edit a file, it is automatically refreshed here.\n\n"
        + "\n\n".join(file_sections)
        + "\n</opened-files>"
    )


def _render_session_notes(session_notes: list[str] | None) -> str:
    """Notes that persist across tasks within the same CLI session."""
    if not session_notes:
        return ""
    lines = [f"{i+1}. {n}" for i, n in enumerate(session_notes)]
    return (
        "<session-notes>\nNotes from previous tasks in this session:\n"
        + "\n".join(lines)
        + "\n</session-notes>"
    )


def _render_notes(state: LoopState) -> str:
    """The scratchpad notes the model has saved via ``add_note`` this task."""
    if not state.notes:
        return ""
    lines = [f"{i+1}. {n}" for i, n in enumerate(state.notes)]
    return (
        "<notes>\nYour notes from previous steps:\n"
        + "\n".join(lines)
        + "\n</notes>"
    )


def _render_note_nudge(state: LoopState, small_model: bool) -> str:
    """Build the note-taking nudge block, or empty if none applies.

    Two independent tiers fire:
    - ``recent``: model has been tool-calling without saving notes.
    - ``zero``:   a step completed and not a single note exists.
    Both can fire in the same iteration; they address different
    failure modes. Kept separate in the rendered output for clarity.
    """
    blocks: list[str] = []

    if state.tool_calls_since_last_note >= 4 and state.total_tool_calls >= 4:
        if small_model:
            blocks.append("⚠ SAVE NOTES NOW — call add_note with file paths and findings.")
        else:
            blocks.append(
                "<note-reminder>\n"
                "You have made multiple tool calls without saving notes. Your context resets "
                "each step — anything not in add_note will be lost. Save key facts NOW: "
                "file paths, function locations, decisions made, values discovered.\n"
                "</note-reminder>"
            )

    if state.history and not state.notes and state.total_tool_calls >= 4:
        if small_model:
            blocks.append("⚠ WARNING: ZERO notes saved. Call add_note NOW or you will lose all context.")
        else:
            blocks.append(
                "<note-warning>\n"
                "WARNING: You have completed step(s) but have ZERO notes saved. "
                "Your context from previous steps is limited to ~150-token summaries. "
                "Critical details (file paths, line numbers, function signatures, decisions) "
                "MUST be saved via add_note or they are permanently lost.\n"
                "</note-warning>"
            )

    return "\n\n".join(blocks)


def _render_current_action(active: Any, state: LoopState, small_model: bool) -> str:
    """Render the ``<current-action>`` block for the active step.

    Small models get a terse "DO NOW" form; regular models get an
    explicit scope constraint listing the next pending steps as
    off-limits. Extracted from ``build_iteration_prompt`` so the caller
    no longer carries both variants inline.
    """
    if small_model:
        guidance = f"\n{active.explanation}" if active.explanation else ""
        detail = f"\n\n{active.detail}" if getattr(active, "detail", "") else ""
        return (
            f"<current-action>\nDO NOW: Step {active.index} — {active.title}"
            f"{guidance}{detail}\n</current-action>"
        )

    scope_warning = ""
    next_pending = [s for s in state.plan.steps if s.status == "pending"]
    if next_pending:
        off_limits = ", ".join(f'"{s.title}"' for s in next_pending[:3])
        scope_warning = (
            f"\n\nSCOPE CONSTRAINT: This step is ONLY about: {active.title}\n"
            f"Do NOT work on future steps: {off_limits}\n"
            f"If you discover that this step requires work from future steps, "
            f"call step_complete with status='continue' and add new steps."
        )
    guidance = f"\n\n{active.explanation}" if active.explanation else ""
    detail = f"\n\n{active.detail}" if getattr(active, "detail", "") else ""
    return (
        f"<current-action>\nStep {active.index}: {active.title}"
        f"{guidance}{detail}{scope_warning}\n</current-action>"
    )

