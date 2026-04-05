"""Phase 2: Investigation — answers questions by reading the codebase."""

from __future__ import annotations

import logging
from typing import Any

from infinidev.engine.loop import LoopEngine
from infinidev.engine.engine_logging import log as _log, DIM, RESET, CYAN, YELLOW
from infinidev.engine.phases.question_generator import _generate_questions, _generate_followups
from infinidev.prompts.phases import PhaseStrategy

logger = logging.getLogger(__name__)

# Tools allowed during INVESTIGATE phase (read-only)
_READ_ONLY_TOOLS = {
    "read_file", "list_directory", "glob", "code_search",
    "project_structure", "find_definition", "find_references",
    "list_symbols", "search_symbols", "get_symbol_code",
    "read_findings", "search_findings",
    "web_search", "web_fetch", "execute_command",
}

_MAX_FOLLOWUP_DEPTH = 2


def _investigate(agent: Any,
    questions: list[dict[str, Any]],
    strategy: PhaseStrategy,
    all_tools: list | None,
    verbose: bool,
) -> list[dict[str, str]]:
    """Run one mini-LoopEngine per question with read-only tools."""
    # Filter to read-only tools — INVESTIGATE must NOT modify files
    if all_tools:
        read_tools = [
            t for t in all_tools
            if getattr(t, 'name', '') in _READ_ONLY_TOOLS
        ]
    else:
        agent_tools = getattr(agent, 'tools', []) or []
        read_tools = [
            t for t in agent_tools
            if getattr(t, 'name', '') in _READ_ONLY_TOOLS
        ] if agent_tools else []

    answers: list[dict[str, str]] = []
    all_notes: list[str] = []  # ALL notes across ALL questions
    previous_text = ""

    for i, q in enumerate(questions):
        q_text = q["question"]
        if verbose:
            _log(f"  {CYAN}Q{i+1}/{len(questions)}: {q_text[:80]}{RESET}")

        # Build previous answers context
        if answers:
            prev_lines = "\n".join(
                f"  Q: {a['question']}\n  A: {a['answer']}"
                for a in answers
            )
            previous_text = f"## PREVIOUS ANSWERS\n{prev_lines}"

        # Format the investigate prompt
        inv_prompt = strategy.investigate_prompt.replace(
            "{{q_num}}", str(i + 1)
        ).replace(
            "{{q_total}}", str(len(questions))
        ).replace(
            "{{question}}", q_text
        ).replace(
            "{{previous_answers}}", previous_text
        )

        from infinidev.config.llm import _is_small_model as _is_sm
        _max_iters = 2 if _is_sm() else 3

        engine = LoopEngine()
        result = engine.execute(
            agent=agent,
            task_prompt=(inv_prompt, "Answer the question with add_note."),
            verbose=verbose,
            task_tools=read_tools,
            max_iterations=_max_iters,
            max_total_tool_calls=strategy.investigate_max_tool_calls,
            max_tool_calls_per_action=strategy.investigate_max_tool_calls,
            nudge_threshold=strategy.investigate_max_tool_calls - 2,
            summarizer_enabled=False,
            identity_override=strategy.investigate_identity or None,
        )

        # Collect ALL notes from the engine
        if engine._last_state and engine._last_state.notes:
            for note in engine._last_state.notes:
                if note not in all_notes:
                    all_notes.append(note)

        # Use all notes from this question as the answer summary
        answer_text = result or "No answer found."
        if engine._last_state and engine._last_state.notes:
            answer_text = " | ".join(engine._last_state.notes)

        answers.append({
            "question": q_text,
            "answer": answer_text[:800],
        })

        if verbose:
            note_count = len(engine._last_state.notes) if engine._last_state else 0
            _log(f"    {DIM}→ {note_count} notes: {answer_text[:100]}{RESET}")

    return answers, all_notes

# ── Iterative investigation (merged phases 1+2) ────────────────────

_MAX_FOLLOWUP_DEPTH = 2  # Max chain depth for follow-up questions


def _investigate_iteratively(agent: Any,
    description: str,
    strategy: PhaseStrategy,
    all_tools: list | None,
    verbose: bool,
    max_questions: int,
    skip_investigate: bool = False,
) -> tuple[list[dict[str, str]], list[str]]:
    """Interleave question generation and investigation.

    1. Generate seed questions
    2. Investigate each, then ask for follow-ups
    3. Investigate follow-ups (up to _MAX_FOLLOWUP_DEPTH)
    4. Return all answers + notes
    """
    # Phase 1: Seed questions
    if verbose:
        _log(f"\n{BOLD}❓ Phase 1: QUESTIONS{RESET}")

    seed_questions = _generate_questions(
        agent, description, strategy, verbose,
        max_questions=max_questions,
    )

    if verbose:
        _log(f"  {DIM}{len(seed_questions)} seed questions generated{RESET}")
        for i, q in enumerate(seed_questions):
            _log(f"    {DIM}{i+1}. {q['question'][:80]}{RESET}")

    if skip_investigate or not seed_questions:
        return [], []

    if verbose:
        _log(f"\n{BOLD}🔍 Phase 2: INVESTIGATE{RESET}")

    # Build read-only tool set
    if all_tools:
        read_tools = [
            t for t in all_tools
            if getattr(t, 'name', '') in _READ_ONLY_TOOLS
        ]
    else:
        agent_tools = getattr(agent, 'tools', []) or []
        read_tools = [
            t for t in agent_tools
            if getattr(t, 'name', '') in _READ_ONLY_TOOLS
        ] if agent_tools else []

    answers: list[dict[str, str]] = []
    all_notes: list[str] = []
    total_investigated = 0

    def _investigate_one(question: dict, label: str) -> None:
        """Investigate a single question and collect results."""
        nonlocal total_investigated
        q_text = question["question"]

        if verbose:
            _log(f"  {CYAN}{label}: {q_text[:80]}{RESET}")

        # Build previous answers context
        previous_text = ""
        if answers:
            prev_lines = "\n".join(
                f"  Q: {a['question']}\n  A: {a['answer']}"
                for a in answers
            )
            previous_text = f"## PREVIOUS ANSWERS\n{prev_lines}"

        inv_prompt = strategy.investigate_prompt.replace(
            "{{q_num}}", str(total_investigated + 1)
        ).replace(
            "{{q_total}}", str(max_questions)
        ).replace(
            "{{question}}", q_text
        ).replace(
            "{{previous_answers}}", previous_text
        )

        from infinidev.config.llm import _is_small_model as _is_sm2
        _max_iters2 = 2 if _is_sm2() else 3

        engine = LoopEngine()
        result = engine.execute(
            agent=agent,
            task_prompt=(inv_prompt, "Answer the question with add_note."),
            verbose=verbose,
            task_tools=read_tools,
            max_iterations=_max_iters2,
            max_total_tool_calls=strategy.investigate_max_tool_calls,
            max_tool_calls_per_action=strategy.investigate_max_tool_calls,
            nudge_threshold=strategy.investigate_max_tool_calls - 2,
            summarizer_enabled=False,
            identity_override=strategy.investigate_identity or None,
        )

        # Collect notes
        if engine._last_state and engine._last_state.notes:
            for note in engine._last_state.notes:
                if note not in all_notes:
                    all_notes.append(note)

        answer_text = result or "No answer found."
        if engine._last_state and engine._last_state.notes:
            answer_text = " | ".join(engine._last_state.notes)

        answers.append({
            "question": q_text,
            "answer": answer_text[:800],
        })
        total_investigated += 1

        if verbose:
            note_count = len(engine._last_state.notes) if engine._last_state else 0
            _log(f"    {DIM}→ {note_count} notes: {answer_text[:100]}{RESET}")

    def _investigate_with_followups(question: dict, label_prefix: str, depth: int) -> None:
        """Investigate a question, then recursively investigate follow-ups."""
        _investigate_one(question, label_prefix)

        # Check budget and depth
        if total_investigated >= max_questions or depth >= _MAX_FOLLOWUP_DEPTH:
            return

        # Generate follow-ups
        followups = _generate_followups(
            agent, description, answers, all_notes, strategy, verbose,
        )

        if not followups:
            return

        remaining_budget = max_questions - total_investigated
        followups = followups[:remaining_budget]

        if verbose:
            _log(f"    {DIM}↳ {len(followups)} follow-up(s) generated{RESET}")

        for j, fq in enumerate(followups):
            if total_investigated >= max_questions:
                break
            fu_label = f"{label_prefix} → F{j+1}"
            _investigate_with_followups(fq, fu_label, depth + 1)

    # Investigate each seed question with follow-ups
    for i, q in enumerate(seed_questions):
        if total_investigated >= max_questions:
            break
        _investigate_with_followups(q, f"Q{i+1}/{len(seed_questions)}", depth=0)

    if verbose:
        _log(f"  {DIM}Investigation complete: {len(answers)} answers, {len(all_notes)} notes{RESET}")

    return answers, all_notes
