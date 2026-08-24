"""Phase 2: Investigation — answers questions by reading the codebase."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from infinidev.engine.engine_logging import (
    BOLD,
    CYAN,
    DIM,
    RESET,
    log as _log,
)
from infinidev.engine.loop import LoopEngine
from infinidev.engine.phases.question_generator import (
    _generate_followups,
    _generate_questions,
)
from infinidev.prompts.phases import PhaseStrategy

logger = logging.getLogger(__name__)

# Tools allowed during INVESTIGATE phase (read-only)
_READ_ONLY_TOOLS = {
    "read_file", "list_directory", "glob", "code_search",
    "project_structure", "find_references",
    "list_symbols", "search_symbols", "get_symbol_code",
    "web_search", "web_fetch",
    # execute_command is allowed for running tests / inspection ONLY —
    # it must never be used to edit, move, or delete files during INVESTIGATE.
    "execute_command",
}

_MAX_FOLLOWUP_DEPTH = 2  # Max chain depth for follow-up questions


def _investigate_iteratively(agent: Any,
    description: str,
    strategy: PhaseStrategy,
    all_tools: list | None,
    verbose: bool,
    max_questions: int,
    skip_investigate: bool = False,
    prompt_configuration: Any | None = None,
    loop_engine: LoopEngine | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    """Interleave question generation and investigation.

    1. Generate seed questions
    2. Investigate each, then ask for follow-ups
    3. Investigate follow-ups (up to _MAX_FOLLOWUP_DEPTH)
    4. Return all answers + notes
    """
    engine = loop_engine or LoopEngine()

    def _cancelled() -> bool:
        return bool(
            getattr(engine, "is_cancelled", False)
            or (cancel_check is not None and cancel_check())
        )

    # Phase 1: Seed questions
    if verbose:
        _log(f"\n{BOLD}❓ Phase 1: QUESTIONS{RESET}")

    seed_questions = _generate_questions(
        agent,
        description,
        strategy,
        verbose,
        max_questions=max_questions,
        cancel_check=_cancelled,
    )
    if _cancelled():
        return [], []

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

    def _investigate_one(question: dict, label: str) -> bool:
        """Investigate a single question and collect results."""
        nonlocal total_investigated
        if _cancelled():
            return False

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
            prompt_configuration=prompt_configuration,
        )
        if _cancelled():
            return False

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
            _log(f"    {DIM}Notes ({note_count}): {answer_text[:100]}{RESET}")
        return True

    def _investigate_with_followups(question: dict, label_prefix: str, depth: int) -> None:
        """Investigate a question, then recursively investigate follow-ups."""
        if _cancelled() or not _investigate_one(question, label_prefix):
            return

        # Check budget and depth
        if (
            _cancelled()
            or total_investigated >= max_questions
            or depth >= _MAX_FOLLOWUP_DEPTH
        ):
            return

        # Generate follow-ups
        followups = _generate_followups(
            agent,
            description,
            answers,
            all_notes,
            strategy,
            verbose,
            cancel_check=_cancelled,
        )

        if not followups:
            return

        remaining_budget = max_questions - total_investigated
        followups = followups[:remaining_budget]

        if verbose:
            _log(f"    {DIM}↳ {len(followups)} follow-up(s) generated{RESET}")

        for j, fq in enumerate(followups):
            if _cancelled() or total_investigated >= max_questions:
                break
            fu_label = f"{label_prefix} / F{j+1}"
            _investigate_with_followups(fq, fu_label, depth + 1)

    # Investigate each seed question with follow-ups
    for i, q in enumerate(seed_questions):
        if _cancelled() or total_investigated >= max_questions:
            break
        _investigate_with_followups(q, f"Q{i+1}/{len(seed_questions)}", depth=0)

    if verbose:
        _log(f"  {DIM}Investigation complete: {len(answers)} answers, {len(all_notes)} notes{RESET}")

    return answers, all_notes
