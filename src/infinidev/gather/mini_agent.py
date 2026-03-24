"""Answer a single investigation question using the LoopEngine.

Reuses the full LoopEngine infrastructure (FC detection, manual mode,
tool dispatch, retries, step_complete protocol) but with:
- Read-only tools only
- A lightweight identity prompt
- Lower iteration/tool limits
"""

from __future__ import annotations

import logging
from typing import Any

from infinidev.gather.models import Question, QuestionResult

logger = logging.getLogger(__name__)

# Tools allowed during gathering (read-only)
READ_ONLY_TOOL_NAMES = {
    "read_file", "list_directory", "code_search", "glob",
    "execute_command",
    "search_findings", "read_findings",
    "web_search", "web_fetch", "code_search_web",
    "find_documentation",
}

_INVESTIGATOR_IDENTITY = """\
## Identity

You are a codebase investigator answering a specific question about a project.
Use tools to explore the codebase and find the answer. Be thorough but efficient.

## Rules

- Read the files that matter, skip the ones that don't.
- Your answer should be factual and specific: file paths, line numbers, function names, class names.
- Do NOT modify any files. You are only investigating.
- When you have enough information, call step_complete with status="done" and put your full answer in final_answer.
"""


def answer_question(
    question: Question,
    ticket_description: str,
    prior_answers: list[QuestionResult],
    agent: Any,
) -> QuestionResult:
    """Answer a single question using LoopEngine with read-only tools.

    Creates a temporary agent configuration and runs LoopEngine.execute()
    with filtered tools and a focused prompt.
    """
    from infinidev.config.settings import settings
    from infinidev.engine.loop_engine import LoopEngine

    # Filter to read-only tools
    read_only_tools = [t for t in agent.tools if t.name in READ_ONLY_TOOL_NAMES]
    if not read_only_tools:
        return QuestionResult(
            question_id=question.id,
            question_text=question.question,
            answer="No read-only tools available for investigation.",
        )

    # Build focused task prompt
    prompt_parts = [
        f"Answer this question: {question.question}",
        "",
        question.context_prompt.format(ticket_description=ticket_description),
    ]

    # Add prior answers as context (truncated)
    if prior_answers:
        prompt_parts.append("\n## Previously Gathered Information")
        for pa in prior_answers:
            summary = pa.answer[:200] + ("..." if len(pa.answer) > 200 else "")
            prompt_parts.append(f"- {pa.question_text}: {summary}")

    task_description = "\n".join(prompt_parts)
    expected_output = (
        "Provide a thorough, factual answer to the question. "
        "Include specific file paths, line numbers, function/class names, "
        "and code patterns found."
    )

    # Save and override agent settings for the investigation
    original_identity = getattr(agent, "_system_prompt_identity", None)
    original_backstory = agent.backstory
    original_max_iter = settings.LOOP_MAX_ITERATIONS
    original_max_tools = settings.LOOP_MAX_TOTAL_TOOL_CALLS
    original_max_per_action = settings.LOOP_MAX_TOOL_CALLS_PER_ACTION
    original_nudge = settings.LOOP_STEP_NUDGE_THRESHOLD
    original_summarizer = settings.LOOP_SUMMARIZER_ENABLED
    original_gather = settings.GATHER_ENABLED

    try:
        # Configure for lightweight investigation — generous tool limits
        agent._system_prompt_identity = _INVESTIGATOR_IDENTITY
        agent.backstory = "Codebase investigator. Reads code, answers questions."
        settings.LOOP_MAX_ITERATIONS = 8
        settings.LOOP_MAX_TOTAL_TOOL_CALLS = question.max_tool_calls
        settings.LOOP_MAX_TOOL_CALLS_PER_ACTION = question.max_tool_calls  # No per-step limit
        settings.LOOP_STEP_NUDGE_THRESHOLD = 0  # No nudge during gathering
        settings.LOOP_SUMMARIZER_ENABLED = False  # No summarizer for sub-questions
        settings.GATHER_ENABLED = False  # No nested gather

        engine = LoopEngine()
        result = engine.execute(
            agent=agent,
            task_prompt=(task_description, expected_output),
            verbose=True,
            task_tools=read_only_tools,
        )

        return QuestionResult(
            question_id=question.id,
            question_text=question.question,
            answer=(result or "No answer produced.").strip(),
            tool_calls_used=getattr(engine, "_last_total_tool_calls", 0),
        )

    except Exception as exc:
        logger.warning("Question %s failed: %s", question.id, str(exc)[:200])
        return QuestionResult(
            question_id=question.id,
            question_text=question.question,
            answer=f"Investigation failed: {exc}",
        )

    finally:
        # Restore original settings
        agent._system_prompt_identity = original_identity
        agent.backstory = original_backstory
        settings.LOOP_MAX_ITERATIONS = original_max_iter
        settings.LOOP_MAX_TOTAL_TOOL_CALLS = original_max_tools
        settings.LOOP_MAX_TOOL_CALLS_PER_ACTION = original_max_per_action
        settings.LOOP_STEP_NUDGE_THRESHOLD = original_nudge
        settings.LOOP_SUMMARIZER_ENABLED = original_summarizer
        settings.GATHER_ENABLED = original_gather
