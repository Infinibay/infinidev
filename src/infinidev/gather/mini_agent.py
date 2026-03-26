"""Answer investigation questions using the LoopEngine with shared state.

Reuses the full LoopEngine infrastructure but with:
- Read-only tools only
- A lightweight identity prompt
- Shared LoopState between questions (opened_files, history, notes persist)
- Prior answers included in full (not truncated)
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
    "find_definition", "find_references", "list_symbols",
    "search_symbols", "get_symbol_code", "project_structure",
}

_INVESTIGATOR_IDENTITY = """\
## Identity

You are a codebase investigator. You ONLY gather information — you do NOT plan fixes,
write code, or suggest solutions. Just find facts and report them.

## Rules

- ONLY INVESTIGATE. Do NOT plan implementation steps, do NOT propose fixes, do NOT write code.
- Your job is to answer ONE specific question with facts from the codebase.
- PREFER semantic tools: find_definition, find_references, get_symbol_code, search_symbols, list_symbols, project_structure.
  These are faster and more precise than code_search or grep.
- Use find_definition(name) to locate where a function/class is defined.
- Use find_references(name) to find ALL usages of a symbol.
- Use get_symbol_code(name) to read the full source code of a function/method/class.
- Use list_symbols(file_path) to see what's in a file without reading it entirely.
- Use project_structure(path) to see directory contents with descriptions.
- Use project_structure(path) FIRST to understand the project layout before diving into files.
- Use code_search only for text patterns that aren't symbol names (error messages, strings).
- Be EFFICIENT: 5-10 tool calls should be enough. Don't keep searching if you have the answer.
- Files from previous questions are already cached — do NOT re-read them.
- This is a SINGLE STEP task. Do NOT create a plan with multiple steps.
  Use tools to investigate, then call step_complete with status="done" and final_answer.
- Do NOT use step_complete with status="continue". Always use status="done".
- Your answer must be factual: file paths, line numbers, function names, class names, code snippets.
"""


class GatherSession:
    """Manages shared state across multiple investigation questions.

    Keeps opened_files, history, and notes persistent between questions
    so the agent doesn't re-read files or lose context.
    """

    def __init__(self):
        self._last_state_dict: dict | None = None
        self._engine = None

    def answer_question(
        self,
        question: Question,
        ticket_description: str,
        prior_answers: list[QuestionResult],
        agent: Any,
    ) -> QuestionResult:
        """Answer a single question, sharing state with previous questions."""
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

        # Add ALL prior answers in full (not truncated)
        if prior_answers:
            prompt_parts.append("\n## Previously Gathered Information")
            prompt_parts.append("(These questions have already been answered — do NOT re-investigate them.)\n")
            for pa in prior_answers:
                prompt_parts.append(f"### Q: {pa.question_text}")
                prompt_parts.append(pa.answer)
                prompt_parts.append("")

        task_description = "\n".join(prompt_parts)
        expected_output = (
            "Provide a thorough, factual answer to the question. "
            "Include specific file paths, line numbers, function/class names, "
            "and code patterns found."
        )

        # Save and override agent settings
        original_identity = getattr(agent, "_system_prompt_identity", None)
        original_backstory = agent.backstory
        original_gather = settings.GATHER_ENABLED

        try:
            agent._system_prompt_identity = _INVESTIGATOR_IDENTITY
            agent.backstory = "Codebase investigator. Reads code, answers questions."
            settings.GATHER_ENABLED = False

            engine = LoopEngine()

            # Build resume_state from previous session state
            resume = None
            if self._last_state_dict:
                resume = self._last_state_dict.copy()
                # Reset iteration/tool counters but keep opened_files, history, notes
                resume["iteration_count"] = 0
                resume["total_tool_calls"] = 0
                resume["current_step_index"] = 0
                resume["last_prompt_tokens"] = 0
                resume["last_completion_tokens"] = 0
                resume["tool_calls_since_last_note"] = 0
                # Clear plan (new question = new plan)
                resume["plan"] = {"steps": []}

            result = engine.execute(
                agent=agent,
                task_prompt=(task_description, expected_output),
                verbose=True,
                task_tools=read_only_tools,
                resume_state=resume,
                max_iterations=1,
                max_total_tool_calls=question.max_tool_calls,
                max_tool_calls_per_action=question.max_tool_calls,
                nudge_threshold=0,
                summarizer_enabled=True,
            )

            # Save state for next question
            if engine._last_state:
                self._last_state_dict = engine._last_state.model_dump()

            return QuestionResult(
                question_id=question.id,
                question_text=question.question,
                answer=(result or "No answer produced.").strip(),
                tool_calls_used=engine._last_total_tool_calls,
            )

        except Exception as exc:
            logger.warning("Question %s failed: %s", question.id, str(exc)[:200])
            return QuestionResult(
                question_id=question.id,
                question_text=question.question,
                answer=f"Investigation failed: {exc}",
            )

        finally:
            agent._system_prompt_identity = original_identity
            agent.backstory = original_backstory
            settings.GATHER_ENABLED = original_gather


# Backward-compatible function interface
def answer_question(
    question: Question,
    ticket_description: str,
    prior_answers: list[QuestionResult],
    agent: Any,
    *,
    session: GatherSession | None = None,
) -> QuestionResult:
    """Answer a single question. Optionally pass a GatherSession for state sharing."""
    if session is None:
        session = GatherSession()
    return session.answer_question(question, ticket_description, prior_answers, agent)
