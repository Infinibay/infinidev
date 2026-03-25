"""Orchestrator for the information gathering phase.

Flow:
0. Synthesize a complete ticket from chat history + user input
1. Classify the ticket type
2. Get fixed questions for the type
3. Answer each fixed question (sequentially, each feeds the next)
4. Generate dynamic follow-up questions
5. Answer each dynamic question
6. Compile the brief
"""

from __future__ import annotations

import json
import logging
from typing import Any

from infinidev.config.settings import settings
from infinidev.gather.classifier import classify_ticket
from infinidev.gather.compiler import compile_brief
from infinidev.gather.mini_agent import answer_question
from infinidev.gather.models import (
    ClassificationResult,
    GatherBrief,
    Question,
    QuestionResult,
    TicketType,
)
from infinidev.gather.questions import get_questions_for_type

logger = logging.getLogger(__name__)


def _parse_dynamic_questions_result(text: str, max_items: int) -> list[Question]:
    """Parse dynamic questions from LLM output — tries JSON first, then text extraction."""
    import re

    # Try JSON parsing first
    parsed = _extract_json_object(text)
    if parsed and "questions" in parsed:
        items = parsed["questions"]
    else:
        items = _extract_json_array(text)

    if items:
        questions = []
        for item in items[:max_items]:
            if isinstance(item, dict) and "question" in item:
                questions.append(Question(
                    id=item.get("id", f"dynamic_{len(questions)}"),
                    question=item["question"],
                    context_prompt=item.get("context_prompt", item["question"] + "\n\n{ticket_description}"),
                ))
        if questions:
            return questions

    # Fallback: extract questions from plain text (numbered list or bullet points)
    questions = []
    lines = text.strip().splitlines()
    for line in lines:
        line = line.strip()
        # Match patterns like "1. Question text" or "- Question text" or "* Question text"
        match = re.match(r'^(?:\d+[.)]\s*|[-*•]\s*|["""])', line)
        if match or (line.endswith("?") and len(line) > 20):
            q_text = re.sub(r'^[\d.)\-*•"""\s]+', '', line).strip().rstrip('"').strip()
            if q_text and len(q_text) > 15 and q_text.endswith("?"):
                questions.append(Question(
                    id=f"dynamic_{len(questions)}",
                    question=q_text,
                    context_prompt=q_text + "\n\n{ticket_description}",
                ))
                if len(questions) >= max_items:
                    break

    return questions


def _extract_json_object(text: str) -> dict | None:
    """Extract a JSON object from text."""
    import re
    text = re.sub(r"```json?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _extract_json_array(text: str) -> list | None:
    """Extract a JSON array from text that may contain other content."""
    import re

    # Strip markdown code fences
    text = re.sub(r"```json?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Try the whole thing
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Find array in text
    for match in re.finditer(r"\[.*?\]", text, re.DOTALL):
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            continue

    return None

_SYNTHESIZER_SYSTEM_PROMPT = """\
Given a conversation history and the user's latest message, produce a complete, \
self-contained task description that any developer could understand without additional context.

Include: what needs to be done, why, any technical details mentioned, and the expected outcome.

Output ONLY the description text. No JSON, no markdown formatting, no preamble.
"""

_DYNAMIC_QUESTIONS_SYSTEM_PROMPT = """\
You are preparing to implement a code change. Given the ticket and information gathered so far, \
generate additional questions that need to be answered before implementation begins.

Call step_complete with status="done". In final_answer, output a JSON array of question objects:
[{"id": "short_id", "question": "The question text", "context_prompt": "Detailed investigation prompt with {ticket_description} placeholder"}]

If no additional questions are needed, output an empty array: []
"""

_DYNAMIC_QUESTIONS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "generate_questions",
        "description": "Generate additional investigation questions. Pass an empty array if no more questions are needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Short identifier"},
                            "question": {"type": "string", "description": "The question text"},
                            "context_prompt": {"type": "string", "description": "Expanded investigation prompt"},
                        },
                        "required": ["id", "question"],
                    },
                    "description": "List of questions to investigate. Empty array if none needed.",
                },
            },
            "required": ["questions"],
        },
    },
}


def run_gather(
    user_input: str,
    chat_history: list[dict],
    analyst_result: Any | None,
    agent: Any,
) -> GatherBrief:
    """Run the complete information gathering phase.

    Returns a GatherBrief with all gathered context. Never raises —
    falls back gracefully on any error.
    """

    # Step -1: Index the project for code intelligence
    try:
        from infinidev.config.settings import settings as _s
        if _s.CODE_INTEL_ENABLED:
            import os
            from infinidev.tools.base.context import get_current_workspace_path
            workspace = get_current_workspace_path() or os.environ.get("INFINIDEV_WORKSPACE") or os.getcwd()
            from infinidev.code_intel.indexer import index_directory
            from infinidev.code_intel.index import clear_project
            from infinidev.code_intel.query import get_index_stats
            logger.info("Gather: indexing %s for code intelligence...", workspace)
            # Clear old index to avoid stale data from other projects
            clear_project(1)
            stats = index_directory(1, workspace)
            db_stats = get_index_stats(1)
            logger.info(
                "Gather: indexed %d files, %d symbols, %d references in %dms",
                stats["files_indexed"], stats["symbols_total"],
                db_stats.get("references", 0), stats["elapsed_ms"],
            )
    except Exception as exc:
        logger.warning("Gather: code intel indexing failed: %s", str(exc)[:200])

    # Step 0: Synthesize ticket description
    logger.info("Gather: synthesizing ticket description...")
    analyst_spec = None
    if analyst_result and hasattr(analyst_result, "specification"):
        analyst_spec = analyst_result.specification

    ticket_description = _synthesize_ticket(
        user_input, chat_history, analyst_spec, agent,
    )
    logger.info("Gather: ticket synthesized (%d chars)", len(ticket_description))

    # Step 1: Classify
    logger.info("Gather: classifying ticket...")
    classification = classify_ticket(ticket_description, analyst_spec, agent=agent)
    logger.info(
        "Gather: classified as %s (%s)",
        classification.ticket_type.value,
        classification.reasoning[:80],
    )

    # Step 2: Fixed questions — use shared session for state persistence
    from infinidev.gather.mini_agent import GatherSession

    questions = get_questions_for_type(classification.ticket_type)
    logger.info("Gather: %d fixed questions for type %s", len(questions), classification.ticket_type.value)

    session = GatherSession()  # Shared state: opened_files, history, notes persist between questions
    fixed_answers: list[QuestionResult] = []
    for i, q in enumerate(questions):
        logger.info("Gather: [%d/%d] %s", i + 1, len(questions), q.question[:80])
        try:
            result = answer_question(
                q, ticket_description, fixed_answers, agent, session=session,
            )
            result.phase = "fixed"
            fixed_answers.append(result)
            logger.info(
                "Gather: [%d/%d] answered (%d tool calls, %d chars)",
                i + 1, len(questions), result.tool_calls_used, len(result.answer),
            )
            preview = result.answer[:300].replace("\n", " ")
            logger.info("Gather: [%d/%d] answer: %s", i + 1, len(questions), preview)
        except Exception as exc:
            logger.warning("Gather: question %s failed: %s", q.id, str(exc)[:200])
            fixed_answers.append(QuestionResult(
                question_id=q.id,
                question_text=q.question,
                answer=f"Could not answer: {exc}",
                phase="fixed",
            ))

    # Step 3: Dynamic questions — same session continues
    logger.info("Gather: generating dynamic questions...")
    dynamic_questions = _generate_dynamic_questions(
        ticket_description, classification, fixed_answers, agent,
    )
    logger.info("Gather: %d dynamic questions generated", len(dynamic_questions))

    dynamic_answers: list[QuestionResult] = []
    all_prior = fixed_answers
    for i, q in enumerate(dynamic_questions):
        logger.info("Gather: [dynamic %d/%d] %s", i + 1, len(dynamic_questions), q.question[:80])
        try:
            result = answer_question(
                q, ticket_description, all_prior + dynamic_answers, agent, session=session,
            )
            result.phase = "dynamic"
            dynamic_answers.append(result)
            logger.info(
                "Gather: [dynamic %d/%d] answered (%d tool calls)",
                i + 1, len(dynamic_questions), result.tool_calls_used,
            )
            preview = result.answer[:300].replace("\n", " ")
            logger.info("Gather: [dynamic %d/%d] answer: %s", i + 1, len(dynamic_questions), preview)
        except Exception as exc:
            logger.warning("Gather: dynamic question %s failed: %s", q.id, str(exc)[:200])
            dynamic_answers.append(QuestionResult(
                question_id=q.id,
                question_text=q.question,
                answer=f"Could not answer: {exc}",
                phase="dynamic",
            ))

    # Step 4: Compile
    brief = compile_brief(ticket_description, classification, fixed_answers, dynamic_answers)
    logger.info("Gather: complete. %s", brief.summary())
    return brief


def _synthesize_ticket(
    user_input: str,
    chat_history: list[dict],
    analyst_spec: dict | None,
    agent: Any,
) -> str:
    """Synthesize a complete ticket description using LoopEngine."""
    # If user_input is already detailed (>200 chars) and no chat history, use directly
    if len(user_input) > 200 and not chat_history:
        return user_input

    from infinidev.config.settings import settings
    from infinidev.engine.loop_engine import LoopEngine

    parts = []
    if chat_history:
        for msg in chat_history[-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:500]
            parts.append(f"{role}: {content}")
    parts.append(f"User's latest message: {user_input}")
    if analyst_spec:
        parts.append(f"Analyst specification: {json.dumps(analyst_spec, indent=2)[:1000]}")

    original_identity = getattr(agent, "_system_prompt_identity", None)
    original_backstory = agent.backstory
    original_max_iter = settings.LOOP_MAX_ITERATIONS
    original_max_tools = settings.LOOP_MAX_TOTAL_TOOL_CALLS
    original_max_per_action = settings.LOOP_MAX_TOOL_CALLS_PER_ACTION
    original_nudge = settings.LOOP_STEP_NUDGE_THRESHOLD
    original_summarizer = settings.LOOP_SUMMARIZER_ENABLED
    original_gather = settings.GATHER_ENABLED

    try:
        agent._system_prompt_identity = _SYNTHESIZER_SYSTEM_PROMPT
        agent.backstory = "Ticket synthesizer."
        settings.LOOP_MAX_ITERATIONS = 2
        settings.LOOP_MAX_TOTAL_TOOL_CALLS = 10
        settings.LOOP_MAX_TOOL_CALLS_PER_ACTION = 10
        settings.LOOP_STEP_NUDGE_THRESHOLD = 0
        settings.LOOP_SUMMARIZER_ENABLED = False
        settings.GATHER_ENABLED = False

        engine = LoopEngine()
        result = engine.execute(
            agent=agent,
            task_prompt=("\n\n".join(parts), "Output ONLY the self-contained task description."),
            verbose=False,
            task_tools=[],
        )
        if result and result.strip():
            return result.strip()

    except Exception as exc:
        logger.warning("Ticket synthesis failed: %s", str(exc)[:200])

    finally:
        agent._system_prompt_identity = original_identity
        agent.backstory = original_backstory
        settings.LOOP_MAX_ITERATIONS = original_max_iter
        settings.LOOP_MAX_TOTAL_TOOL_CALLS = original_max_tools
        settings.LOOP_MAX_TOOL_CALLS_PER_ACTION = original_max_per_action
        settings.LOOP_STEP_NUDGE_THRESHOLD = original_nudge
        settings.LOOP_SUMMARIZER_ENABLED = original_summarizer
        settings.GATHER_ENABLED = original_gather

    # Fallback
    fallback = user_input
    if analyst_spec:
        summary = analyst_spec.get("summary", "")
        if summary:
            fallback = f"{summary}\n\n{user_input}"
    return fallback


def _generate_dynamic_questions(
    ticket_description: str,
    classification: ClassificationResult,
    fixed_answers: list[QuestionResult],
    agent: Any,
) -> list[Question]:
    """Generate dynamic follow-up questions using LoopEngine."""
    max_dynamic = settings.GATHER_MAX_DYNAMIC_QUESTIONS

    from infinidev.config.settings import settings as _settings
    from infinidev.engine.loop_engine import LoopEngine

    context_parts = [
        f"Ticket type: {classification.ticket_type.value}",
        f"Ticket: {ticket_description[:1000]}",
        "Information gathered so far:",
    ]
    for a in fixed_answers:
        context_parts.append(f"- {a.question_text}: {a.answer[:300]}")

    original_identity = getattr(agent, "_system_prompt_identity", None)
    original_backstory = agent.backstory
    original_max_iter = _settings.LOOP_MAX_ITERATIONS
    original_max_tools = _settings.LOOP_MAX_TOTAL_TOOL_CALLS
    original_max_per_action = _settings.LOOP_MAX_TOOL_CALLS_PER_ACTION
    original_nudge = _settings.LOOP_STEP_NUDGE_THRESHOLD
    original_summarizer = _settings.LOOP_SUMMARIZER_ENABLED
    original_gather = _settings.GATHER_ENABLED

    try:
        agent._system_prompt_identity = _DYNAMIC_QUESTIONS_SYSTEM_PROMPT
        agent.backstory = "Question generator."
        _settings.LOOP_MAX_ITERATIONS = 2
        _settings.LOOP_MAX_TOTAL_TOOL_CALLS = 10
        _settings.LOOP_MAX_TOOL_CALLS_PER_ACTION = 10
        _settings.LOOP_STEP_NUDGE_THRESHOLD = 0
        _settings.LOOP_SUMMARIZER_ENABLED = False
        _settings.GATHER_ENABLED = False

        engine = LoopEngine()
        result = engine.execute(
            agent=agent,
            task_prompt=("\n".join(context_parts), "Output a JSON array of questions."),
            verbose=False,
            task_tools=[],
        )

        logger.info("Dynamic questions result: %s", (result or "")[:300])

        if result:
            questions = _parse_dynamic_questions_result(result, max_dynamic)
            return questions

    except Exception as exc:
        logger.warning("Dynamic question generation failed: %s", str(exc)[:200])
        return []
