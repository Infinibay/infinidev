"""Phase 1: Question generation for the phase engine."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from infinidev.engine.llm_client import call_llm
from infinidev.engine.engine_logging import emit_loop_event, log as _log, DIM, RESET, YELLOW
from infinidev.prompts.phases import PhaseStrategy

logger = logging.getLogger(__name__)


def _generate_questions_text_mode(
    agent: Any,
    description: str,
    strategy: PhaseStrategy,
    verbose: bool,
    max_questions: int,
) -> list[dict[str, Any]]:
    """Generate questions via text output for small models.

    Single LLM call — asks for a numbered list, parses it.
    Far more reliable than multi-round tool calling for <40B models.
    """
    from infinidev.config.llm import get_litellm_params
    from infinidev.engine.loop.context import build_system_prompt

    q_min = strategy.questions_min

    prompt = (
        f"You are preparing to work on a task. Generate investigation "
        f"questions to understand the codebase before implementing.\n\n"
        f"Task: {description}\n\n"
        f"{strategy.questions_prompt}\n\n"
        f"Output a NUMBERED LIST of {q_min}-{max_questions} questions.\n"
        f"Each question should be answerable by reading code or running commands.\n\n"
        f"Example format:\n"
        f"1. Where is the auth module and what function handles login?\n"
        f"2. Are there existing tests for the login flow?\n"
        f"3. What is the current test baseline?\n\n"
        f"Output ONLY the numbered list."
    )

    llm_params = get_litellm_params()
    system_prompt = build_system_prompt(
        agent.backstory,
        identity_override=getattr(agent, '_system_prompt_identity', None),
        small_model=True,
    )

    _pid = getattr(agent, "project_id", 0)
    _aid = getattr(agent, "agent_id", "")

    def _on_thinking(text: str) -> None:
        emit_loop_event("loop_thinking_chunk", _pid, _aid, {"text": text})

    try:
        response = call_llm(
            llm_params,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            on_thinking_chunk=_on_thinking,
        )
    except Exception as exc:
        logger.warning("Text-mode question generation failed: %s", str(exc)[:200])
        return []

    text = response.choices[0].message.content or ""
    # Strip thinking tags
    text = re.sub(
        r"<(?:think|thinking)>.*?</(?:think|thinking)>",
        "", text, flags=re.DOTALL | re.IGNORECASE,
    )

    questions: list[dict[str, Any]] = []
    for match in re.finditer(r'^\s*\d+\s*[.):\-]\s*(.+)', text, re.MULTILINE):
        q = match.group(1).strip().rstrip("?") + "?"
        if len(q) >= 10:
            questions.append({"question": q, "intent": "general"})

    if verbose and questions:
        _log(f"  {DIM}Text-mode questions: {len(questions)} parsed{RESET}")

    return questions[:max_questions]


def _generate_questions(agent: Any,
    description: str,
    strategy: PhaseStrategy,
    verbose: bool,
    max_questions: int | None = None,
) -> list[dict[str, Any]]:
    """Generate questions using generate_question tool in a mini-loop.

    Same pattern as _generate_plan: model calls generate_question once
    per question, then step_complete(done) when finished.
    For small models, tries text-mode (numbered list) first.
    """
    from infinidev.config.llm import get_litellm_params, _is_small_model
    from infinidev.engine.loop.context import build_system_prompt
    from infinidev.engine.loop.tools import (
        STEP_COMPLETE_SCHEMA, GENERATE_QUESTION_SCHEMA,
    )
    from infinidev.engine.formats.tool_call_parser import parse_step_complete_args

    q_max = max_questions or strategy.questions_max
    q_min = strategy.questions_min

    # Small models: try text-mode first (more reliable than tool calling)
    if _is_small_model():
        if verbose:
            _log(f"  {DIM}Using text-mode question generation (small model){RESET}")
        text_qs = _generate_questions_text_mode(agent, description, strategy, verbose, q_max)
        if len(text_qs) >= q_min:
            return text_qs
        if verbose and text_qs:
            _log(f"  {YELLOW}⚠ Text-mode produced {len(text_qs)} questions (need {q_min}), falling back to tool mode{RESET}")

    user_prompt = (
        f"You are preparing to work on a task. Generate investigation "
        f"questions that will help you understand the codebase and create "
        f"an implementation plan.\n\n"
        f"Task: {description}\n\n"
        f"{strategy.questions_prompt}\n\n"
        f"Call generate_question once per question ({q_min}-{q_max} questions).\n"
        f"Call step_complete with status='done' when finished."
    )

    llm_params = get_litellm_params()
    system_prompt = build_system_prompt(
        agent.backstory,
        identity_override=getattr(agent, '_system_prompt_identity', None),
    )

    tools = [GENERATE_QUESTION_SCHEMA, STEP_COMPLETE_SCHEMA]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Build streaming callbacks so thinking tokens are visible in the TUI
    _pid = getattr(agent, "project_id", 0)
    _aid = getattr(agent, "agent_id", "")

    def _on_thinking(text: str) -> None:
        emit_loop_event("loop_thinking_chunk", _pid, _aid, {"text": text})

    def _on_stream_status(phase: str, tokens: int, tool_name: str | None) -> None:
        emit_loop_event("loop_stream_status", _pid, _aid, {
            "phase": phase, "tokens": tokens, "tool_name": tool_name,
        })

    collected: list[dict[str, Any]] = []
    max_rounds = q_max + 3  # headroom for retries

    for round_num in range(max_rounds):
        try:
            response = call_llm(llm_params, messages, tools=tools, tool_choice="auto",
                                on_thinking_chunk=_on_thinking, on_stream_status=_on_stream_status)
        except Exception as exc:
            logger.warning("Question generation failed (round %d): %s", round_num + 1, str(exc)[:200])
            break

        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            if verbose:
                _log(f"  {DIM}Round {round_num + 1}: no tool calls, stopping{RESET}")
            break

        done = False
        for tc in tool_calls:
            fn_name = tc.function.name
            if fn_name == "generate_question":
                try:
                    args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                except (json.JSONDecodeError, TypeError):
                    args = {}

                q_text = args.get("question", "")
                q_intent = args.get("intent", "general")

                if q_text and len(q_text) >= 10:
                    collected.append({"question": q_text, "intent": q_intent})
                    if verbose:
                        _log(f"  {DIM}Q{len(collected)}: {q_text[:80]}{RESET}")

                # Feed back confirmation
                messages.append({"role": "assistant", "tool_calls": [
                    {"id": getattr(tc, "id", f"q_{round_num}"),
                     "type": "function",
                     "function": {"name": "generate_question", "arguments": tc.function.arguments}}
                ]})
                messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(tc, "id", f"q_{round_num}"),
                    "content": f"Question #{len(collected)} recorded. "
                               f"Generate more or call step_complete(status='done').",
                })

            elif fn_name == "step_complete":
                done = True
                messages.append({"role": "assistant", "tool_calls": [
                    {"id": getattr(tc, "id", f"sc_{round_num}"),
                     "type": "function",
                     "function": {"name": "step_complete", "arguments": tc.function.arguments}}
                ]})
                messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(tc, "id", f"sc_{round_num}"),
                    "content": "Questions phase complete.",
                })
                break

        if done or len(collected) >= q_max:
            break

    if len(collected) < q_min:
        if verbose:
            _log(f"  {YELLOW}Using fallback questions ({len(collected)} < {q_min}){RESET}")
        return [{"question": q, "intent": "fallback"} for q in strategy.fallback_questions]

    return collected

# ── Phase 2: Investigate each question ────────────────────────────


def _generate_followups(agent: Any,
    description: str,
    answers: list[dict[str, str]],
    all_notes: list[str],
    strategy: PhaseStrategy,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Ask LLM if follow-up questions are needed based on investigation so far."""
    from infinidev.config.llm import get_litellm_params
    from infinidev.engine.loop.context import build_system_prompt
    from infinidev.engine.loop.tools import (
        STEP_COMPLETE_SCHEMA, GENERATE_QUESTION_SCHEMA,
    )
    from infinidev.prompts.phases.investigate import FOLLOWUP_PROMPT

    answers_text = "\n".join(
        f"  Q: {a['question']}\n  A: {a['answer']}"
        for a in answers
    )
    notes_text = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(all_notes)) if all_notes else "(none)"

    user_prompt = FOLLOWUP_PROMPT.format(
        answers_text=answers_text,
        notes_text=notes_text,
        description=description,
    )

    llm_params = get_litellm_params()
    system_prompt = build_system_prompt(
        agent.backstory,
        identity_override=strategy.investigate_identity or None,
    )

    tools = [GENERATE_QUESTION_SCHEMA, STEP_COMPLETE_SCHEMA]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    _pid = getattr(agent, "project_id", 0)
    _aid = getattr(agent, "agent_id", "")

    def _on_thinking(text: str) -> None:
        emit_loop_event("loop_thinking_chunk", _pid, _aid, {"text": text})

    def _on_stream_status(phase: str, tokens: int, tool_name: str | None) -> None:
        emit_loop_event("loop_stream_status", _pid, _aid, {
            "phase": phase, "tokens": tokens, "tool_name": tool_name,
        })

    collected: list[dict[str, Any]] = []
    # Single round — follow-up generation should be quick
    max_rounds = 3

    for round_num in range(max_rounds):
        try:
            response = call_llm(llm_params, messages, tools=tools, tool_choice="auto",
                                on_thinking_chunk=_on_thinking, on_stream_status=_on_stream_status)
        except Exception as exc:
            logger.warning("Follow-up generation failed: %s", str(exc)[:200])
            break

        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            break

        done = False
        for tc in tool_calls:
            fn_name = tc.function.name
            if fn_name == "generate_question":
                try:
                    args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                except (json.JSONDecodeError, TypeError):
                    args = {}

                q_text = args.get("question", "")
                q_intent = args.get("intent", "followup")

                if q_text and len(q_text) >= 10:
                    collected.append({"question": q_text, "intent": q_intent})

                messages.append({"role": "assistant", "tool_calls": [
                    {"id": getattr(tc, "id", f"fu_{round_num}"),
                     "type": "function",
                     "function": {"name": "generate_question", "arguments": tc.function.arguments}}
                ]})
                messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(tc, "id", f"fu_{round_num}"),
                    "content": f"Follow-up #{len(collected)} recorded. "
                               f"Generate more or call step_complete(status='done').",
                })

            elif fn_name == "step_complete":
                done = True
                break

        if done or len(collected) >= 2:
            break

    return collected

# ── Phase 3: Generate plan ────────────────────────────────────────
