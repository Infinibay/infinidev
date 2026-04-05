"""Phase 3: Plan generation for the phase engine."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from infinidev.engine.llm_client import call_llm
from infinidev.engine.engine_logging import emit_loop_event, log as _log, DIM, BOLD, RESET, YELLOW, RED
from infinidev.engine.phases.plan_validator import validate_questions, format_rejection
from infinidev.prompts.phases import PhaseStrategy
from infinidev.prompts.phases.plan import PLANNER_IDENTITY as _PLANNER_IDENTITY

logger = logging.getLogger(__name__)


def _generate_plan(agent: Any,
    description: str,
    answers: list[dict[str, str]],
    all_notes: list[str],
    strategy: PhaseStrategy,
    all_tools: list | None,
    verbose: bool,
    test_checkpoint: Any | None = None,
) -> list[dict[str, Any]]:
    """Use LoopEngine to build the plan incrementally.

    The model uses step_complete(next_steps=[...]) to add steps to the
    plan one at a time. When it says "done", the accumulated plan steps
    become our execution plan.

    Custom mini-loop: calls LLM repeatedly, collects next_steps from
    step_complete, but NEVER activates/executes them. Stops when model
    says done or max rounds reached.
    """
    from infinidev.engine.llm_client import call_llm
    from infinidev.config.llm import get_litellm_params
    from infinidev.engine.loop.context import build_system_prompt
    from infinidev.engine.loop.tools import STEP_COMPLETE_SCHEMA, ADD_STEP_SCHEMA, MODIFY_STEP_SCHEMA, REMOVE_STEP_SCHEMA, build_tool_schemas
    from infinidev.engine.formats.tool_call_parser import parse_step_complete_args

    answers_text = "\n".join(
        f"  Q: {a['question']}\n  A: {a['answer']}"
        for a in answers
    )

    notes_text = ""
    if all_notes:
        notes_lines = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(all_notes))
        notes_text = f"\n## DETAILED NOTES FROM INVESTIGATION\n{notes_lines}\n"

    baseline_str = ""
    if strategy.auto_test and test_checkpoint:
        passed, total = test_checkpoint.run()
        if total > 0:
            baseline_str = f"\nTest baseline: {passed}/{total} passing\n"

    user_prompt = (
        f"{strategy.plan_prompt}\n\n"
        f"## YOUR TASK\n{description}\n\n"
        f"## YOUR INVESTIGATION RESULTS\n{answers_text}\n"
        f"{notes_text}"
        f"{baseline_str}"
    )

    llm_params = get_litellm_params()
    identity = strategy.plan_identity or _PLANNER_IDENTITY
    system_prompt = build_system_prompt(
        "Software engineering planner.",
        identity_override=identity,
    )

    # Only offer plan management + step_complete — no read/write tools
    tools = [ADD_STEP_SCHEMA, MODIFY_STEP_SCHEMA, REMOVE_STEP_SCHEMA, STEP_COMPLETE_SCHEMA]

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

    collected_steps: list[dict[str, Any]] = []
    max_rounds = 5

    for round_num in range(max_rounds):
        try:
            response = call_llm(llm_params, messages, tools=tools, tool_choice="required",
                                on_thinking_chunk=_on_thinking, on_stream_status=_on_stream_status)
        except Exception as exc:
            logger.warning("Plan LLM call failed (round %d): %s", round_num + 1, str(exc)[:200])
            if verbose:
                _log(f"  {RED}⚠ LLM error: {str(exc)[:80]}{RESET}")
            break

        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            # Model returned text instead of tool call — try to parse
            content = (getattr(message, "content", None) or "").strip()
            if content:
                from infinidev.engine.formats.tool_call_parser import parse_text_tool_calls, ManualToolCall
                import json as _json
                parsed = parse_text_tool_calls(content)
                if parsed:
                    tool_calls = [
                        ManualToolCall(id=f"plan_{round_num}", name=pc["name"],
                                      arguments=_json.dumps(pc["arguments"]) if isinstance(pc["arguments"], dict) else str(pc["arguments"]))
                        for pc in parsed
                    ]

        if not tool_calls:
            if verbose:
                _log(f"  {DIM}Round {round_num + 1}: no tool calls, stopping{RESET}")
            break

        # Build assistant message with all tool calls
        from infinidev.engine.formats.tool_call_parser import safe_json_loads as _safe_json
        assistant_tool_calls = []
        for tc in tool_calls:
            tc_id = tc.id if hasattr(tc, "id") else f"plan_{round_num}_{len(assistant_tool_calls)}"
            assistant_tool_calls.append({
                "id": tc_id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })
        messages.append({"role": "assistant", "tool_calls": assistant_tool_calls})

        # Process each tool call
        steps_added_this_round = 0
        plan_done = False
        for tc in tool_calls:
            tc_id = tc.id if hasattr(tc, "id") else f"plan_{round_num}_{tool_calls.index(tc)}"
            name = tc.function.name

            if name == "add_step":
                try:
                    args = _safe_json(tc.function.arguments) if isinstance(tc.function.arguments, str) else (tc.function.arguments or {})
                    if isinstance(args, dict):
                        title = args.get("title", args.get("description", ""))
                        desc = args.get("description", "")
                        if title:
                            collected_steps.append({
                                "step": len(collected_steps) + 1,
                                "title": title,
                                "description": desc if title != desc else "",
                                "files": [],
                            })
                            steps_added_this_round += 1
                except Exception:
                    pass
                messages.append({"role": "tool", "tool_call_id": tc_id,
                                 "content": f'{{"status": "added", "total_steps": {len(collected_steps)}}}'})

            elif name in ("modify_step", "remove_step"):
                messages.append({"role": "tool", "tool_call_id": tc_id,
                                 "content": '{"status": "updated"}'})

            elif name == "step_complete":
                result = parse_step_complete_args(tc.function.arguments)
                messages.append({"role": "tool", "tool_call_id": tc_id,
                                 "content": f"Plan has {len(collected_steps)} steps. Add more with add_step or finish with status='done'."})
                if result.status == "done":
                    plan_done = True

        if verbose and steps_added_this_round:
            _log(f"  {DIM}Round {round_num + 1}: +{steps_added_this_round} steps (total: {len(collected_steps)}){RESET}")

        if plan_done:
            if verbose:
                _log(f"  {DIM}Plan complete: {len(collected_steps)} steps{RESET}")
            break

    if verbose:
        _log(f"  {DIM}Plan generated: {len(collected_steps)} steps{RESET}")

    if len(collected_steps) < strategy.plan_min_steps:
        if verbose:
            _log(f"  {YELLOW}⚠ Only {len(collected_steps)} steps (need {strategy.plan_min_steps}){RESET}")
        return []

    return collected_steps

# ── Phase 4: Execute plan ─────────────────────────────────────────
