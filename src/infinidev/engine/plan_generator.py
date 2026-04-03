"""Phase 3: Plan generation for the phase engine."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from infinidev.engine.llm_client import call_llm
from infinidev.engine.engine_logging import log as _log, DIM, BOLD, RESET, YELLOW, RED
from infinidev.engine.plan_validator import validate_questions, format_rejection
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
    from infinidev.engine.loop.tools import STEP_COMPLETE_SCHEMA, build_tool_schemas
    from infinidev.engine.tool_call_parser import parse_step_complete_args

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

    # Only offer step_complete as a tool — no read/write tools
    tools = [STEP_COMPLETE_SCHEMA]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    collected_steps: list[dict[str, Any]] = []
    max_rounds = 5

    for round_num in range(max_rounds):
        try:
            response = call_llm(llm_params, messages, tools=tools, tool_choice="required")
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
                from infinidev.engine.tool_call_parser import parse_text_tool_calls, ManualToolCall
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

        # Process step_complete calls — collect next_steps, don't execute them
        for tc in tool_calls:
            if tc.function.name == "step_complete":
                result = parse_step_complete_args(tc.function.arguments)

                # Collect new steps
                for op in result.next_steps:
                    if op.op == "add":
                        collected_steps.append({
                            "step": len(collected_steps) + 1,
                            "description": op.description,
                            "files": [],
                        })

                if verbose:
                    new_count = len(result.next_steps)
                    _log(f"  {DIM}Round {round_num + 1}: +{new_count} steps (total: {len(collected_steps)}){RESET}")

                # Add tool response to conversation
                messages.append({"role": "assistant", "tool_calls": [
                    {"id": tc.id if hasattr(tc, "id") else f"plan_{round_num}",
                     "type": "function",
                     "function": {"name": "step_complete", "arguments": tc.function.arguments}}
                ]})
                messages.append({"role": "tool", "tool_call_id": tc.id if hasattr(tc, "id") else f"plan_{round_num}",
                                 "content": f"Plan updated. {len(collected_steps)} steps so far. Call step_complete again to add more, or with status='done' to finish."})

                # If model said done, stop
                if result.status == "done":
                    if verbose:
                        _log(f"  {DIM}Plan complete: {len(collected_steps)} steps{RESET}")
                    break
        else:
            continue  # no break in inner loop → continue outer
        break  # inner loop broke (done) → break outer too

    if verbose:
        _log(f"  {DIM}Plan generated: {len(collected_steps)} steps{RESET}")

    if len(collected_steps) < strategy.plan_min_steps:
        if verbose:
            _log(f"  {YELLOW}⚠ Only {len(collected_steps)} steps (need {strategy.plan_min_steps}){RESET}")
        return []

    return collected_steps

# ── Phase 4: Execute plan ─────────────────────────────────────────
