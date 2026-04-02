"""Phase-based execution: QUESTIONS → INVESTIGATE → PLAN → EXECUTE.

1. QUESTIONS: Model generates questions about the task (direct LLM call)
2. INVESTIGATE: Engine creates 1 step per question (read-only LoopEngine)
3. PLAN: Model generates granular plan from answers (direct LLM call)
4. EXECUTE: Step-by-step implementation with test verification (LoopEngine)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from infinidev.engine.loop_engine import LoopEngine
from infinidev.engine.llm_client import call_llm
from infinidev.prompts.phases import get_strategy, PhaseStrategy
from infinidev.engine.plan_validator import (
    validate_questions, format_rejection,
)
from infinidev.prompts.phases.plan import PLANNER_IDENTITY as _PLANNER_IDENTITY
from infinidev.engine.test_checkpoint import TestCheckpoint
from infinidev.engine.engine_logging import (
    log as _log,
    DIM, BOLD, RESET, CYAN, GREEN, YELLOW, RED,
)

logger = logging.getLogger(__name__)

# Tools allowed during INVESTIGATE phase (read-only)
_READ_ONLY_TOOLS = {
    "read_file", "list_directory", "glob", "code_search",
    "project_structure", "find_definition", "find_references",
    "list_symbols", "search_symbols", "get_symbol_code",
    "search_knowledge", "read_findings", "search_findings",
    "web_search", "web_fetch", "execute_command",
}

# _PLANNER_IDENTITY imported from prompts.phases.plan


class PhaseEngine:
    """Four-phase execution: QUESTIONS → INVESTIGATE → PLAN → EXECUTE."""

    def __init__(self) -> None:
        self._last_engine: LoopEngine | None = None
        self._test_checkpoint: TestCheckpoint | None = None

    def execute(
        self,
        agent: Any,
        task_prompt: tuple[str, str],
        task_type: str = "feature",
        *,
        verbose: bool = True,
        task_tools: list | None = None,
        test_command: str | None = None,
        depth_config: Any | None = None,
    ) -> str:
        from infinidev.gather.models import DepthLevel, DepthConfig, DEPTH_CONFIGS

        description, expected_output = task_prompt

        # Init test checkpoint
        from infinidev.tools.base.context import get_current_workspace_path
        workdir = get_current_workspace_path()
        self._test_checkpoint = TestCheckpoint(test_command, workdir)

        # ── Step 0: CLASSIFY (if no depth provided) ──────────────────
        if depth_config is None:
            classification = self._classify(agent, description, verbose)
            task_type = classification.ticket_type.value
            depth_config = DEPTH_CONFIGS.get(classification.depth, DEPTH_CONFIGS[DepthLevel.standard])
            if verbose:
                _log(f"\n{BOLD}{CYAN}⚡ Phase Engine{RESET} — type: {task_type}, depth: {classification.depth.value}")
                if classification.depth_reasoning:
                    _log(f"  {DIM}{classification.depth_reasoning}{RESET}")
        else:
            if verbose:
                _log(f"\n{BOLD}{CYAN}⚡ Phase Engine{RESET} — type: {task_type}, depth: (provided)")

        strategy = get_strategy(task_type)

        # ── MINIMAL: single free LoopEngine run ──────────────────────
        if depth_config.skip_questions and depth_config.skip_investigate and depth_config.plan_min_steps <= 1:
            return self._execute_minimal(agent, description, expected_output, strategy, task_tools, depth_config, verbose)

        # ── LIGHT: skip questions/investigate, go straight to plan+execute
        answers: list[dict[str, str]] = []
        all_notes: list[str] = []

        if not depth_config.skip_questions:
            # Override investigate budget from depth config
            strategy.investigate_max_tool_calls = depth_config.investigate_max_tool_calls

            answers, all_notes = self._investigate_iteratively(
                agent, description, strategy, task_tools, verbose,
                max_questions=depth_config.questions_max,
                skip_investigate=depth_config.skip_investigate,
            )

        # ── Phase 3: PLAN ─────────────────────────────────────────────
        if verbose:
            _log(f"\n{BOLD}📋 Phase 3: PLAN{RESET}")

        strategy.plan_min_steps = depth_config.plan_min_steps

        plan_steps = self._generate_plan(
            agent, description, answers, all_notes, strategy, task_tools, verbose,
        )

        if not plan_steps:
            return "Failed to generate a valid plan."

        if verbose:
            _log(f"  {DIM}Plan: {len(plan_steps)} steps{RESET}")
            for s in plan_steps:
                files_str = ", ".join(s.get("files", [])) or "(verify)"
                _log(f"    {DIM}{s['step']}. {s['description'][:70]} [{files_str}]{RESET}")

        # ── Phase 4: EXECUTE (with re-plan loop) ─────────────────────
        for plan_round in range(depth_config.replan_max_rounds):
            if verbose:
                round_label = f" (round {plan_round + 1})" if plan_round > 0 else ""
                _log(f"\n{BOLD}🔨 Phase 4: EXECUTE{round_label}{RESET}")

            result = self._execute_plan(
                agent, description, expected_output, answers, all_notes,
                plan_steps, strategy, task_tools, depth_config, verbose,
            )

            # Check test progress
            if strategy.auto_test and self._test_checkpoint:
                passed, total = self._test_checkpoint.run()
                if verbose and total > 0:
                    _log(f"\n  {BOLD}{self._test_checkpoint.progress_str()}{RESET}")

                if total == 0 or passed == total:
                    break

                if plan_round < depth_config.replan_max_rounds - 1:
                    if verbose:
                        _log(f"\n{BOLD}📋 Re-planning: {passed}/{total} tests passing...{RESET}")

                    all_notes.append(f"PROGRESS: {passed}/{total} tests passing after round {plan_round + 1}")

                    plan_steps = self._generate_plan(
                        agent, description, answers, all_notes, strategy, task_tools, verbose,
                    )
                    if not plan_steps:
                        break

                    if verbose:
                        _log(f"  {DIM}Re-plan: {len(plan_steps)} new steps{RESET}")
                        for s in plan_steps:
                            _log(f"    {DIM}{s['step']}. {s['description'][:70]}{RESET}")
            else:
                break

        return result

    # ── Step 0: Classify ──────────────────────────────────────────────

    def _classify(self, agent: Any, description: str, verbose: bool) -> Any:
        """Run ticket classification to determine task_type and depth."""
        from infinidev.gather.classifier import classify_ticket
        from infinidev.gather.models import ClassificationResult, DepthLevel

        if verbose:
            _log(f"\n{BOLD}🏷️  Step 0: CLASSIFY{RESET}")

        result = classify_ticket(description, agent=agent)

        if verbose:
            _log(f"  {DIM}Type: {result.ticket_type.value} — {result.reasoning}{RESET}")
            _log(f"  {DIM}Depth: {result.depth.value} — {result.depth_reasoning}{RESET}")

        return result

    # ── Minimal mode: single free LoopEngine run ─────────────────────

    def _execute_minimal(
        self,
        agent: Any,
        description: str,
        expected_output: str,
        strategy: PhaseStrategy,
        task_tools: list | None,
        depth_config: Any,
        verbose: bool,
    ) -> str:
        """Minimal depth: single LoopEngine run with no phase separation."""
        if verbose:
            _log(f"\n{BOLD}🔨 EXECUTE (minimal — single run){RESET}")

        engine = LoopEngine()
        result = engine.execute(
            agent=agent,
            task_prompt=(description, expected_output),
            verbose=verbose,
            task_tools=task_tools,
            max_iterations=50,
            max_total_tool_calls=1000,
            max_tool_calls_per_action=0,
            nudge_threshold=0,
            summarizer_enabled=True,
            identity_override=strategy.execute_identity or None,
        )

        self._last_engine = engine

        if strategy.auto_test and self._test_checkpoint:
            passed, total = self._test_checkpoint.run()
            if verbose and total > 0:
                _log(f"\n  {BOLD}{self._test_checkpoint.progress_str()}{RESET}")

        return result or ""

    # ── Phase 1: Generate questions ───────────────────────────────────

    def _generate_questions(
        self,
        agent: Any,
        description: str,
        strategy: PhaseStrategy,
        verbose: bool,
        max_questions: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate questions using generate_question tool in a mini-loop.

        Same pattern as _generate_plan: model calls generate_question once
        per question, then step_complete(done) when finished.
        """
        from infinidev.config.llm import get_litellm_params
        from infinidev.engine.loop_context import build_system_prompt
        from infinidev.engine.loop_tools import (
            STEP_COMPLETE_SCHEMA, GENERATE_QUESTION_SCHEMA,
        )
        from infinidev.engine.tool_call_parser import parse_step_complete_args

        q_max = max_questions or strategy.questions_max
        q_min = strategy.questions_min

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

        collected: list[dict[str, Any]] = []
        max_rounds = q_max + 3  # headroom for retries

        for round_num in range(max_rounds):
            try:
                response = call_llm(llm_params, messages, tools=tools, tool_choice="auto")
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

    def _investigate(
        self,
        agent: Any,
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

            engine = LoopEngine()
            result = engine.execute(
                agent=agent,
                task_prompt=(inv_prompt, "Answer the question with add_note."),
                verbose=verbose,
                task_tools=read_tools,
                max_iterations=3,
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

    def _investigate_iteratively(
        self,
        agent: Any,
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

        seed_questions = self._generate_questions(
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

            engine = LoopEngine()
            result = engine.execute(
                agent=agent,
                task_prompt=(inv_prompt, "Answer the question with add_note."),
                verbose=verbose,
                task_tools=read_tools,
                max_iterations=3,
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
            if total_investigated >= max_questions or depth >= self._MAX_FOLLOWUP_DEPTH:
                return

            # Generate follow-ups
            followups = self._generate_followups(
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

    def _generate_followups(
        self,
        agent: Any,
        description: str,
        answers: list[dict[str, str]],
        all_notes: list[str],
        strategy: PhaseStrategy,
        verbose: bool,
    ) -> list[dict[str, Any]]:
        """Ask LLM if follow-up questions are needed based on investigation so far."""
        from infinidev.config.llm import get_litellm_params
        from infinidev.engine.loop_context import build_system_prompt
        from infinidev.engine.loop_tools import (
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

        collected: list[dict[str, Any]] = []
        # Single round — follow-up generation should be quick
        max_rounds = 3

        for round_num in range(max_rounds):
            try:
                response = call_llm(llm_params, messages, tools=tools, tool_choice="auto")
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

    def _generate_plan(
        self,
        agent: Any,
        description: str,
        answers: list[dict[str, str]],
        all_notes: list[str],
        strategy: PhaseStrategy,
        all_tools: list | None,
        verbose: bool,
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
        from infinidev.engine.loop_context import build_system_prompt
        from infinidev.engine.loop_tools import STEP_COMPLETE_SCHEMA, build_tool_schemas
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
        if strategy.auto_test and self._test_checkpoint:
            passed, total = self._test_checkpoint.run()
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

    def _execute_plan(
        self,
        agent: Any,
        description: str,
        expected_output: str,
        answers: list[dict[str, str]],
        all_notes: list[str],
        plan_steps: list[dict[str, Any]],
        strategy: PhaseStrategy,
        all_tools: list | None,
        depth_config: Any,
        verbose: bool,
    ) -> str:
        answers_text = "\n".join(
            f"  Q: {a['question']}\n  A: {a['answer']}"
            for a in answers
        )
        notes_text = ""
        if all_notes:
            notes_text = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(all_notes))
        completed: list[str] = []
        last_result = ""

        for step in plan_steps:
            step_num = step["step"]
            step_desc = step["description"]
            step_files = step.get("files", [])
            total = len(plan_steps)

            files_str = ", ".join(step_files) if step_files else "(verification step)"
            completed_str = "\n".join(
                f"  ✓ {s}" for s in completed
            ) if completed else "  (none yet)"

            progress_str = ""
            regression_warning = ""
            if self._test_checkpoint and self._test_checkpoint.total > 0:
                progress_str = f"\n{self._test_checkpoint.progress_str()}\n"
                rw = self._test_checkpoint.regression_warning()
                if rw:
                    regression_warning = f"\n{rw}\n"

            # Use conditional prompt generation for small models
            from infinidev.config.llm import _is_small_model
            if _is_small_model():
                from infinidev.prompts.tool_hints import (
                    build_execute_prompt, get_available_tool_names,
                )
                _tool_names = get_available_tool_names(all_tools)
                step_prompt = build_execute_prompt(
                    available_tools=_tool_names,
                    step_num=step_num,
                    total_steps=total,
                    step_description=step_desc,
                    step_files=files_str,
                )
            else:
                step_prompt = strategy.execute_prompt.replace(
                    "{{step_num}}", str(step_num)
                ).replace(
                    "{{total_steps}}", str(total)
                ).replace(
                    "{{step_description}}", step_desc
                ).replace(
                    "{{step_files}}", files_str
                )

            notes_section = f"## NOTES\n{notes_text}\n\n" if notes_text else ""
            depth_suffix = depth_config.prompt_suffix if depth_config else ""
            full_prompt = (
                f"{step_prompt}\n\n"
                f"{notes_section}"
                f"## INVESTIGATION RESULTS\n{answers_text}\n\n"
                f"## COMPLETED STEPS\n{completed_str}\n"
                f"{progress_str}{regression_warning}"
                f"{depth_suffix}"
            )

            if verbose:
                _log(f"\n  {CYAN}Step {step_num}/{total}: {step_desc[:80]}{RESET}")

            engine = LoopEngine()
            result = engine.execute(
                agent=agent,
                task_prompt=(full_prompt, expected_output),
                verbose=verbose,
                task_tools=all_tools,
                max_iterations=50,
                max_total_tool_calls=1000,
                max_tool_calls_per_action=0,
                nudge_threshold=0,
                summarizer_enabled=not (depth_config and depth_config.aggressive_summarizer),
                identity_override=strategy.execute_identity or None,
                allow_only_add_steps=depth_config.allow_only_add_steps if depth_config else True,
                reject_write_on_existing=depth_config.reject_write_on_existing if depth_config else False,
                require_test_before_complete=depth_config.require_test_before_complete if depth_config else False,
            )

            self._last_engine = engine
            last_result = result or step_desc
            completed.append(f"Step {step_num}: {step_desc}: {(result or 'done')[:80]}")

            # Auto-test after code-modifying steps
            if strategy.auto_test and step_files and self._test_checkpoint:
                passed, total_tests = self._test_checkpoint.run()
                if verbose and total_tests > 0:
                    progress = self._test_checkpoint.progress_str()
                    color = RED if self._test_checkpoint.has_regression() else GREEN
                    _log(f"    {color}{progress}{RESET}")

        return last_result

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _clean_llm_text(response: Any) -> str:
        """Extract and clean text from LLM response."""
        content = response.choices[0].message.content or ""
        content = content.strip()
        # Strip <think> blocks
        content = re.sub(
            r"<(?:think|thinking)>.*?</(?:think|thinking)>",
            "", content, flags=re.DOTALL | re.IGNORECASE,
        )
        content = content.strip()
        # Strip markdown code fences
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        if content.startswith("json"):
            content = content[4:].strip()
        return content

    def get_changed_files_summary(self) -> str:
        if self._last_engine:
            return self._last_engine.get_changed_files_summary()
        return ""

    def has_file_changes(self) -> bool:
        if self._last_engine:
            return self._last_engine.has_file_changes()
        return False
