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
from infinidev.engine.phase_prompts import get_strategy, PhaseStrategy
from infinidev.engine.plan_validator import (
    validate_questions, format_rejection,
)
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
    ) -> str:
        strategy = get_strategy(task_type)
        description, expected_output = task_prompt

        if verbose:
            _log(f"\n{BOLD}{CYAN}⚡ Phase Engine{RESET} — strategy: {task_type}")

        # Init test checkpoint
        from infinidev.tools.base.context import get_current_workspace_path
        workdir = get_current_workspace_path()
        self._test_checkpoint = TestCheckpoint(test_command, workdir)

        # ── Phase 1: QUESTIONS ────────────────────────────────────────
        if verbose:
            _log(f"\n{BOLD}❓ Phase 1: QUESTIONS{RESET}")

        questions = self._generate_questions(agent, description, strategy, verbose)

        if verbose:
            _log(f"  {DIM}{len(questions)} questions generated{RESET}")
            for i, q in enumerate(questions):
                _log(f"    {DIM}{i+1}. {q['question'][:80]}{RESET}")

        # ── Phase 2: INVESTIGATE ──────────────────────────────────────
        if verbose:
            _log(f"\n{BOLD}🔍 Phase 2: INVESTIGATE{RESET}")

        answers, all_notes = self._investigate(
            agent, questions, strategy, task_tools, verbose,
        )

        if verbose:
            _log(f"  {DIM}{len(answers)} answers, {len(all_notes)} total notes{RESET}")

        # ── Phase 3: PLAN ─────────────────────────────────────────────
        if verbose:
            _log(f"\n{BOLD}📋 Phase 3: PLAN{RESET}")

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

        # ── Phase 4: EXECUTE ──────────────────────────────────────────
        if verbose:
            _log(f"\n{BOLD}🔨 Phase 4: EXECUTE{RESET}")

        result = self._execute_plan(
            agent, description, expected_output, answers, all_notes,
            plan_steps, strategy, task_tools, verbose,
        )

        # Final test run
        if strategy.auto_test and self._test_checkpoint:
            passed, total = self._test_checkpoint.run()
            if verbose and total > 0:
                _log(f"\n  {BOLD}Final: {self._test_checkpoint.progress_str()}{RESET}")

        return result

    # ── Phase 1: Generate questions ───────────────────────────────────

    def _generate_questions(
        self,
        agent: Any,
        description: str,
        strategy: PhaseStrategy,
        verbose: bool,
    ) -> list[dict[str, Any]]:
        """Direct LLM call to generate investigation questions."""
        from infinidev.config.llm import get_litellm_params
        from infinidev.engine.loop_context import build_system_prompt

        prompt = (
            f"You are preparing to work on a task. Before starting, generate "
            f"questions that will give you everything you need to create a "
            f"detailed implementation plan.\n\n"
            f"Task: {description}\n\n"
            f"{strategy.questions_prompt}\n\n"
            f"Output ONLY a JSON array of questions. No other text.\n"
            f"Each question: {{\"question\": \"...\", \"intent\": \"...\"}}\n"
            f"Generate {strategy.questions_min}-{strategy.questions_max} questions."
        )

        prompt = "/no_think\n" + prompt

        llm_params = get_litellm_params()
        system_prompt = build_system_prompt(
            agent.backstory,
            identity_override=getattr(agent, '_system_prompt_identity', None),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(2):
            try:
                response = call_llm(llm_params, messages)
                content = self._clean_llm_text(response)

                is_valid, questions, errors = validate_questions(
                    content, strategy.questions_min, strategy.questions_max,
                )
                if is_valid:
                    return questions

                if verbose:
                    for err in errors:
                        _log(f"  {YELLOW}⚠ {err}{RESET}")

                if attempt == 0:
                    rejection = format_rejection(errors)
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": rejection})

            except Exception as exc:
                logger.warning("Question generation failed: %s", str(exc)[:200])

        # Fallback to default questions
        if verbose:
            _log(f"  {YELLOW}Using fallback questions{RESET}")
        return [{"question": q, "intent": "fallback"} for q in strategy.fallback_questions]

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
        # Filter to read-only tools
        if all_tools:
            read_tools = [
                t for t in all_tools
                if getattr(t, 'name', '') in _READ_ONLY_TOOLS
            ]
        else:
            read_tools = None

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

        Read-only tools are available so the model can re-check files.
        """
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

        prompt = (
            f"You are creating an implementation plan. Use step_complete with "
            f"next_steps to ADD steps to the plan. Each step_complete call should "
            f"add 2-4 new steps. Keep going until the plan covers the full task, "
            f"then call step_complete with status='done'.\n\n"
            f"You can use read-only tools (read_file, glob, code_search) if you "
            f"need to check something while planning.\n\n"
            f"{strategy.plan_prompt}\n\n"
            f"## YOUR TASK\n{description}\n\n"
            f"## YOUR INVESTIGATION RESULTS\n{answers_text}\n"
            f"{notes_text}"
            f"{baseline_str}"
        )

        # Filter to read-only tools
        if all_tools:
            plan_tools = [
                t for t in all_tools
                if getattr(t, 'name', '') in _READ_ONLY_TOOLS
            ]
        else:
            plan_tools = None

        engine = LoopEngine()
        engine.execute(
            agent=agent,
            task_prompt=(prompt, "Build a complete implementation plan using step_complete(next_steps=[...])."),
            verbose=verbose,
            task_tools=plan_tools,
            max_iterations=50,
            max_total_tool_calls=1000,
            max_tool_calls_per_action=0,  # unlimited per step
            nudge_threshold=0,  # don't nudge during planning
            summarizer_enabled=False,
        )

        # Extract ALL plan steps (pending + done) from the engine's state.
        # The engine "executes" plan steps but in PLAN phase that just means
        # the model added them. We collect everything as our execution plan.
        steps = []
        if engine._last_state and engine._last_state.plan.steps:
            for s in engine._last_state.plan.steps:
                steps.append({
                    "step": s.index,
                    "description": s.description,
                    "files": [],
                })

        if verbose:
            _log(f"  {DIM}Plan generated: {len(steps)} steps{RESET}")

        if len(steps) < strategy.plan_min_steps:
            if verbose:
                _log(f"  {YELLOW}⚠ Only {len(steps)} steps (need {strategy.plan_min_steps}){RESET}")
            return []

        return steps

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
            full_prompt = (
                f"{step_prompt}\n\n"
                f"{notes_section}"
                f"## INVESTIGATION RESULTS\n{answers_text}\n\n"
                f"## COMPLETED STEPS\n{completed_str}\n"
                f"{progress_str}{regression_warning}"
            )

            if verbose:
                _log(f"\n  {CYAN}Step {step_num}/{total}: {step_desc[:80]}{RESET}")

            engine = LoopEngine()
            result = engine.execute(
                agent=agent,
                task_prompt=(full_prompt, expected_output),
                verbose=verbose,
                task_tools=all_tools,
                max_iterations=5,
                max_total_tool_calls=strategy.execute_max_tool_calls_per_step,
                max_tool_calls_per_action=strategy.execute_max_tool_calls_per_step,
                nudge_threshold=strategy.execute_max_tool_calls_per_step - 3,
                summarizer_enabled=True,
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
