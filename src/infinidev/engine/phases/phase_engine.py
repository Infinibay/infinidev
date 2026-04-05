"""Phase-based execution: QUESTIONS → INVESTIGATE → PLAN → EXECUTE.

Orchestrates four phases by delegating to specialized components:
- QuestionGenerator: generates investigation questions
- Investigator: answers questions by reading the codebase
- PlanGenerator: creates an implementation plan from findings
- PlanExecutor: executes the plan step-by-step

Each component is in its own module (question_generator.py, investigator.py,
plan_generator.py, plan_executor.py).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from infinidev.engine.loop import LoopEngine
from infinidev.engine.test_checkpoint import TestCheckpoint
from infinidev.engine.engine_logging import (
    log as _log,
    DIM, BOLD, RESET, CYAN, GREEN, YELLOW, RED,
)
from infinidev.prompts.phases import get_strategy, PhaseStrategy

# Phase components
from infinidev.engine.phases.question_generator import (
    _generate_questions,
    _generate_followups,
)
from infinidev.engine.phases.investigator import (
    _investigate,
    _investigate_iteratively,
)
from infinidev.engine.phases.plan_generator import _generate_plan
from infinidev.engine.phases.plan_executor import (
    _execute_minimal,
    _execute_plan,
)

logger = logging.getLogger(__name__)


class PhaseEngine:
    """Four-phase execution: QUESTIONS → INVESTIGATE → PLAN → EXECUTE.

    Thin orchestrator that delegates each phase to its component module.
    """

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

        # ── Step 0: CLASSIFY ─────────────────────────────────────
        from infinidev.config.llm import _is_small_model
        if depth_config is None:
            if _is_small_model():
                # Skip classification LLM call for small models — saves
                # an entire round-trip and small models get it wrong anyway.
                from infinidev.gather.models import ClassificationResult, TicketType
                classification = ClassificationResult(
                    ticket_type=TicketType.feature,
                    reasoning="Small model — skipped classification.",
                    depth=DepthLevel.standard,
                    depth_reasoning="Small model default.",
                )
                if verbose:
                    _log(f"\n{BOLD}{CYAN}⚡ Phase Engine{RESET} — type: feature, depth: standard (small model — skipped classify)")
            else:
                classification = self._classify(agent, description, verbose)
            task_type = classification.ticket_type.value
            depth_config = DEPTH_CONFIGS.get(classification.depth, DEPTH_CONFIGS[DepthLevel.standard])
            if verbose and not _is_small_model():
                _log(f"\n{BOLD}{CYAN}⚡ Phase Engine{RESET} — type: {task_type}, depth: {classification.depth.value}")
                if classification.depth_reasoning:
                    _log(f"  {DIM}{classification.depth_reasoning}{RESET}")
        else:
            if verbose:
                _log(f"\n{BOLD}{CYAN}⚡ Phase Engine{RESET} — type: {task_type}, depth: (provided)")

        strategy = get_strategy(task_type)

        # ── MINIMAL: single free LoopEngine run ─────────────────
        if depth_config.skip_questions and depth_config.skip_investigate and depth_config.plan_min_steps <= 1:
            result, engine = _execute_minimal(agent, description, expected_output, strategy,
                                              task_tools, depth_config, verbose)
            self._last_engine = engine
            if strategy.auto_test and self._test_checkpoint:
                passed, total = self._test_checkpoint.run()
                if verbose and total > 0:
                    _log(f"\n  {BOLD}{self._test_checkpoint.progress_str()}{RESET}")
            return result

        # ── Phase 1+2: QUESTIONS + INVESTIGATE ──────────────────
        answers: list[dict[str, str]] = []
        all_notes: list[str] = []

        if not depth_config.skip_questions:
            strategy.investigate_max_tool_calls = depth_config.investigate_max_tool_calls

            answers, all_notes = _investigate_iteratively(
                agent, description, strategy, task_tools, verbose,
                max_questions=depth_config.questions_max,
                skip_investigate=depth_config.skip_investigate,
            )

        # ── Phase 3: PLAN ───────────────────────────────────────
        if verbose:
            _log(f"\n{BOLD}📋 Phase 3: PLAN{RESET}")

        strategy.plan_min_steps = depth_config.plan_min_steps

        plan_steps = _generate_plan(
            agent, description, answers, all_notes, strategy, task_tools, verbose,
            test_checkpoint=self._test_checkpoint,
        )

        if not plan_steps:
            return "Failed to generate a valid plan."

        if verbose:
            _log(f"  {DIM}Plan: {len(plan_steps)} steps{RESET}")
            for s in plan_steps:
                files_str = ", ".join(s.get("files", [])) or "(verify)"
                _log(f"    {DIM}{s['step']}. {s.get('title', s.get('explanation', ''))[:70]} [{files_str}]{RESET}")

        # ── Phase 4: EXECUTE (with re-plan loop) ────────────────
        result = ""
        for plan_round in range(depth_config.replan_max_rounds):
            if verbose:
                round_label = f" (round {plan_round + 1})" if plan_round > 0 else ""
                _log(f"\n{BOLD}🔨 Phase 4: EXECUTE{round_label}{RESET}")

            result, engine = _execute_plan(
                agent, description, expected_output, answers, all_notes,
                plan_steps, strategy, task_tools, depth_config, verbose,
                test_checkpoint=self._test_checkpoint,
            )
            self._last_engine = engine

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

                    plan_steps = _generate_plan(
                        agent, description, answers, all_notes, strategy, task_tools, verbose,
                    )
                    if not plan_steps:
                        break

                    if verbose:
                        _log(f"  {DIM}Re-plan: {len(plan_steps)} new steps{RESET}")
                        for s in plan_steps:
                            _log(f"    {DIM}{s['step']}. {s.get('title', s.get('explanation', ''))[:70]}{RESET}")
            else:
                break

        return result

    # ── Classify ─────────────────────────────────────────────────

    def _classify(self, agent: Any, description: str, verbose: bool) -> Any:
        """Run ticket classification to determine task_type and depth."""
        from infinidev.gather.classifier import classify_ticket

        if verbose:
            _log(f"\n{BOLD}🏷️  Step 0: CLASSIFY{RESET}")

        result = classify_ticket(description, agent=agent)

        if verbose:
            _log(f"  {DIM}Type: {result.ticket_type.value} — {result.reasoning}{RESET}")
            _log(f"  {DIM}Depth: {result.depth.value} — {result.depth_reasoning}{RESET}")

        return result

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _clean_llm_text(response: Any) -> str:
        """Extract and clean text from LLM response."""
        content = response.choices[0].message.content or ""
        content = content.strip()
        content = re.sub(
            r"<(?:think|thinking)>.*?</(?:think|thinking)>",
            "", content, flags=re.DOTALL | re.IGNORECASE,
        )
        content = content.strip()
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
