"""Validate model-generated plans and questions against strategy constraints.

Checks that plans are granular enough, have specific file references,
include test verification steps, and aren't too vague.
Also validates question lists from the QUESTIONS phase.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from infinidev.engine.phase_prompts import PhaseStrategy

logger = logging.getLogger(__name__)


def validate_plan(
    plan_json: str | list[dict],
    strategy: PhaseStrategy,
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    """Validate a model-generated plan.

    Args:
        plan_json: JSON string or parsed list of step dicts
        strategy: The PhaseStrategy with constraints

    Returns:
        (is_valid, parsed_steps, error_messages)
    """
    # Parse JSON if needed
    if isinstance(plan_json, str):
        try:
            from infinidev.engine.tool_call_parser import safe_json_loads
            steps = safe_json_loads(plan_json)
        except (json.JSONDecodeError, TypeError):
            # Try to extract JSON array from text with surrounding content
            steps = _extract_json_array(plan_json)
            if steps is None:
                return False, [], ["Could not parse plan as JSON. Output a JSON array of steps."]
    else:
        steps = plan_json

    if not isinstance(steps, list):
        return False, [], ["Plan must be a JSON array of step objects."]

    errors: list[str] = []

    # Check minimum step count
    impl_steps = [s for s in steps if s.get("files")]
    if len(impl_steps) < strategy.plan_min_steps:
        errors.append(
            f"Plan has only {len(impl_steps)} implementation steps. "
            f"Need at least {strategy.plan_min_steps}. Break large steps into smaller ones."
        )

    # Validate each step
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"Step {i + 1}: must be a JSON object with 'description' and 'files' keys.")
            continue

        desc = step.get("description", "")
        files = step.get("files", [])

        # Description checks
        if not desc:
            errors.append(f"Step {i + 1}: missing 'description'.")
        elif len(desc) < 20:
            errors.append(
                f"Step {i + 1}: description too short ({len(desc)} chars). "
                f"Be specific about what to do and where."
            )

        # Vague description check
        vague_patterns = [
            "implement the", "fix the bugs", "write the code",
            "set up", "implement everything", "finish",
            "do the rest", "complete the",
        ]
        if desc and any(p in desc.lower() for p in vague_patterns):
            if len(desc) < 60:  # Short + vague = bad
                errors.append(
                    f"Step {i + 1}: description is too vague: \"{desc}\". "
                    f"Name specific files, functions, and what changes."
                )

        # Files check for implementation steps
        if files and len(files) > strategy.plan_max_step_files:
            errors.append(
                f"Step {i + 1}: touches {len(files)} files (max {strategy.plan_max_step_files}). "
                f"Split into smaller steps."
            )

    # Check for test/verification steps
    if strategy.auto_test:
        test_steps = [
            s for s in steps
            if any(
                kw in s.get("description", "").lower()
                for kw in ["run test", "pytest", "npm test", "verify", "check progress"]
            )
        ]
        if len(impl_steps) >= 3 and len(test_steps) == 0:
            errors.append(
                "Plan has no test/verification steps. "
                "Add 'run tests' steps after every 2-3 implementation steps."
            )

    # Normalize steps (add step numbers if missing)
    normalized = []
    for i, step in enumerate(steps):
        if isinstance(step, dict):
            normalized.append({
                "step": step.get("step", i + 1),
                "description": step.get("description", f"Step {i + 1}"),
                "files": step.get("files", []),
            })

    is_valid = len(errors) == 0
    return is_valid, normalized, errors


def _extract_json_array(text: str) -> list | None:
    """Try to extract a JSON array from text that might contain other content."""
    # Find the first [ and matching ]
    start = text.find("[")
    if start == -1:
        return None

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def validate_questions(
    questions_json: str | list[dict],
    min_count: int = 2,
    max_count: int = 10,
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    """Validate model-generated questions.

    Returns (is_valid, parsed_questions, error_messages).
    """
    if isinstance(questions_json, str):
        try:
            from infinidev.engine.tool_call_parser import safe_json_loads
            questions = safe_json_loads(questions_json)
        except (json.JSONDecodeError, TypeError):
            questions = _extract_json_array(questions_json)
            if questions is None:
                return False, [], ["Could not parse questions as JSON. Output a JSON array."]
    else:
        questions = questions_json

    if not isinstance(questions, list):
        return False, [], ["Questions must be a JSON array."]

    errors: list[str] = []

    if len(questions) < min_count:
        errors.append(f"Too few questions ({len(questions)}). Generate at least {min_count}.")

    if len(questions) > max_count:
        errors.append(f"Too many questions ({len(questions)}). Maximum is {max_count}.")
        questions = questions[:max_count]

    vague_patterns = [
        "what is this project", "what language", "how does everything",
        "can you explain", "what should i do", "tell me about",
    ]

    validated = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            errors.append(f"Question {i + 1}: must be a JSON object with 'question' key.")
            continue

        text = q.get("question", "")
        if not text or len(text) < 15:
            errors.append(f"Question {i + 1}: too short or empty. Be specific.")
            continue

        if any(p in text.lower() for p in vague_patterns):
            errors.append(f"Question {i + 1}: too vague: \"{text}\". Ask about something specific.")
            continue

        validated.append({
            "question": text,
            "intent": q.get("intent", "general"),
        })

    is_valid = len(errors) == 0 and len(validated) >= min_count
    return is_valid, validated, errors


def format_rejection(errors: list[str]) -> str:
    """Format validation errors into a re-prompt message for the model."""
    lines = ["Your plan was REJECTED for the following reasons:\n"]
    for i, err in enumerate(errors, 1):
        lines.append(f"  {i}. {err}")
    lines.append("\nFix these issues and output a new plan as a JSON array.")
    return "\n".join(lines)
