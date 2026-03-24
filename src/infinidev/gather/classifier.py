"""Ticket type classification using the LoopEngine."""

from __future__ import annotations

import json
import logging
from typing import Any

from infinidev.gather.models import ClassificationResult, TicketType

logger = logging.getLogger(__name__)

_CLASSIFIER_IDENTITY = """\
## Identity

You are a ticket classifier. Your ONLY job is to classify a software task into one type.

Types:
- bug: Something is broken, produces errors, or behaves incorrectly
- feature: New functionality to add or existing functionality to extend
- refactor: Restructure/reorganize existing code without changing behavior
- sysadmin: Infrastructure, deployment, configuration, or system administration
- other: Anything that doesn't fit the above

## Rules

- Read the ticket carefully.
- Call step_complete with status="done" immediately.
- In final_answer, output ONLY this JSON: {"ticket_type": "bug|feature|refactor|sysadmin|other", "reasoning": "1 sentence why", "keywords": ["key", "terms"]}
- Do NOT use any tools. Just classify and respond.
"""


def classify_ticket(
    ticket_description: str,
    analyst_spec: dict | None = None,
    agent: Any = None,
) -> ClassificationResult:
    """Classify a ticket into a type using LoopEngine.

    Falls back to TicketType.other on any error.
    """
    fallback = ClassificationResult(
        ticket_type=TicketType.other,
        reasoning="Classification failed — using default.",
    )

    if agent is None:
        return fallback

    from infinidev.config.settings import settings
    from infinidev.engine.loop_engine import LoopEngine

    # Save original settings
    original_identity = getattr(agent, "_system_prompt_identity", None)
    original_backstory = agent.backstory
    original_max_iter = settings.LOOP_MAX_ITERATIONS
    original_max_tools = settings.LOOP_MAX_TOTAL_TOOL_CALLS
    original_max_per_action = settings.LOOP_MAX_TOOL_CALLS_PER_ACTION
    original_nudge = settings.LOOP_STEP_NUDGE_THRESHOLD
    original_summarizer = settings.LOOP_SUMMARIZER_ENABLED
    original_gather = settings.GATHER_ENABLED

    try:
        agent._system_prompt_identity = _CLASSIFIER_IDENTITY
        agent.backstory = "Ticket classifier."
        settings.LOOP_MAX_ITERATIONS = 2
        settings.LOOP_MAX_TOTAL_TOOL_CALLS = 10
        settings.LOOP_MAX_TOOL_CALLS_PER_ACTION = 10
        settings.LOOP_STEP_NUDGE_THRESHOLD = 0
        settings.LOOP_SUMMARIZER_ENABLED = False
        settings.GATHER_ENABLED = False

        task_desc = f"Classify this ticket:\n\n{ticket_description[:3000]}"
        if analyst_spec:
            task_desc += f"\n\nAnalyst specification:\n{json.dumps(analyst_spec, indent=2)[:1000]}"

        engine = LoopEngine()
        result = engine.execute(
            agent=agent,
            task_prompt=(task_desc, "Output JSON with ticket_type, reasoning, and keywords."),
            verbose=False,
            task_tools=[],  # No tools needed, just step_complete
        )

        logger.info("Classifier raw result: %s", (result or "")[:300])

        if result:
            parsed = _extract_json(result)
            if parsed:
                return ClassificationResult(
                    ticket_type=TicketType(parsed.get("ticket_type", "other")),
                    reasoning=str(parsed.get("reasoning", ""))[:300],
                    keywords=list(parsed.get("keywords", []))[:10],
                )

            # Try to detect type from text
            result_lower = result.lower()
            for tt in TicketType:
                if tt.value in result_lower:
                    return ClassificationResult(ticket_type=tt, reasoning=result[:300])

    except Exception as exc:
        logger.warning("Ticket classification failed: %s", str(exc)[:200])

    finally:
        agent._system_prompt_identity = original_identity
        agent.backstory = original_backstory
        settings.LOOP_MAX_ITERATIONS = original_max_iter
        settings.LOOP_MAX_TOTAL_TOOL_CALLS = original_max_tools
        settings.LOOP_MAX_TOOL_CALLS_PER_ACTION = original_max_per_action
        settings.LOOP_STEP_NUDGE_THRESHOLD = original_nudge
        settings.LOOP_SUMMARIZER_ENABLED = original_summarizer
        settings.GATHER_ENABLED = original_gather

    return fallback


def _extract_json(text: str) -> dict | None:
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

    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None
