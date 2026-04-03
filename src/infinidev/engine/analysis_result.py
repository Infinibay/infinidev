"""Pre-development analysis engine.

Runs the analyst as a full agent loop with tool access to explore the
codebase before producing a specification. Handles:
- Passthrough for simple requests (greetings, questions, quick tasks)
- Clarifying questions for ambiguous/incomplete requests
- Web research for external API/library references
- Full specification generation for complex requests
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of the pre-development analysis phase."""

    action: str  # "passthrough" | "ask" | "proceed" | "research"

    # Original user input (always present)
    original_input: str = ""

    # For passthrough: reason it was passed through
    reason: str = ""

    # For ask: questions to present to the user
    questions: list[dict[str, Any]] = field(default_factory=list)
    context: str = ""

    # For proceed: the enriched specification
    specification: dict[str, Any] = field(default_factory=dict)

    # For research: queries to search and reason
    research_queries: list[str] = field(default_factory=list)
    research_reason: str = ""

    # Flow to route to: "develop" | "research" | "document" | "sysadmin" | "explore" | "done"
    flow: str = "develop"

    # The enriched task prompt for the developer (built from specification)
    enriched_prompt: str = ""

    def build_flow_prompt(self) -> tuple[str, str]:
        """Build the (description, expected_output) tuple for the developer loop.

        For passthrough: returns the original input as-is.
        For proceed: returns an enriched prompt with the specification.
        """
        if self.action == "passthrough":
            return (self.original_input, "Complete the task and report findings.")

        if self.action != "proceed" or not self.specification:
            return (self.original_input, "Complete the task and report findings.")

        spec = self.specification
        parts = []

        # Original request
        parts.append(f"## User Request\n{self.original_input}")

        # Summary
        if spec.get("summary"):
            parts.append(f"## Analysis Summary\n{spec['summary']}")

        # Requirements
        reqs = spec.get("requirements", [])
        if reqs:
            req_lines = [f"- {r}" for r in reqs]
            parts.append("## Requirements\n" + "\n".join(req_lines))

        # Hidden requirements
        hidden = spec.get("hidden_requirements", [])
        if hidden:
            hidden_lines = [f"- {h}" for h in hidden]
            parts.append(
                "## Identified Hidden Requirements\n"
                "These were not explicitly stated but are logical consequences "
                "of the request:\n" + "\n".join(hidden_lines)
            )

        # Assumptions
        assumptions = spec.get("assumptions", [])
        if assumptions:
            assumption_lines = [f"- {a}" for a in assumptions]
            parts.append("## Assumptions\n" + "\n".join(assumption_lines))

        # Out of scope
        oos = spec.get("out_of_scope", [])
        if oos:
            oos_lines = [f"- {o}" for o in oos]
            parts.append("## Out of Scope\n" + "\n".join(oos_lines))

        # Technical notes
        if spec.get("technical_notes"):
            parts.append(f"## Technical Notes\n{spec['technical_notes']}")

        description = "\n\n".join(parts)
        expected = "Complete the task according to the specification above and report findings."

        return (description, expected)

    # Backward-compatible alias
    build_developer_prompt = build_flow_prompt

    def format_questions_for_user(self) -> str:
        """Format questions for display to the user."""
        if not self.questions:
            return ""

        parts = []
        if self.context:
            parts.append(f"I've analyzed your request. {self.context}\n")

        parts.append(
            "Before I proceed, I need to clarify a few things:\n"
        )

        for i, q in enumerate(self.questions, 1):
            parts.append(f"**Question {i}:** {q.get('question', '')}")
            if q.get("why"):
                parts.append(f"  *Why this matters:* {q['why']}")
            options = q.get("options", [])
            if options:
                for opt in options:
                    parts.append(f"  - {opt}")
            parts.append("")

        return "\n".join(parts)


