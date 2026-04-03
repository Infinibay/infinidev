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


class AnalysisEngine:
    """Pre-development analysis engine.

    Runs the analyst as a full agent loop with tool access so it can
    explore the codebase before producing a specification.
    """

    def __init__(self) -> None:
        self._qa_history: list[dict[str, str]] = []
        self._analysis_rounds: int = 0
        self._max_rounds: int = 3

    def reset(self) -> None:
        """Reset state for a new user request."""
        self._qa_history = []
        self._analysis_rounds = 0

    def add_answer(self, question_context: str, answer: str) -> None:
        """Record a user's answer to a clarifying question."""
        self._qa_history.append({
            "question_context": question_context,
            "answer": answer,
        })

    def analyze(
        self,
        user_input: str,
        *,
        session_summaries: list[str] | None = None,
    ) -> AnalysisResult:
        """Analyze the user's request using a full agent loop with tools.

        Args:
            user_input: The raw user input.
            session_summaries: Previous conversation summaries for context.

        Returns:
            AnalysisResult with action type and relevant data.
        """
        from infinidev.config.llm import get_litellm_params
        from infinidev.flows.event_listeners import event_bus

        llm_params = get_litellm_params()
        if llm_params is None:
            logger.warning("AnalysisEngine: no LLM params, passing through")
            return AnalysisResult(
                action="passthrough",
                original_input=user_input,
                reason="No LLM configured",
            )

        self._analysis_rounds += 1

        event_bus.emit("analysis_start", 0, "", {
            "round": self._analysis_rounds,
            "input": user_input[:200],
        })

        try:
            result = self._run_analyst_loop(user_input, session_summaries)
        except Exception as e:
            logger.warning("AnalysisEngine: agent loop failed (%s), passing through", e)
            result = AnalysisResult(
                action="passthrough",
                original_input=user_input,
                reason=f"Analysis failed: {e}",
            )

        # Handle research action — perform web search and re-run
        if result.action == "research":
            event_bus.emit("analysis_research", 0, "", {
                "queries": result.research_queries,
                "reason": result.research_reason,
            })
            try:
                research_results = self._perform_research(result.research_queries)
                enriched_input = user_input + "\n\n" + research_results
                result = self._run_analyst_loop(enriched_input, session_summaries)

                # Prevent infinite research loops
                if result.action == "research":
                    logger.warning("AnalysisEngine: second research request, forcing passthrough")
                    result = AnalysisResult(
                        action="passthrough",
                        original_input=user_input,
                        reason="Research loop prevented — passing through",
                    )
                # Restore original input in result
                result.original_input = user_input
            except Exception as e:
                logger.warning("AnalysisEngine: research loop failed (%s), passing through", e)
                result = AnalysisResult(
                    action="passthrough",
                    original_input=user_input,
                    reason=f"Research failed: {e}",
                )

        event_bus.emit("analysis_complete", 0, "", {
            "action": result.action,
            "round": self._analysis_rounds,
        })

        return result

    def _run_analyst_loop(
        self,
        user_input: str,
        session_summaries: list[str] | None,
    ) -> AnalysisResult:
        """Run the analyst as a full agent loop with tool access."""
        from infinidev.agents.base import InfinidevAgent  # noqa: deferred to avoid circular
        from infinidev.engine.loop import LoopEngine  # noqa: deferred to avoid circular
        from infinidev.prompts.analyst.system import (  # noqa
            ANALYST_BACKSTORY,
            ANALYST_GOAL,
            ANALYST_SYSTEM_PROMPT,
        )

        # Create analyst agent with all tools but analyst identity
        analyst_agent = InfinidevAgent(
            agent_id="analyst",
            role="analyst",
            name="Analyst",
            goal=ANALYST_GOAL,
            backstory=ANALYST_BACKSTORY,
        )
        analyst_agent._session_summaries = session_summaries
        analyst_agent._system_prompt_identity = ANALYST_SYSTEM_PROMPT

        # Build task prompt for the analyst
        task_description = self._build_analysis_prompt(user_input, session_summaries)
        expected_output = (
            "A JSON object with your analysis result. The JSON must have an "
            '"action" field set to one of: "passthrough", "ask", "research", '
            'or "proceed". See the system prompt for the exact format of each action type. '
            "Output ONLY the JSON object as your final_answer."
        )

        # Run the loop engine
        analyst_agent.activate_context()
        engine = LoopEngine()
        try:
            raw_output = engine.execute(
                agent=analyst_agent,
                task_prompt=(task_description, expected_output),
                verbose=True,
            )
        finally:
            analyst_agent.deactivate()

        return self._parse_response(raw_output or "", user_input)

    def _build_analysis_prompt(
        self,
        user_input: str,
        session_summaries: list[str] | None,
    ) -> str:
        """Build the user prompt for the analysis agent loop."""
        parts = []

        # Session context
        if session_summaries:
            numbered = "\n".join(
                f"{i+1}. {s}" for i, s in enumerate(session_summaries)
            )
            parts.append(
                f"## Conversation History\n"
                f"Previous interactions in this session:\n{numbered}"
            )

        # Previous Q&A (from earlier analysis rounds)
        if self._qa_history:
            qa_lines = []
            for qa in self._qa_history:
                qa_lines.append(f"**Q:** {qa['question_context']}")
                qa_lines.append(f"**A:** {qa['answer']}")
            parts.append(
                "## Previous Clarifications\n"
                "The user already answered these questions. Do NOT re-ask:\n"
                + "\n".join(qa_lines)
            )

        # Analysis round info
        if self._analysis_rounds > 1:
            remaining = self._max_rounds - self._analysis_rounds
            parts.append(
                f"## Analysis Round {self._analysis_rounds}\n"
                f"This is round {self._analysis_rounds} of analysis. "
                f"You have {remaining} rounds remaining. "
                f"If you cannot resolve ambiguities, make reasonable assumptions "
                f"and produce a specification."
            )

            # Force specification on last round
            if remaining <= 0:
                parts.append(
                    "**FINAL ROUND**: You MUST produce a specification now. "
                    "Make reasonable assumptions for any unresolved questions."
                )

        # The actual request
        parts.append(f"## User Request\n{user_input}")

        # Instructions
        parts.append(
            "## Instructions\n"
            "1. Start by exploring the codebase to understand the project structure "
            "and the code relevant to this request. Use list_directory, read_file, "
            "code_search, and glob tools.\n"
            "2. Based on what you find, analyze the request and produce your "
            "result as a JSON object in your final_answer.\n"
            "3. Do NOT write or modify any files. You are only analyzing."
        )

        return "\n\n".join(parts)

    def _parse_response(self, raw: str, original_input: str) -> AnalysisResult:
        """Parse the LLM response into an AnalysisResult."""
        # Try to extract JSON from the response
        raw = raw.strip()

        # Handle markdown code blocks
        if raw.startswith("```"):
            # Strip ```json and ``` markers
            lines = raw.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.strip() == "```" and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            raw = "\n".join(json_lines)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    logger.warning("AnalysisEngine: could not parse response as JSON")
                    return AnalysisResult(
                        action="passthrough",
                        original_input=original_input,
                        reason="Could not parse analysis response",
                    )
            else:
                return AnalysisResult(
                    action="passthrough",
                    original_input=original_input,
                    reason="No JSON found in analysis response",
                )

        action = data.get("action", "passthrough")

        if action == "passthrough":
            return AnalysisResult(
                action="passthrough",
                original_input=original_input,
                reason=data.get("reason", ""),
                flow="done",
            )

        elif action == "ask":
            return AnalysisResult(
                action="ask",
                original_input=original_input,
                questions=data.get("questions", []),
                context=data.get("context", ""),
            )

        elif action == "research":
            return AnalysisResult(
                action="research",
                original_input=original_input,
                research_queries=data.get("queries", []),
                research_reason=data.get("reason", ""),
            )

        elif action == "proceed":
            flow = data.get("flow", "develop")
            # Validate flow against registry + "done"
            valid_flows = {"develop", "research", "document", "sysadmin", "done"}
            if flow not in valid_flows:
                logger.warning("AnalysisEngine: invalid flow '%s', defaulting to 'develop'", flow)
                flow = "develop"
            return AnalysisResult(
                action="proceed",
                original_input=original_input,
                specification=data.get("specification", {}),
                flow=flow,
            )

        else:
            logger.warning("AnalysisEngine: unknown action '%s', passing through", action)
            return AnalysisResult(
                action="passthrough",
                original_input=original_input,
                reason=f"Unknown action: {action}",
            )

    def _perform_research(self, queries: list[str]) -> str:
        """Perform web research for the given queries.

        Args:
            queries: Search queries (max 3 will be used).

        Returns:
            Formatted markdown string with research results.
        """
        from infinidev.tools.web.backends import search_ddg, fetch_with_trafilatura

        queries = queries[:3]  # Cap at 3
        parts = ["## Research Results\n"]

        for query in queries:
            parts.append(f"### Query: {query}\n")
            try:
                results = search_ddg(query, num_results=3)
                if not results:
                    parts.append("No results found.\n")
                    continue

                for i, r in enumerate(results):
                    title = r.get("title", "No title")
                    url = r.get("url", "")
                    snippet = r.get("snippet", "")
                    parts.append(f"**{title}**")
                    if url:
                        parts.append(f"URL: {url}")
                    if snippet:
                        parts.append(f"Snippet: {snippet}")

                    # Fetch full content only for the first result of each query
                    if i == 0 and url:
                        try:
                            content = fetch_with_trafilatura(url, timeout=10)
                            if content:
                                content = content[:2000]
                                parts.append(f"Content excerpt: {content}")
                        except Exception:
                            pass

                    parts.append("")

            except Exception as e:
                logger.warning("Research query failed for '%s': %s", query, e)
                parts.append("No results found.\n")

        return "\n".join(parts)

    @property
    def can_ask_more(self) -> bool:
        """Whether we can do another analysis round (haven't hit max)."""
        return self._analysis_rounds < self._max_rounds
