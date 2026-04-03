"""Step summarization for the loop engine."""

from __future__ import annotations

import json
import logging
from typing import Any

from infinidev.engine.loop_models import LoopState, StepResult

logger = logging.getLogger(__name__)


_SUMMARIZER_SYSTEM_PROMPT = """\
You are a step summarizer for a coding agent. Analyze the raw step data and produce a structured JSON summary.
The summary helps the agent remember what happened and plan the next step effectively.

Output EXACTLY this JSON format (no markdown, no code fences, just JSON):
{
  "files_to_preload": ["path1", "path2"],
  "changes_made": "Files modified: what changed and why. Include brief diffs if possible.",
  "discovered": "Relevant classes, files, function signatures, architecture patterns, web content, command results found.",
  "pending": "What still needs doing: code to fix/implement, problems found, things to investigate.",
  "anti_patterns": "What went wrong or was wasteful. Failed approaches, dead ends, repeated errors that should NOT be repeated.",
  "summary": "1-2 sentences: what was done, how, and why."
}

Rules:
- files_to_preload: ONLY files the NEXT step will need to read/edit. Max 5 paths.
- Keep each text field under 150 tokens. Focus on FACTS, not narration.
- anti_patterns: Look for repeated failed tool calls, re-reading same files, loops without progress.
- If no anti-patterns were observed, set it to empty string.
"""


def _summarize_step(
    messages: list[dict],
    task_description: str,
    state: LoopState,
    step_result: "StepResult",
    llm_params: dict,
) -> dict:
    """Make a dedicated LLM call to produce a structured step summary.

    Returns a dict with keys: summary, files_to_preload, changes_made,
    discovered, pending, anti_patterns. Falls back to step_result.summary
    on any error.
    """
    fallback = {
        "summary": step_result.summary,
        "files_to_preload": [],
        "changes_made": "",
        "discovered": "",
        "pending": "",
        "anti_patterns": "",
    }

    # Build the user prompt with raw step data
    parts = [f"<task>\n{task_description}\n</task>"]

    # Current plan state
    plan_text = state.plan.render() if state.plan.steps else "No plan yet."
    parts.append(f"<plan>\n{plan_text}\n</plan>")

    # Next pending steps
    next_pending = [s for s in state.plan.steps if s.status == "pending"]
    if next_pending:
        next_lines = [f"- {s.description}" for s in next_pending[:5]]
        parts.append(f"<next-steps>\n{chr(10).join(next_lines)}\n</next-steps>")

    # Previous summaries for context
    if state.history:
        prev = [f"- Step {r.step_index}: {r.summary}" for r in state.history[-3:]]
        parts.append(f"<previous-summaries>\n{chr(10).join(prev)}\n</previous-summaries>")

    # Raw step messages (truncated)
    max_input = settings.LOOP_SUMMARIZER_MAX_INPUT_TOKENS
    step_msgs_text = _truncate_step_messages(messages, max_input)
    parts.append(f"<step-messages>\n{step_msgs_text}\n</step-messages>")

    user_prompt = "\n\n".join(parts)

    # Make the summarizer LLM call (no tools)
    summarizer_messages = [
        {"role": "system", "content": _SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        import litellm
        summarizer_params = {k: v for k, v in llm_params.items() if k != "tool_choice"}
        summarizer_params.pop("tools", None)
        response = litellm.completion(
            **summarizer_params,
            messages=summarizer_messages,
            max_tokens=500,
        )
        content = response.choices[0].message.content or ""

        # Try to parse as JSON
        import json as _json
        # Strip markdown code fences if present
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()

        parsed = _safe_json_loads(clean)
        if isinstance(parsed, dict):
            return {
                "summary": str(parsed.get("summary", step_result.summary))[:500],
                "files_to_preload": list(parsed.get("files_to_preload", []))[:5],
                "changes_made": str(parsed.get("changes_made", ""))[:500],
                "discovered": str(parsed.get("discovered", ""))[:500],
                "pending": str(parsed.get("pending", ""))[:500],
                "anti_patterns": str(parsed.get("anti_patterns", ""))[:500],
            }
    except Exception as exc:
        logger.debug("Summarizer call failed, using fallback: %s", str(exc)[:200])

    return fallback


def _truncate_step_messages(messages: list[dict], max_tokens: int) -> str:
    """Truncate step messages to fit within a token budget.

    Keeps tool call names and arguments, truncates tool results.
    Rough estimate: 1 token ≈ 4 chars.
    """
    max_chars = max_tokens * 4
    parts = []
    total_chars = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue  # Skip system prompt (already in summarizer context)

        if role == "tool":
            # Truncate tool results to 500 chars each
            tool_id = msg.get("tool_call_id", "")
            truncated = content[:500] + ("..." if len(content) > 500 else "")
            line = f"[Tool result {tool_id}]: {truncated}"
        elif role == "assistant":
            # Keep tool call info
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_lines = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args = str(fn.get("arguments", ""))[:300]
                    tc_lines.append(f"  → {name}({args})")
                line = "Assistant tool calls:\n" + "\n".join(tc_lines)
            else:
                line = f"Assistant: {content[:500]}"
        elif role == "user":
            line = f"User: {content[:500]}"
        else:
            line = f"{role}: {content[:300]}"

        line_len = len(line)
        if total_chars + line_len > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                parts.append(line[:remaining] + "...[truncated]")
            break
        parts.append(line)
        total_chars += line_len

    return "\n".join(parts)


def _synthesize_final(state: LoopState) -> str:
    """Synthesize a final answer from accumulated history when loop exhausts iterations."""
    if not state.history:
        return "No actions were completed."

    parts = ["Task execution summary (iteration limit reached):"]
    for record in state.history:
        parts.append(f"- Step {record.step_index}: {record.summary}")
    return "\n".join(parts)
