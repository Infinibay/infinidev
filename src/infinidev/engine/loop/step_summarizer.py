"""Step summarization for the loop engine."""

from __future__ import annotations

import logging
from typing import Any

from infinidev.config.settings import settings
from infinidev.engine.formats.tool_call_parser import safe_json_loads as _safe_json_loads
from infinidev.engine.loop.models import LoopState, StepResult
from infinidev.prompts.profiles import (
    EffectivePromptConfiguration,
    resolve_prompt_fragment,
)

logger = logging.getLogger(__name__)


_SUMMARIZER_GUIDANCE = """\
You are a step summarizer for a coding agent. Analyze the raw step data and produce a structured JSON summary.
The summary helps the agent remember what happened and plan the next step effectively.
"""

_SUMMARIZER_OUTPUT_CONTRACT = """\
Output EXACTLY this JSON format (no markdown, no code fences, just JSON):
{
  "files_to_preload": ["path1", "path2"],
  "changes_made": "Files modified: what changed and why. Include brief diffs if possible.",
  "discovered": "Classes, files, function signatures, architecture patterns, web content, and command results found in this step.",
  "pending": "What still needs doing: code to fix/implement, problems found, things to investigate.",
  "anti_patterns": "What went wrong or was wasteful. Failed approaches, dead ends, and repeated errors that must NOT be repeated.",
  "summary": "1-2 sentences: what was done, how, and why."
}

Rules:
- files_to_preload: ONLY files the NEXT step will need to read/edit. Max 5 paths.
- Keep each text field under 150 tokens. Focus on FACTS, not narration.
- anti_patterns: Look for repeated failed tool calls, re-reading same files, loops without progress.
- If no anti-patterns were observed, set it to empty string.
"""


_MAX_SUMMARY_FIELD_CHARS = 500
_MAX_FILES_TO_PRELOAD = 5


def _summary_text(document: dict, field: str, fallback: str = "") -> str:
    """Return one bounded text field, rejecting non-string model output."""
    value = document.get(field)
    if not isinstance(value, str):
        return fallback
    value = value.strip()
    return (value or fallback)[:_MAX_SUMMARY_FIELD_CHARS]


def _summary_files(document: dict) -> list[str]:
    """Return up to five unique, non-empty path strings in model order."""
    value = document.get("files_to_preload")
    if not isinstance(value, list):
        return []

    paths: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        path = item.strip()
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
        if len(paths) >= _MAX_FILES_TO_PRELOAD:
            break
    return paths


def _normalize_summary(document: dict, fallback_summary: str) -> dict:
    """Normalize the untrusted JSON object returned by the summarizer model."""
    return {
        "summary": _summary_text(document, "summary", fallback_summary),
        "files_to_preload": _summary_files(document),
        "changes_made": _summary_text(document, "changes_made"),
        "discovered": _summary_text(document, "discovered"),
        "pending": _summary_text(document, "pending"),
        "anti_patterns": _summary_text(document, "anti_patterns"),
    }


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
        next_lines = [f"- {s.title}" for s in next_pending[:5]]
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

    # Make the summarizer LLM call (no tools). The JSON contract is atomic;
    # only role/method guidance can be disabled.
    configuration = (
        llm_params.get("_prompt_configuration")
        or EffectivePromptConfiguration.compile()
    )
    guidance = resolve_prompt_fragment(
        "summary.step_guidance",
        "summarize",
        _SUMMARIZER_GUIDANCE,
        configuration=configuration,
    )
    system_prompt = "\n\n".join(
        part for part in (guidance, _SUMMARIZER_OUTPUT_CONTRACT) if part
    )
    summarizer_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        import litellm
        summarizer_params = {
            key: value
            for key, value in llm_params.items()
            if key not in {"tool_choice", "_prompt_configuration"}
        }
        summarizer_params.pop("tools", None)
        # Hard timeout: the summarizer is a "best effort" enrichment — if the
        # local model takes longer than this we'd rather fall back to the raw
        # summary than double the wall-clock cost of every step. Configurable
        # via INFINIDEV_LOOP_SUMMARIZER_TIMEOUT.
        timeout_s = float(getattr(settings, "LOOP_SUMMARIZER_TIMEOUT", 30) or 30)
        summarizer_params["timeout"] = timeout_s

        # The summarizer's system prompt is fully static and fires on
        # every completed step — apply provider-aware prompt caching so
        # the fixed prefix is a cache hit from the 2nd call onward.
        from infinidev.config.prompt_cache import apply_prompt_caching
        call_kwargs = {
            **summarizer_params,
            "messages": summarizer_messages,
            "max_tokens": 500,
        }
        apply_prompt_caching(call_kwargs, settings.LLM_PROVIDER)
        response = litellm.completion(**call_kwargs)
        content = response.choices[0].message.content or ""

        # Try to parse as JSON. Every field remains untrusted model output
        # until _normalize_summary validates its type and bound.
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
            return _normalize_summary(parsed, step_result.summary)
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
                    tc_lines.append(f"  Tool: {name}({args})")
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


_FINAL_HEADERS = {
    "cancelled": "Task stopped by the user. Completed before stopping:",
    "exhausted": "Task execution summary (iteration limit reached):",
}


def _synthesize_final(state: LoopState, status: str = "exhausted") -> str:
    """Build a final answer from history when no step produced one.

    The header names the reason. A cancelled run reported "iteration limit
    reached", which is not what happened and is read by the reviewer, the
    end-of-task hooks and next turn's work summary.
    """
    if not state.history:
        return (
            "Task stopped by the user before any step completed."
            if status == "cancelled" else "No actions were completed."
        )

    parts = [_FINAL_HEADERS.get(status, _FINAL_HEADERS["exhausted"])]
    for record in state.history:
        parts.append(f"- Step {record.step_index}: {record.summary}")
    return "\n".join(parts)
