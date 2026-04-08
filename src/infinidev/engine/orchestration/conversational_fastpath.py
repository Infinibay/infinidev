"""Pre-planning step: the agent decides chat-vs-work via step_complete.

Every user turn starts here. We run a single LLM call with the
following restrictions:

  * Only ONE tool is exposed: ``step_complete``. The model has no
    way to read files, search code, or run anything else — there is
    nothing to call BUT the terminator.
  * The system prompt explains the choice to the model neutrally:
    pick ``status="done"`` for a conversational reply, or
    ``status="continue"`` to unlock the rest of the toolbox in the
    next iteration.

The two outcomes:

  status="done"      → ``final_answer`` is the user-facing reply,
                       the pipeline returns ``flow="done"`` and
                       skips the analyst entirely.
  status="continue"  → ``summary`` is a one-line "I'm starting X"
                       message shown to the user, the pipeline
                       falls through to the analyst with the
                       message already on screen.

Why this design (vs my earlier custom ``respond`` tool):

  * Reuses the engine's existing step_complete schema and parser —
    no parallel infrastructure to maintain.
  * The model has been seeing step_complete for the entire session,
    so the prompt format is in-distribution.
  * The "tools restricted to step_complete" idea generalises: if we
    later want a "preamble step" with read-only access, we just
    add ``read_file`` to the allowed list.

Prompt-bias warning (the user surfaced this — preserved as a code
comment because it's easy to forget):

  Do NOT frame ``continue`` as "the way to help" or ``done`` as
  "ending too early". The model will read either as a moral
  imperative and tilt every decision the same way. Both outcomes
  are equally valid; the prompt below presents them symmetrically.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from infinidev.engine.analysis.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────


_PREAMBLE_SYSTEM_PROMPT = """You are Infinidev, a coding assistant. \
This is the very first turn after the user wrote a message. Your job \
is to look at what they said and decide which kind of reply matches it.

You have ONE tool available: ``step_complete``. There are no other \
tools right now. You will use ``step_complete`` exactly once and the \
``status`` field decides what happens next.

There are two equally valid options:

Option A: status="done"
  Pick this when the user's message is a reply that does NOT require \
touching any file or running any command. Examples:
    • Greetings: "hola", "hi", "buenas tardes"
    • Thanks / acknowledgements: "gracias", "perfect", "ok"
    • Goodbyes: "chau", "bye", "see you"
    • Smalltalk / questions about you: "who are you?", "are you there?"
    • Simple non-code questions you can answer from memory:
      "what is Python?", "what does this CLI do?"
  When you pick "done", put the actual reply text in ``final_answer``. \
The user sees ``final_answer`` and the conversation ends here. \
Conversational replies are FIRST-CLASS responses — picking "done" is \
the correct, expected outcome for this category. It is NOT abandoning \
the user.

Option B: status="continue"
  Pick this when the user's message asks you to look at code, change \
code, run tests, refactor, debug, search the codebase, or in any way \
inspect the project. Examples:
    • "fix the auth bug in src/auth.py"
    • "explain how the login flow works" (you need to read the code)
    • "add a unit test for parseDate"
    • "refactor verify_token"
    • "what's in this directory?"
  When you pick "continue", put a SHORT one-sentence preview of what \
you're about to do in ``summary``. The user sees the summary as your \
first message, then the system unlocks the rest of the toolbox \
(read_file, edit_symbol, execute_command, etc.) on the next turn so \
you can do the actual work.

CRITICAL RULE — self-referential follow-ups ALWAYS pick "continue":

  If the user is asking you to elaborate on, explain, expand, justify, \
defend, or give details about something YOU said in a previous turn \
(recommendations, findings, code you wrote, an analysis you produced, \
files you mentioned), you MUST pick "continue". Even if the previous \
turn is shown to you above in RECENT CONVERSATION, the visible content \
is a TRUNCATED snapshot — the real findings live in the project's \
files, the knowledge base, and the conversation database. Answering \
from the truncated snapshot is HALLUCINATION, not memory.

Phrases that signal a self-referential follow-up (Spanish + English):
    • "explica/explain/elabora/expand on/dame mas detalle/give me more detail"
    • "por que dijiste/why did you say/justifica/justify"
    • "que significa esa recomendacion/what do you mean by"
    • "como llegaste a/how did you reach/show me where"
    • "muestrame/show me/cita/cite the file/the line"
    • "ampliame/extend/dive deeper into"
  When you see these AND the topic refers to anything from RECENT \
CONVERSATION above → status="continue", summary="Voy a re-leer mis \
hallazgos previos y ampliar". The unlocked toolbox lets you read the \
real files instead of guessing.

The cost of a wrong "done" here is HALLUCINATION — the user catches \
you inventing facts. The cost of a wrong "continue" is just a few \
extra seconds of latency. When in doubt, pick "continue".

Both options are normal. There is NO bias toward one over the other — \
pick whichever literally matches what the user wrote. If they sent a \
greeting, "done" is correct. If they sent a task, "continue" is \
correct. Neither is "playing it safe" or "trying harder".

Always answer in the user's language (Spanish or English). Be brief — \
both ``final_answer`` and ``summary`` should be 1-2 sentences, no more.

Call ``step_complete`` immediately. Do not write content, do not think \
out loud, do not use any other tool — there are no other tools."""


# Restricted tool list for the preamble step. We use a CUSTOM
# step_complete schema (instead of the engine's main one) because
# the engine's description is written for in-loop use — it warns
# "After finishing the current step objective AND verifying the
# outcome" — and the model interprets that as "I haven't done
# anything yet, I shouldn't call this". For the preamble step the
# tool MUST be callable from a fresh start, so we describe it
# accordingly. Same parameter shape as the real one, so the
# downstream parser is unchanged.
_PREAMBLE_STEP_COMPLETE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "step_complete",
        "description": (
            "Call this tool exactly once per turn. It is the ONLY "
            "tool available right now. Pick the status that matches "
            "the user's message: 'done' for a conversational reply "
            "(greeting / thanks / smalltalk / simple answer), "
            "'continue' to unlock the rest of the toolbox so you can "
            "do real work (file reads, edits, tests). The user sees "
            "the reply text immediately either way."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["done", "continue"],
                    "description": (
                        "'done' = conversational reply, no further "
                        "work needed. 'continue' = real work needed, "
                        "the system will unlock the rest of the "
                        "toolbox on the next turn."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "When status='continue': a SHORT (1 sentence) "
                        "preview of what you're about to do, e.g. "
                        "'I'll read src/auth.py to find the bug'."
                    ),
                },
                "final_answer": {
                    "type": "string",
                    "description": (
                        "When status='done': your conversational reply "
                        "to the user (1-2 sentences). Match their "
                        "language (Spanish or English)."
                    ),
                },
            },
            "required": ["status"],
        },
    },
}


def _get_preamble_tools() -> list[dict]:
    return [_PREAMBLE_STEP_COMPLETE_SCHEMA]


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────


_MAX_PREAMBLE_LEN = 1500


def run_preamble_step(
    user_input: str,
    session_summaries: Optional[list[str]] = None,
    session_id: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Run the pre-planning step and return ``(status, message)``.

    Returns:
        ``(status, message)`` where:
            * ``status`` is ``"done"`` or ``"continue"``
            * ``message`` is the reply text (final_answer for done,
              summary for continue)
        or ``None`` when the call failed (network error, LLM
        returned no parseable tool call, etc). Callers should fall
        through to the normal analyst on None.
    """
    if not user_input or not user_input.strip():
        return None
    if len(user_input) > _MAX_PREAMBLE_LEN:
        return None

    try:
        return _preamble_via_llm(user_input, session_summaries, session_id)
    except Exception as exc:
        logger.debug("Preamble step failed (falling through): %s", exc)
        return None


def try_conversational_fastpath(
    user_input: str,
    session_summaries: Optional[list[str]] = None,
    session_id: Optional[str] = None,
) -> Optional[tuple[AnalysisResult, str, bool]]:
    """Backwards-compatible entry point used by ``pipeline.run_task``.

    Returns ``(analysis, reply, continue_planning)`` so the caller can
    show the reply unconditionally and only short-circuit when
    ``continue_planning=False``. ``None`` falls through.
    """
    decision = run_preamble_step(user_input, session_summaries, session_id)
    if decision is None:
        return None

    status, message = decision
    if not message:
        return None

    continue_planning = status == "continue"
    flow = "develop" if continue_planning else "done"
    result = AnalysisResult(
        action="passthrough",
        original_input=user_input,
        reason=message,
        flow=flow,
    )
    return result, message, continue_planning


# ─────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────


def _preamble_via_llm(
    user_input: str,
    session_summaries: Optional[list[str]],
    session_id: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Make the litellm call and parse the step_complete tool args.

    *session_id* is the preferred way to fetch context: when it is
    provided we read the last few turns at full fidelity from the DB
    via :func:`get_recent_turns_full`. *session_summaries* is the
    legacy fallback (200-char snippets) kept for callers that don't
    have a session_id at hand.
    """
    import litellm

    from infinidev.config.llm import get_litellm_params_for_behavior

    params = get_litellm_params_for_behavior()

    user_lines: list[str] = []

    # Prefer the full-content fetch when we have a session id. The
    # 200-char ``summary`` field that ``get_recent_summaries`` reads
    # is not enough context for the preamble: when a user asks
    # "explain those recommendations", the agent's previous reply
    # IS the recommendations, and truncating it to 200 chars makes
    # the model hallucinate the rest.
    full_turns: list[tuple[str, str]] = []
    if session_id:
        try:
            from infinidev.db.service import get_recent_turns_full
            full_turns = get_recent_turns_full(
                session_id, limit=6, max_chars_per_turn=2000
            )
        except Exception:
            full_turns = []

    if full_turns:
        user_lines.append(
            "RECENT CONVERSATION (full content of the last few turns "
            "— use this to detect self-referential follow-ups):"
        )
        for role, content in full_turns:
            tag = "USER" if role == "user" else "AGENT"
            user_lines.append(f"<turn role=\"{tag}\">")
            user_lines.append(content)
            user_lines.append("</turn>")
        user_lines.append("")
    elif session_summaries:
        # Legacy path — only used by callers that don't supply a
        # session_id (tests, single-prompt mode without history).
        recent = [s for s in session_summaries[-3:] if s]
        if recent:
            user_lines.append("RECENT SESSION CONTEXT (truncated snippets):")
            for s in recent:
                user_lines.append(f"  - {s}")
            user_lines.append("")

    user_lines.append(f"User just wrote: {user_input.strip()}")
    user_lines.append("")
    user_lines.append("Call step_complete now.")

    messages = [
        {"role": "system", "content": _PREAMBLE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_lines)},
    ]

    call_kwargs = dict(params)
    call_kwargs["messages"] = messages
    call_kwargs["tools"] = _get_preamble_tools()
    call_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": "step_complete"},
    }
    call_kwargs["max_tokens"] = 1500
    # temperature=0 is critical: this is a CLASSIFICATION decision,
    # not a creative one. Determinism makes the test reliable and
    # eliminates the 1-in-10 misfire we observed where the model
    # picks continue for a clear greeting.
    call_kwargs["temperature"] = 0.0
    call_kwargs.setdefault("stream", False)
    # Hint thinking models to skip CoT; ignored when the chat
    # template doesn't honour the kwarg.
    extra_body = dict(call_kwargs.get("extra_body") or {})
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs.setdefault("enable_thinking", False)
    chat_template_kwargs.setdefault("thinking", False)
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    call_kwargs["extra_body"] = extra_body

    response = litellm.completion(**call_kwargs)
    args = _extract_step_complete_args(response)
    if args is None:
        return None

    status = (args.get("status") or "").lower().strip()
    if status not in ("done", "continue"):
        # Models occasionally return blocked / explore — treat as
        # fall-through so the analyst handles them properly.
        return None

    if status == "done":
        message = (args.get("final_answer") or args.get("summary") or "").strip()
    else:  # continue
        message = (args.get("summary") or args.get("final_answer") or "").strip()

    if not message:
        return None
    return status, message


# ─────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────


def _extract_step_complete_args(response) -> Optional[dict]:
    """Pull the step_complete tool arguments out of the response.

    Returns the parsed argument dict, or None when the model didn't
    actually call a tool. Tolerant to provider differences in how
    arguments are serialised (string JSON vs dict).
    """
    try:
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            return None
        tc = tool_calls[0]
        fn = getattr(tc, "function", None)
        if fn is None:
            return None
        raw_args = getattr(fn, "arguments", None) or "{}"
        if isinstance(raw_args, dict):
            return raw_args
        try:
            obj = json.loads(raw_args)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    except Exception:
        return None
    return None


__all__ = ["run_preamble_step", "try_conversational_fastpath"]
