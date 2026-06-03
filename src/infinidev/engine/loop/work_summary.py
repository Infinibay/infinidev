"""End-of-task work summary — the developer loop's hand-off to the chat.

The loop engine runs in a plan-execute-summarize cycle whose context is
*discarded* the moment the task ends: raw tool output, opened-file
caches, and the per-step ``ActionRecord`` history all live in
``LoopState`` and vanish when ``execute()`` returns. The only thing the
next user turn normally sees is the short ``final_answer`` shown in the
chat — so the chat agent has to start cold, re-reading files it just
edited to answer a simple follow-up.

This module closes that gap. After a task finishes it distils the loop's
own accumulated state into a compact, durable summary that the pipeline
stores as a **hidden conversation turn** (``role="work_summary"``). The
user never sees it (it is filtered out of the UI repaint), but the next
chat-agent turn reads it back as context — so "why did you change X?" or
"continue where you left off" works without re-investigation.

The summary deliberately captures, per the product requirement:
  * which files were modified and *why*;
  * what was done in each file (prose, not code — code only when it is
    genuinely load-bearing);
  * challenges / problems hit along the way (so they are not repeated);
  * anything else relevant enough to remember.

Synthesis uses the single configured model when enabled; a deterministic
assembly from the same source data is the always-available fallback, so
a slow or unavailable model never costs us the summary.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infinidev.engine.file_change_tracker import FileChangeTracker
    from infinidev.engine.loop.loop_state import LoopState

logger = logging.getLogger(__name__)

# Stable header so the chat agent (and a human reading the DB) can
# recognise these turns. Kept in one place — the renderer in chat_agent
# and any future consumer should reference this rather than re-typing it.
WORK_SUMMARY_ROLE = "work_summary"

_SYNTHESIS_SYSTEM_PROMPT = """\
You write a short hand-off note for a coding assistant that just finished \
a task. The note is stored privately and read by the assistant at the \
START of the next turn so it does not have to re-investigate work it just \
did. The user never sees it.

Write plain prose (markdown bullets allowed). Be concrete and honest — \
report what actually happened, including failures and unfinished parts. \
Do NOT paste code unless a specific snippet is genuinely essential to \
understand a change. Do NOT pad. Cover, in this order and only if it \
applies:

1. **Files changed & why** — each modified/created/deleted file and the \
reason it was touched.
2. **What was done in each file** — a one-line prose summary of the change \
per file (not the code).
3. **Challenges / problems** — anything that went wrong, was tricky, was \
worked around, or remains broken. These matter most: they stop the next \
turn from repeating mistakes.
4. **Other notes worth remembering** — decisions, follow-ups, things the \
user should be reminded of.

Aim for 120-250 words. If almost nothing happened, say so in one line."""


def build_work_summary(
    state: "LoopState | None",
    file_tracker: "FileChangeTracker | None",
    *,
    final_answer: str,
    status: str,
) -> str | None:
    """Build the hidden end-of-task work summary, or ``None`` to skip.

    Returns ``None`` when there is nothing worth recording (no file
    changes and no substantive step history) — e.g. a read-only or
    immediately-aborted task — so we never store empty hand-off turns.

    Honours ``LOOP_WORK_SUMMARY_ENABLED`` (off → always ``None``) and
    ``LOOP_WORK_SUMMARY_USE_LLM`` (off → deterministic assembly only).
    """
    from infinidev.config import settings

    if not getattr(settings, "LOOP_WORK_SUMMARY_ENABLED", True):
        return None

    facts = _collect_facts(state, file_tracker)
    # Nothing changed and no meaningful work recorded — not worth a turn.
    if not facts["files"] and not facts["steps"]:
        return None

    deterministic = _render_deterministic(facts, status=status)

    if getattr(settings, "LOOP_WORK_SUMMARY_USE_LLM", True):
        synthesized = _synthesize_with_llm(
            facts, final_answer=final_answer, status=status,
        )
        if synthesized:
            return _wrap(synthesized)

    return _wrap(deterministic)


# ── Fact collection ─────────────────────────────────────────────────────

def _collect_facts(
    state: "LoopState | None",
    file_tracker: "FileChangeTracker | None",
) -> dict[str, Any]:
    """Pull the raw materials out of the loop state and file tracker.

    Keeps this side effect-free and serialisable so both the LLM prompt
    and the deterministic renderer consume the same structured data.
    """
    files: list[dict[str, Any]] = []
    if file_tracker is not None:
        for path in file_tracker.get_all_paths():
            files.append({
                "path": path,
                "action": file_tracker.get_action(path),
                "reasons": file_tracker.get_reasons(path) or [],
            })

    steps: list[dict[str, Any]] = []
    if state is not None:
        for rec in state.history:
            steps.append({
                "summary": (rec.summary or "").strip(),
                "changes": (rec.changes_made or "").strip(),
                "discovered": (rec.discovered_context or "").strip(),
                "pending": (rec.pending_items or "").strip(),
                "anti_patterns": (rec.anti_patterns or "").strip(),
            })

    notes = list(state.notes) if state is not None else []

    return {"files": files, "steps": steps, "notes": notes}


# ── Deterministic rendering (fallback + LLM input) ──────────────────────

def _render_deterministic(facts: dict[str, Any], *, status: str) -> str:
    """Assemble a structured summary directly from the collected facts.

    Always available (no model needed). Doubles as the structured
    context block fed to the LLM synthesiser.
    """
    lines: list[str] = [f"Task ended with status: {status}.", ""]

    if facts["files"]:
        lines.append("Files changed:")
        for f in facts["files"]:
            reasons = "; ".join(f["reasons"]) if f["reasons"] else "no reason recorded"
            lines.append(f"- {f['path']} ({f['action']}) — {reasons}")
        lines.append("")

    # Per-step narrative: what was done, and any trouble.
    challenges: list[str] = []
    did: list[str] = []
    for i, s in enumerate(facts["steps"], 1):
        body = s["changes"] or s["summary"]
        if body:
            did.append(f"- Step {i}: {body}")
        for trouble in (s["anti_patterns"], s["pending"]):
            if trouble:
                challenges.append(f"- {trouble}")

    if did:
        lines.append("What was done:")
        lines.extend(did)
        lines.append("")

    if challenges:
        lines.append("Challenges / problems encountered:")
        lines.extend(challenges)
        lines.append("")

    if facts["notes"]:
        lines.append("Other notes:")
        lines.extend(f"- {n}" for n in facts["notes"][:10])
        lines.append("")

    return "\n".join(lines).strip()


# ── LLM synthesis ───────────────────────────────────────────────────────

def _synthesize_with_llm(
    facts: dict[str, Any], *, final_answer: str, status: str,
) -> str | None:
    """Ask the configured model to distil the facts into a clean note.

    Returns ``None`` on any failure so the caller falls back to the
    deterministic render — generating the summary must never raise into
    the task's finish path.
    """
    try:
        import litellm
        from infinidev.config.llm import get_litellm_params

        structured = _render_deterministic(facts, status=status)
        user_content = (
            "Here is the raw record of the task I just finished. Write the "
            "hand-off note.\n\n"
            f"Final answer shown to the user:\n{final_answer.strip() or '(none)'}\n\n"
            f"Raw record:\n{structured}"
        )

        params = get_litellm_params()
        params["messages"] = [
            {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        params.setdefault("temperature", 0.2)
        params["stream"] = False
        params["max_tokens"] = 600

        response = litellm.completion(**params)
        text = (response.choices[0].message.content or "").strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 — must never break finish()
        logger.warning("work summary LLM synthesis failed, using fallback: %s", exc)
        return None


# ── Framing ─────────────────────────────────────────────────────────────

def _wrap(body: str) -> str:
    """Frame the body so the reader knows it is an internal work record.

    The tags make it unmistakable to the chat agent that this is a
    factual log of completed work to build on — not a user message to
    answer and not text to echo back verbatim.
    """
    return (
        "<work-summary>\n"
        "Internal record of work completed in the previous task "
        "(not shown to the user). Use it for continuity; reground "
        "specifics with tools before relying on them.\n\n"
        f"{body}\n"
        "</work-summary>"
    )
