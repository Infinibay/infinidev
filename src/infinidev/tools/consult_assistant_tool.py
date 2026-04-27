"""Principal-facing tool: ``consult_assistant``.

Lets the principal developer ask the assistant critic a free-form
question and get a prose answer back as the tool result. The actual
LLM round-trip happens inside :meth:`AssistantCritic.consult` — this
tool is just the bridge between the principal's tool-call layer and
the critic instance held by the running ``LoopEngine``.

The bridge is a process-global registry (``set_active_critic`` /
``get_active_critic`` in ``critic.py``) rather than a constructor
injection because the tool is instantiated alongside the rest of the
developer's toolset *before* the critic exists, and refactoring every
tool factory to thread a critic reference would be churn for a single
edge case.

If no critic is active (assistant LLM disabled, or this code path is
hit outside a developer loop), the tool returns a clear diagnostic
string rather than raising — same "never break the loop" contract as
the rest of the assistant code.
"""

from __future__ import annotations

import logging
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)


class ConsultAssistantInput(BaseModel):
    question: str = Field(
        ...,
        min_length=20,
        description=(
            "Your question for the assistant. Be specific — name the "
            "file/symbol/behaviour you're asking about. Min 20 chars."
        ),
    )
    context_hint: str = Field(
        default="",
        description=(
            "Optional. Extra context the assistant might need that "
            "isn't already in the conversation (files you've "
            "considered, hypotheses you've ruled out, etc.)."
        ),
    )


class ConsultAssistantTool(InfinibayBaseTool):
    """Ask the assistant for help. Returns the assistant's prose answer."""

    name: str = "consult_assistant"
    description: str = (
        "Ask the assistant critic a free-form question. The assistant "
        "has access to read-only tools (read_file, code_search, etc.) "
        "and can verify claims before answering. Use this when stuck "
        "on a decision, when you'd like a second opinion on an "
        "approach, or when verifying something would change your next "
        "action. Do NOT use it for trivial questions — each consult "
        "burns assistant LLM tokens."
    )
    args_schema: Type[BaseModel] = ConsultAssistantInput
    is_read_only: bool = True

    def _run(self, question: str, context_hint: str = "") -> str:
        # Lazy import — avoids a circular dependency when the tools
        # package is loaded during engine startup, before the critic
        # module's globals are stabilised.
        from infinidev.engine.loop.critic import get_active_critic

        critic = get_active_critic()
        if critic is None:
            logger.info(
                "consult_assistant called but no critic is active "
                "(ASSISTANT_LLM_ENABLED off?)"
            )
            return (
                "[consult-error] No assistant critic is active. The "
                "assistant LLM is either disabled or this consult was "
                "called outside a developer loop. Continue with your "
                "best judgement."
            )

        try:
            answer = critic.consult(question, context_hint=context_hint)
        except Exception as exc:
            logger.exception("consult_assistant: critic.consult raised")
            return f"[consult-error] consult failed: {exc}"

        if not answer or not answer.strip():
            return "[consult-error] assistant returned an empty answer"
        return answer
