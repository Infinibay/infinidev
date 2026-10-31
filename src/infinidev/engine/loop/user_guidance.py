"""Literal user guidance retained independently of derived Step summaries."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from infinidev.engine.loop.loop_state import LoopState
    from infinidev.engine.loop.user_message_injector import InjectedUserMessage


class UserGuidance(BaseModel):
    """One user message and image references, without image payloads."""

    text: str
    attachment_paths: tuple[str, ...] = ()
    message_id: str = ""


def retain_user_guidance(
    state: LoopState | None, messages: list[InjectedUserMessage],
) -> None:
    """Retain each drained message exactly once and in reception order."""
    if state is None:
        return
    retained_ids = {item.message_id for item in state.user_guidance if item.message_id}
    for message in messages:
        if message.message_id in retained_ids:
            continue
        state.user_guidance.append(UserGuidance(
            text=message.text,
            attachment_paths=tuple(str(attachment.path) for attachment in message.attachments),
            message_id=message.message_id,
        ))
        retained_ids.add(message.message_id)


def render_user_guidance(guidance: list[UserGuidance]) -> str:
    """Project literal instructions separately from summaries and observations."""
    if not guidance:
        return ""
    parts = [
        '<user-guidance-history authority="USER_LITERAL" scope="current-task">',
        "These user messages were received during this Task. Keep applying "
        "them when rebuilding context; later messages may revise earlier instructions. "
        "Plans and summaries do not replace their literal wording. Do not acknowledge "
        "them again merely because this history is repeated.",
    ]
    for index, message in enumerate(guidance, 1):
        parts.append(f'<message sequence="{index}"><text>{escape(message.text)}</text>')
        if message.attachment_paths:
            parts.append(
                '<attachment-references authority="runtime">'
                "Image references only: pixel payloads are not retained in this history. "
                "Consult the original attachment when accessible; otherwise request "
                "it again before relying on visual details.\n"
                + "\n".join(escape(path) for path in message.attachment_paths)
                + "</attachment-references>"
            )
        parts.append("</message>")
    parts.append("</user-guidance-history>")
    return "\n".join(parts)
