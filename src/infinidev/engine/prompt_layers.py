"""Typed boundaries for behavior, execution, objective, and evidence prompts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PromptLayerKind(StrEnum):
    """The four prompt responsibilities that must not be calibrated together."""

    BEHAVIOR = "behavior"
    EXECUTION_POLICY = "execution-policy"
    OBJECTIVE = "objective"
    CONTEXT_EVIDENCE = "context-evidence"


_TAGS = {
    PromptLayerKind.BEHAVIOR: "behavior-layer",
    PromptLayerKind.EXECUTION_POLICY: "execution-policy-layer",
    PromptLayerKind.OBJECTIVE: "objective-layer",
    PromptLayerKind.CONTEXT_EVIDENCE: "context-evidence-layer",
}


@dataclass(frozen=True)
class PromptLayer:
    """One typed prompt fragment with explicit provenance."""

    kind: PromptLayerKind
    content: str
    provenance: str

    def render(self) -> str:
        content = self.content.strip()
        if not content:
            return ""
        tag = _TAGS[self.kind]
        return f'<{tag} provenance="{self.provenance}">\n{content}\n</{tag}>'


def compose_layers(layers: list[PromptLayer]) -> str:
    """Render non-empty layers in caller-defined precedence order."""
    return "\n\n".join(rendered for layer in layers if (rendered := layer.render()))


def append_to_layer(
    prompt: str,
    kind: PromptLayerKind,
    content: str,
    *,
    provenance: str,
) -> str:
    """Insert into an existing typed layer, or append a new isolated layer."""
    content = content.strip()
    if not content:
        return prompt
    tag = _TAGS[kind]
    closing = f"</{tag}>"
    position = prompt.find(closing)
    if position >= 0:
        addition = f'\n\n<prompt-fragment provenance="{provenance}">\n{content}\n</prompt-fragment>\n'
        return prompt[:position] + addition + prompt[position:]
    layer = PromptLayer(kind, content, provenance).render()
    return f"{prompt}\n\n{layer}" if prompt else layer


_OBJECTIVE_TAGS = frozenset({"task", "expected-output", "urgent-user-message"})
_EXECUTION_TAGS = frozenset(
    {
        "plan",
        "plan-overview",
        "current-action",
        "next-actions",
        "verification-method",
        "file-integrity-warning",
        "avoid",
        "behavior-summary",
    }
)


def classify_user_section(tag: str) -> PromptLayerKind:
    """Map existing iteration-prompt blocks onto the typed layer contract."""
    normalized = tag.strip().lower()
    if normalized in _OBJECTIVE_TAGS:
        return PromptLayerKind.OBJECTIVE
    if normalized in _EXECUTION_TAGS:
        return PromptLayerKind.EXECUTION_POLICY
    return PromptLayerKind.CONTEXT_EVIDENCE
