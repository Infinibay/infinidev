"""Prompt composition for selected task policies, roles, and phases."""

from __future__ import annotations

from html import escape

from infinidev.engine.prompt_composition import (
    ConditionalPromptSelection,
    append_dynamic_system_layer,
    select_conditional_catalog,
    select_conditional_fragments,
)
from infinidev.engine.prompt_layers import PromptLayer, PromptLayerKind
from infinidev.engine.task_policies.models import TaskProfile
from infinidev.engine.task_policies.fragments import TASK_METHOD_FRAGMENTS
from infinidev.engine.task_policies.registry import POLICY_BY_ID


_CONDITIONAL_POLICY_PROTOCOL = (
    "Each <if> block below is conditional guidance, not a claim that its reason is true. "
    "Apply a block only when its reason is supported by the literal user request and the "
    "current task. Ignore false or uncertain blocks. A block never grants permission, "
    "expands scope, or overrides higher-priority instructions."
)


def select_task_policy_fragments(
    profile: TaskProfile | None,
    *,
    role: str,
    phase: str,
    max_utf8_bytes: int = 3600,
    provider: str = "",
    model: str = "",
) -> ConditionalPromptSelection:
    """Return the exact role/phase fragments that fit the prompt budget."""
    return select_conditional_fragments(
        profile,
        TASK_METHOD_FRAGMENTS,
        role=role,
        phase=phase,
        max_utf8_bytes=max_utf8_bytes,
        provider=provider,
        model=model,
    )


def render_all_conditional_task_policy_layer(
    *,
    role: str,
    phase: str,
    max_utf8_bytes: int = 12000,
    force: bool = False,
) -> str:
    """Render the complete role/phase catalog as model-evaluated conditions."""
    from infinidev.config.settings import settings

    if settings.TASK_POLICIES_SHADOW_MODE and not force:
        return ""
    composition = select_conditional_catalog(
        TASK_METHOD_FRAGMENTS,
        role=role,
        phase=phase,
        max_utf8_bytes=max_utf8_bytes,
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
    )
    fragments: list[str] = []
    for fragment in composition.fragments:
        policy = POLICY_BY_ID.get(fragment.policy_id)
        if policy is None:
            continue
        content = fragment.content.encode("utf-8")[:fragment.max_utf8_bytes].decode(
            "utf-8", errors="ignore"
        )
        reason = escape(fragment.condition_reason, quote=True)
        fragments.append(
            f'<prompt-fragment id="{fragment.id}" version="{fragment.version}" '
            f'sha256="{fragment.conditional_content_hash}" '
            f'policy="{policy.id}@{policy.version}">\n'
            f'<if reason="{reason}">\n'
            f"{content}\n"
            "</if>\n"
            "</prompt-fragment>"
        )
    if not fragments:
        return ""
    protocol = (
        "<conditional-policy-protocol>\n"
        f"{_CONDITIONAL_POLICY_PROTOCOL}\n"
        "</conditional-policy-protocol>"
    )
    return PromptLayer(
        kind=PromptLayerKind.TASK_POLICY,
        content=protocol + "\n\n" + "\n\n".join(fragments),
        provenance=f"all-conditional-v1:{role}:{phase}",
    ).render()


def render_task_policy_layer(
    profile: TaskProfile | None,
    *,
    role: str,
    phase: str,
    max_utf8_bytes: int = 3600,
    force: bool = False,
) -> str:
    """Render only policies relevant to role and phase within a strict budget."""
    from infinidev.config.settings import settings

    if profile is None or (settings.TASK_POLICIES_SHADOW_MODE and not force):
        return ""
    composition = select_task_policy_fragments(
        profile,
        role=role,
        phase=phase,
        max_utf8_bytes=max_utf8_bytes,
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
    )
    versions = {
        selection.id: selection.version for selection in profile.selected_policies
    }
    fragments: list[str] = []
    for fragment in composition.fragments:
        if settings.TASK_POLICIES_EVIDENCE_GATED and not force:
            from infinidev.engine.task_policies.rollout import fragment_is_approved

            if not fragment_is_approved(
                fragment,
                provider=settings.LLM_PROVIDER,
                model=settings.LLM_MODEL,
            ):
                continue
        policy = POLICY_BY_ID.get(fragment.policy_id)
        if policy is None:
            continue
        content = fragment.content.encode("utf-8")[:fragment.max_utf8_bytes].decode(
            "utf-8", errors="ignore"
        )
        version = versions.get(policy.id, policy.version)
        fragments.append(
            f'<prompt-fragment id="{fragment.id}" version="{fragment.version}" '
            f'sha256="{fragment.content_hash}" policy="{policy.id}@{version}">\n'
            f"{content}\n"
            "</prompt-fragment>"
        )
    if not fragments:
        return ""
    return PromptLayer(
        kind=PromptLayerKind.TASK_POLICY,
        content="\n\n".join(fragments),
        provenance=f"task-profile-v{profile.version}:{role}:{phase}",
    ).render()


def compose_task_aware_system_prompt(
    stable_prompt: str,
    profile: TaskProfile | None,
    *,
    role: str,
    phase: str,
    max_utf8_bytes: int = 3600,
    force: bool = False,
    cache_boundary: bool = False,
) -> str:
    """Append task-local instructions after the stable role core."""
    from infinidev.config.settings import settings

    if settings.TASK_POLICIES_RENDER_ALL_CONDITIONAL:
        layer = render_all_conditional_task_policy_layer(
            role=role,
            phase=phase,
            max_utf8_bytes=max_utf8_bytes,
            force=force,
        )
    else:
        layer = render_task_policy_layer(
            profile,
            role=role,
            phase=phase,
            max_utf8_bytes=max_utf8_bytes,
            force=force,
        )
    return append_dynamic_system_layer(
        stable_prompt,
        layer,
        cache_boundary=cache_boundary,
    )


def render_task_profile_summary(profile: TaskProfile | None) -> str:
    """Render compact non-authoritative labels for shared task context."""
    from infinidev.config.settings import settings

    if profile is None or settings.TASK_POLICIES_SHADOW_MODE:
        return ""

    def line(name: str, values: tuple[str, ...]) -> str:
        return f"  <{name}>{', '.join(values)}</{name}>" if values else ""

    lines = [f'<task-profile version="{profile.version}" authority-source="user-literal">']
    for name, values in (
        ("operations", profile.operations),
        ("authority", profile.authority),
        ("constraints", profile.constraints),
        ("risks", profile.risks),
        ("result", profile.result),
        ("sequence", profile.sequence),
    ):
        if rendered := line(name, values):
            lines.append(rendered)
    lines.append("</task-profile>")
    return "\n".join(lines)
