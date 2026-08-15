"""Typed conditional composition and content-free prompt measurements."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from infinidev.engine.task_policies.models import TaskProfile


CACHE_BREAKPOINT_MARKER = "<!--__INFINIDEV_CACHE_BREAK__-->"


@dataclass(frozen=True)
class ConditionalPromptFragment:
    """One bounded instruction selected from a vetted task profile."""

    id: str
    policy_id: str
    content: str
    roles: frozenset[str]
    phases: frozenset[str]
    condition_reason: str = ""
    priority: int = 0
    max_utf8_bytes: int = 900
    requires_operations: frozenset[str] = frozenset()
    requires_constraints: frozenset[str] = frozenset()
    requires_authority: frozenset[str] = frozenset()
    excludes_operations: frozenset[str] = frozenset()
    excludes_constraints: frozenset[str] = frozenset()
    model_routes: frozenset[str] = frozenset()
    excluded_model_routes: frozenset[str] = frozenset()
    version: int = 1

    @property
    def content_hash(self) -> str:
        payload = f"{self.id}\0{self.version}\0{self.content}".encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def conditional_content_hash(self) -> str:
        """Hash the condition together with the instruction it controls."""
        payload = f"{self.id}\0{self.version}\0{self.condition_reason}\0{self.content}".encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ConditionalPromptSelection:
    """Auditable result of selecting and budgeting conditional fragments."""

    fragments: tuple[ConditionalPromptFragment, ...]
    omitted: tuple[tuple[str, str], ...]
    used_utf8_bytes: int


def _clip_utf8(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore").strip()


def _model_route_matches(selector: str, *, provider: str, model: str) -> bool:
    """Match ``provider:model-family`` without coupling fragments to SDK prefixes."""
    selector_provider, separator, selector_model = selector.casefold().partition(":")
    if not separator:
        selector_model = selector_provider
        selector_provider = "*"
    route_provider = provider.casefold().strip()
    route_model = model.casefold().strip().split("/")[-1]
    provider_matches = selector_provider in {"", "*", route_provider}
    model_matches = (
        selector_model in {"", "*"}
        or route_model == selector_model
        or route_model.startswith(f"{selector_model}-")
    )
    return provider_matches and model_matches


def select_conditional_fragments(
    profile: "TaskProfile | None",
    fragments: Iterable[ConditionalPromptFragment],
    *,
    role: str,
    phase: str,
    max_utf8_bytes: int,
    provider: str = "",
    model: str = "",
) -> ConditionalPromptSelection:
    """Select only applicable fragments, then enforce a deterministic budget."""
    if profile is None:
        return ConditionalPromptSelection((), (), 0)
    operations = set(profile.operations)
    constraints = set(profile.constraints)
    authority = set(profile.authority)
    selected_policies = {item.id for item in profile.selected_policies}
    eligible: list[ConditionalPromptFragment] = []
    omitted: list[tuple[str, str]] = []
    for fragment in fragments:
        reason = ""
        if fragment.policy_id not in selected_policies:
            reason = "policy-not-selected"
        elif role not in fragment.roles:
            reason = "role-mismatch"
        elif phase not in fragment.phases:
            reason = "phase-mismatch"
        elif not fragment.requires_operations.issubset(operations):
            reason = "missing-operation"
        elif not fragment.requires_constraints.issubset(constraints):
            reason = "missing-constraint"
        elif not fragment.requires_authority.issubset(authority):
            reason = "missing-authority"
        elif fragment.excludes_operations & operations:
            reason = "excluded-operation"
        elif fragment.excludes_constraints & constraints:
            reason = "excluded-constraint"
        elif fragment.model_routes and not provider and not model:
            reason = "model-route-unavailable"
        elif fragment.model_routes and not any(
            _model_route_matches(selector, provider=provider, model=model)
            for selector in fragment.model_routes
        ):
            reason = "model-route-mismatch"
        elif any(
            _model_route_matches(selector, provider=provider, model=model)
            for selector in fragment.excluded_model_routes
        ):
            reason = "excluded-model-route"
        if reason:
            omitted.append((fragment.id, reason))
        else:
            eligible.append(fragment)

    eligible.sort(key=lambda item: (-item.priority, item.id))
    chosen: list[ConditionalPromptFragment] = []
    used = 0
    for fragment in eligible:
        content = _clip_utf8(fragment.content, fragment.max_utf8_bytes)
        size = len(content.encode("utf-8"))
        if not content:
            omitted.append((fragment.id, "empty"))
            continue
        if used + size > max(0, max_utf8_bytes):
            omitted.append((fragment.id, "budget"))
            continue
        chosen.append(fragment)
        used += size
    return ConditionalPromptSelection(
        fragments=tuple(chosen),
        omitted=tuple(omitted),
        used_utf8_bytes=used,
    )


def select_conditional_catalog(
    fragments: Iterable[ConditionalPromptFragment],
    *,
    role: str,
    phase: str,
    max_utf8_bytes: int,
    provider: str = "",
    model: str = "",
) -> ConditionalPromptSelection:
    """Select the complete role/phase catalog without task-profile gating."""
    eligible: list[ConditionalPromptFragment] = []
    omitted: list[tuple[str, str]] = []
    for fragment in fragments:
        reason = ""
        if role not in fragment.roles:
            reason = "role-mismatch"
        elif phase not in fragment.phases:
            reason = "phase-mismatch"
        elif not fragment.condition_reason.strip():
            reason = "missing-condition-reason"
        elif fragment.model_routes and not provider and not model:
            reason = "model-route-unavailable"
        elif fragment.model_routes and not any(
            _model_route_matches(selector, provider=provider, model=model)
            for selector in fragment.model_routes
        ):
            reason = "model-route-mismatch"
        elif any(
            _model_route_matches(selector, provider=provider, model=model)
            for selector in fragment.excluded_model_routes
        ):
            reason = "excluded-model-route"
        if reason:
            omitted.append((fragment.id, reason))
        else:
            eligible.append(fragment)

    eligible.sort(key=lambda item: (-item.priority, item.id))
    chosen: list[ConditionalPromptFragment] = []
    used = 0
    for fragment in eligible:
        content = _clip_utf8(fragment.content, fragment.max_utf8_bytes)
        if not content:
            omitted.append((fragment.id, "empty"))
            continue
        rendered = (
            f'<if reason="{fragment.condition_reason}">\n'
            f"{content}\n"
            "</if>"
        )
        size = len(rendered.encode("utf-8"))
        if used + size > max(0, max_utf8_bytes):
            omitted.append((fragment.id, "budget"))
            continue
        chosen.append(fragment)
        used += size
    return ConditionalPromptSelection(
        fragments=tuple(chosen),
        omitted=tuple(omitted),
        used_utf8_bytes=used,
    )


def append_dynamic_system_layer(
    stable_prompt: str,
    dynamic_layer: str,
    *,
    cache_boundary: bool,
) -> str:
    """Append task-local guidance without contaminating the cacheable prefix."""
    dynamic_layer = dynamic_layer.strip()
    if not dynamic_layer:
        return stable_prompt
    if cache_boundary and CACHE_BREAKPOINT_MARKER not in stable_prompt:
        return f"{stable_prompt}\n\n{CACHE_BREAKPOINT_MARKER}\n\n{dynamic_layer}"
    return f"{stable_prompt}\n\n{dynamic_layer}"


_TOP_LEVEL_OPEN = re.compile(
    r"(?:\A|\n\n)<([a-z][a-z0-9-]*)(?:\s[^>]*)?>\n",
    re.IGNORECASE,
)
_TASK_FRAGMENT_OPEN = re.compile(
    r'<prompt-fragment id="(?P<id>[^"]+)" version="(?P<version>\d+)" '
    r'sha256="(?P<sha256>[a-f0-9]{64})" policy="(?P<policy>[^"]+)">'
)


def user_section_chars(prompt: str) -> dict[str, int]:
    """Measure top-level XML-like blocks without retaining their contents."""
    counts: dict[str, int] = {}
    covered = 0
    for match in _TOP_LEVEL_OPEN.finditer(prompt):
        tag = match.group(1).lower()
        start = match.start()
        if prompt.startswith("\n\n", start):
            start += 2
        close = re.compile(rf"\n</{re.escape(tag)}>(?=\n\n|\Z)", re.IGNORECASE)
        closing = close.search(prompt, match.end())
        if closing is None:
            continue
        end = closing.end()
        counts[tag] = counts.get(tag, 0) + end - start
        covered += end - start
    counts["unclassified"] = max(0, len(prompt) - covered)
    return dict(sorted(counts.items()))


def measure_prompt_composition(
    system_prompt: str,
    user_prompt: str,
    tool_schemas: list[dict[str, Any]] | None,
    *,
    iteration: int,
) -> dict[str, Any]:
    """Return exact character counts for one request's static components."""
    encoded_tools = json.dumps(
        tool_schemas or [], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    user_sections = user_section_chars(user_prompt)
    from infinidev.engine.prompt_layers import classify_user_section

    layer_chars: dict[str, int] = {}
    for tag, count in user_sections.items():
        if tag == "unclassified":
            continue
        layer = classify_user_section(tag).value
        layer_chars[layer] = layer_chars.get(layer, 0) + count
    stable_system, boundary, dynamic_system = system_prompt.partition(
        CACHE_BREAKPOINT_MARKER
    )
    if not boundary:
        stable_system = system_prompt
        dynamic_system = ""
    conditional_fragments = [
        {
            "id": match.group("id"),
            "version": int(match.group("version")),
            "sha256": match.group("sha256"),
            "policy": match.group("policy"),
        }
        for match in _TASK_FRAGMENT_OPEN.finditer(system_prompt)
    ]
    return {
        "iteration": iteration,
        "system_chars": len(system_prompt),
        "stable_system_chars": len(stable_system),
        "dynamic_system_chars": len(dynamic_system),
        "conditional_fragment_ids": [
            fragment["id"] for fragment in conditional_fragments
        ],
        "conditional_fragments": conditional_fragments,
        "user_chars": len(user_prompt),
        "tool_schema_chars": len(encoded_tools),
        "request_static_chars": len(system_prompt) + len(user_prompt) + len(encoded_tools),
        "user_sections": user_sections,
        "user_layer_chars": dict(sorted(layer_chars.items())),
    }


def measure_request_payload(
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]] | None,
    *,
    mode: str,
    sequence: int,
) -> dict[str, Any]:
    """Measure the complete message transcript immediately before dispatch."""
    encoded_messages = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    encoded_tools = json.dumps(
        tool_schemas or [], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    role_chars: dict[str, int] = {}
    for message in messages:
        role = str(message.get("role", "unknown"))
        content = json.dumps(
            message.get("content"), ensure_ascii=False, separators=(",", ":"), default=str
        )
        role_chars[role] = role_chars.get(role, 0) + len(content)
    return {
        "sequence": sequence,
        "mode": mode,
        "message_count": len(messages),
        "message_payload_chars": len(encoded_messages),
        "tool_schema_chars": len(encoded_tools),
        "request_payload_chars": len(encoded_messages) + len(encoded_tools),
        "message_content_chars_by_role": dict(sorted(role_chars.items())),
    }
