"""Load curated opt-in workflow guidance from packaged JSON resources."""

from __future__ import annotations

from html import escape
from importlib.resources import files
import json
import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from infinidev.prompts.profiles import EffectivePromptConfiguration


CAPABILITY_RESOURCE_PACKAGE: Final = "infinidev.prompts.capabilities"
_CAPABILITY_ID_PATTERN: Final = re.compile(r"capability\.[a-z][a-z0-9_]*")
_CAPABILITY_FIELDS: Final = frozenset({"id", "reason", "guidance"})


def _required_string(document: dict[str, object], field: str, source: str) -> str:
    """Return one required, non-empty string from a capability document."""
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"Optional capability {source} field {field!r} must be a non-empty string"
        )
    return value.strip()


def _capability_from_document(document: object, source: str) -> tuple[str, str]:
    """Validate and render one JSON capability document."""
    if not isinstance(document, dict):
        raise RuntimeError(f"Optional capability {source} must contain a JSON object")
    unknown_fields = set(document) - _CAPABILITY_FIELDS
    missing_fields = _CAPABILITY_FIELDS - set(document)
    if unknown_fields or missing_fields:
        details: list[str] = []
        if missing_fields:
            details.append(f"missing {sorted(missing_fields)}")
        if unknown_fields:
            details.append(f"unknown {sorted(unknown_fields)}")
        raise RuntimeError(
            f"Optional capability {source} has invalid fields: {', '.join(details)}"
        )

    name = _required_string(document, "id", source)
    if _CAPABILITY_ID_PATTERN.fullmatch(name) is None:
        raise RuntimeError(
            f"Optional capability {source} field 'id' must match capability.snake_case_name"
        )
    reason = _required_string(document, "reason", source)
    guidance_lines = document.get("guidance")
    if (
        not isinstance(guidance_lines, list)
        or not guidance_lines
        or any(not isinstance(line, str) for line in guidance_lines)
    ):
        raise RuntimeError(
            f"Optional capability {source} field 'guidance' must be a non-empty string array"
        )
    guidance = "\n".join(guidance_lines).strip()
    if not guidance:
        raise RuntimeError(
            f"Optional capability {source} field 'guidance' must contain non-empty text"
        )
    body = f'<if reason="{escape(reason, quote=True)}">\n{guidance}\n</if>'
    return name, body


def _load_optional_capabilities() -> tuple[tuple[str, str], ...]:
    """Load capability resources in filename order and reject ambiguous catalogs."""
    resource_root = files(CAPABILITY_RESOURCE_PACKAGE)
    resources = sorted(
        (resource for resource in resource_root.iterdir() if resource.name.endswith(".json")),
        key=lambda resource: resource.name,
    )
    if not resources:
        raise RuntimeError(
            f"Optional capability package {CAPABILITY_RESOURCE_PACKAGE} contains no JSON files"
        )

    capabilities: list[tuple[str, str]] = []
    names: set[str] = set()
    for resource in resources:
        try:
            document = json.loads(resource.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as err:
            raise RuntimeError(
                f"Could not load optional capability {resource.name}: {err}"
            ) from err
        capability = _capability_from_document(document, resource.name)
        if capability[0] in names:
            raise RuntimeError(f"Duplicate optional capability id: {capability[0]}")
        names.add(capability[0])
        capabilities.append(capability)
    return tuple(capabilities)


# The public tuple remains stable for prompt composition, slash commands, and starter profiles.
OPTIONAL_CAPABILITIES: tuple[tuple[str, str], ...] = _load_optional_capabilities()


def render_optional_capabilities(
    configuration: EffectivePromptConfiguration,
) -> str:
    """Render enabled capabilities while keeping every capability opt-in by default."""
    from infinidev.prompts.profiles import resolve_prompt_fragment

    enabled: list[str] = []
    for name, body in OPTIONAL_CAPABILITIES:
        fragment = resolve_prompt_fragment(
            name,
            "develop",
            body,
            configuration=configuration,
            default_enabled=False,
        )
        if fragment:
            enabled.append(fragment)
    return "\n\n".join(enabled)
