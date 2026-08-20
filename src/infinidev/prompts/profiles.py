"""JSON prompt-profile loading and resolution.

Profiles live at ``.infinidev/prompts.json`` and only override registered prompt
fragments. A missing profile preserves the built-in prompt composition.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, TypeAlias

from infinidev.config.settings import get_base_dir, settings

logger = logging.getLogger(__name__)

Scalar: TypeAlias = str | int | float


@dataclass(frozen=True)
class PromptProfile:
    """Resolved setting for one named prompt fragment."""

    enabled: bool = True
    parameters: Mapping[str, Scalar] | None = None


@dataclass(frozen=True)
class EffectivePromptConfiguration:
    """Immutable prompt-profile snapshot for one engine execution."""

    profiles: Mapping[str, Mapping[str, PromptProfile]]

    @classmethod
    def compile(cls, path: Path | None = None) -> "EffectivePromptConfiguration":
        """Read, validate, and resolve the active model's overrides once."""
        document = load_prompt_profiles(path)
        models = document.get("models", {})
        if models is not None and not isinstance(models, dict):
            raise PromptProfileError("Prompt profile 'models' must be an object")
        models = models or {}

        provider = settings.LLM_PROVIDER
        model = settings.LLM_MODEL
        scopes: list[tuple[str, Mapping[str, object]]] = [("general", document)]
        for key in (provider, f"{provider}/{model}"):
            scope = models.get(key, {})
            if not isinstance(scope, dict):
                raise PromptProfileError(f"Prompt profile model {key!r} must be an object")
            scopes.append((key, scope))

        phase_names = {
            phase
            for _scope_name, scope in scopes
            for phase in scope
            if phase != "models" and isinstance(phase, str)
        }
        effective: dict[str, Mapping[str, PromptProfile]] = {}
        for phase in phase_names:
            merged: dict[str, PromptProfile] = {}
            for _scope_name, scope in scopes:
                merged.update(_parse_entries(scope.get(phase, {}), phase))
            effective[phase] = MappingProxyType(merged)
        return cls(MappingProxyType(effective))

    def resolve(self, phase: str, name: str) -> PromptProfile:
        """Return one compiled setting, defaulting to an enabled fragment."""
        return self.profiles.get(phase, {}).get(name, PromptProfile())


class PromptProfileError(ValueError):
    """Raised when a recognized prompt-profile value has an invalid type."""


def get_prompt_profile_path() -> Path:
    """Return the project-local prompt-profile path."""
    return get_base_dir() / "prompts.json"


def _parse_entries(entries: object, section: str) -> dict[str, PromptProfile]:
    """Validate one phase section and return its named settings."""
    if not isinstance(entries, dict):
        logger.warning("Ignoring unknown prompt-profile section %r", section)
        return {}

    result: dict[str, PromptProfile] = {}
    for name, value in entries.items():
        if not isinstance(name, str):
            logger.warning("Ignoring non-string prompt-profile name in %s", section)
            continue
        if isinstance(value, bool):
            result[name] = PromptProfile(enabled=value)
        elif isinstance(value, dict) and all(
            isinstance(parameter, str)
            and isinstance(parameter_value, (str, int, float))
            and not isinstance(parameter_value, bool)
            for parameter, parameter_value in value.items()
        ):
            result[name] = PromptProfile(parameters=MappingProxyType(dict(value)))
        else:
            raise PromptProfileError(
                f"Prompt setting {section}.{name!r} must be a boolean or object of string/number values"
            )
    return result


def load_prompt_profiles(path: Path | None = None) -> dict[str, object]:
    """Load the JSON document, returning no overrides when it is absent."""
    profile_path = path or get_prompt_profile_path()
    if not profile_path.exists():
        return {}
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise PromptProfileError(f"Invalid JSON in {profile_path}: {err.msg}") from err
    if not isinstance(data, dict):
        raise PromptProfileError("Prompt profile root must be an object")
    return data


def resolve_prompt_profile(
    phase: str, name: str, path: Path | None = None,
) -> PromptProfile:
    """Resolve exact model, provider, then general phase profile settings.

    ``models`` maps either ``provider/model`` or ``provider`` to the same
    phase sections as the top level. This lets profile files be shared while
    preserving a clear exact-model override.
    """
    return EffectivePromptConfiguration.compile(path).resolve(phase, name)


def apply_prompt_profile(
    prompt: str | None, phase: str, name: str, *, profile: PromptProfile | None = None,
) -> str | None:
    """Disable a fragment or append its declared scalar parameters."""
    if prompt is None:
        return None
    resolved = profile or resolve_prompt_profile(phase, name)
    if not resolved.enabled:
        return None
    if not resolved.parameters:
        return prompt
    parameters = json.dumps(dict(resolved.parameters), ensure_ascii=False, sort_keys=True)
    return f"{prompt}\n\n<prompt-profile name=\"{name}\">{parameters}</prompt-profile>"


def resolve_prompt_fragment(
    name: str,
    phase: str,
    default: str,
    variant: str | None = None,
    *,
    configuration: EffectivePromptConfiguration | None = None,
) -> str | None:
    """Return a configured variant or its built-in default fragment."""
    profile = (
        configuration.resolve(phase, name)
        if configuration is not None
        else resolve_prompt_profile(phase, name)
    )
    return apply_prompt_profile(variant or default, phase, name, profile=profile)
