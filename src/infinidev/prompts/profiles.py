"""JSON prompt-profile loading and resolution.

Profiles live in ``~/.infinidev/prompts/*.json`` with an optional project-local
``.infinidev/prompts.json`` override. They only affect registered prompt fragments;
an empty catalog preserves the built-in prompt composition.
"""

from __future__ import annotations

import errno
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, TypeAlias

from infinidev.config.settings import get_base_dir, settings
from infinidev.prompts.catalog import STARTER_PROMPT_PROFILES, materialize_starter_prompt_profiles

logger = logging.getLogger(__name__)

Scalar: TypeAlias = str | int | float


@dataclass(frozen=True)
class PromptProfile:
    """Resolved setting for one named prompt fragment."""

    enabled: bool = True
    enabled_by_default: bool = True
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

    def resolve(
        self, phase: str, name: str, *, default_enabled: bool = True,
    ) -> PromptProfile:
        """Return one compiled setting, falling back to the fragment's built-in state."""
        return self.profiles.get(phase, {}).get(
            name,
            PromptProfile(
                enabled=default_enabled,
                enabled_by_default=default_enabled,
            ),
        )


class PromptProfileError(ValueError):
    """Raised when a recognized prompt-profile value has an invalid type."""


def get_prompt_profile_path() -> Path:
    """Return the legacy project-local prompt-profile path."""
    return get_base_dir() / "prompts.json"


def get_prompt_catalog_path() -> Path:
    """Return the user-level directory containing shared prompt profiles."""
    return Path.home() / ".infinidev" / "prompts"


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
            result[name] = PromptProfile(enabled=value, enabled_by_default=value)
        elif isinstance(value, dict) and _is_structured_profile(value):
            enabled_by_default = value.get("enabled_by_default", True)
            enabled = value.get("enabled", enabled_by_default)
            parameters = value.get("parameters")
            if not isinstance(enabled_by_default, bool) or not isinstance(enabled, bool):
                raise PromptProfileError(
                    f"Prompt setting {section}.{name!r} must use boolean enabled fields"
                )
            if parameters is not None and not _valid_parameters(parameters):
                raise PromptProfileError(
                    f"Prompt setting {section}.{name!r} has invalid parameters"
                )
            unknown = set(value) - {"enabled", "enabled_by_default", "parameters"}
            if unknown:
                raise PromptProfileError(
                    f"Prompt setting {section}.{name!r} has unknown fields: "
                    f"{', '.join(sorted(unknown))}"
                )
            result[name] = PromptProfile(
                enabled=enabled,
                enabled_by_default=enabled_by_default,
                parameters=(
                    MappingProxyType(dict(parameters)) if parameters is not None else None
                ),
            )
        elif _valid_parameters(value):
            result[name] = PromptProfile(parameters=MappingProxyType(dict(value)))
        else:
            raise PromptProfileError(
                f"Prompt setting {section}.{name!r} must be a boolean or object "
                "(scalar parameters or a structured profile)"
            )
    return result


def _is_structured_profile(value: Mapping[object, object]) -> bool:
    """Return whether an object uses any explicit profile-control field."""
    return bool({"enabled", "enabled_by_default", "parameters"} & set(value))


def _valid_parameters(value: object) -> bool:
    """Return whether ``value`` is a flat map of scalar prompt parameters."""
    return isinstance(value, dict) and all(
        isinstance(parameter, str)
        and isinstance(parameter_value, (str, int, float))
        and not isinstance(parameter_value, bool)
        for parameter, parameter_value in value.items()
    )


def _load_prompt_profile(path: Path) -> dict[str, object]:
    """Read one profile document and attribute validation errors to its source."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise PromptProfileError(f"Invalid JSON in {path}: {err.msg}") from err
    if not isinstance(data, dict):
        raise PromptProfileError(f"Prompt profile root in {path} must be an object")
    return data


def _merge_scope(target: dict[str, object], source: Mapping[str, object]) -> None:
    """Merge phase maps at fragment granularity, with ``source`` winning."""
    for phase, entries in source.items():
        existing = target.get(phase)
        if isinstance(existing, dict) and isinstance(entries, dict):
            existing.update(entries)
        else:
            target[phase] = entries


def _merge_document(target: dict[str, object], source: Mapping[str, object]) -> None:
    """Merge one catalog document without replacing unrelated phase entries."""
    for key, value in source.items():
        if key != "models":
            _merge_scope(target, {key: value})
            continue
        existing_models = target.setdefault("models", {})
        if not isinstance(existing_models, dict) or not isinstance(value, dict):
            target["models"] = value
            continue
        for model, scope in value.items():
            existing_scope = existing_models.get(model)
            if isinstance(existing_scope, dict) and isinstance(scope, dict):
                _merge_scope(existing_scope, scope)
            else:
                existing_models[model] = scope


def load_prompt_profiles(path: Path | None = None) -> dict[str, object]:
    """Load one explicit profile or the shared catalog plus project override.

    Catalog files are merged in filename order. The legacy project-local profile is
    applied last so existing repositories keep their previous, most-local behavior.
    """
    if path is not None:
        return _load_prompt_profile(path) if path.exists() else {}

    catalog_path = get_prompt_catalog_path()
    packaged: dict[str, str] = {}
    try:
        materialize_starter_prompt_profiles(catalog_path)
    except OSError as err:
        if err.errno not in {errno.EACCES, errno.EPERM, errno.EROFS}:
            raise
        logger.warning("Cannot publish prompt starters in %s: %s", catalog_path, err)
        packaged = dict(STARTER_PROMPT_PROFILES)

    # Read-only installations use the same defaults and filename precedence
    # as a writable catalog; existing user files still replace their starters.
    profile_paths = {path.name: path for path in catalog_path.glob("*.json")}
    merged: dict[str, object] = {}
    for filename in sorted(packaged.keys() | profile_paths.keys()):
        document = (
            _load_prompt_profile(profile_paths[filename])
            if filename in profile_paths
            else json.loads(packaged[filename])
        )
        _merge_document(merged, document)

    project_path = get_prompt_profile_path()
    if project_path.exists():
        _merge_document(merged, _load_prompt_profile(project_path))
    return merged


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
    default_enabled: bool = True,
) -> str | None:
    """Return a configured variant or its built-in default fragment."""
    profile = (
        configuration.resolve(phase, name, default_enabled=default_enabled)
        if configuration is not None
        else EffectivePromptConfiguration.compile().resolve(
            phase,
            name,
            default_enabled=default_enabled,
        )
    )
    return apply_prompt_profile(variant or default, phase, name, profile=profile)
