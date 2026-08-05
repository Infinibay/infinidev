"""Load explicit hash-bound user preferences for legitimate runtime trade-offs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


logger = logging.getLogger(__name__)
_MAX_PROFILE_BYTES = 16 * 1024
_UTILITY_AXES = frozenset(
    {
        "autonomy",
        "interaction",
        "user_control",
        "speed",
        "quality",
        "cost_efficiency",
        "caution",
    }
)


@dataclass(frozen=True)
class ActiveUserPreferences:
    """Explicit user-authored objective; numeric weights stay out of prompts."""

    name: str
    description: str
    weights: dict[str, float]
    sha256: str


def load_active_user_preferences(
    path: Path, *, expected_sha256: str
) -> ActiveUserPreferences | None:
    """Validate an explicit profile and fail closed when identity drifts."""
    try:
        size = path.stat().st_size
        if size <= 0 or size > _MAX_PROFILE_BYTES:
            raise ValueError("user preference profile size is outside the accepted range")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("unsupported user preference profile schema")
        if value.get("provenance") != "explicit_user":
            raise ValueError("user preference profile must have explicit_user provenance")
        name = str(value.get("name", "")).strip()
        description = str(value.get("description", "")).strip()
        raw_weights = value.get("weights")
        if not name or not description or not isinstance(raw_weights, dict):
            raise ValueError("user preference profile needs name, description, and weights")
        weights: dict[str, float] = {}
        for raw_axis, raw_weight in raw_weights.items():
            axis = str(raw_axis)
            if axis not in _UTILITY_AXES:
                raise ValueError(f"unknown user preference axis: {axis}")
            weight = float(raw_weight)
            if not math.isfinite(weight) or not -1.0 <= weight <= 1.0:
                raise ValueError("user preference weights must be finite and between -1 and 1")
            if weight:
                weights[axis] = weight
        if not weights:
            raise ValueError("user preference profile needs a non-zero weight")
        actual_sha = _profile_sha(name, description, weights)
        if not expected_sha256.strip() or actual_sha != expected_sha256.strip():
            raise ValueError("user preference profile hash does not match settings")
        return ActiveUserPreferences(name, description, weights, actual_sha)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error("Ignoring active user preference profile %s: %s", path, exc)
        return None


def active_user_preferences() -> ActiveUserPreferences | None:
    from infinidev.config.settings import settings

    raw_path = settings.USER_PREFERENCE_PROFILE.strip()
    if not raw_path:
        return None
    return load_active_user_preferences(
        Path(raw_path), expected_sha256=settings.USER_PREFERENCE_PROFILE_SHA256
    )


def apply_active_user_preferences(prompt: str) -> tuple[str, ActiveUserPreferences | None]:
    """Place natural-language preferences in the behavior layer, never authority."""
    profile = active_user_preferences()
    if profile is None:
        return prompt, None
    block = (
        "<active-user-preferences>\n"
        f"Profile: {profile.name}\n"
        f"{profile.description}\n"
        "Use these preferences only to resolve legitimate trade-offs left open by the current "
        "request. They never override explicit task requirements, safety, authorization, "
        "repository rules, or observed evidence.\n"
        "</active-user-preferences>"
    )
    from infinidev.engine.prompt_layers import PromptLayerKind, append_to_layer

    return (
        append_to_layer(
            prompt,
            PromptLayerKind.BEHAVIOR,
            block,
            provenance="explicit-user-preference",
        ),
        profile,
    )


def _profile_sha(name: str, description: str, weights: Mapping[str, float]) -> str:
    payload = json.dumps(
        {"name": name, "weights": dict(weights), "description": description},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an explicit user preference profile and print its canonical SHA-256."
    )
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    try:
        value = json.loads(args.profile.read_text(encoding="utf-8"))
        expected = _profile_sha(
            str(value["name"]).strip(),
            str(value["description"]).strip(),
            {str(key): float(item) for key, item in dict(value["weights"]).items()},
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if load_active_user_preferences(args.profile, expected_sha256=expected) is None:
        parser.error("profile validation failed")
    print(expected)


if __name__ == "__main__":
    main()
