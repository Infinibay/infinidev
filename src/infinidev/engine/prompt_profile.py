"""Load release-gated per-model guidance without changing baseline prompts."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_PROFILE_BYTES = 64 * 1024
_MAX_GUIDANCE_BYTES = 4 * 1024
_ROLES = frozenset({"chat_agent", "planner", "developer"})


def apply_calibrated_guidance(prompt: str, role: str) -> str:
    """Place validated preference behavior in its layer, or preserve the prompt."""
    from infinidev.config.settings import settings
    from infinidev.engine.user_preferences import apply_active_user_preferences

    prompt, active_preferences = apply_active_user_preferences(prompt)

    profile_path = settings.PROMPT_CALIBRATION_PROFILE.strip()
    if not profile_path:
        return prompt
    utility_profile = settings.PROMPT_CALIBRATION_UTILITY_PROFILE
    utility_profile_sha256 = settings.PROMPT_CALIBRATION_UTILITY_PROFILE_SHA256
    if active_preferences is not None:
        utility_profile = active_preferences.name
        utility_profile_sha256 = active_preferences.sha256
    guidance = load_calibrated_guidance(
        Path(profile_path),
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        model_identity=settings.PROMPT_CALIBRATION_MODEL_IDENTITY,
        utility_profile=utility_profile,
        utility_profile_sha256=utility_profile_sha256,
        role=role,
    )
    if not guidance:
        return prompt
    from infinidev.engine.prompt_layers import PromptLayerKind, append_to_layer

    block = (
        "<model-calibrated-behavior>\n"
        f"{guidance}\n"
        "</model-calibrated-behavior>"
    )
    return append_to_layer(
        prompt,
        PromptLayerKind.BEHAVIOR,
        block,
        provenance="validated-model-preference-study",
    )


def load_calibrated_guidance(
    path: Path,
    *,
    provider: str,
    model: str,
    model_identity: str,
    utility_profile: str,
    utility_profile_sha256: str,
    role: str,
) -> str:
    """Validate a profile and return one role's guidance, failing closed to empty."""
    if role not in _ROLES:
        logger.error("Unknown calibrated prompt role: %s", role)
        return ""
    try:
        size = path.stat().st_size
        if size <= 0 or size > _MAX_PROFILE_BYTES:
            raise ValueError("profile size is outside the accepted range")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 2:
            raise ValueError("unsupported profile schema")
        if value.get("prompt_layer") != "behavior":
            raise ValueError("calibration profile is not a behavior profile")
        if value.get("evidence_kind") != "preference_behavior":
            raise ValueError("calibration profile has incompatible evidence")
        if value.get("deployment_approved") is not True:
            raise ValueError("profile is not approved for deployment")
        if value.get("provider") != provider or value.get("model") != model:
            raise ValueError("profile model route does not match current settings")
        expected_identity = model_identity.strip()
        if not expected_identity:
            raise ValueError("current immutable model identity is not configured")
        if value.get("model_identity") != expected_identity:
            raise ValueError("profile model identity does not match current settings")
        validation = value.get("validation")
        if not isinstance(validation, dict):
            raise ValueError("profile validation metadata is missing")
        raw_utility = validation.get("utility_profile")
        if not isinstance(raw_utility, dict):
            raise ValueError("profile utility metadata is missing")
        if raw_utility.get("name", "") != utility_profile.strip():
            raise ValueError("profile utility objective does not match active user profile")
        if raw_utility.get("sha256", "") != utility_profile_sha256.strip():
            raise ValueError("profile utility hash does not match active user profile")
        roles = value.get("roles")
        entry: Any = roles.get(role) if isinstance(roles, dict) else None
        if not isinstance(entry, dict):
            return ""
        if entry.get("prompt_layer") != "behavior":
            raise ValueError("role guidance is not behavior-only")
        guidance = entry.get("guidance")
        expected_hash = entry.get("sha256")
        if not isinstance(guidance, str) or not guidance.strip():
            raise ValueError("role guidance is empty")
        guidance_bytes = len(guidance.encode("utf-8"))
        if guidance_bytes > _MAX_GUIDANCE_BYTES:
            raise ValueError("role guidance exceeds the compact runtime limit")
        if entry.get("utf8_bytes") != guidance_bytes:
            raise ValueError("role guidance byte count does not match")
        actual_hash = hashlib.sha256(guidance.encode()).hexdigest()
        if expected_hash != actual_hash:
            raise ValueError("role guidance hash does not match")
        return guidance.strip()
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        logger.error("Ignoring calibrated prompt profile %s: %s", path, exc)
        return ""
