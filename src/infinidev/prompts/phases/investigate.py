"""Investigation prompts loaded from packaged JSON phase resources."""

from __future__ import annotations

from infinidev.prompts._phase_resources import (
    load_phase_prompt_bundle,
    load_shared_phase_prompts,
)


_SHARED = load_shared_phase_prompts()
_BUG = load_phase_prompt_bundle("bug")
_FEATURE = load_phase_prompt_bundle("feature")
_REFACTOR = load_phase_prompt_bundle("refactor")
_OTHER = load_phase_prompt_bundle("other")

_INVESTIGATE_RULES = _SHARED.investigate_rules
FOLLOWUP_PROMPT = _SHARED.followup_prompt
BUG_INVESTIGATE = _BUG.investigate_prompt
BUG_INVESTIGATE_IDENTITY = _BUG.investigate_identity
FEATURE_INVESTIGATE = _FEATURE.investigate_prompt
FEATURE_INVESTIGATE_IDENTITY = _FEATURE.investigate_identity
REFACTOR_INVESTIGATE = _REFACTOR.investigate_prompt
REFACTOR_INVESTIGATE_IDENTITY = _REFACTOR.investigate_identity
OTHER_INVESTIGATE = _OTHER.investigate_prompt
OTHER_INVESTIGATE_IDENTITY = _OTHER.investigate_identity
