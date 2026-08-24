"""Planning prompts loaded from packaged JSON phase resources."""

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

PLANNER_IDENTITY = _SHARED.planner_identity
BUG_PLAN = _BUG.plan_prompt
BUG_PLAN_IDENTITY = _BUG.plan_identity
FEATURE_PLAN = _FEATURE.plan_prompt
FEATURE_PLAN_IDENTITY = _FEATURE.plan_identity
REFACTOR_PLAN = _REFACTOR.plan_prompt
REFACTOR_PLAN_IDENTITY = _REFACTOR.plan_identity
OTHER_PLAN = _OTHER.plan_prompt
OTHER_PLAN_IDENTITY = _OTHER.plan_identity
