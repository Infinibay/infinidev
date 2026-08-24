"""Execution prompts loaded from packaged JSON phase resources."""

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

_EDIT_CONTRACT = _SHARED.execute_edit_contract
BUG_EXECUTE = _BUG.execute_prompt
BUG_EXECUTE_IDENTITY = _BUG.execute_identity
FEATURE_EXECUTE = _FEATURE.execute_prompt
FEATURE_EXECUTE_IDENTITY = _FEATURE.execute_identity
REFACTOR_EXECUTE = _REFACTOR.execute_prompt
REFACTOR_EXECUTE_IDENTITY = _REFACTOR.execute_identity
OTHER_EXECUTE = _OTHER.execute_prompt
OTHER_EXECUTE_IDENTITY = _OTHER.execute_identity
