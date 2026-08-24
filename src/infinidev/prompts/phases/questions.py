"""Question prompts loaded from packaged JSON phase resources."""

from __future__ import annotations

from infinidev.prompts._phase_resources import load_phase_prompt_bundle


_BUG = load_phase_prompt_bundle("bug")
_FEATURE = load_phase_prompt_bundle("feature")
_REFACTOR = load_phase_prompt_bundle("refactor")
_OTHER = load_phase_prompt_bundle("other")

BUG_QUESTIONS = _BUG.questions_prompt
BUG_FALLBACK = list(_BUG.fallback_questions)
FEATURE_QUESTIONS = _FEATURE.questions_prompt
FEATURE_FALLBACK = list(_FEATURE.fallback_questions)
REFACTOR_QUESTIONS = _REFACTOR.questions_prompt
REFACTOR_FALLBACK = list(_REFACTOR.fallback_questions)
OTHER_QUESTIONS = _OTHER.questions_prompt
OTHER_FALLBACK = list(_OTHER.fallback_questions)
