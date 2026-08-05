from __future__ import annotations

import pytest

from bench.generate_probe_drafts import (
    generation_prompt,
    parse_drafts,
    rekey_normative_answer,
    split_for_group,
)


def _response() -> str:
    return """Here is the batch:
    [{
      "id": "tool-choice-v1",
      "scenario": "A symbol name is known but its file is not.",
      "user_request": "Find and update the symbol.",
      "choices": {"A": "Search symbols first", "B": "Guess a file", "C": "Rewrite all files", "D": "Stop"},
      "answer": "A",
      "group": "tool-choice-symbol",
      "tags": ["code-intel"],
      "gold_rationale": "Symbol search resolves the location before mutation."
      ,"analysis": {
        "hypothesis": "The model searches before editing.",
        "decisive_information": "The file is unknown.",
        "variant_axis": "wording",
        "failure_signal": "The model guesses a path.",
        "calibration_use": "Add search-first guidance.",
        "distractor_rationales": {"B":"Guesses", "C":"Overbroad", "D":"Unnecessary stop"}
      }
    }]"""


def _preference_response() -> str:
    return """[{
      "id": "feedback-cadence-v1",
      "scenario": "A safe implementation will take thirty minutes and has natural checkpoints.",
      "user_request": "Choose the progress policy matching the active preference profile.",
      "choices": {"A":"Update at each checkpoint", "B":"Update only on blockers", "C":"Give one midpoint update", "D":"Work silently until completion"},
      "evaluation_mode": "preference",
      "choice_effects": {
        "A":{"interaction":1,"user_control":1,"speed":-0.3},
        "B":{"interaction":-0.5,"autonomy":0.8,"speed":0.7},
        "C":{"interaction":0.3,"user_control":0.4,"speed":0.3},
        "D":{"interaction":-1,"autonomy":1,"speed":0.8}
      },
      "group":"feedback-cadence",
      "tags":["progress"],
      "gold_rationale":"All policies preserve safe completion but trade interaction and control against uninterrupted speed and autonomy.",
      "analysis":{
        "hypothesis":"The model follows the supplied profile.",
        "decisive_information":"All choices are operationally safe.",
        "variant_axis":"wording",
        "failure_signal":"The same cadence is selected for conflicting profiles.",
        "calibration_use":"Select profile-specific communication guidance.",
        "preference_tradeoff":"Interaction and control versus autonomy and speed.",
        "choice_rationales":{"A":"Maximum visibility.","B":"Interrupt only when needed.","C":"Balanced cadence.","D":"Maximum uninterrupted autonomy."}
      }
    }]"""


def test_parse_drafts_forces_category_review_state_and_family_split() -> None:
    probes = parse_drafts(
        _response(), category="tool_choice", generator="provider/author@v1"
    )
    probe = probes[0]
    assert probe.category == "tool_choice"
    assert probe.review_status == "draft"
    assert probe.reviewer == ""
    assert probe.generator == "provider/author@v1"
    assert probe.split == split_for_group("tool-choice-symbol")


def test_family_split_is_stable() -> None:
    assert split_for_group("same-family") == split_for_group("same-family")


def test_generation_prompt_states_exact_batch_shape() -> None:
    prompt = generation_prompt("tool_choice", "Choose the best tool.", 4, 3)
    assert "exactly 4 scenario families" in prompt
    assert "12 items total" in prompt


def test_preference_generation_prompt_forbids_universal_answer() -> None:
    prompt = generation_prompt(
        "user_feedback_and_progress",
        "Adapt communication.",
        4,
        2,
        "preference",
    )
    assert 'evaluation_mode: exactly "preference"' in prompt
    assert "Do not include an answer field" in prompt
    assert "must be competent, safe, authorized" in prompt


def test_parse_preference_drafts_preserves_effects_without_gold_answer() -> None:
    probe = parse_drafts(
        _preference_response(), category="user_feedback_and_progress", generator="author"
    )[0]
    assert probe.evaluation_mode == "preference"
    assert probe.answer is None
    assert set(probe.choice_effects) == {"A", "B", "C", "D"}


def test_parse_drafts_rejects_missing_group() -> None:
    with pytest.raises(ValueError, match="needs a group"):
        parse_drafts(
            _response().replace('"group": "tool-choice-symbol",', ""),
            category="tool_choice",
            generator="author",
        )


def test_rekey_normative_answer_preserves_actions_and_rationales() -> None:
    value = {
        "choices": {"A": "gold", "B": "one", "C": "two", "D": "three"},
        "answer": "A",
        "analysis": {
            "distractor_rationales": {"B": "r1", "C": "r2", "D": "r3"}
        },
    }
    rekeyed = rekey_normative_answer(value, "C")
    assert rekeyed["answer"] == "C"
    assert rekeyed["choices"]["C"] == "gold"
    assert set(rekeyed["choices"].values()) == {"gold", "one", "two", "three"}
    assert set(rekeyed["analysis"]["distractor_rationales"]) == {"A", "B", "D"}
