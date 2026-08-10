"""Tests for the prompt style variant system."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from infinidev.engine.loop.prompt.text import BEHAVIOR_GUIDELINES
from infinidev.prompts.variants import (
    _REGISTRY,
    get_variant,
    registered_names,
    resolve_style,
)


# ── resolve_style ────────────────────────────────────────────────────────

class TestResolveStyle:
    def test_auto_defaults_to_generalized(self):
        with patch("infinidev.config.settings.settings") as mock_settings:
            mock_settings.PROMPT_STYLE = "auto"
            assert resolve_style() == "generalized"

    def test_explicit_full(self):
        with patch("infinidev.config.settings.settings") as mock_settings:
            mock_settings.PROMPT_STYLE = "full"
            assert resolve_style() == "full"

    def test_explicit_generalized(self):
        with patch("infinidev.config.settings.settings") as mock_settings:
            mock_settings.PROMPT_STYLE = "generalized"
            assert resolve_style() == "generalized"

    def test_explicit_coding(self):
        with patch("infinidev.config.settings.settings") as mock_settings:
            mock_settings.PROMPT_STYLE = "coding"
            assert resolve_style() == "coding"


# ── get_variant ──────────────────────────────────────────────────────────

class TestGetVariant:
    def test_full_returns_prompt(self):
        result = get_variant("loop.identity", "full")
        assert result is not None
        assert len(result) > 1000  # full prompts are detailed

    def test_returns_none_for_missing(self):
        assert get_variant("nonexistent.prompt", "generalized") is None

    def test_returns_prompt_for_generalized(self):
        result = get_variant("loop.identity", "generalized")
        assert result is not None
        assert len(result) > 100

    def test_returns_prompt_for_coding(self):
        result = get_variant("loop.identity", "coding")
        assert result is not None
        assert "class " in result or "def " in result


class TestGeneralizedBehaviorContract:
    def test_identity_distinguishes_interest_from_authority(self):
        prompt = get_variant("loop.identity", "generalized")
        assert prompt is not None
        assert "interest, hypotheticals, examples" in prompt
        assert "Future or conditional approval is not current permission" in prompt

    def test_identity_does_not_broaden_ambiguous_target(self):
        prompt = get_variant("loop.identity", "generalized")
        assert prompt is not None
        assert "never choose one or broaden the target to all candidates" in prompt

    def test_protocol_separates_requirements_from_defaults(self):
        prompt = get_variant("loop.protocol", "generalized")
        assert prompt is not None
        assert "literal user requirements" in prompt
        assert "Defaults guide HOW to work" in prompt

    def test_protocol_requires_evidence_driven_bounded_retries(self):
        prompt = get_variant("loop.protocol", "generalized")
        assert prompt is not None
        assert "Retry a failure only when it produced new evidence" in prompt
        assert "never loop until success or conceal material failures" in prompt

    def test_develop_flow_uses_proportional_defaults(self):
        prompt = get_variant("flow.develop.identity", "generalized")
        assert prompt is not None
        assert "Inspect only enough evidence for the next reversible code decision" in prompt
        assert "After every edit" not in prompt
        assert "production-ready" not in prompt

    def test_phases_do_not_restore_ritual_verification(self):
        bug = get_variant("phase.bug.execute", "generalized")
        feature = get_variant("phase.feature.execute", "generalized")
        refactor = get_variant("phase.refactor.plan", "generalized")
        assert bug is not None
        assert feature is not None
        assert refactor is not None
        assert "after 3 attempts" not in bug
        assert "3 consecutive edits" not in feature
        assert '"run full test suite" after every step' not in refactor


class TestCrossStyleBehaviorContract:
    def test_shared_behavior_defines_authority_and_retry_invariants(self):
        assert "Future or conditional approval is not current permission" in BEHAVIOR_GUIDELINES
        assert "Do not choose one or broaden the target to all" in BEHAVIOR_GUIDELINES
        assert "Keep literal user requirements separate" in BEHAVIOR_GUIDELINES
        assert "Retry a failure only when it supplied new evidence" in BEHAVIOR_GUIDELINES
        assert "Correctness comes from that feedback loop" in BEHAVIOR_GUIDELINES

    def test_shared_behavior_keeps_the_literal_task_authoritative(self):
        assert "The literal active task is authoritative" in BEHAVIOR_GUIDELINES
        assert "not a competing product owner" in BEHAVIOR_GUIDELINES
        assert "then stop" not in BEHAVIOR_GUIDELINES

    @pytest.mark.parametrize("style", ["full", "coding", "extra_simple"])
    def test_develop_flow_does_not_require_verification_after_every_edit(self, style):
        prompt = get_variant("flow.develop.identity", style)
        assert prompt is not None
        assert "After EVERY edit" not in prompt
        assert "Verify EVERY edit" not in prompt
        assert "Verify every edit" not in prompt

    @pytest.mark.parametrize("style", ["full", "coding", "extra_simple"])
    def test_refactor_phase_does_not_require_full_suite_after_every_change(self, style):
        execute = get_variant("phase.refactor.execute", style)
        plan = get_variant("phase.refactor.plan", style)
        assert execute is not None
        assert plan is not None
        combined = execute + plan
        assert "run the full test suite (not just one test)" not in combined
        assert "Run ALL tests after every change" not in combined
        assert "run_full_test_suite_after_EVERY_step" not in combined
        assert "Run tests after every step" not in combined

    def test_coding_and_extra_simple_encode_authority_without_prose_layer(self):
        coding = get_variant("loop.identity", "coding")
        minimal = get_variant("loop.identity", "extra_simple")
        assert coding is not None
        assert minimal is not None
        assert "future_or_conditional_permission != current_permission" in coding
        assert "future permission is not current permission" in minimal

    def test_extra_simple_keeps_pending_steps_mutable(self):
        protocol = get_variant("loop.protocol", "extra_simple")
        assert protocol is not None
        assert "add, modify, or remove pending steps" in protocol
        assert "do not stop after planning" in protocol


# ── Registration completeness ────────────────────────────────────────────

class TestRegistrationCompleteness:
    """Verify both variants cover the same prompt names."""

    EXPECTED_NAMES = {
        "loop.identity",
        "loop.protocol",
        "flow.develop.identity",
        "flow.document.identity",
        "flow.sysadmin.identity",
        "flow.explore.identity",
        "flow.brainstorm.identity",
        "phase.bug.execute",
        "phase.feature.execute",
        "phase.refactor.execute",
        "phase.other.execute",
        "phase.bug.execute_identity",
        "phase.feature.execute_identity",
        "phase.refactor.execute_identity",
        "phase.other.execute_identity",
        "phase.bug.plan",
        "phase.feature.plan",
        "phase.refactor.plan",
        "phase.other.plan",
        "phase.bug.plan_identity",
        "phase.feature.plan_identity",
        "phase.refactor.plan_identity",
        "phase.other.plan_identity",
        "phase.planner.identity",
        "phase.investigate.rules",
        "phase.bug.investigate_identity",
        "phase.feature.investigate_identity",
        "phase.refactor.investigate_identity",
        "phase.other.investigate_identity",
    }

    def test_full_covers_all_expected(self):
        full_names = registered_names("full")
        # phase.investigate.rules is a shared prefix only used by generalized/coding;
        # in full, it's baked into the per-type investigate prompts.
        expected = self.EXPECTED_NAMES - {"phase.investigate.rules"}
        missing = expected - full_names
        assert not missing, f"Missing in full: {missing}"

    def test_generalized_covers_all_expected(self):
        gen_names = registered_names("generalized")
        missing = self.EXPECTED_NAMES - gen_names
        assert not missing, f"Missing in generalized: {missing}"

    def test_coding_covers_all_expected(self):
        cod_names = registered_names("coding")
        missing = self.EXPECTED_NAMES - cod_names
        assert not missing, f"Missing in coding: {missing}"

    def test_extra_simple_covers_all_expected(self):
        es_names = registered_names("extra_simple")
        missing = self.EXPECTED_NAMES - es_names
        assert not missing, f"Missing in extra_simple: {missing}"

    def test_generalized_subset_of_coding(self):
        """Every generalized prompt should have a coding counterpart."""
        gen_names = registered_names("generalized")
        cod_names = registered_names("coding")
        missing = gen_names - cod_names
        assert not missing, f"In generalized but not coding: {missing}"


# ── Size reduction ───────────────────────────────────────────────────────

class TestSizeReduction:
    """Generalized variants should be significantly shorter than full."""

    @pytest.mark.parametrize("name", [
        "loop.identity",
        "loop.protocol",
        "flow.develop.identity",
        "flow.sysadmin.identity",
    ])
    def test_generalized_shorter_than_full(self, name):
        full = get_variant(name, "full")
        gen = get_variant(name, "generalized")
        assert full is not None
        assert gen is not None
        ratio = len(gen) / len(full)
        assert ratio < 0.5, f"{name}: generalized is {ratio:.0%} of full (expected <50%)"


# ── Phase strategy integration ───────────────────────────────────────────

class TestPhaseStrategyIntegration:
    def test_full_returns_same_content(self):
        with patch("infinidev.prompts.variants.resolve_style", return_value="full"):
            from infinidev.prompts.phases import get_strategy, STRATEGIES
            s = get_strategy("bug")
            original = STRATEGIES["bug"]
            assert s.execute_prompt == original.execute_prompt
            assert s.plan_prompt == original.plan_prompt
            assert s.questions_min == original.questions_min

    def test_generalized_returns_variant_prompts(self):
        with patch("infinidev.prompts.variants.resolve_style", return_value="generalized"):
            from infinidev.prompts.phases import get_strategy, STRATEGIES
            s = get_strategy("bug")
            original = STRATEGIES["bug"]
            # Should be different prompt text
            assert s.execute_prompt != original.execute_prompt
            # But same numeric limits
            assert s.questions_min == original.questions_min
            assert s.execute_max_tool_calls_per_step == original.execute_max_tool_calls_per_step

    def test_unknown_type_defaults_to_feature(self):
        with patch("infinidev.prompts.variants.resolve_style", return_value="full"):
            from infinidev.prompts.phases import get_strategy, STRATEGIES
            s = get_strategy("unknown_type")
            assert s.execute_prompt == STRATEGIES["feature"].execute_prompt


# ── Flow identity integration ────────────────────────────────────────────

class TestFlowIdentityIntegration:
    def test_full_returns_original(self):
        with patch("infinidev.prompts.variants.resolve_style", return_value="full"):
            from infinidev.prompts.flows import get_flow_identity
            from infinidev.prompts.flows.develop import DEVELOP_IDENTITY
            identity = get_flow_identity("develop")
            assert identity == DEVELOP_IDENTITY

    def test_generalized_returns_variant(self):
        with patch("infinidev.prompts.variants.resolve_style", return_value="generalized"):
            from infinidev.prompts.flows import get_flow_identity
            identity = get_flow_identity("develop")
            assert len(identity) < 5000  # Much shorter than full

    def test_coding_returns_pseudocode(self):
        with patch("infinidev.prompts.variants.resolve_style", return_value="coding"):
            from infinidev.prompts.flows import get_flow_identity
            identity = get_flow_identity("develop")
            assert "class " in identity or "def " in identity


# ── Template placeholders preserved ──────────────────────────────────────

class TestPlaceholdersPreserved:
    """Phase prompts must keep {{placeholder}} variables for runtime substitution."""

    @pytest.mark.parametrize("style", ["generalized", "coding", "extra_simple"])
    @pytest.mark.parametrize("task_type", ["bug", "feature", "refactor", "other"])
    def test_execute_has_step_placeholders(self, style, task_type):
        prompt = get_variant(f"phase.{task_type}.execute", style)
        assert prompt is not None
        assert "{{step_num}}" in prompt
        assert "{{total_steps}}" in prompt
        assert "{{step_title}}" in prompt
        assert "{{step_files}}" in prompt

    @pytest.mark.parametrize("style", ["generalized", "coding", "extra_simple"])
    def test_investigate_has_question_placeholders(self, style):
        prompt = get_variant("phase.investigate.rules", style)
        assert prompt is not None
        assert "{{q_num}}" in prompt
        assert "{{q_total}}" in prompt
        assert "{{question}}" in prompt
