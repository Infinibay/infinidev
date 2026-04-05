"""Tests for the prompt style variant system."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from infinidev.prompts.variants import (
    resolve_style,
    get_variant,
    registered_names,
    _REGISTRY,
)


# ── resolve_style ────────────────────────────────────────────────────────

class TestResolveStyle:
    def test_auto_small_model(self):
        with patch("infinidev.config.settings.settings") as mock_settings:
            mock_settings.PROMPT_STYLE = "auto"
            with patch("infinidev.config.llm._is_small_model", return_value=True):
                assert resolve_style() == "generalized"

    def test_auto_large_model(self):
        with patch("infinidev.config.settings.settings") as mock_settings:
            mock_settings.PROMPT_STYLE = "auto"
            with patch("infinidev.config.llm._is_small_model", return_value=False):
                assert resolve_style() == "full"

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
    def test_returns_none_for_full(self):
        assert get_variant("loop.identity", "full") is None

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


# ── Registration completeness ────────────────────────────────────────────

class TestRegistrationCompleteness:
    """Verify both variants cover the same prompt names."""

    EXPECTED_NAMES = {
        "loop.identity",
        "loop.protocol",
        "flow.develop.identity",
        "flow.research.identity",
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

    def test_generalized_covers_all_expected(self):
        gen_names = registered_names("generalized")
        missing = self.EXPECTED_NAMES - gen_names
        assert not missing, f"Missing in generalized: {missing}"

    def test_coding_covers_all_expected(self):
        cod_names = registered_names("coding")
        missing = self.EXPECTED_NAMES - cod_names
        assert not missing, f"Missing in coding: {missing}"

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
        "flow.research.identity",
        "flow.sysadmin.identity",
    ])
    def test_generalized_shorter_than_full(self, name):
        gen = get_variant(name, "generalized")
        assert gen is not None

        # Compare against known full prompt sizes (approximate)
        full_sizes = {
            "loop.identity": 5000,
            "loop.protocol": 7000,
            "flow.develop.identity": 10000,
            "flow.research.identity": 4000,
            "flow.sysadmin.identity": 5000,
        }
        full_size = full_sizes[name]
        ratio = len(gen) / full_size
        assert ratio < 0.5, f"{name}: generalized is {ratio:.0%} of full (expected <50%)"


# ── Phase strategy integration ───────────────────────────────────────────

class TestPhaseStrategyIntegration:
    def test_full_returns_original(self):
        with patch("infinidev.prompts.variants.resolve_style", return_value="full"):
            from infinidev.prompts.phases import get_strategy, STRATEGIES
            s = get_strategy("bug")
            assert s is STRATEGIES["bug"]

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
            assert s is STRATEGIES["feature"]


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

    @pytest.mark.parametrize("style", ["generalized", "coding"])
    @pytest.mark.parametrize("task_type", ["bug", "feature", "refactor", "other"])
    def test_execute_has_step_placeholders(self, style, task_type):
        prompt = get_variant(f"phase.{task_type}.execute", style)
        assert prompt is not None
        assert "{{step_num}}" in prompt
        assert "{{total_steps}}" in prompt
        assert "{{step_description}}" in prompt
        assert "{{step_files}}" in prompt

    @pytest.mark.parametrize("style", ["generalized", "coding"])
    def test_investigate_has_question_placeholders(self, style):
        prompt = get_variant("phase.investigate.rules", style)
        assert prompt is not None
        assert "{{q_num}}" in prompt
        assert "{{q_total}}" in prompt
        assert "{{question}}" in prompt
