"""Tests for the adaptive analysis depth system."""

import pytest
from infinidev.gather.models import (
    DepthLevel, DepthConfig, DEPTH_CONFIGS,
    ClassificationResult, TicketType,
)


class TestDepthLevel:
    def test_enum_values(self):
        assert DepthLevel.minimal.value == "minimal"
        assert DepthLevel.light.value == "light"
        assert DepthLevel.standard.value == "standard"
        assert DepthLevel.deep.value == "deep"

    def test_from_string(self):
        assert DepthLevel("minimal") == DepthLevel.minimal
        assert DepthLevel("standard") == DepthLevel.standard

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            DepthLevel("invalid")


class TestDepthConfigs:
    def test_all_levels_present(self):
        assert set(DEPTH_CONFIGS.keys()) == {
            DepthLevel.minimal, DepthLevel.light,
            DepthLevel.standard, DepthLevel.deep,
        }

    def test_minimal_skips_everything(self):
        cfg = DEPTH_CONFIGS[DepthLevel.minimal]
        assert cfg.skip_questions is True
        assert cfg.skip_investigate is True
        assert cfg.reject_write_on_existing is False
        assert cfg.require_test_before_complete is False

    def test_light_skips_questions_investigate(self):
        cfg = DEPTH_CONFIGS[DepthLevel.light]
        assert cfg.skip_questions is True
        assert cfg.skip_investigate is True
        assert cfg.reject_write_on_existing is False

    def test_standard_runs_everything(self):
        cfg = DEPTH_CONFIGS[DepthLevel.standard]
        assert cfg.skip_questions is False
        assert cfg.skip_investigate is False
        assert cfg.reject_write_on_existing is False
        assert cfg.require_test_before_complete is False

    def test_deep_has_guardrails(self):
        cfg = DEPTH_CONFIGS[DepthLevel.deep]
        assert cfg.skip_questions is False
        assert cfg.skip_investigate is False
        assert cfg.reject_write_on_existing is True
        assert cfg.require_test_before_complete is True
        assert cfg.auto_revert_on_regression is True
        assert cfg.aggressive_summarizer is True

    def test_depth_ordering_questions(self):
        """Deeper levels allow more questions."""
        assert DEPTH_CONFIGS[DepthLevel.minimal].questions_max == 0
        assert DEPTH_CONFIGS[DepthLevel.light].questions_max == 0
        assert DEPTH_CONFIGS[DepthLevel.standard].questions_max > 0
        assert DEPTH_CONFIGS[DepthLevel.deep].questions_max >= DEPTH_CONFIGS[DepthLevel.standard].questions_max

    def test_deep_prompt_suffix(self):
        cfg = DEPTH_CONFIGS[DepthLevel.deep]
        assert "STRICT RULES" in cfg.prompt_suffix
        assert "MUST run tests" in cfg.prompt_suffix


class TestClassificationResult:
    def test_default_depth(self):
        result = ClassificationResult()
        assert result.depth == DepthLevel.standard

    def test_with_depth(self):
        result = ClassificationResult(
            ticket_type=TicketType.bug,
            depth=DepthLevel.minimal,
            depth_reasoning="Simple typo fix",
        )
        assert result.depth == DepthLevel.minimal
        assert result.depth_reasoning == "Simple typo fix"

    def test_depth_in_dict(self):
        result = ClassificationResult(
            ticket_type=TicketType.feature,
            depth=DepthLevel.deep,
        )
        d = result.model_dump()
        assert d["depth"] == "deep"


class TestClassifierParsing:
    def test_parse_with_depth(self):
        from infinidev.gather.classifier import _extract_json
        text = '{"ticket_type": "bug", "reasoning": "test", "keywords": ["x"], "depth": "light", "depth_reasoning": "small fix"}'
        parsed = _extract_json(text)
        assert parsed is not None
        assert parsed["depth"] == "light"

    def test_parse_without_depth_defaults(self):
        from infinidev.gather.classifier import _extract_json
        text = '{"ticket_type": "feature", "reasoning": "test", "keywords": []}'
        parsed = _extract_json(text)
        assert parsed is not None
        assert parsed.get("depth", "standard") == "standard"
