"""Tests for the multi-flow system."""

import json
import pytest

from infinidev.engine.flows import FlowConfig, FLOW_REGISTRY, get_flow_config, register_flow
from infinidev.engine.analysis.analysis_engine import AnalysisEngine, AnalysisResult

# Ensure flows are registered
import infinidev.prompts.flows  # noqa: F401


class TestFlowRegistry:
    """Test flow registration and retrieval."""

    def test_all_flows_registered(self):
        expected = {"develop", "research", "document", "sysadmin", "explore", "brainstorm"}
        assert expected == set(FLOW_REGISTRY.keys())

    def test_get_flow_config_develop(self):
        config = get_flow_config("develop")
        assert config.name == "develop"
        assert config.run_review is True
        assert config.identity_prompt  # non-empty
        assert config.expected_output  # non-empty
        assert config.backstory  # non-empty

    def test_get_flow_config_research(self):
        config = get_flow_config("research")
        assert config.name == "research"
        assert config.run_review is False
        assert "researcher" in config.identity_prompt.lower()

    def test_get_flow_config_document(self):
        config = get_flow_config("document")
        assert config.name == "document"
        assert config.run_review is False
        assert "documentation" in config.identity_prompt.lower()

    def test_get_flow_config_sysadmin(self):
        config = get_flow_config("sysadmin")
        assert config.name == "sysadmin"
        assert config.run_review is False
        assert "system administrator" in config.identity_prompt.lower()

    def test_get_flow_config_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown flow"):
            get_flow_config("nonexistent")

    def test_each_identity_prompt_nonempty(self):
        for name, config in FLOW_REGISTRY.items():
            assert config.identity_prompt.strip(), f"Flow {name} has empty identity"

    def test_each_expected_output_nonempty(self):
        for name, config in FLOW_REGISTRY.items():
            assert config.expected_output.strip(), f"Flow {name} has empty expected_output"

    def test_develop_has_own_identity(self):
        """Develop flow should have its own dedicated identity prompt."""
        config = get_flow_config("develop")
        assert "software engineer" in config.identity_prompt.lower()
        assert "read before writing" in config.identity_prompt.lower() or \
               "read the relevant code" in config.identity_prompt.lower() or \
               "read the specific files" in config.identity_prompt.lower()


class TestAnalysisResultFlow:
    """Test flow field in AnalysisResult."""

    def test_default_flow_is_develop(self):
        result = AnalysisResult(action="proceed", original_input="test")
        assert result.flow == "develop"

    def test_passthrough_has_flow_done(self):
        engine = AnalysisEngine()
        result = engine._parse_response(
            '{"action": "passthrough", "reason": "greeting"}', "hello"
        )
        assert result.action == "passthrough"
        assert result.flow == "done"

    def test_proceed_extracts_flow(self):
        engine = AnalysisEngine()
        result = engine._parse_response(
            json.dumps({
                "action": "proceed",
                "flow": "research",
                "specification": {"summary": "Research task"},
            }),
            "search for info",
        )
        assert result.action == "proceed"
        assert result.flow == "research"

    def test_proceed_invalid_flow_defaults_to_develop(self):
        engine = AnalysisEngine()
        result = engine._parse_response(
            json.dumps({
                "action": "proceed",
                "flow": "invalid_flow",
                "specification": {"summary": "Test"},
            }),
            "test",
        )
        assert result.action == "proceed"
        assert result.flow == "develop"

    def test_proceed_missing_flow_defaults_to_develop(self):
        engine = AnalysisEngine()
        result = engine._parse_response(
            json.dumps({
                "action": "proceed",
                "specification": {"summary": "Test"},
            }),
            "test",
        )
        assert result.action == "proceed"
        assert result.flow == "develop"

    def test_proceed_flow_sysadmin(self):
        engine = AnalysisEngine()
        result = engine._parse_response(
            json.dumps({
                "action": "proceed",
                "flow": "sysadmin",
                "specification": {"summary": "Install postgres"},
            }),
            "install postgres",
        )
        assert result.flow == "sysadmin"

    def test_proceed_flow_document(self):
        engine = AnalysisEngine()
        result = engine._parse_response(
            json.dumps({
                "action": "proceed",
                "flow": "document",
                "specification": {"summary": "Write API docs"},
            }),
            "document the API",
        )
        assert result.flow == "document"

    def test_build_flow_prompt_alias(self):
        """build_developer_prompt should be an alias for build_flow_prompt."""
        result = AnalysisResult(
            action="proceed",
            original_input="test",
            specification={"summary": "Test"},
        )
        assert result.build_developer_prompt() == result.build_flow_prompt()
