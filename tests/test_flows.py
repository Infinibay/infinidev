"""Tests for the multi-flow system.

The legacy AnalysisResult-driven flow-routing tests were removed in
Commit 7 of the pipeline redesign — flow routing is now gone (the
chat-agent-first pipeline always uses ``develop`` for escalations;
``/explore``, ``/brainstorm`` go through ``run_flow_task`` directly).
These tests only cover the flow registry itself.
"""

import pytest

from infinidev.engine.flows import FLOW_REGISTRY, get_flow_config

# Ensure flows are registered
import infinidev.prompts.flows  # noqa: F401


class TestFlowRegistry:
    def test_known_flows_registered(self):
        # develop + sysadmin are escalation-capable; explore/brainstorm
        # remain as slash-command-only flows. research/document survive
        # as registered flows until Commit 9 deletes them.
        names = set(FLOW_REGISTRY.keys())
        assert {"develop", "sysadmin", "explore", "brainstorm"}.issubset(names)

    def test_get_flow_config_develop(self):
        config = get_flow_config("develop")
        assert config.name == "develop"
        assert config.run_review is True
        assert config.identity_prompt
        assert config.expected_output
        assert config.backstory

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
        config = get_flow_config("develop")
        assert "software engineer" in config.identity_prompt.lower()
        assert "read before writing" in config.identity_prompt.lower() or \
               "read the relevant code" in config.identity_prompt.lower() or \
               "read the specific files" in config.identity_prompt.lower()
