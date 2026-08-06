"""Tests for Graph prompts and the live beta-routing guarantee.

Explicit Graph selection stays pinned to Graph; Auto may choose it only when
its separate feature flag is enabled.
"""

from __future__ import annotations

from infinidev.config.settings import settings
from infinidev.engine.engines.routing import ENGINE_GRAPH_BETA, select_engine
from infinidev.engine.orchestration.escalation_packet import EscalationPacket
from infinidev.prompts.graph import GRAPH_PROTOCOL, GRAPH_VOCABULARY


class TestGraphPrompts:
    def test_vocabulary_defines_core_roles(self):
        for concept in (
            "Requirement", "Hypothesis", "Decision", "Work", "Verification",
            "Evidence", "Blocker", "Checkpoint", "Lifecycle", "Verdict",
            "Freshness",
        ):
            assert concept in GRAPH_VOCABULARY, concept

    def test_vocabulary_keeps_authority_labels(self):
        for label in ("USER_LITERAL", "DERIVED", "OBSERVED_EVIDENCE"):
            assert label in GRAPH_VOCABULARY, label

    def test_protocol_is_guidance_not_stone(self):
        # The project's prompt policy: guidance, room to breathe — not a wall
        # of absolute commands.
        assert "guidance" in GRAPH_PROTOCOL.lower()
        assert "not" in GRAPH_PROTOCOL and "stone" in GRAPH_PROTOCOL

    def test_protocol_prefers_words_and_honesty(self):
        # Meaning-loaded language and honest blocking over false completion.
        assert "checkpoint" in GRAPH_PROTOCOL.lower()
        assert "budget" in GRAPH_PROTOCOL.lower()
        assert "contradict" in GRAPH_PROTOCOL.lower()


class TestGraphConnected:
    def test_explicit_graph_beta_stays_pinned(self, monkeypatch):
        monkeypatch.setattr(settings, "AUTO_ENGINE_ALLOW_GRAPH", False)
        escalation = EscalationPacket(
            user_request="Do a thing", understanding="do it"
        )
        selection = select_engine(escalation, mode="graph_beta")
        assert selection.engine == ENGINE_GRAPH_BETA
        assert selection.requested_mode == "graph_beta"
        assert not selection.fallback_note

    def test_auto_selects_graph_when_enabled(self, monkeypatch):
        monkeypatch.setattr(settings, "AUTO_ENGINE_ALLOW_GRAPH", True)
        escalation = EscalationPacket(
            user_request="Investigate alternatives for a branching solution.",
            understanding="complex",
        )
        selection = select_engine(escalation, mode="auto")
        assert selection.engine == ENGINE_GRAPH_BETA
        assert selection.requested_mode == "auto"
