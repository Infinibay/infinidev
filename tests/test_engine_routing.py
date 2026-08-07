"""Tests for engine routing / selection (docs/GRAPH_ENGINE_BETA_DESIGN.md §8.4)."""

from __future__ import annotations

import pytest

from infinidev.config.settings import settings
from infinidev.engine.engines.routing import (
    ENGINE_GRAPH_BETA,
    ENGINE_REACT,
    ENGINE_STAGED,
    ENGINE_TASK,
    normalize_mode,
    select_engine,
)
from infinidev.engine.orchestration.escalation_packet import EscalationPacket


def _packet(text: str) -> EscalationPacket:
    return EscalationPacket(user_request=text, understanding=text)


class TestNormalizeMode:
    @pytest.mark.parametrize("mode", ["auto", "task", "react", "staged", "graph_beta"])
    def test_valid_modes_pass_through(self, mode):
        assert normalize_mode(mode) == mode

    @pytest.mark.parametrize("mode", ["", None, "bogus", "graph", "REACT "])
    def test_invalid_modes_resolve_to_task(self, mode):
        # "REACT " has trailing space + wrong case; normalize lower/strips.
        if mode == "REACT ":
            assert normalize_mode(mode) == "react"
        else:
            assert normalize_mode(mode) == ENGINE_TASK


class TestExplicitModes:
    def test_explicit_react(self):
        selection = select_engine(_packet("do a thing"), mode="react")
        assert selection.engine == ENGINE_REACT
        assert selection.requested_mode == "react"
        assert "user_selected_react" in selection.reasons

    def test_explicit_task(self):
        selection = select_engine(_packet("do a thing"), mode="task")
        assert selection.engine == ENGINE_TASK
        assert "user_selected_task" in selection.reasons

    def test_explicit_staged(self):
        selection = select_engine(_packet("do a thing"), mode="staged")
        assert selection.engine == ENGINE_STAGED
        assert "user_selected_staged" in selection.reasons

    def test_explicit_graph_beta_is_pinned_even_when_auto_graph_is_disabled(
        self, monkeypatch
    ):
        monkeypatch.setattr(settings, "AUTO_ENGINE_ALLOW_GRAPH", False)

        selection = select_engine(_packet("do a thing"), mode="graph_beta")

        assert selection.engine == ENGINE_GRAPH_BETA
        assert selection.requested_mode == ENGINE_GRAPH_BETA
        assert not selection.fallback_note
        assert "user_selected_graph_beta" in selection.reasons


class TestAutoClassifier:
    def test_trivial_request_prefers_task(self):
        selection = select_engine(_packet("What is the capital of France?"), mode="auto")
        assert selection.engine == ENGINE_TASK
        assert selection.requested_mode == "auto"
        assert 0.0 <= selection.confidence <= 1.0

    def test_long_multistep_request_prefers_task(self):
        text = (
            "Migrate the authentication layer to JWT. "
            "1. Add middleware to all endpoints. 2. Update every route. "
            "3. And then write tests. " + ("Extra context. " * 60)
        )
        selection = select_engine(_packet(text), mode="auto")
        assert selection.engine == ENGINE_TASK

    def test_branching_request_prefers_graph_when_enabled(self, monkeypatch):
        monkeypatch.setattr(settings, "AUTO_ENGINE_ALLOW_GRAPH", True)
        selection = select_engine(
            _packet("Investigate alternatives and compare their trade-offs."),
            mode="auto",
        )
        assert selection.engine == ENGINE_GRAPH_BETA
        assert selection.requested_mode == "auto"

    def test_branching_request_does_not_use_graph_when_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "AUTO_ENGINE_ALLOW_GRAPH", False)
        selection = select_engine(
            _packet("Investigate alternatives and compare their trade-offs."),
            mode="auto",
        )
        assert selection.engine != ENGINE_GRAPH_BETA

    def test_selection_payload_shape(self):
        selection = select_engine(_packet("What is X?"), mode="auto")
        payload = selection.to_payload()
        assert set(payload) == {
            "engine", "requested_mode", "confidence", "reasons", "risks",
            "reconsider_if", "estimated_overhead", "fallback_note",
        }
        assert isinstance(payload["reasons"], list)
        assert payload["reasons"]
