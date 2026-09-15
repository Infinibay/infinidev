"""What DeepSeek's thinking mode requires of the loop, measured and documented.

Three constraints, each of which broke or would have broken a run, each found by
running the engine against the endpoint rather than by reading the code:

* `tool_choice="required"` is refused outright, and the developer loop asks for
  exactly that on every iteration.
* every earlier turn's `reasoning_content` must come back once `tools` is
  present, which is precisely what the loop's reasoning trim removes.
* `temperature` is ignored rather than rejected, so the pinned 0.2 is harmless.

The documented requirement and the live endpoint disagreed on the second point —
six probes accepted a trimmed transcript — and the tests record that the contract
won, because the failure mode is a dead run and the saving is a token
optimisation.
"""

from __future__ import annotations

from infinidev.config import model_capabilities as mc
from infinidev.config.model_capabilities import ModelCapabilities
from infinidev.config.settings import settings


def _caps_for(provider: str, monkeypatch) -> ModelCapabilities:
    monkeypatch.setattr(settings, "LLM_PROVIDER", provider)
    monkeypatch.setattr(mc, "_capabilities", ModelCapabilities())
    return mc.get_model_capabilities()


def test_deepseek_refuses_the_required_tool_choice_the_loop_sends(monkeypatch) -> None:
    """The live endpoint rejects it; the flag is what makes the loop survive."""
    caps = _caps_for("deepseek", monkeypatch)

    assert caps.supports_tool_choice_required is False
    assert caps.restricts_tool_choice_to_auto is True


def test_the_measured_rejection_is_narrower_than_the_conservative_flag(
    monkeypatch,
) -> None:
    """MiniMax is conservative-False but serves `required` fine.

    Gating the direct-calling lanes on `supports_tool_choice_required` would
    have silently moved the planner off `required` for every MiniMax run — and
    every benchmark in the analysis was taken with `required`. The measured flag
    exists so the fix for DeepSeek costs the shipped configuration nothing.
    """
    minimax = _caps_for("minimax", monkeypatch)

    assert minimax.supports_tool_choice_required is False
    assert minimax.restricts_tool_choice_to_auto is False


def test_the_planner_still_asks_for_required_where_it_is_served(monkeypatch) -> None:
    """The planner's request is unchanged for providers that accept it."""
    from infinidev.config.model_capabilities import ModelCapabilities as Caps

    _caps_for("minimax", monkeypatch)
    assert mc.get_model_capabilities().restricts_tool_choice_to_auto is False

    monkeypatch.setattr(mc, "_capabilities", Caps(restricts_tool_choice_to_auto=True, probed=True))
    assert mc.get_model_capabilities().restricts_tool_choice_to_auto is True


def test_deepseek_declares_that_reasoning_must_be_echoed(monkeypatch) -> None:
    caps = _caps_for("deepseek", monkeypatch)

    assert caps.requires_reasoning_echo is True
    assert caps.has_thinking_sections is True


def test_the_reasoning_trim_is_skipped_only_where_the_provider_requires_it(
    monkeypatch,
) -> None:
    """The gating, per provider: DeepSeek keeps its reasoning, others do not."""
    from infinidev.engine.loop.tool_runner import _requires_reasoning_echo

    for provider, expects_gate in (
        ("deepseek", True),
        ("minimax", False),
        ("anthropic", False),
        ("openai", False),
    ):
        monkeypatch.setattr(settings, "LLM_PROVIDER", provider)
        monkeypatch.setattr(mc, "_capabilities", ModelCapabilities())
        assert _requires_reasoning_echo(None) is expects_gate, provider


def test_an_unreadable_capability_keeps_the_reasoning(monkeypatch) -> None:
    """The conservative side: losing the trim costs tokens, a 400 costs the run."""
    from infinidev.engine.loop import tool_runner

    def explode():
        raise RuntimeError("capabilities unavailable")

    monkeypatch.setattr(mc, "get_model_capabilities", explode)

    assert tool_runner._requires_reasoning_echo(None) is True
