"""Detect native tool support without imposing manual mode on compatible servers."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.config import model_capabilities as mc
from infinidev.config.settings import settings


@pytest.mark.parametrize('native', [True, False])
def test_compatible_route_uses_probe_evidence_and_caches_it(monkeypatch, native):
    monkeypatch.setattr(settings, 'LLM_PROVIDER', 'openai_compatible')
    monkeypatch.setattr('infinidev.config.llm.get_litellm_params',
                        lambda: {'model': 'custom_openai/protocol-fixture'})
    monkeypatch.setattr(mc, '_detect_vision_support', lambda: False)
    requests = []
    def completion(**kwargs):
        requests.append(kwargs)
        calls = [SimpleNamespace(function=SimpleNamespace(
            name='calculator', arguments='{"expression":"2+2"}'))] if native else None
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            tool_calls=calls, content=''))])
    monkeypatch.setattr('litellm.completion', completion)
    caps = mc.get_model_capabilities()
    assert caps.supports_function_calling is native
    assert caps.supports_tool_choice_required is native
    assert caps.probed
    assert requests and requests[0]['tools'][0]['function']['name'] == 'calculator'
    count = len(requests)
    assert mc.get_model_capabilities() is caps
    assert len(requests) == count


def test_compatible_probe_failure_preserves_manual_fallback(monkeypatch):
    monkeypatch.setattr(settings, 'LLM_PROVIDER', 'openai_compatible')
    monkeypatch.setattr('infinidev.config.llm.get_litellm_params', lambda: {'model': 'fixture'})
    def unavailable(params):
        raise RuntimeError('probe unavailable')
    monkeypatch.setattr(mc, '_run_probes', unavailable)
    caps = mc._detect_openai_compatible_capabilities()
    assert caps.probed
    assert not caps.supports_function_calling
    assert not caps.supports_tool_choice_required
