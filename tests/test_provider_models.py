"""Regression tests for provider model catalogs."""

import re
from types import SimpleNamespace

import pytest

from infinidev.config.providers import fetch_models, get_provider


def test_kimi_catalog_includes_k3() -> None:
    models = fetch_models("kimi")
    assert "moonshot/kimi-k3" in models


def test_zai_catalog_includes_glm_52() -> None:
    models = fetch_models("zai")
    assert "zai/glm-5.2" in models


def test_minimax_catalog_includes_m3() -> None:
    models = fetch_models("minimax")
    assert "minimax/MiniMax-M3" in models


def test_minimax_m3_litellm_metadata_uses_its_documented_context() -> None:
    import litellm

    from infinidev.config import llm  # noqa: F401 - registers custom model metadata

    assert litellm.model_cost["minimax/MiniMax-M3"]["max_input_tokens"] == 1_000_000


def test_custom_model_registration_updates_stale_context_without_replacing_pricing(
    monkeypatch,
) -> None:
    import litellm

    from infinidev.config.llm import _register_custom_models

    monkeypatch.setattr(litellm, "model_cost", {
        "minimax/MiniMax-M3": {
            "max_input_tokens": 512_000,
            "max_output_tokens": 12345,
            "input_cost_per_token": 0.000001,
        },
    })

    _register_custom_models()

    assert litellm.model_cost["minimax/MiniMax-M3"] == {
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 12345,
        "input_cost_per_token": 0.000001,
    }


def test_catalog_models_use_provider_prefixes() -> None:
    assert get_provider("kimi").prefix == "moonshot/"
    assert get_provider("zai").prefix == "zai/"
    assert get_provider("minimax").prefix == "minimax/"


def test_qwen_token_plan_is_a_separate_fixed_provider() -> None:
    provider = get_provider("qwen_subscription")

    assert provider.display_name == "Qwen Token Plan (Subscription)"
    assert provider.prefix == "custom_openai/"
    assert provider.base_url_editable is False
    assert provider.default_base_url == (
        "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    )
    assert "custom_openai/qwen3.8-max-preview" in fetch_models(
        "qwen_subscription"
    )
    assert "qwen3.8-max-preview" not in get_provider("qwen").static_models


def test_qwen_token_plan_transport_overrides_stale_metered_base() -> None:
    from infinidev.config.llm import apply_provider_transport

    params = {
        "model": "qwen/qwen3.8-max-preview",
        "api_base": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "api_key": "subscription-key",
    }
    apply_provider_transport(params, "qwen_subscription")

    assert params == {
        "model": "custom_openai/qwen3.8-max-preview",
        "api_base": (
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        ),
        "api_key": "subscription-key",
        "extra_body": {"enable_thinking": True, "reasoning_effort": "medium"},
    }


def test_qwen_token_plan_has_known_tool_capabilities() -> None:
    from infinidev.config.model_capabilities import _PROVIDER_PRESETS

    caps = _PROVIDER_PRESETS["qwen_subscription"]
    assert caps.supports_function_calling is True
    assert caps.supports_tool_choice_required is True
    assert caps.needs_schema_sanitization is True
    assert caps.probed is True


# ── Catalogs age silently; these are the guards ──────────────────────


def test_anthropic_catalog_offers_no_retired_or_deprecated_models() -> None:
    """A listed model the API refuses is worse than one that is missing.

    Offering it means the user picks it, configures a key, and gets a 404
    on their first turn with nothing pointing at the model choice.
    """
    retired_or_deprecated = {
        "claude-opus-4-0",
        "claude-sonnet-4-0",
        "claude-opus-4-1",
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022",
    }
    offered = set(get_provider("anthropic").static_models)
    assert not (offered & retired_or_deprecated)


def test_anthropic_catalog_uses_aliases_not_dated_snapshots() -> None:
    """An alias follows the current snapshot; a dated id freezes and 404s.

    The generation-5 models have no dated variant at all, so a dated entry
    here is always either stale or invented.
    """
    for model in get_provider("anthropic").static_models:
        assert not re.search(r"-\d{8}$", model), model


def test_anthropic_catalog_includes_generation_5() -> None:
    offered = set(get_provider("anthropic").static_models)
    assert {"claude-opus-5", "claude-sonnet-5"} <= offered


def test_kimi_catalog_excludes_the_sunsetting_k2_line() -> None:
    """`kimi-k2.5` and the K2 previews sunset 2026-08-31 (new accounts
    already cannot use them)."""
    offered = set(get_provider("kimi").static_models)
    assert not any(m.startswith("kimi-k2-") or m == "kimi-k2.5" for m in offered)


# Catalog entries whose context window is still unknown. Each one shows "?"
# in the status line and disables the loop's context budget for that model,
# so this list should shrink, never grow. It holds real values rather than a
# guess on purpose: an over-stated window is worse than an unknown one,
# because the loop packs context until the backend truncates it silently.
#
#   qwen:*  LiteLLM indexes DashScope under prefixes that never match this
#           provider's `custom_openai/`; entries remain here only when the
#           current Alibaba catalog does not publish a precise window.
#   others  single models litellm's map has not picked up.
_CONTEXT_WINDOW_GAPS = {
    "qwen:qwen3-32b",
    "qwen:qwen3-30b-a3b",
    "qwen:qwen3-235b-a22b",
    "qwen:qwen3-coder-flash",
    "qwen:qwen3-max",
    "qwen:qwen-max",
    "qwen:qwen-plus",
    "qwen:qwen-turbo",
    "qwen:qwen-flash",
    "qwen:qwq-plus",
    "qwen_subscription:qwen3-coder-flash",
    "qwen_subscription:qwen3-max",
    "qwen_subscription:qwen-plus",
    "qwen_subscription:qwen-flash",
}


def _is_known_gap(provider_id: str, model: str) -> bool:
    return f"{provider_id}:*" in _CONTEXT_WINDOW_GAPS or (
        f"{provider_id}:{model}" in _CONTEXT_WINDOW_GAPS
    )


def test_catalog_models_resolve_a_context_window() -> None:
    """Guards the drift between the catalog and the context-window map.

    A model can be added to `static_models` and work fine for chat while
    silently reporting an unknown window — which shows as "?" and drops the
    loop's context budget to zero. This fails the moment the two disagree.
    """
    from infinidev.config.providers import PROVIDERS
    from infinidev.engine.loop.model_context import get_model_context_window

    missing: list[str] = []
    for pid, provider in PROVIDERS.items():
        for model in provider.static_models:
            if _is_known_gap(pid, model):
                continue
            window = get_model_context_window(
                {"model": f"{provider.prefix}{model}"}, pid
            )
            if not window:
                missing.append(f"{pid}:{model}")

    assert not missing, f"no context window for: {missing}"


def test_documented_context_windows_win_over_litellm(monkeypatch) -> None:
    """A known-but-stale LiteLLM row must not override provider documentation."""
    import litellm

    from infinidev.engine.loop.model_context import get_model_context_window

    monkeypatch.setitem(
        litellm.model_cost,
        "openai/gpt-5.4-mini",
        {"max_input_tokens": 272_000},
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "minimax/MiniMax-M2.5",
        {"max_input_tokens": 1_000_000},
    )

    assert get_model_context_window(
        {"model": "openai/gpt-5.4-mini"}, "openai"
    ) == 400_000
    assert get_model_context_window(
        {"model": "minimax/MiniMax-M2.5"}, "minimax"
    ) == 204_800


def test_openai_api_gpt_56_uses_the_full_model_window() -> None:
    from infinidev.engine.loop.model_context import get_model_context_window

    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        assert get_model_context_window(
            {"model": f"openai/{model}"}, "openai"
        ) == 1_050_000


# ── DeepSeek ─────────────────────────────────────────────────────────
#
# The registry had no `deepseek` entry even though `reasoning.py` and the
# prefix map in `llm.py` both already had branches for the id. `get_provider`
# falls back to *ollama* for an unknown id, so selecting it resolved to
# http://localhost:11434 and every discovery call died with "Connection
# refused" — a silence, not an error anybody could act on.


def test_deepseek_is_a_registered_native_provider() -> None:
    provider = get_provider("deepseek")

    assert provider.id == "deepseek"
    assert provider.prefix == "deepseek/"
    assert provider.default_base_url == "https://api.deepseek.com/v1"
    # Native, so no api_base is sent and LiteLLM routes by the prefix.
    assert provider.is_native is True
    assert provider.model_list_format == "openai"


def test_deepseek_does_not_resolve_to_ollama() -> None:
    """The fallback that made the gap silent."""
    provider = get_provider("deepseek")

    assert provider.id != "ollama"
    assert "localhost" not in provider.default_base_url
    assert provider.prefix != "ollama_chat/"


def test_deepseek_catalog_carries_the_served_model_names() -> None:
    """These are the two ids the endpoint reports from POST /models."""
    models = fetch_models("deepseek")

    assert "deepseek/deepseek-flash" in models
    assert "deepseek/deepseek-v4-pro" in models


def test_deepseek_models_are_in_litellms_cost_map() -> None:
    """No custom registration needed: LiteLLM already prices both."""
    import litellm

    flash = litellm.model_cost["deepseek/deepseek-flash"]
    assert flash["input_cost_per_token"] == 3e-07
    assert flash["max_input_tokens"] == 1_000_000
    assert flash["supports_reasoning"] is True
    assert "deepseek/deepseek-v4-pro" in litellm.model_cost


# ── Context window for backends no catalog knows ─────────────────────
#
# A self-hosted OpenAI-compatible server (llama.cpp, vLLM, or any gateway the
# user points `openai_compatible` at) has no LiteLLM row, so every lookup came
# back None: the status line showed "?" and the loop ran with no context
# budget at all. Two mechanisms fix that — a `LLM_CONTEXT_WINDOW` declaration
# and a probe of the server itself — and both must stay off the path of the
# providers that already resolve a window.


@pytest.fixture
def context_probe(monkeypatch):
    """Reset the context-window caches and stub out the probe's HTTP layer."""
    import httpx

    from infinidev.config.settings import settings
    from infinidev.engine.loop import model_context

    monkeypatch.setattr(settings, "LLM_CONTEXT_WINDOW", 0)
    monkeypatch.setattr(model_context, "_OLLAMA_CTX_CACHE", {})
    monkeypatch.setattr(model_context, "_SELF_HOSTED_CTX_CACHE", {})

    calls: list[str] = []
    posts: list[str] = []

    class _Response:
        def __init__(self, payload: object, status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def json(self) -> object:
            return self._payload

    def install(responses: dict[str, object]) -> None:
        def fake_get(url: str, headers: object = None, timeout: object = None):
            calls.append(url)
            payload = responses.get(url)
            if isinstance(payload, Exception):
                raise payload
            if payload is None:
                return _Response({}, status_code=404)
            return _Response(payload)

        def fake_post(url: str, json: object = None, timeout: object = None):
            # Keeps the Ollama branch off the developer's real localhost.
            posts.append(url)
            return _Response({}, status_code=404)

        monkeypatch.setattr(httpx, "get", fake_get)
        monkeypatch.setattr(httpx, "post", fake_post)

    return SimpleNamespace(calls=calls, posts=posts, install=install)


def test_llm_context_window_defaults_to_autodetection(monkeypatch) -> None:
    """Zero means "figure it out", so the default must not disable probing."""
    from infinidev.config.settings import Settings

    monkeypatch.delenv("INFINIDEV_LLM_CONTEXT_WINDOW", raising=False)
    assert Settings().LLM_CONTEXT_WINDOW == 0


def test_llm_context_window_override_wins_over_the_catalogs(
    context_probe, monkeypatch
) -> None:
    """The user's declaration beats a probe or a catalog row that may be wrong."""
    from infinidev.config.settings import settings
    from infinidev.engine.loop.model_context import (
        get_model_context_window,
        get_model_max_context_window,
    )

    monkeypatch.setattr(settings, "LLM_CONTEXT_WINDOW", 131_072)

    # A model with a documented override (400 000) and one with a litellm row:
    # both report the declared window instead.
    assert get_model_context_window(
        {"model": "openai/gpt-5.4-mini"}, "openai"
    ) == 131_072
    assert get_model_max_context_window(
        {"model": "openai/gpt-5.4-mini"}, "openai"
    ) == 131_072
    assert not context_probe.calls


def test_llm_context_window_override_wins_over_ollama(context_probe, monkeypatch) -> None:
    """It also short-circuits the Ollama `/api/show` fetch, which is a POST."""
    from infinidev.config.settings import settings
    from infinidev.engine.loop.model_context import get_model_context_window

    monkeypatch.setattr(settings, "LLM_CONTEXT_WINDOW", 262_144)

    assert get_model_context_window(
        {"model": "ollama_chat/qwen3.8:latest", "base_url": "http://localhost:11434"},
        "ollama",
    ) == 262_144
    assert not context_probe.posts


def test_self_hosted_probe_reads_llama_cpp_props(context_probe) -> None:
    """llama-server publishes the allocated `n_ctx` on `/props`, at the root."""
    from infinidev.engine.loop.model_context import get_model_context_window

    context_probe.install(
        {
            "http://localhost:8080/props": {
                "default_generation_settings": {"n_ctx": 32_768},
            },
        }
    )

    window = get_model_context_window(
        {
            "model": "custom_openai/my-model.Q4_K_M.gguf",
            "base_url": "http://localhost:8080/v1",
        },
        "llama_cpp",
    )

    assert window == 32_768
    # Both sources are read in one round so the field-priority rule below has
    # something to rank; the memo keeps it to a single round.
    assert context_probe.calls == [
        "http://localhost:8080/props",
        "http://localhost:8080/v1/models/my-model.Q4_K_M.gguf",
    ]


def test_self_hosted_probe_reads_vllm_model_card(context_probe) -> None:
    """vLLM has no `/props`; its configured limit is `max_model_len`."""
    from infinidev.engine.loop.model_context import get_model_context_window

    context_probe.install(
        {
            "http://localhost:8000/v1/models/strata-code-7b": {
                "id": "strata-code-7b",
                "object": "model",
                "max_model_len": 262_144,
            },
        }
    )

    window = get_model_context_window(
        {
            "model": "custom_openai/strata-code-7b",
            "base_url": "http://localhost:8000/v1",
        },
        "vllm",
    )

    assert window == 262_144
    # The `/props` miss must not abort the model-card attempt.
    assert context_probe.calls == [
        "http://localhost:8000/props",
        "http://localhost:8000/v1/models/strata-code-7b",
    ]


def test_unregistered_provider_id_still_gets_probed(context_probe) -> None:
    """A gateway registered as `custom_openai/...` under an unknown id.

    `get_provider` falls back for an unregistered id, so the probe must key on
    the model prefix as well — otherwise a custom backend is sized by nothing.
    """
    from infinidev.engine.loop.model_context import (
        get_model_context_window,
        get_model_max_context_window,
    )

    context_probe.install(
        {
            "https://gateway.internal/props": {
                "default_generation_settings": {"n_ctx": 65_536},
            },
        }
    )
    params = {
        "model": "custom_openai/strata-proprietary-9b",
        "base_url": "https://gateway.internal/v1",
    }

    assert get_model_context_window(params, "strata") == 65_536
    assert get_model_max_context_window(params, "strata") == 65_536


def test_self_hosted_probe_prefers_the_served_window(context_probe) -> None:
    """A trained length the server may not serve ranks below a served limit."""
    from infinidev.engine.loop.model_context import get_model_context_window

    context_probe.install(
        {
            "http://localhost:8080/props": {"default_generation_settings": {"n_ctx": 32_768}},
            "http://localhost:8080/v1/models/qwen3": {
                "details": {"max_context_length": 131_072},
                "max_position_embeddings": 40_960,
            },
        }
    )

    window = get_model_context_window(
        {"model": "custom_openai/qwen3", "base_url": "http://localhost:8080/v1"},
        "llama_cpp",
    )

    assert window == 131_072


def test_self_hosted_probe_returns_none_when_nothing_is_advertised(context_probe) -> None:
    """Unknown stays unknown: a guessed window silently truncates context."""
    from infinidev.engine.loop.model_context import get_model_context_window

    context_probe.install(
        {
            "http://localhost:9000/v1/models/strata-proprietary-9b": {
                "id": "strata-proprietary-9b",
                "object": "model",
                "created": 1_700_000_000,
            },
        }
    )

    assert get_model_context_window(
        {
            "model": "custom_openai/strata-proprietary-9b",
            "base_url": "http://localhost:9000/v1",
        },
        "openai_compatible",
    ) is None


def test_self_hosted_probe_ignores_bogus_fields(context_probe) -> None:
    """Booleans are ints in Python, and a token id is not a context window."""
    from infinidev.engine.loop.model_context import get_model_context_window

    context_probe.install(
        {
            "http://localhost:9000/v1/models/strata-proprietary-9b": {
                "max_model_len": True,
                "n_ctx": 7,
                "context_length": 2**40,
            },
        }
    )

    assert get_model_context_window(
        {
            "model": "custom_openai/strata-proprietary-9b",
            "base_url": "http://localhost:9000/v1",
        },
        "openai_compatible",
    ) is None


def test_self_hosted_probe_is_memoized(context_probe) -> None:
    """The context bar recomputes on every refresh; the probe must not follow."""
    from infinidev.engine.loop.model_context import get_model_context_window

    context_probe.install(
        {"http://localhost:8080/props": {"default_generation_settings": {"n_ctx": 32_768}}}
    )
    params = {"model": "custom_openai/qwen3", "base_url": "http://localhost:8080/v1"}

    assert get_model_context_window(params, "llama_cpp") == 32_768
    assert get_model_context_window(params, "llama_cpp") == 32_768
    assert len(context_probe.calls) == 2


def test_self_hosted_probe_caches_the_unknown_result(context_probe) -> None:
    """A server that answers nothing must not be probed on every rebuild."""
    import httpx

    from infinidev.engine.loop.model_context import get_model_context_window

    context_probe.install(
        {
            "http://localhost:9000/props": httpx.ConnectError("connection refused"),
            "http://localhost:9000/v1/models/strata-proprietary-9b": httpx.ConnectError(
                "connection refused"
            ),
        }
    )
    params = {
        "model": "custom_openai/strata-proprietary-9b",
        "base_url": "http://localhost:9000/v1",
    }

    assert get_model_context_window(params, "openai_compatible") is None
    assert get_model_context_window(params, "openai_compatible") is None
    assert len(context_probe.calls) == 3


def test_cataloged_providers_are_never_probed_over_http(context_probe) -> None:
    """The `/props` + model-card probe is for servers with no size catalog.

    Running it against a cloud API would cost a request per context rebuild and
    could read an unrelated number in a vendor payload as a context window.
    """
    from infinidev.engine.loop.model_context import get_model_context_window

    context_probe.install({"https://api.openai.com/v1/props": {"n_ctx": 128}})

    assert get_model_context_window({"model": "openai/gpt-5.4-mini"}, "openai") == 400_000
    # Ollama answers nothing here, so its trained length stays unknown.
    assert get_model_context_window(
        {"model": "ollama_chat/qwen3.8:latest", "base_url": "http://localhost:11434"},
        "ollama",
    ) is None
    assert not context_probe.calls
    assert context_probe.posts == ["http://localhost:11434/api/show"]


def test_self_hosted_probe_reads_strata_model_list(context_probe):
    from infinidev.engine.loop.model_context import get_model_context_window

    context_probe.install({
        "http://localhost:8080/v1/models": {"data": [
            {"id": "other-model", "meta": {"n_ctx": 1_000_000}},
            {"id": "local-qwen", "meta": {"n_ctx": 262_144}},
        ]},
    })

    assert get_model_context_window({
        "model": "custom_openai/local-qwen", "base_url": "http://localhost:8080/v1",
    }, "llama_cpp") == 262_144


def test_model_list_never_borrows_another_models_context(context_probe):
    from infinidev.engine.loop.model_context import get_model_context_window

    context_probe.install({
        "http://localhost:8080/v1/models": {"data": [
            {"id": "other-model", "meta": {"n_ctx": 1_000_000}},
        ]},
    })

    assert get_model_context_window({
        "model": "custom_openai/unlisted-local", "base_url": "http://localhost:8080/v1",
    }, "llama_cpp") is None
