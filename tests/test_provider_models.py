"""Regression tests for provider model catalogs."""

import re

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
