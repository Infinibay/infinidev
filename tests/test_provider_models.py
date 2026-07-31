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


def test_catalog_models_use_provider_prefixes() -> None:
    assert get_provider("kimi").prefix == "moonshot/"
    assert get_provider("zai").prefix == "zai/"
    assert get_provider("minimax").prefix == "minimax/"


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
#   qwen:*  litellm indexes DashScope under prefixes that never match this
#           provider's `custom_openai/`; only the 3.6 tier has a documented
#           1M window, so the rest have no verifiable source.
#   others  single models litellm's map has not picked up.
_CONTEXT_WINDOW_GAPS = {
    "qwen:*",
    "mistral:ministral-3b-latest",
    "gmi:deepseek-ai/DeepSeek-R1",
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
