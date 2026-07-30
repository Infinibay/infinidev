"""Regression tests for provider model catalogs."""

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
