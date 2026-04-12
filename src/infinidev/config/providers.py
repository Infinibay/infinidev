"""LLM provider registry — configuration and model discovery per provider."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 8  # seconds for API calls


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    id: str
    display_name: str
    prefix: str                              # LiteLLM model prefix
    default_base_url: str
    api_key_required: bool = True
    base_url_editable: bool = False
    model_list_format: str = "static"        # ollama, openai, anthropic, gemini, static
    static_models: list[str] = field(default_factory=list)
    is_native: bool = False                  # LiteLLM handles endpoint natively


# ── Provider Registry ────────────────────────────────────────────────

PROVIDERS: dict[str, ProviderConfig] = {
    "ollama": ProviderConfig(
        id="ollama",
        display_name="Ollama (Local)",
        prefix="ollama_chat/",
        default_base_url="http://localhost:11434",
        api_key_required=False,
        base_url_editable=True,
        model_list_format="ollama",
    ),
    "openai": ProviderConfig(
        id="openai",
        display_name="OpenAI",
        prefix="openai/",
        default_base_url="https://api.openai.com/v1",
        model_list_format="openai",
        is_native=True,
        static_models=[
            "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
            "o3", "o3-pro", "o3-mini", "o4-mini",
        ],
    ),
    "anthropic": ProviderConfig(
        id="anthropic",
        display_name="Claude (Anthropic)",
        prefix="anthropic/",
        default_base_url="https://api.anthropic.com",
        model_list_format="anthropic",
        is_native=True,
        static_models=[
            "claude-opus-4-6", "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101",
            "claude-sonnet-4-0", "claude-opus-4-0",
        ],
    ),
    "gemini": ProviderConfig(
        id="gemini",
        display_name="Gemini (Google)",
        prefix="gemini/",
        default_base_url="https://generativelanguage.googleapis.com",
        model_list_format="gemini",
        is_native=True,
        static_models=[
            "gemini-3.1-pro-preview", "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
        ],
    ),
    "zai": ProviderConfig(
        id="zai",
        display_name="Z.AI (Zhipu/GLM)",
        prefix="zai/",
        default_base_url="https://api.z.ai/api/paas/v4",
        model_list_format="openai",
        is_native=True,
        static_models=[
            "glm-5", "glm-5-turbo",
            "glm-4.7", "glm-4.6", "glm-4.5",
            "glm-4.5-flash", "glm-4.5-air",
        ],
    ),
    "zai_coding": ProviderConfig(
        id="zai_coding",
        display_name="Z.AI Coding Plan",
        prefix="zai/",
        default_base_url="https://api.z.ai/api/coding/paas/v4",
        model_list_format="openai",
        is_native=False,
        static_models=[
            "glm-5", "glm-5-turbo",
            "glm-4.7", "glm-4.6", "glm-4.5",
            "glm-4.5-flash", "glm-4.5-air",
        ],
    ),
    "kimi": ProviderConfig(
        id="kimi",
        display_name="Kimi (Moonshot)",
        prefix="moonshot/",
        default_base_url="https://api.moonshot.ai/v1",
        model_list_format="static",
        static_models=[
            "kimi-k2.5",
            "kimi-k2-thinking", "kimi-k2-thinking-turbo",
            "kimi-k2-0905-preview", "kimi-k2-turbo-preview",
        ],
    ),
    "minimax": ProviderConfig(
        id="minimax",
        display_name="Minimax",
        prefix="minimax/",
        default_base_url="https://api.minimax.io/v1",
        model_list_format="static",
        static_models=[
            "MiniMax-M2.7", "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5", "MiniMax-M2.1",
        ],
    ),
    "openrouter": ProviderConfig(
        id="openrouter",
        display_name="OpenRouter",
        prefix="openrouter/",
        default_base_url="https://openrouter.ai/api/v1",
        model_list_format="openai",
        is_native=True,
    ),
    "llama_cpp": ProviderConfig(
        id="llama_cpp",
        display_name="llama.cpp Server",
        prefix="custom_openai/",
        default_base_url="http://localhost:8080/v1",
        api_key_required=False,
        base_url_editable=True,
        # llama.cpp serves a single model from a .gguf file path; /v1/models
        # returns a synthetic name that is not user-meaningful. Treat the
        # model field as free-text so the user can type the actual gguf name.
        model_list_format="free_text",
    ),
    "vllm": ProviderConfig(
        id="vllm",
        display_name="vLLM Server",
        prefix="custom_openai/",
        default_base_url="http://localhost:8000/v1",
        api_key_required=False,
        base_url_editable=True,
        model_list_format="openai",
    ),
    "openai_compatible": ProviderConfig(
        id="openai_compatible",
        display_name="OpenAI Compatible",
        prefix="custom_openai/",
        default_base_url="",
        base_url_editable=True,
        model_list_format="free_text",
    ),
    "qwen": ProviderConfig(
        id="qwen",
        display_name="Qwen (Alibaba)",
        prefix="custom_openai/",
        default_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        model_list_format="openai",
        static_models=[
            "qwen3.6-plus",
            "qwen3.5-plus", "qwen3.5-flash",
            "qwen3.5-397b-a17b", "qwen3.5-122b-a10b",
            "qwen3-max", "qwen3-coder-plus", "qwen3-coder-flash",
            "qwen3-235b-a22b", "qwen3-32b", "qwen3-30b-a3b",
            "qwen-max", "qwen-plus", "qwen-turbo", "qwen-flash",
            "qwq-plus",
        ],
    ),
}


def get_provider(provider_id: str) -> ProviderConfig:
    """Get provider config by ID. Falls back to ollama."""
    return PROVIDERS.get(provider_id, PROVIDERS["ollama"])


def list_provider_ids() -> list[str]:
    """Return ordered list of provider IDs."""
    return list(PROVIDERS.keys())


# ── Model Discovery ──────────────────────────────────────────────────

def fetch_models(
    provider_id: str,
    api_key: str = "",
    base_url: str = "",
) -> list[str]:
    """Fetch available models for a provider. Returns prefixed model names."""
    provider = get_provider(provider_id)
    # For providers with a fixed URL, always use their default — the passed
    # base_url likely belongs to a previously-selected provider (e.g. Ollama).
    if not provider.base_url_editable and provider.default_base_url:
        url = provider.default_base_url.rstrip("/")
    else:
        url = base_url.rstrip("/") if base_url else provider.default_base_url.rstrip("/")

    if not url:
        return [f"{provider.prefix}{m}" for m in provider.static_models]

    try:
        if provider.model_list_format == "ollama":
            return _fetch_ollama(url, provider.prefix)
        elif provider.model_list_format == "openai":
            return _fetch_openai(url, api_key, provider.prefix)
        elif provider.model_list_format == "anthropic":
            return _fetch_anthropic(url, api_key, provider.prefix)
        elif provider.model_list_format == "gemini":
            return _fetch_gemini(url, api_key, provider.prefix)
        else:
            return [f"{provider.prefix}{m}" for m in provider.static_models]
    except Exception as exc:
        logger.warning("Failed to fetch models for %s: %s", provider_id, str(exc)[:200])
        # Fall back to static list
        if provider.static_models:
            return [f"{provider.prefix}{m}" for m in provider.static_models]
        return []


def _fetch_ollama(base_url: str, prefix: str) -> list[str]:
    """GET {base_url}/api/tags → models[].name"""
    resp = httpx.get(f"{base_url}/api/tags", timeout=_TIMEOUT)
    resp.raise_for_status()
    models = resp.json().get("models", [])
    return [f"{prefix}{m['name']}" for m in models]


def _fetch_openai(base_url: str, api_key: str, prefix: str) -> list[str]:
    """GET {base_url}/models with Bearer auth → data[].id"""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = httpx.get(f"{base_url}/models", headers=headers, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return [f"{prefix}{m['id']}" for m in data]


def _fetch_anthropic(base_url: str, api_key: str, prefix: str) -> list[str]:
    """GET {base_url}/v1/models with x-api-key header → data[].id"""
    headers = {
        "anthropic-version": "2023-06-01",
    }
    if api_key:
        headers["x-api-key"] = api_key
    resp = httpx.get(f"{base_url}/v1/models", headers=headers, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return [f"{prefix}{m['id']}" for m in data]


def _fetch_gemini(base_url: str, api_key: str, prefix: str) -> list[str]:
    """GET {base_url}/v1/models?key={api_key} → models[].name"""
    params = {}
    if api_key:
        params["key"] = api_key
    resp = httpx.get(f"{base_url}/v1/models", params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    models = resp.json().get("models", [])
    result = []
    for m in models:
        name = m.get("name", "")
        # Gemini returns "models/gemini-pro" — strip prefix
        if name.startswith("models/"):
            name = name[len("models/"):]
        if name:
            result.append(f"{prefix}{name}")
    return result
