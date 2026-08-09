"""Centralized LLM configuration for Infinidev CLI."""

from __future__ import annotations

import logging
import os
import re
import warnings
from typing import Any

from infinidev.config.settings import settings

logger = logging.getLogger(__name__)

# LiteLLM's Responses stream logger serializes an intermediate chunk before it
# has converted ``usage`` from a dict into ``ResponseAPIUsage``. The final
# ModelResponse exposes a valid ``Usage`` instance, but Pydantic otherwise
# prints this same harmless warning for every request. Keep the filter exact so
# unrelated serializer warnings remain visible.
warnings.filterwarnings(
    "ignore",
    message=r"(?s)^Pydantic serializer warnings:.*Expected `ResponseAPIUsage`",
    category=UserWarning,
    module=r"pydantic\.main",
)

# Warn-once guard for the response normalizer: it fires on every completion, so
# repeated failures would spam the log. Warn loudly the first time it regresses
# (a real bug), DEBUG thereafter.
_normalizer_warned = False


# ── Register models missing from LiteLLM's built-in database ─────────
# LiteLLM rejects requests for unknown models with wrong context-window
# limits.  We register them once at import time so every call_llm() and
# capability probe works correctly.


def _register_custom_models() -> None:
    """Add model entries that LiteLLM doesn't ship yet."""
    try:
        import litellm

        _M27_BASE = {
            "litellm_provider": "minimax",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_tool_choice": True,
            "supports_prompt_caching": True,
            "supports_reasoning": True,
            "supports_system_messages": True,
            "max_input_tokens": 204_800,
            "max_output_tokens": 8192,
            "input_cost_per_token": 3e-07,
            "output_cost_per_token": 1.2e-06,
            "cache_read_input_token_cost": 3e-08,
            "cache_creation_input_token_cost": 3.75e-07,
        }
        _M3_BASE = {
            **_M27_BASE,
            # M3 is not part of the M2 API family's 204.8k window. Keeping
            # this metadata in sync with the loop's documented override
            # prevents any LiteLLM caller outside LoopEngine from silently
            # treating a 1M-context agent as a 204.8k one.
            "max_input_tokens": 1_000_000,
        }

        # The highspeed variants are the ones litellm's map keeps missing —
        # it indexes MiniMax's fast tier under a different suffix, so every
        # `-highspeed` id the provider actually serves needs registering here
        # or the request is rejected as an unknown model.
        custom = {
            "minimax/MiniMax-M3": _M3_BASE,
            "minimax/MiniMax-M2.7": {**_M27_BASE},
            "minimax/MiniMax-M2.7-highspeed": {**_M27_BASE},
            "minimax/MiniMax-M2.5-highspeed": {**_M27_BASE},
            "minimax/MiniMax-M2.1-highspeed": {**_M27_BASE},
        }

        for model_id, info in custom.items():
            if model_id not in litellm.model_cost:
                litellm.model_cost[model_id] = info
                logger.debug("Registered custom model: %s", model_id)
    except Exception as exc:
        logger.debug("Could not register custom models: %s", exc)


_register_custom_models()


def _install_global_response_normalizer() -> None:
    """Wrap ``litellm.completion`` once at import time so every caller
    — ``engine.llm_client.call_llm``, ``engine.orchestration.chat_agent``,
    ``engine.analysis.planner``, review, summariser, etc. — gets
    <think>...</think> blocks lifted out of ``message.content`` and
    into ``message.reasoning_content`` before touching them.

    Why a global wrapper instead of per-site calls: there are 8+
    places in the codebase that call ``litellm.completion`` directly,
    and new ones appear any time someone writes a helper that needs a
    one-off LLM call. Patching every site is churn and the next new
    one will re-introduce the leak. Normalising at the LiteLLM
    boundary is one edit and self-maintains.

    Streaming responses (generator) are passed through unchanged;
    callers that consume streams and assemble content must call
    ``strip_think_blocks`` on the assembled text themselves.

    The same boundary also repairs the one backend that refuses to answer
    a non-streaming request at all — see ``_needs_forced_streaming`` in the
    ChatGPT-subscription section below. Those helpers are defined further
    down the module and resolved when a request runs, not when this
    installs, so they live beside the rest of that provider's knowledge.
    """
    try:
        import litellm

        if getattr(litellm, "_infinidev_response_normalizer_installed", False):
            return

        _original = litellm.completion

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            # Evaluation campaigns activate this boundary-wide pacer through a
            # ContextVar. Keeping it here covers every direct LiteLLM caller,
            # not only the main developer-loop client.
            from infinidev.engine.subscription_safety import pace_llm_request

            pace_llm_request()
            if _is_codex_request(kwargs):
                kwargs = _sanitized_for_codex(kwargs)
            if _needs_forced_streaming(kwargs):
                response = _completion_via_forced_stream(_original, args, kwargs)
            else:
                response = _original(*args, **kwargs)
            if kwargs.get("stream"):
                return response
            try:
                from infinidev.engine.loop.llm_caller import (
                    promote_embedded_think as _promote,
                )

                choices = getattr(response, "choices", None) or []
                for choice in choices:
                    msg = getattr(choice, "message", None)
                    if msg is not None:
                        _promote(msg)
            except Exception as exc:
                # Load-bearing normalizer (8+ call sites depend on <think>
                # promotion). Warn loudly the first time it regresses so a real
                # bug surfaces; DEBUG afterward to avoid per-response spam. Must
                # not break completion, so keep the catch.
                global _normalizer_warned
                if not _normalizer_warned:
                    _normalizer_warned = True
                    logger.warning(
                        "response normalizer failed (first occurrence): %s",
                        exc,
                        exc_info=True,
                    )
                else:
                    logger.debug("response normalizer skipped: %s", exc)
            return response

        litellm.completion = _wrapped
        litellm._infinidev_response_normalizer_installed = True
    except Exception as exc:
        logger.warning(
            "Could not install response normalizer: %s",
            exc,
            exc_info=True,
        )


_install_global_response_normalizer()


def _extract_provider(model: str) -> str:
    """Extract provider prefix from a LiteLLM model string."""
    if "/" in model:
        return model.split("/", 1)[0].lower()
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    return ""


_NATIVE_MODEL_PREFIXES = {"deepseek", "anthropic", "gemini", "openai"}


def _is_native_provider_id(provider_id: str, model: str) -> bool:
    """Whether LiteLLM reaches this provider's endpoint without an api_base."""
    from infinidev.config.providers import PROVIDERS, get_provider

    provider = get_provider(provider_id)
    if provider.is_native:
        return True
    # A registered provider's own flag settles it. The prefix heuristic below
    # only exists for ids that predate the registry, and consulting it anyway
    # would misroute openai_subscription: its models carry the `openai/`
    # prefix but must reach chatgpt.com, where the OAuth token is valid —
    # api.openai.com would reject it as a malformed key.
    if provider_id in PROVIDERS:
        return False
    return _extract_provider(model) in _NATIVE_MODEL_PREFIXES


def _is_native_provider(model: str) -> bool:
    """Return True if LiteLLM handles this provider's endpoint natively."""
    return _is_native_provider_id(settings.LLM_PROVIDER, model)


def _resolved_base_url(provider_id: str, configured_base_url: str) -> str:
    """Return the endpoint owned by *provider_id* for every LLM lane."""
    from infinidev.config.providers import resolve_base_url

    return resolve_base_url(provider_id, configured_base_url)


# ── ChatGPT subscription (Codex) ─────────────────────────────────────

CHATGPT_SUBSCRIPTION_PROVIDER = "openai_subscription"
QWEN_SUBSCRIPTION_PROVIDER = "qwen_subscription"

# `openai/` selects LiteLLM's OpenAI transport, `responses/` flips it onto
# the Responses API. The Codex backend serves no /chat/completions endpoint,
# so without the second half every request 404s.
_SUBSCRIPTION_PREFIX = "openai/responses/"


def _normalize_subscription_model(model: str) -> str:
    """Force the ``openai/responses/`` prefix onto a subscription model.

    Users type (and settings files inherit) bare slugs like ``gpt-5.5``, or
    an ``openai/`` prefix copied from the metered provider. Both are the
    right *model* pointed at the wrong *protocol*, and the resulting 404 says
    nothing useful — so normalise instead of failing, exactly as the
    ``ollama/`` → ``ollama_chat/`` correction below does.
    """
    if model.startswith(_SUBSCRIPTION_PREFIX):
        return model
    bare = model
    for prefix in ("responses/", "openai/"):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    return f"{_SUBSCRIPTION_PREFIX}{bare}"


def _apply_chatgpt_subscription(params: dict[str, Any], provider_id: str) -> None:
    """Point *params* at the Codex backend with a live OAuth credential.

    Mutates in place, and is a no-op for every other provider. Called last by
    each ``get_litellm_params*`` builder so it can override the api_key and
    api_base those functions derived from settings — under this provider both
    are wrong by construction: there is no key to configure, and the base URL
    is fixed by the backend rather than chosen by the user.
    """
    if provider_id != CHATGPT_SUBSCRIPTION_PROVIDER:
        return

    from infinidev.config import openai_oauth
    from infinidev.config.providers import get_provider

    params["model"] = _normalize_subscription_model(params.get("model", ""))

    # Resolves (and refreshes) on every call. Cheap in the common case — a
    # stat, a read and a clock comparison — and a run that outlives its token
    # is otherwise a 401 halfway through a step.
    creds = openai_oauth.resolve()
    params["api_key"] = creds.access_token
    params["api_base"] = get_provider(provider_id).default_base_url

    headers = params.setdefault("extra_headers", {})
    headers.update(openai_oauth.request_headers(account_id=creds.account_id))

    # The Responses API defaults `store` to true, which would file every
    # Infinidev turn into server-side conversation history. Off, explicitly.
    extra_body = params.setdefault("extra_body", {})
    extra_body.setdefault("store", False)

    # GPT-5.x reasoning models reject an explicit temperature. The developer
    # loop pins 0.2 for tool-calling stability (see get_litellm_params), which
    # is right for local models and a 400 here.
    params.pop("temperature", None)


def _apply_qwen_subscription(params: dict[str, Any], provider_id: str) -> None:
    """Pin Qwen Token Plan requests to its subscription-only transport."""
    if provider_id != QWEN_SUBSCRIPTION_PROVIDER:
        return

    from infinidev.config.providers import get_provider

    model = params.get("model", "")
    bare = model
    for prefix in ("custom_openai/", "qwen/", "openai/"):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    params["model"] = f"custom_openai/{bare}"
    # Never inherit a metered DashScope URL from an older configuration:
    # Token Plan credentials are deliberately not interchangeable with it.
    params["api_base"] = get_provider(provider_id).default_base_url


def apply_provider_transport(params: dict[str, Any], provider_id: str) -> None:
    """Apply provider-specific authentication and transport to standalone calls.

    Most Infinidev call builders invoke the provider adapters as their final
    step. Offline evaluation runners build their own generation parameters,
    so this public boundary keeps fixed subscription endpoints and any
    provider-specific authentication policy consistent without copying it.
    """
    _apply_chatgpt_subscription(params, provider_id)
    _apply_qwen_subscription(params, provider_id)


def _codex_api_base() -> str:
    """The Codex backend's base URL, normalised for comparison.

    Read from the provider registry rather than written here twice, so the
    URL has exactly one definition.
    """
    from infinidev.config.providers import get_provider

    base = get_provider(CHATGPT_SUBSCRIPTION_PROVIDER).default_base_url or ""
    return base.rstrip("/").lower()


def _is_codex_request(kwargs: dict[str, Any]) -> bool:
    """Whether this request is bound for the Codex backend.

    The api_base decides it, not the model string: ``openai/responses/`` says
    which protocol is spoken, and every restriction repaired below belongs to
    the *host* that is listening.
    """
    api_base = str(kwargs.get("api_base") or "").rstrip("/").lower()
    if not api_base:
        return False
    try:
        return api_base == _codex_api_base()
    except Exception as exc:  # provider registry unavailable: behave as before
        logger.debug("codex-backend check skipped: %s", exc)
        return False


# Parameters the Codex backend refuses. ``max_tokens`` is the one that bites:
# LiteLLM renders it as the Responses API's ``max_output_tokens``, which this
# backend answers with "Unsupported parameter: max_output_tokens", and nine
# call sites in the engine set it. ``temperature`` is popped by
# _apply_chatgpt_subscription and then put back by the planner's own
# ``setdefault(0.1)``, so it has to be caught here too. The rest are rejected
# by the Responses transport and cost nothing to drop.
_CODEX_UNSUPPORTED_PARAMS = (
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "presence_penalty",
    "frequency_penalty",
)


def _sanitized_for_codex(kwargs: dict[str, Any]) -> dict[str, Any]:
    """A copy of *kwargs* without the parameters this backend rejects.

    A copy, because callers reuse their kwargs dict across loop iterations
    and a caller that finds its own settings edited underneath it is a bug
    that surfaces far from here.
    """
    dropped = [k for k in _CODEX_UNSUPPORTED_PARAMS if k in kwargs]
    if not dropped:
        return kwargs
    logger.debug("dropping unsupported Codex params: %s", ", ".join(dropped))
    return {k: v for k, v in kwargs.items() if k not in _CODEX_UNSUPPORTED_PARAMS}


def _needs_forced_streaming(kwargs: dict[str, Any]) -> bool:
    """Whether this request goes to a backend that answers streams only.

    The Codex backend rejects every non-streaming request with HTTP 400 and
    ``{"detail":"Stream must be set to true"}``.

    Five call sites ask for a whole response rather than a stream (the
    analyst planner, both spec-elaborator passes, the council loop and the
    work summariser), and every one of them 400s under this provider. Each
    also inherits ``num_retries=3``, so a single such call burns four
    requests against the user's plan before the error surfaces.
    """
    if kwargs.get("stream"):
        return False
    return _is_codex_request(kwargs)


def _completion_via_forced_stream(
    original: Any, args: tuple, kwargs: dict[str, Any]
) -> Any:
    """Request the stream the backend insists on, return the whole response.

    ``stream_chunk_builder`` reassembles the chunks into the same
    ``ModelResponse`` a non-streaming call produces, tool calls and usage
    included, so the caller never learns that its request was rewritten.
    """
    import litellm

    stream_kwargs = dict(kwargs)
    stream_kwargs["stream"] = True
    chunks = list(original(*args, **stream_kwargs))

    rebuilt = litellm.stream_chunk_builder(chunks, messages=kwargs.get("messages"))
    if rebuilt is None:
        raise RuntimeError(
            f"The Codex backend streamed {len(chunks)} chunks that rebuilt "
            "into no response. Retry, or set LLM_PROVIDER to a metered "
            "provider for this run."
        )
    return rebuilt


def _get_model_size_b(model: str | None = None) -> int:
    """Extract model size in billions from model name.

    Parses patterns like 'qwen2.5-coder:7b', 'llama3.1:8b', 'mistral-7b-instruct'.
    Returns 0 if size cannot be detected.
    """
    model = model or settings.LLM_MODEL or ""
    match = re.search(r"(\d+)\s*[bB]\b", model.lower())
    if match:
        return int(match.group(1))
    return 0


_SMALL_MODEL_NAME_HINTS = (
    # Explicit local / open-weight families that fit on consumer GPUs.
    # Listed lowercase; matched as substrings of the model id.
    "glm-4.7-flash",
    "glm-4-flash",
    "glm-flash",
    "gemma2",
    "gemma3",
    "gemma4",
    "qwen2.5-coder",
    "qwen3",
    "qwen3.5",
    "mistral-small",
    "mistral-7b",
    "mixtral-8x7b",
    "nemotron-3-super",
    "nemotron-cascade",
    "lfm2",
    "gpt-oss:20b",
    # Generic "small" markers
    ":flash",
    "-flash",
    "-mini",
    "-tiny",
    "-small",
    "haiku",
)


def _is_small_model(model: str | None = None) -> bool:
    """Return True if the model is in the "small" tier (<~40B effective).

    Detection order:
      1. Explicit size suffix in the name (e.g. "qwen3:9b" → 9 < 40 → True).
      2. A model the Codex catalog publishes is hosted, whatever its name
         suggests. ``gpt-5.4-mini`` matched the generic ``-mini`` marker and
         was forced to "low" reasoning plus the trimmed small-model toolset,
         even though the catalog states it accepts low, medium, high and
         xhigh. A published capability outranks a substring guess.
      3. Substring match against ``_SMALL_MODEL_NAME_HINTS`` for known
         local/open-weight families that don't carry a size in their tag
         (e.g. ``glm-4.7-flash:latest`` — previously classified as large).
      4. Default False (treat unknown as large; safer for hosted big models).
    """
    name = (model or settings.LLM_MODEL or "").lower()
    size = _get_model_size_b(name)
    if 0 < size < 40:
        return True
    if _has_published_reasoning_levels(name):
        return False
    for hint in _SMALL_MODEL_NAME_HINTS:
        if hint in name:
            return True
    return False


def _has_published_reasoning_levels(name: str) -> bool:
    """Whether the Codex catalog names this model and its reasoning levels.

    Read from the catalog rather than from ``settings.LLM_PROVIDER`` so the
    answer is about the model in hand — the review extractor and the council
    can each point at a different one within a single run.
    """
    try:
        from infinidev.config.codex_catalog import reasoning_levels

        return bool(reasoning_levels(name.rsplit("/", 1)[-1]))
    except Exception as exc:  # catalog unreadable: fall through to the hints
        logger.debug("catalog check skipped for %s: %s", name, exc)
        return False


def get_litellm_params_for_review_extractor() -> dict[str, Any]:
    """Build litellm params for the review extractor (Pass A).

    Each ``REVIEW_EXTRACTOR_LLM_*`` setting is optional and falls back to
    the matching ``LLM_*`` main setting when empty. Use this to point the
    factual-extraction pass at a small/fast model while the judge keeps
    running on the main one.
    """
    model = (settings.REVIEW_EXTRACTOR_LLM_MODEL or "").strip() or settings.LLM_MODEL
    if not model:
        raise RuntimeError(
            "No review-extractor model and no main LLM_MODEL configured."
        )

    if model.startswith("ollama/"):
        model = "ollama_chat/" + model[len("ollama/") :]

    provider_id = (
        settings.REVIEW_EXTRACTOR_LLM_PROVIDER or ""
    ).strip() or settings.LLM_PROVIDER
    api_key = (
        settings.REVIEW_EXTRACTOR_LLM_API_KEY or ""
    ).strip() or settings.LLM_API_KEY
    base_url = (
        settings.REVIEW_EXTRACTOR_LLM_BASE_URL or ""
    ).strip() or settings.LLM_BASE_URL

    params: dict[str, Any] = {"model": model}
    if api_key:
        params["api_key"] = api_key

    is_native = _is_native_provider_id(provider_id, model)
    base_url = _resolved_base_url(provider_id, base_url)
    if base_url and not is_native:
        params["api_base"] = base_url

    if settings.LLM_TIMEOUT:
        params["timeout"] = float(settings.LLM_TIMEOUT)

    if provider_id == "ollama" and settings.OLLAMA_NUM_CTX > 0:
        params["num_ctx"] = settings.OLLAMA_NUM_CTX

    from importlib.metadata import version as _pkg_version

    try:
        _version = _pkg_version("infinidev")
    except Exception:
        _version = "0.1.0"
    params["extra_headers"] = {
        "User-Agent": f"infinidev/{_version}",
        "X-Client-Name": "infinidev-review-extractor",
        "X-Client-Version": _version,
    }

    apply_provider_transport(params, provider_id)

    return params


def get_litellm_params_for_behavior() -> dict[str, Any]:
    """Build litellm params for the behavior-checker judge.

    Each ``BEHAVIOR_LLM_*`` setting is optional and falls back to the
    matching ``LLM_*`` main setting when empty. This lets users point the
    judge at a small/fast model (e.g. ``ollama/qwen2.5:3b``) without
    affecting the main agent. Returns the same shape as
    :func:`get_litellm_params`.
    """
    model = (settings.BEHAVIOR_LLM_MODEL or "").strip() or settings.LLM_MODEL
    if not model:
        raise RuntimeError("No behavior model and no main LLM_MODEL configured.")

    if model.startswith("ollama/"):
        model = "ollama_chat/" + model[len("ollama/") :]

    provider_id = (
        settings.BEHAVIOR_LLM_PROVIDER or ""
    ).strip() or settings.LLM_PROVIDER
    api_key = (settings.BEHAVIOR_LLM_API_KEY or "").strip() or settings.LLM_API_KEY
    base_url = (settings.BEHAVIOR_LLM_BASE_URL or "").strip() or settings.LLM_BASE_URL

    params: dict[str, Any] = {"model": model}
    if api_key:
        params["api_key"] = api_key

    # Mirror the native-provider rule: only pass api_base for non-native
    # providers, otherwise litellm routes to the wrong endpoint.
    is_native = _is_native_provider_id(provider_id, model)
    base_url = _resolved_base_url(provider_id, base_url)
    if base_url and not is_native:
        params["api_base"] = base_url

    if settings.LLM_TIMEOUT:
        params["timeout"] = float(settings.LLM_TIMEOUT)

    # num_ctx only matters for Ollama-style local providers
    if provider_id == "ollama" and settings.OLLAMA_NUM_CTX > 0:
        params["num_ctx"] = settings.OLLAMA_NUM_CTX

    from importlib.metadata import version as _pkg_version

    try:
        _version = _pkg_version("infinidev")
    except Exception:
        _version = "0.1.0"
    params["extra_headers"] = {
        "User-Agent": f"infinidev/{_version}",
        "X-Client-Name": "infinidev-behavior",
        "X-Client-Version": _version,
    }

    apply_provider_transport(params, provider_id)

    return params


def get_litellm_params_for_assistant() -> dict[str, Any]:
    """Build litellm params for the assistant pair-programming critic.

    Each ``ASSISTANT_LLM_*`` setting is optional and falls back to the
    matching ``LLM_*`` main setting when empty. Designed for a setup
    where the principal runs on GPU0 and the critic runs on a second
    Ollama instance pinned to GPU1 (different ``api_base``), so both
    31B-class models stay hot in VRAM without tensor splitting.
    """
    model = (settings.ASSISTANT_LLM_MODEL or "").strip() or settings.LLM_MODEL
    if not model:
        raise RuntimeError("No assistant model and no main LLM_MODEL configured.")

    if model.startswith("ollama/"):
        model = "ollama_chat/" + model[len("ollama/") :]

    provider_id = (
        settings.ASSISTANT_LLM_PROVIDER or ""
    ).strip() or settings.LLM_PROVIDER
    api_key = (settings.ASSISTANT_LLM_API_KEY or "").strip() or settings.LLM_API_KEY
    base_url = (settings.ASSISTANT_LLM_BASE_URL or "").strip() or settings.LLM_BASE_URL

    params: dict[str, Any] = {"model": model}
    if api_key:
        params["api_key"] = api_key

    is_native = _is_native_provider_id(provider_id, model)
    base_url = _resolved_base_url(provider_id, base_url)
    if base_url and not is_native:
        params["api_base"] = base_url

    if settings.ASSISTANT_LLM_TIMEOUT:
        params["timeout"] = float(settings.ASSISTANT_LLM_TIMEOUT)

    if provider_id == "ollama" and settings.OLLAMA_NUM_CTX > 0:
        params["num_ctx"] = settings.OLLAMA_NUM_CTX

    from importlib.metadata import version as _pkg_version

    try:
        _version = _pkg_version("infinidev")
    except Exception:
        _version = "0.1.0"
    params["extra_headers"] = {
        "User-Agent": f"infinidev/{_version}",
        "X-Client-Name": "infinidev-assistant",
        "X-Client-Version": _version,
    }

    apply_provider_transport(params, provider_id)

    return params


def get_litellm_params() -> dict[str, Any]:
    """Return kwargs suitable for ``litellm.completion(**params, messages=...)``."""
    model = settings.LLM_MODEL
    if not model:
        raise RuntimeError("INFINIDEV_LLM_MODEL is not set.")

    # Auto-correct ollama/ → ollama_chat/ so the /api/chat endpoint is used
    # (ollama/ hits /api/generate which has no function-calling support).
    if model.startswith("ollama/"):
        model = "ollama_chat/" + model[len("ollama/") :]
        logger.info(
            "Auto-corrected model prefix: ollama/ → ollama_chat/ (required for tool calling)"
        )

    params: dict[str, Any] = {"model": model}

    if settings.LLM_API_KEY:
        params["api_key"] = settings.LLM_API_KEY

    if not _is_native_provider(model):
        base_url = _resolved_base_url(settings.LLM_PROVIDER, settings.LLM_BASE_URL)
        if base_url:
            params["api_base"] = base_url

    if settings.LLM_TIMEOUT:
        params["timeout"] = float(settings.LLM_TIMEOUT)

    # Retry transient provider errors (e.g. OpenRouter mid-stream
    # "Network connection lost"). LiteLLM retries APIError / Timeout /
    # RateLimitError / ServiceUnavailableError automatically.
    if settings.LLM_NUM_RETRIES > 0:
        params["num_retries"] = settings.LLM_NUM_RETRIES
        params["retry_strategy"] = "exponential_backoff_retry"

    # Pass num_ctx for Ollama to control KV cache allocation.
    # Models like gemma4 default to 262k context which hangs on consumer GPUs.
    if settings.LLM_PROVIDER == "ollama" and settings.OLLAMA_NUM_CTX > 0:
        params["num_ctx"] = settings.OLLAMA_NUM_CTX

    # Pin the developer loop temperature. Without this, local Ollama models
    # fall back to their Modelfile default (often 0.8–1.0), which destabilises
    # tool-calling JSON and mid-edit structured output. Stages that want a
    # different value (chat_agent, planner, review) use `setdefault` on top
    # of a fresh call dict, so they are unaffected. Set LLM_TEMPERATURE < 0
    # to opt out and defer to the model/provider default.
    if settings.LLM_TEMPERATURE >= 0:
        params["temperature"] = float(settings.LLM_TEMPERATURE)

    # Identify Infinidev to providers via HTTP headers.
    # Providers track client identity for analytics, rate-limit fairness,
    # and partnership eligibility.
    from importlib.metadata import version as _pkg_version

    try:
        _version = _pkg_version("infinidev")
    except Exception:
        _version = "0.1.0"
    params["extra_headers"] = {
        "User-Agent": f"infinidev/{_version}",
        "X-Client-Name": "infinidev",
        "X-Client-Version": _version,
        "anthropic-client-name": "infinidev",
        "anthropic-client-version": _version,
    }

    # Disable thinking for Qwen3+ family served via OpenAI-compatible
    # backends. Qwen3's Jinja template wraps tool calls inside
    # <think>...</think> when thinking is on, which --reasoning-format
    # deepseek then extracts into reasoning_content — trapping the
    # tool call outside the native tool_calls slot. Passing
    # chat_template_kwargs={"enable_thinking": false} per-request
    # bypasses the think block entirely and lets tool_calls emit
    # cleanly to the structured field. For an agent loop the think
    # pass is redundant anyway — plan/summarize stages already own
    # structured reasoning.
    _openai_compat = {"llama_cpp", "vllm", "openai_compatible", "gmi"}
    if settings.LLM_PROVIDER in _openai_compat and "qwen3" in model.lower():
        extra = params.setdefault("extra_body", {})
        kwargs_map = extra.setdefault("chat_template_kwargs", {})
        kwargs_map.setdefault("enable_thinking", False)

    # MiniMax M2-family (M2, M2.1, M2.5, M2.7, ...) emits reasoning as
    # <think>...</think> blocks in message.content by default. Without
    # `reasoning_split: true` the tags stay in content and the TUI
    # displays them as chat text (same class of leak we fixed for
    # Qwen). With the flag on, MiniMax's server extracts the think
    # block into `reasoning_content` so tool_calls and final text
    # remain clean. The flag is recognised by MinimaxChatConfig in
    # LiteLLM's provider layer.
    if settings.LLM_PROVIDER == "minimax":
        extra = params.setdefault("extra_body", {})
        extra.setdefault("reasoning_split", True)

    apply_provider_transport(params, settings.LLM_PROVIDER)

    return params
