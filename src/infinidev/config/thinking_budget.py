"""Thinking budget — provider-aware reasoning token limits.

Translates the user-facing THINKING_BUDGET preset into provider-specific
LLM parameters. Each provider has its own mechanism:

    Provider        | Mechanism
    ────────────────┼──────────────────────────────────────────
    Anthropic       | thinking.budget_tokens  (dedicated field)
    OpenAI (o-series)| reasoning_effort "low"/"medium"/"high"
    Gemini          | thinking_config.thinking_budget (tokens)
    DeepSeek        | max_tokens (total, includes reasoning)
    Ollama/Qwen     | /think vs /no_think prompt tag + max_tokens
    llama.cpp       | max_tokens
    vLLM            | max_tokens
    OpenRouter      | provider-dependent, uses max_tokens
    Others          | max_tokens (universal fallback)
"""

from __future__ import annotations

import logging
from typing import Any

from infinidev.config.settings import settings

logger = logging.getLogger(__name__)

# ── Preset definitions ──────────────────────────────────────────────

# Token budgets per preset.  These are *thinking* tokens, not total
# output tokens.  The actual parameter sent depends on the provider.
_PRESET_TOKENS: dict[str, int] = {
    "low": 1024,
    "medium": 4096,
    "high": 16384,
    # Reasoning tiers the Codex catalog publishes above "high". A user who
    # picks one on a subscription model and later switches provider gets the
    # deepest budget that provider understands rather than a silent drop to
    # "medium", which is where an unmapped preset used to land.
    "xhigh": 32768,
    "max": 65536,
    "ultra": 0,  # 0 = no limit
}

# OpenAI o-series models use a keyword instead of token count
_OPENAI_EFFORT: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",  # o-series tops out at high
    "max": "high",
    "ultra": "high",
}

# Reasoning depth, shallowest first. The Codex catalog names these per model;
# the order is what turns "give me the deepest this model has" into a pick.
_EFFORT_ORDER: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max", "ultra")

# Presets that mean "as deep as this model goes" rather than a fixed budget.
_UNCAPPED_PRESETS = frozenset({"ultra", "max"})


def _resolve_tokens() -> int:
    """Return the thinking budget in tokens from settings."""
    preset = settings.THINKING_BUDGET.lower().strip()
    if preset == "custom":
        return max(0, settings.THINKING_BUDGET_TOKENS)
    return _PRESET_TOKENS.get(preset, _PRESET_TOKENS["medium"])


# Thinking tokens are typically 2-5x the response tokens, so the output cap
# (thinking + response) gets headroom over the raw thinking budget.
_THINKING_HEADROOM_MULT = 3


def _local_ctx() -> int:
    """Effective local context window used to clamp output budgets."""
    return getattr(settings, "OLLAMA_NUM_CTX", 0) or 16384


def _budget_with_headroom(tokens: int, ctx: int) -> int:
    """Output-token cap (thinking + response) that cannot crowd out the prompt.

    Without clamping, a 'high' preset (16384 thinking tokens * 3) asks for
    49152 output tokens — more than an entire 16k local window, so the
    backend silently clamps to num_ctx and leaves no room for the prompt.
    When ``ctx`` is a known local window we cap the budget at ~75% of it so
    the prompt still fits; ``ctx=0`` (unknown / large hosted window) means
    no clamp.
    """
    raw = tokens * _THINKING_HEADROOM_MULT
    if ctx and ctx > 0:
        return min(raw, max(1024, int(ctx * 0.75)))
    return raw


def _subscription_provider() -> str:
    """The ChatGPT-subscription provider id (imported lazily: config.llm
    imports settings, which this module also imports)."""
    from infinidev.config.llm import CHATGPT_SUBSCRIPTION_PROVIDER

    return CHATGPT_SUBSCRIPTION_PROVIDER


def subscription_efforts(model: str | None = None) -> list[str]:
    """Reasoning levels *this* model publishes, shallowest first.

    Empty when there is no catalog to read. Exposed because the ``/effort``
    command offers exactly these, and offering a level the model does not
    have is a 400 the user has no way to predict.
    """
    from infinidev.config.codex_catalog import reasoning_levels

    slug = (model or settings.LLM_MODEL or "").rsplit("/", 1)[-1]
    supported = set(reasoning_levels(slug))
    return [level for level in _EFFORT_ORDER if level in supported]


def _subscription_effort(model: str, preset: str, tokens: int) -> str:
    """Map a thinking preset onto a reasoning effort the model accepts.

    The catalog is consulted rather than assumed: sending an effort a model
    does not list is a 400, and the levels are per-model.  With no catalog on
    disk the mapping falls back to the o-series keywords, which every GPT-5.x
    model accepts.

    Two behaviours worth knowing, both learned from the backend:

    A preset that IS one of the model's own levels is passed through
    untouched, which is what lets ``/effort max`` mean ``max`` instead of
    being rounded to the nearest generic tier.

    ``ultra`` means "the deepest this model offers" and resolves against the
    catalog rather than to a fixed keyword.  It used to hardcode ``xhigh``,
    and that is precisely the level LiteLLM refuses on ``gpt-5.6-*``: its
    version check recognises "gpt-5.4+" but not the named 5.6 variants, so
    the request died before reaching a backend that would have accepted
    ``max`` and ``ultra`` happily.
    """
    from infinidev.config.codex_catalog import reasoning_levels

    supported = reasoning_levels(model.rsplit("/", 1)[-1])

    # The user named a level this model publishes. Nothing to translate.
    if preset in supported:
        return preset

    effort = _OPENAI_EFFORT.get(preset, "medium")
    if preset == "custom":
        if tokens <= 1024:
            effort = "low"
        elif tokens <= 8192:
            effort = "medium"
        else:
            effort = "high"

    if not supported:
        return effort

    if preset in _UNCAPPED_PRESETS:
        deepest = [level for level in _EFFORT_ORDER if level in supported]
        if deepest:
            return deepest[-1]

    if effort not in supported:
        return "medium" if "medium" in supported else supported[0]
    return effort


def _is_openai_reasoning_model(model: str) -> bool:
    """Return True if the model is an OpenAI o-series reasoning model."""
    # o1, o1-mini, o1-pro, o3, o3-mini, o3-pro, o4-mini, etc.
    # Use regex word boundary to avoid matching "gpt-4o1" or "photo1".
    import re
    return bool(re.search(r'\bo[134](-|\b)', model, re.IGNORECASE))


# ── Public API ──────────────────────────────────────────────────────

def apply_thinking_budget(
    kwargs: dict[str, Any],
    provider_id: str,
    model: str,
    *,
    enabled: bool | None = None,
) -> None:
    """Mutate *kwargs* in-place to apply the thinking budget.

    Called from ``call_llm()`` after basic kwargs are assembled but
    before the LLM call is made.
    """
    # ── Master toggle ────────────────────────────────────────────
    thinking_enabled = settings.THINKING_ENABLED if enabled is None else enabled
    if not thinking_enabled:
        return _disable_thinking(kwargs, provider_id, model)

    # ── Small models: force low thinking to prevent reasoning bloat ──
    from infinidev.config.llm import _is_small_model
    if _is_small_model(model):
        preset = "low"
        tokens = _PRESET_TOKENS["low"]  # 1024
    else:
        preset = settings.THINKING_BUDGET.lower().strip()
        tokens = _resolve_tokens()

    # ── Anthropic ────────────────────────────────────────────────
    if provider_id == "anthropic":
        if preset == "ultra" or tokens == 0:
            # Let the model think freely — Anthropic's max is 128k
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 128_000}
        elif preset == "low" and tokens <= 1024:
            # Disable extended thinking entirely for low budget
            kwargs.pop("thinking", None)
        else:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": max(1024, tokens)}
        return

    # ── ChatGPT subscription (Codex) ─────────────────────────────
    # Every model the subscription serves is a GPT-5.x reasoning model, so
    # there is nothing to sniff for — but the levels differ per model, and
    # the catalog says which. That is what lets "ultra" reach xhigh where the
    # model has it instead of being flattened to high like the o-series.
    if provider_id == _subscription_provider():
        kwargs["reasoning_effort"] = _subscription_effort(model, preset, tokens)
        return

    # ── OpenAI (o-series reasoning models) ───────────────────────
    if provider_id == "openai" and _is_openai_reasoning_model(model):
        effort = _OPENAI_EFFORT.get(preset, "medium")
        if preset == "custom":
            # Map custom token count to effort keyword
            if tokens <= 1024:
                effort = "low"
            elif tokens <= 8192:
                effort = "medium"
            else:
                effort = "high"
        kwargs["reasoning_effort"] = effort
        return

    # ── Gemini ───────────────────────────────────────────────────
    if provider_id == "gemini":
        if preset == "ultra" or tokens == 0:
            kwargs["thinking"] = {"thinking_budget": -1}  # -1 = no limit
        elif preset == "low" and tokens <= 1024:
            kwargs["thinking"] = {"thinking_budget": 0}  # disable
        else:
            kwargs["thinking"] = {"thinking_budget": tokens}
        return

    # ── Ollama (Qwen3, QwQ, DeepSeek, etc.) ─────────────────────
    if provider_id == "ollama":
        # Ollama models with thinking use /think and /no_think tags
        # The tag injection is already handled in call_llm for FC mode;
        # here we handle max_tokens budget.
        if preset == "low":
            # Inject /no_think to suppress thinking entirely
            _inject_prompt_tag(kwargs, "/no_think")
        else:
            # Make sure /no_think is NOT present, allow thinking
            _remove_prompt_tag(kwargs, "/no_think")
            if preset != "ultra" and tokens > 0:
                # Output cap (thinking + response), clamped to the local
                # context window so a 'high' preset can't crowd out the prompt.
                kwargs["max_tokens"] = _budget_with_headroom(tokens, _local_ctx())
        return

    # ── DeepSeek (native provider) ───────────────────────────────
    if provider_id in ("deepseek",):
        # DeepSeek reasoning appears in reasoning_content, controlled by max_tokens
        if preset != "ultra" and tokens > 0:
            kwargs["max_tokens"] = tokens * 3  # thinking + response headroom
        return

    # MiniMax can otherwise inherit the generic 3x headroom (12,288 for the
    # default medium preset), exceeding the model metadata's documented
    # 8,192-token output ceiling. Preserve the reasoning preset while
    # enforcing the provider's finite output contract.
    if provider_id == "minimax":
        if preset != "ultra" and tokens > 0:
            kwargs["max_tokens"] = min(tokens * 3, 8_192)
        return

    # ── llama.cpp / vLLM / OpenAI-compatible / OpenRouter / GMI ──
    if provider_id in ("llama_cpp", "vllm", "openai_compatible", "openrouter", "gmi"):
        if preset == "low":
            _inject_prompt_tag(kwargs, "/no_think")
            if tokens > 0:
                kwargs["max_tokens"] = tokens * 2
        elif preset != "ultra" and tokens > 0:
            _remove_prompt_tag(kwargs, "/no_think")
            kwargs["max_tokens"] = _budget_with_headroom(tokens, _local_ctx())
        else:
            _remove_prompt_tag(kwargs, "/no_think")
        return

    # ── Fallback for unknown providers ───────────────────────────
    if preset != "ultra" and tokens > 0:
        kwargs["max_tokens"] = _budget_with_headroom(tokens, _local_ctx())


# ── Prompt tag helpers ──────────────────────────────────────────────

def _inject_prompt_tag(kwargs: dict[str, Any], tag: str) -> None:
    """Inject a tag (e.g. /no_think) into the last user message."""
    msgs = kwargs.get("messages")
    if not msgs:
        return
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            content = msgs[i].get("content", "")
            if tag not in content:
                msgs[i] = {**msgs[i], "content": tag + "\n" + content}
            return


def _remove_prompt_tag(kwargs: dict[str, Any], tag: str) -> None:
    """Remove a prompt tag from user messages if present."""
    msgs = kwargs.get("messages")
    if not msgs:
        return
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            content = msgs[i].get("content", "")
            if tag in content:
                msgs[i] = {**msgs[i], "content": content.replace(tag + "\n", "").replace(tag, "")}
            return


def _disable_thinking(
    kwargs: dict[str, Any], provider_id: str, model: str,
) -> None:
    """Disable thinking/reasoning entirely for all providers.

    Called when ``THINKING_ENABLED=False``.
    """
    # Anthropic: don't send thinking param at all
    if provider_id == "anthropic":
        kwargs.pop("thinking", None)
        return

    # OpenAI o-series and the ChatGPT subscription: reasoning cannot be turned
    # off on these models, so "low" is as close to off as the API goes.
    if provider_id == _subscription_provider():
        kwargs["reasoning_effort"] = "low"
        return
    if provider_id == "openai" and _is_openai_reasoning_model(model):
        kwargs["reasoning_effort"] = "low"
        return

    # Gemini: set budget to 0 (disabled)
    if provider_id == "gemini":
        kwargs["thinking"] = {"thinking_budget": 0}
        return

    # Z.AI GLM-4.5+ exposes an explicit per-request switch. Unlike a prompt
    # tag, this prevents the provider from generating hidden reasoning tokens
    # and is therefore important for latency-sensitive helper calls.
    if provider_id in ("zai", "zai_coding"):
        extra_body = kwargs.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
            kwargs["extra_body"] = extra_body
        extra_body["thinking"] = {"type": "disabled"}
        return

    # Ollama / llama.cpp / vLLM / OpenRouter / compatible / others:
    # inject /no_think prompt tag + cap max_tokens so the model can't
    # spend unlimited tokens on reasoning if it ignores the tag.
    _inject_prompt_tag(kwargs, "/no_think")
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 4096
