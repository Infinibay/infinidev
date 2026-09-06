"""Model-specific reasoning controls shared by requests and the effort command.

Contracts checked against provider documentation on 2026-09-06. See
docs/model-support.md for sources and the distinction between API levels and
local budget presets. Unknown routes never inherit another provider's API.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from infinidev.config.settings import settings

_ORDER = ("off", "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
_FIVE = ("low", "medium", "high", "xhigh", "max")
_THREE = ("low", "medium", "high")
_BUDGETS = {"low": 1024, "medium": 4096, "high": 16384}
_CLAUDE_ADAPTIVE = {
    "claude-fable-5-1", "claude-mythos-5-1", "claude-fable-5", "claude-mythos-5",
    "claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7",
    "claude-opus-4-6", "claude-sonnet-4-6",
}


@dataclass(frozen=True)
class EffortProfile:
    """A route's accepted controls, not a guessed measure of intelligence."""

    choices: tuple[str, ...] = ()
    mechanism: str = "unsupported"
    description: str = "No verified reasoning control for this model and API"
    always_on: bool = False


def effort_profile(provider: str | None = None, model: str | None = None) -> EffortProfile:
    provider = provider or settings.LLM_PROVIDER
    slug = (model or settings.LLM_MODEL).rsplit("/", 1)[-1].lower()
    if provider == "openai_subscription":
        from infinidev.config.codex_catalog import reasoning_levels

        levels = reasoning_levels(slug)
        if levels:
            return EffortProfile(tuple(level for level in _ORDER if level in levels),
                                 "openai", "Live Codex model catalog",
                                 always_on=not bool({"none", "off"} & set(levels)))
        return EffortProfile(_THREE, "openai", "Offline Codex fallback; refresh Codex's catalog "
                             "to discover additional levels", True)
    if provider == "openai":
        slug = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", slug)
        if slug == "gpt-6-astra":
            return EffortProfile(_FIVE, "openai", "reasoning.effort (Responses)", True)
        if slug in {"gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}:
            return EffortProfile(("none", *_FIVE), "openai", "reasoning.effort (Responses)")
        if slug == "gpt-5.5-pro":
            return EffortProfile(("medium", "high", "xhigh"), "openai",
                                 "reasoning.effort (Responses)", True)
        if slug in {"gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.2"}:
            return EffortProfile(("none", "low", "medium", "high", "xhigh"),
                                 "openai", "reasoning_effort")
        if slug == "gpt-5.1":
            return EffortProfile(("none", *_THREE), "openai", "reasoning_effort")
        if slug in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
            return EffortProfile(("minimal", *_THREE), "openai", "reasoning_effort", True)
        if slug in {"o1", "o3", "o3-mini", "o4-mini"}:
            return EffortProfile(_THREE, "openai", "reasoning_effort", True)
    if provider == "anthropic":
        slug = re.sub(r"-\d{8}$", "", slug)
        if slug in _CLAUDE_ADAPTIVE:
            levels = _FIVE if slug not in {"claude-opus-4-6", "claude-sonnet-4-6"} else (
                "low", "medium", "high", "max")
            always = slug.startswith(("claude-fable-", "claude-mythos-"))
            return EffortProfile(levels if always else ("off", *levels),
                                 "anthropic_adaptive", "output_config.effort", always)
        if slug == "claude-opus-4-5":
            return EffortProfile(("off", *_THREE), "anthropic_manual_effort",
                                 "output_config.effort + thinking.budget_tokens")
        if slug in {"claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5",
                    "claude-sonnet-4", "claude-opus-4", "claude-opus-4-1",
                    "claude-3-7-sonnet", "claude-3-7-sonnet-latest"}:
            return EffortProfile(("off", "low", "medium", "high", "custom"),
                                 "anthropic_budget", "thinking.budget_tokens (budget presets)")
    if provider in {"zai", "zai_coding"}:
        if slug in {"glm-5.3", "glm-5.3-flash"}:
            return EffortProfile(("low", "high", "max"), "zai_effort",
                                 "reasoning_effort; thinking is always enabled", True)
        if slug == "glm-5.2":
            return EffortProfile(("off", "high", "max"), "zai_effort",
                                 "reasoning_effort + thinking.type")
        if slug.startswith(("glm-5", "glm-4.7", "glm-4.6", "glm-4.5")):
            return EffortProfile(("off", "on"), "zai_toggle", "thinking.type")
    if provider in {"qwen", "qwen_subscription"}:
        if slug.startswith("qwen3.8"):
            always = "2.4t" in slug
            levels = ("low", "medium", "xhigh")
            return EffortProfile(levels if always else ("off", *levels), "qwen_effort",
                                 "extra_body.reasoning_effort + enable_thinking", always)
        if slug.startswith(("qwen3.7", "qwen3.6", "qwen3.5", "qwen3-max", "qwen3-",
                            "qwen-plus", "qwen-flash", "qwen-turbo", "qwq-plus")):
            if "coder" not in slug and "instruct" not in slug:
                always = "thinking" in slug or slug in {
                    "qwq-plus", "qwen3.7-max-preview", "qwen3.7-max-2026-05-17",
                }
                levels = ("low", "medium", "high", "custom")
                return EffortProfile(levels if always else ("off", *levels), "qwen_budget",
                                     "thinking_budget (budget presets)", always)
    if provider == "ollama":
        if "gpt-oss" in slug:
            return EffortProfile(_THREE, "ollama_effort", "think", True)
        if ("coder" not in slug and "instruct" not in slug
                and any(n in slug for n in ("qwen3", "deepseek-r1", "deepseek-v3.1"))):
            return EffortProfile(("off", "on"), "ollama_toggle", "think (boolean)")
    if provider in {"llama_cpp", "vllm"}:
        if "gpt-oss" in slug:
            return EffortProfile(_THREE, "template_effort", "chat_template_kwargs.reasoning_effort",
                                 True)
        if "qwen3" in slug and "coder" not in slug and "instruct" not in slug:
            return EffortProfile(("off", "on"), "template_toggle",
                                 "chat_template_kwargs.enable_thinking")
    if provider == "gemini":
        if slug.startswith("gemini-3"):
            levels = (("low", "high") if slug == "gemini-3-pro-preview" else
                      _THREE if slug in {"gemini-3.1-pro-preview", "gemini-3.7-flash",
                                         "gemini-3.8-flash"} else ("minimal", *_THREE))
            return EffortProfile(levels, "gemini_level", "thinkingConfig.thinkingLevel", True)
        if slug.startswith("gemini-2.5"):
            levels = ("low", "medium", "high", "custom")
            always = "pro" in slug
            return EffortProfile(levels if always else ("off", *levels), "gemini_budget",
                                 "thinkingConfig.thinkingBudget (budget presets)", always)
    if provider == "deepseek" and slug.startswith("deepseek-v4"):
        return EffortProfile(("off", "low", "high", "max"), "deepseek_effort",
                             "reasoning_effort + thinking.type")
    if provider == "mistral" and slug in {"mistral-small-latest", "mistral-medium-3-5"}:
        return EffortProfile(("none", "high"), "mistral_effort", "reasoning_effort")
    if provider == "openrouter":
        from infinidev.config.openrouter_reasoning import model_reasoning

        model_id = (model or settings.LLM_MODEL).removeprefix("openrouter/")
        metadata = model_reasoning(model_id)
        if metadata is None:
            return EffortProfile(description="OpenRouter reasoning metadata unavailable; "
                                 "no effort selection advertised")
        mandatory = metadata.get("mandatory") is True
        levels = metadata.get("supported_efforts", [])
        levels = _ORDER[1:-1] if levels is None else levels
        choices = tuple(v for v in _ORDER[1:-1] if v in levels
                        and (v != "none" or not mandatory))
        if choices and not mandatory and "none" not in choices:
            choices = ("none", *choices)
        if metadata.get("supports_max_tokens"):
            choices += ("custom",)
        if not choices:
            choices = ("on",) if mandatory else ("off", "on")
        return EffortProfile(choices, "openrouter", "Live OpenRouter reasoning metadata", mandatory)
    return EffortProfile()


def resolve_effort(profile: EffortProfile, requested: str, *, enabled: bool = True) -> str:
    """Map saved preferences across model changes; commands validate exact choices."""
    choices = profile.choices
    if not choices:
        return ""
    if not enabled or requested in {"off", "none"}:
        return next((v for v in ("off", "none", "minimal", "low") if v in choices), choices[0])
    if requested in choices:
        return requested
    if requested in {"ultra", "max"}:
        return next((v for v in reversed(_ORDER) if v in choices), choices[-1])
    if "on" in choices:
        return "on"
    if requested == "custom":
        tokens = settings.THINKING_BUDGET_TOKENS
        requested = "low" if tokens <= 1024 else "medium" if tokens <= 8192 else "high"
    rank = _ORDER.index(requested) if requested in _ORDER else _ORDER.index("medium")
    # GLM has no medium and Qwen has no high. Move to the next supported
    # level instead of sending an invalid value or claiming it was honored.
    return next((v for v in _ORDER[rank:] if v in choices), choices[-1])


def apply_reasoning(kwargs: dict[str, Any], provider: str, model: str,
                    *, enabled: bool | None = None) -> bool:
    """Apply a verified contract; return False when legacy handling owns this route."""
    profile = effort_profile(provider, model)
    if not profile.choices:
        return False
    requested = settings.THINKING_BUDGET.lower().strip()
    enabled = settings.THINKING_ENABLED if enabled is None else enabled
    value = resolve_effort(profile, requested, enabled=enabled)
    off = value in {"off", "none"}
    mechanism = profile.mechanism
    tokens = (settings.THINKING_BUDGET_TOKENS if value == "custom" else
              _BUDGETS.get(value, 16384))
    if mechanism == "openai":
        kwargs["reasoning_effort"] = value
    elif mechanism in {"anthropic_adaptive", "anthropic_manual_effort"}:
        kwargs.pop("reasoning_effort", None)
        extra = kwargs.setdefault("extra_body", {})
        extra["output_config"] = {**extra.get("output_config", {}),
                                  "effort": "low" if off else value}
        kwargs["thinking"] = {"type": "disabled" if off else "adaptive"}
        if mechanism == "anthropic_manual_effort" and not off:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": tokens}
            kwargs["max_tokens"] = max(kwargs.get("max_tokens") or 64000, tokens + 1024)
    elif mechanism == "anthropic_budget":
        if off:
            kwargs["thinking"] = {"type": "disabled"}
        else:
            budget = max(1024, min(tokens, 63000))
            output = kwargs.get("max_tokens") or 64000
            kwargs["max_tokens"] = max(output, budget + 1024)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    elif mechanism in {"zai_effort", "zai_toggle", "deepseek_effort"}:
        kwargs.pop("thinking", None)
        kwargs.pop("reasoning_effort", None)
        extra = kwargs.setdefault("extra_body", {})
        extra["thinking"] = {"type": "disabled" if off else "enabled"}
        if mechanism in {"zai_effort", "deepseek_effort"}:
            if off:
                extra.pop("reasoning_effort", None)
            else:
                extra["reasoning_effort"] = value
    elif mechanism in {"qwen_effort", "qwen_budget"}:
        kwargs.pop("reasoning_effort", None)
        extra = kwargs.setdefault("extra_body", {})
        extra["enable_thinking"] = not off
        extra.pop("thinking_budget", None)
        extra.pop("reasoning_effort", None)
        if not off:
            if mechanism == "qwen_effort":
                extra["reasoning_effort"] = value
            else:
                extra["thinking_budget"] = max(1, tokens)
    elif mechanism.startswith("ollama_"):
        # LiteLLM maps this public argument to the native top-level `think`.
        kwargs["reasoning_effort"] = value if mechanism == "ollama_effort" else (
            "none" if off else "high")
    elif mechanism.startswith("template_"):
        template = kwargs.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})
        if mechanism == "template_effort":
            template["reasoning_effort"] = value
        else:
            template["enable_thinking"] = not off
    elif mechanism == "gemini_level":
        kwargs.pop("thinking", None)
        kwargs["reasoning_effort"] = value
    elif mechanism == "gemini_budget":
        ceiling = 32768 if profile.always_on else 24576
        budget = 0 if off else max(128 if profile.always_on else 512, min(tokens, ceiling))
        kwargs["thinking"] = {"thinking_budget": budget}
    elif mechanism == "mistral_effort":
        kwargs["reasoning_effort"] = value
    elif mechanism == "openrouter":
        reasoning = {"enabled": not off}
        if value == "custom":
            reasoning["max_tokens"] = max(1024, tokens)
        elif value not in {"off", "on", "none"}:
            reasoning["effort"] = value
        kwargs.setdefault("extra_body", {})["reasoning"] = reasoning
    return True


def effort_choices() -> tuple[list[str], bool]:
    profile = effort_profile()
    return list(profile.choices), bool(profile.choices and "presets" not in profile.description)


def effort_listing() -> str:
    """Render one shared CLI/TUI view of the selected route's controls."""
    profile = effort_profile()
    lines = [f"Reasoning effort — {settings.LLM_PROVIDER} / {settings.LLM_MODEL}",
             profile.description]
    current = resolve_effort(profile, settings.THINKING_BUDGET,
                             enabled=settings.THINKING_ENABLED)
    lines.extend(f"  {'>' if value == current else ' '} {value}" for value in profile.choices)
    if profile.choices:
        if current != settings.THINKING_BUDGET:
            lines.append(f"Saved preference: {settings.THINKING_BUDGET}; resolved: {current}")
        lines.extend(["", f"In effect now: {effort_in_effect()}",
                      "Change it with /effort <level>"])
    return "\n".join(lines)


def normalize_model_request(params: dict[str, Any], *, bridge: bool = False) -> None:
    """Repair protocol constraints at the final boundary, including direct callers."""
    model = str(params.get("model", ""))
    slug = model.rsplit("/", 1)[-1]
    if model.startswith("openai/"):
        if slug == "gpt-6-astra" or slug.startswith(("gpt-5.6", "gpt-5.5")):
            params["model"] = f"openai/responses/{slug}"
            extra = params.setdefault("extra_body", {})
            extra.setdefault("store", False)
            included = list(extra.get("include", params.get("include", [])))
            if "reasoning.encrypted_content" not in included:
                included.append("reasoning.encrypted_content")
            extra["include"] = included
        if slug == "gpt-6-astra" or (
            slug.startswith("gpt-5") and params.get("reasoning_effort") != "none"
        ):
            for key in ("temperature", "top_p", "top_logprobs", "logprobs"):
                params.pop(key, None)
        if slug == "gpt-6-astra":
            extra = params.setdefault("extra_body", {})
            for body in (params, extra):
                if "prompt_cache_retention" in body:
                    body.pop("prompt_cache_retention")
                    extra.setdefault("prompt_cache_options", {"ttl": "30m"})
                if isinstance(body.get("include"), list):
                    body["include"] = [v for v in body["include"]
                                       if v != "message.output_text.logprobs"]
        if bridge and "/responses/" in params["model"]:
            effort = params.get("reasoning_effort")
            # The locked SDK silently drops unrecognized string levels such
            # as max. Its documented dict path preserves the provider value.
            if isinstance(effort, str):
                params["reasoning_effort"] = {"effort": effort}
    if model.startswith("anthropic/"):
        profile = effort_profile("anthropic", model)
        response_format = params.get("response_format")
        if (profile.mechanism == "anthropic_adaptive" and isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"):
            from litellm.llms.anthropic.chat.transformation import AnthropicConfig

            output_format = AnthropicConfig().map_response_format_to_anthropic_output_format(
                response_format,
            )
            if output_format is not None:
                params.pop("response_format")
                output = params.setdefault("extra_body", {}).setdefault("output_config", {})
                output["format"] = output_format
        thinking = params.get("thinking", {})
        active = thinking.get("type") in {"adaptive", "enabled"}
        if profile.always_on:
            params["thinking"] = {"type": "adaptive"}
            active = True
        if active:
            # Extended/adaptive thinking rejects forced tool choice. New
            # models also reject sampling overrides while thinking is active.
            choice = params.get("tool_choice")
            if isinstance(choice, dict) or choice in {"required", "any"}:
                params["tool_choice"] = "auto"
        if active or slug in _CLAUDE_ADAPTIVE - {"claude-opus-4-6", "claude-sonnet-4-6"}:
            for key in ("temperature", "top_p", "top_k"):
                params.pop(key, None)


def effort_in_effect() -> str:
    """Describe the same parameters the next request will receive, including nested bodies."""
    profile = effort_profile()
    if not profile.choices:
        return profile.description
    probe: dict[str, Any] = {}
    apply_reasoning(probe, settings.LLM_PROVIDER, settings.LLM_MODEL)
    value = resolve_effort(profile, settings.THINKING_BUDGET,
                           enabled=settings.THINKING_ENABLED)
    suffix = " (reasoning cannot be disabled)" if profile.always_on else ""
    return f"{value}: {json.dumps(probe, sort_keys=True)}{suffix}"
