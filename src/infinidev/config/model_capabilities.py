"""Model capability detection via runtime probing.

Runs lightweight test calls at startup to determine what the configured LLM
actually supports (function calling, tool_choice=required, JSON mode, etc.)
rather than relying on hardcoded provider lists.

For Ollama models, the ``/api/show`` endpoint is queried first: if the model
template lacks tool-calling markers (``.ToolCalls``, ``if .Tools``), the model
is assumed to lack native function-calling support and the system falls back
to manual (text-based) tool calling without ever sending a slow probe request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CapabilityStatus(StrEnum):
    """Conservative support state for a model capability."""

    SUPPORTED = "supported"
    ADVERTISED_UNVERIFIED = "advertised_unverified"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CapabilityEvidence:
    """A reason for a capability decision, suitable for diagnostics."""

    source: str
    detail: str


@dataclass(frozen=True)
class CapabilityAssessment:
    """Immutable status, evidence, and operational restrictions for one capability."""

    status: CapabilityStatus = CapabilityStatus.UNKNOWN
    evidence: tuple[CapabilityEvidence, ...] = ()
    restrictions: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        """Return True only for affirmative evidence; unknown fails closed."""
        return self.status is CapabilityStatus.SUPPORTED


@dataclass(frozen=True)
class ModelRoute:
    """Stable identity of one configured chat-provider route."""

    provider: str
    model: str
    base_url: str = ""

    @classmethod
    def from_settings(cls) -> ModelRoute:
        """Build the current chat route without exposing credentials."""
        from infinidev.config.settings import settings

        return cls(
            provider=settings.LLM_PROVIDER,
            model=settings.LLM_MODEL,
            base_url=(settings.LLM_BASE_URL or "").rstrip("/"),
        )


@dataclass(frozen=True)
class ImageGenerationRoute:
    """Complete non-secret identity of one reviewed image-generation route."""

    provider: str
    model: str
    endpoint: str
    transport: str
    adapter: str
    mechanism: str
    operation: str
    revision: str
    credential_type: str
    account_id: str = ""
    project_id: str = ""
    credential_id: str = ""

    @property
    def base_url(self) -> str:
        """Compatibility alias for provider adapters expecting a base URL."""
        return self.endpoint


_OPENAI_IMAGES_ENDPOINT = "https://api.openai.com/v1"
_OPENAI_IMAGES_TRANSPORT = "https"
_OPENAI_IMAGES_ADAPTER = "litellm.image_generation"
_OPENAI_IMAGES_MECHANISM = "openai_images_api"
_OPENAI_IMAGES_OPERATION = "images.generate"
_OPENAI_IMAGES_REVISION = "2025-04-01"
_OPENAI_IMAGES_CREDENTIAL_TYPE = "api_key"


def _credential_id(api_key: str) -> str:
    """Return a stable non-secret identifier for one credential."""
    if not api_key:
        return ""
    return f"sha256:{hashlib.sha256(api_key.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class ImageGenerationProfile:
    """Versioned, allow-listed contract for one exact generation route."""

    version: int
    provider: str
    model: str
    adapter: str
    response_formats: tuple[str, ...]
    sizes: tuple[str, ...] = ()
    qualities: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()
    max_images: int = 1


@dataclass(frozen=True)
class CapabilitySnapshot:
    """Immutable multimodal capabilities for one exact route/configuration."""

    route: ModelRoute
    image_input: CapabilityAssessment
    image_generation: CapabilityAssessment
    generation_profile: ImageGenerationProfile | None = None
    generation_route: ImageGenerationRoute | None = None

    @property
    def supports_vision(self) -> bool:
        """Temporary compatibility alias; unknown deliberately maps to False."""
        return self.image_input.supported


# Generation is never inferred from a chat model name. Each entry represents a
# reviewed API contract and is keyed by the independently configured route.
def _bare_model(model: str) -> str:
    """Strip a transport prefix without guessing model families."""
    return model.rsplit("/", 1)[-1]


def _is_reviewed_vision_model(model: str) -> bool:
    """Return whether an exact model has reviewed image-input support."""
    return _bare_model(model) == "gpt-5.6-sol"


def _generation_route_from_settings() -> ImageGenerationRoute | None:
    """Return the complete route identity when generation was explicitly configured."""
    from infinidev.config.settings import settings

    provider = settings.IMAGE_GENERATION_PROVIDER.strip()
    model = settings.IMAGE_GENERATION_MODEL.strip()
    api_key = settings.IMAGE_GENERATION_API_KEY.strip()
    base_url = settings.IMAGE_GENERATION_BASE_URL.strip()
    account_id = settings.IMAGE_GENERATION_ACCOUNT_ID.strip()
    project_id = settings.IMAGE_GENERATION_PROJECT_ID.strip()
    if not any((provider, model, api_key, base_url, account_id, project_id)):
        return None
    return ImageGenerationRoute(
        provider=provider,
        model=model,
        endpoint=(base_url or _OPENAI_IMAGES_ENDPOINT).rstrip("/"),
        transport=settings.IMAGE_GENERATION_TRANSPORT.strip(),
        adapter=settings.IMAGE_GENERATION_ADAPTER.strip(),
        mechanism=settings.IMAGE_GENERATION_MECHANISM.strip(),
        operation=settings.IMAGE_GENERATION_OPERATION.strip(),
        revision=settings.IMAGE_GENERATION_REVISION.strip(),
        credential_type=_OPENAI_IMAGES_CREDENTIAL_TYPE if api_key else "",
        account_id=account_id,
        project_id=project_id,
        credential_id=_credential_id(api_key),
    )


def _generation_profile_for_route(
    route: ImageGenerationRoute,
) -> ImageGenerationProfile | None:
    """Match only the reviewed public OpenAI Images API contract."""
    profile = IMAGE_GENERATION_PROFILES.get((route.provider, _bare_model(route.model)))
    if profile is None:
        return None
    expected = (
        route.provider == "openai",
        route.model == profile.model,
        route.endpoint == _OPENAI_IMAGES_ENDPOINT,
        route.transport == _OPENAI_IMAGES_TRANSPORT,
        route.adapter == _OPENAI_IMAGES_ADAPTER == profile.adapter,
        route.mechanism == _OPENAI_IMAGES_MECHANISM,
        route.operation == _OPENAI_IMAGES_OPERATION,
        route.revision == _OPENAI_IMAGES_REVISION,
        route.credential_type == _OPENAI_IMAGES_CREDENTIAL_TYPE,
        bool(route.credential_id),
    )
    return profile if all(expected) else None


def _metadata_image_input_assessment(route: ModelRoute) -> CapabilityAssessment:
    """Resolve image input from reviewed overrides and LiteLLM static metadata."""
    if _is_reviewed_vision_model(route.model):
        return CapabilityAssessment(
            status=CapabilityStatus.SUPPORTED,
            evidence=(CapabilityEvidence(
                "reviewed-model-override",
                "Exact model review confirms image input support.",
            ),),
        )

    try:
        import litellm

        supported = bool(litellm.supports_vision(model=route.model))
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        return CapabilityAssessment(
            evidence=(CapabilityEvidence("litellm-metadata", type(exc).__name__),)
        )
    except Exception as exc:
        logger.debug("LiteLLM vision metadata unavailable: %s", str(exc)[:120])
        return CapabilityAssessment(
            evidence=(CapabilityEvidence("litellm-metadata", type(exc).__name__),)
        )
    return CapabilityAssessment(
        status=(CapabilityStatus.SUPPORTED if supported else CapabilityStatus.UNSUPPORTED),
        evidence=(CapabilityEvidence(
            "litellm-metadata",
            "Static metadata reports image input support."
            if supported else "Static metadata reports no image input support.",
        ),),
    )


def _local_image_input_assessment(route: ModelRoute) -> CapabilityAssessment:
    """Use authoritative local server inspection where an endpoint exposes it."""
    from infinidev.config.settings import settings

    current = ModelRoute.from_settings()
    if route != current:
        return CapabilityAssessment()
    if route.provider == "ollama":
        caps = _detect_ollama_capabilities()
        return CapabilityAssessment(
            status=(
                CapabilityStatus.SUPPORTED
                if caps.supports_vision else CapabilityStatus.UNSUPPORTED
            ),
            evidence=(CapabilityEvidence("ollama-api-show", "Local model manifest."),),
        )
    if route.provider == "llama_cpp":
        supported = _detect_llama_cpp_vision()
        return CapabilityAssessment(
            status=(CapabilityStatus.SUPPORTED if supported else CapabilityStatus.UNSUPPORTED),
            evidence=(CapabilityEvidence("llama-cpp-props", "Local server properties."),),
        )
    return CapabilityAssessment()


IMAGE_GENERATION_PROFILES: dict[tuple[str, str], ImageGenerationProfile] = {
    ("openai", "gpt-image-1"): ImageGenerationProfile(
        version=1,
        provider="openai",
        model="gpt-image-1",
        adapter="litellm.image_generation",
        response_formats=("b64_json", "url"),
        sizes=("1024x1024", "1024x1536", "1536x1024"),
        qualities=("low", "medium", "high"),
        max_images=10,
    ),
    ("openai", "gpt-image-1-mini"): ImageGenerationProfile(
        version=1,
        provider="openai",
        model="gpt-image-1-mini",
        adapter="litellm.image_generation",
        response_formats=("b64_json", "url"),
        sizes=("1024x1024", "1024x1536", "1536x1024"),
        qualities=("low", "medium", "high"),
        max_images=10,
    ),
}


class CapabilityResolver:
    """Resolve and cache conservative multimodal snapshots per exact route."""

    def __init__(
        self,
        metadata_detector: Callable[[ModelRoute], CapabilityAssessment] | None = None,
        local_detector: Callable[[ModelRoute], CapabilityAssessment] | None = None,
    ) -> None:
        self._metadata_detector = metadata_detector or _metadata_image_input_assessment
        self._local_detector = local_detector or _local_image_input_assessment
        self._cache: dict[
            tuple[ModelRoute, ImageGenerationRoute | None], CapabilitySnapshot
        ] = {}
        self._lock = RLock()

    def resolve(
        self,
        route: ModelRoute | None = None,
        generation_route: ImageGenerationRoute | None = None,
    ) -> CapabilitySnapshot:
        """Return a snapshot keyed by the complete non-secret route identity."""
        chat_route = route or ModelRoute.from_settings()
        if generation_route is None:
            generation_route = _generation_route_from_settings()
        key = (chat_route, generation_route)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        local = self._local_detector(chat_route)
        image_input = local
        if local.status is CapabilityStatus.UNKNOWN:
            image_input = self._metadata_detector(chat_route)

        profile = None
        generation = CapabilityAssessment()
        if generation_route is not None:
            profile = _generation_profile_for_route(generation_route)
            if profile is None:
                generation = CapabilityAssessment(
                    status=CapabilityStatus.UNSUPPORTED,
                    evidence=(CapabilityEvidence(
                        "exact-profile-registry",
                        "The configured route does not exactly match the reviewed "
                        "OpenAI Images API contract with an API key.",
                    ),),
                )
            else:
                generation = CapabilityAssessment(
                    status=CapabilityStatus.SUPPORTED,
                    evidence=(CapabilityEvidence(
                        "exact-profile-registry",
                        f"Matched generation profile v{profile.version}.",
                    ),),
                    restrictions=(
                        f"adapter={generation_route.adapter}",
                        f"mechanism={generation_route.mechanism}",
                        f"operation={generation_route.operation}",
                        f"revision={generation_route.revision}",
                    ),
                )
        elif chat_route.provider == "openai_subscription":
            generation = CapabilityAssessment(
                status=CapabilityStatus.UNSUPPORTED,
                evidence=(CapabilityEvidence(
                    "codex-product-boundary",
                    "Codex catalog metadata is not authorization for the public "
                    "OpenAI Images API.",
                ),),
            )

        snapshot = CapabilitySnapshot(
            route=chat_route,
            image_input=image_input,
            image_generation=generation,
            generation_profile=profile,
            generation_route=generation_route,
        )
        with self._lock:
            self._cache[key] = snapshot
        return snapshot

    def invalidate(
        self,
        route: ModelRoute | ImageGenerationRoute | None = None,
    ) -> None:
        """Invalidate every snapshot, or snapshots involving one exact identity."""
        with self._lock:
            if route is None:
                self._cache.clear()
                return
            self._cache = {
                key: value for key, value in self._cache.items() if route not in key
            }


# Timeout (seconds) for probe calls — much shorter than the general LLM
# timeout so a failing probe doesn't block the session for minutes.
_PROBE_TIMEOUT = 30


@dataclass
class ModelCapabilities:
    """Runtime-detected capabilities of the configured LLM."""

    supports_function_calling: bool = True
    supports_tool_choice_required: bool = True
    supports_json_mode: bool = True
    supports_vision: bool = False
    has_thinking_sections: bool = False
    needs_schema_sanitization: bool = False
    probed: bool = False
    probe_duration: float = 0.0


# Module-level singletons: legacy probe capabilities and public multimodal resolver.
_capabilities = ModelCapabilities()
capability_resolver = CapabilityResolver()


def get_capability_snapshot(
    route: ModelRoute | None = None,
    generation_route: ModelRoute | None = None,
) -> CapabilitySnapshot:
    """Return the public immutable capability snapshot for configured routes."""
    return capability_resolver.resolve(route, generation_route)

# Tiny tool used for probing — minimal tokens, clear expected behavior
_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic. Return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate",
                },
            },
            "required": ["expression"],
        },
    },
}

# Tool with anyOf in schema — tests schema sanitization needs
_PROBE_TOOL_ANYOF = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic. Return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate",
                },
                "precision": {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "null"},
                    ],
                    "description": "Decimal places (optional)",
                },
            },
            "required": ["expression"],
        },
    },
}

_PROBE_MESSAGES = [
    {"role": "user", "content": "Calculate 2+2. Use the calculator tool."},
]

# Thinking section markers emitted by various models
_THINKING_MARKERS = ("<thinking>", "<|thinking|>", "<think>")


# Known capability presets per provider — skip probing for these
_PROVIDER_PRESETS: dict[str, ModelCapabilities] = {
    "openai": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=True,
        supports_json_mode=True,
        probed=True,
    ),
    "openai_subscription": ModelCapabilities(
        # Same GPT-5.x models as the metered API, reached over the Responses
        # API. A preset rather than a probe because probing would burn a
        # subscription request at every startup to re-learn a constant.
        supports_function_calling=True,
        supports_tool_choice_required=True,
        supports_json_mode=True,
        probed=True,
    ),
    "anthropic": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=True,
        supports_json_mode=True,
        probed=True,
    ),
    "gemini": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=True,
        supports_json_mode=True,
        probed=True,
    ),
    "zai": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=False,
        supports_json_mode=True,
        probed=True,
    ),
    "kimi": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=False,
        supports_json_mode=True,
        probed=True,
    ),
    "minimax": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=False,
        supports_json_mode=True,
        has_thinking_sections=True,  # MiniMax M2.7 sends reasoning_content like DeepSeek
        probed=True,
    ),
    "mistral": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=True,
        supports_json_mode=True,
        probed=True,
    ),
    "openrouter": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=False,  # varies by model — be conservative
        supports_json_mode=False,             # varies by model — be conservative
        probed=True,
    ),
    "qwen": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=True,
        supports_json_mode=True,
        needs_schema_sanitization=True,  # Qwen rejects anyOf and complex schemas
        probed=True,
    ),
    "qwen_subscription": ModelCapabilities(
        supports_function_calling=True,
        supports_tool_choice_required=True,
        supports_json_mode=True,
        needs_schema_sanitization=True,
        probed=True,
    ),
    "gmi": ModelCapabilities(
        # GMI Cloud serves a mixed catalog (DeepSeek, Qwen, Kimi, GLM,
        # Llama, ...) behind one OpenAI-compatible endpoint, so — like
        # OpenRouter — be conservative on the per-model-variable features
        # instead of probing (which would cost a live call + a valid key
        # at startup). FC itself is supported across the catalog.
        supports_function_calling=True,
        supports_tool_choice_required=False,  # varies by model
        supports_json_mode=False,             # varies by model
        probed=True,
    ),
}


def get_model_capabilities() -> ModelCapabilities:
    """Return model capabilities. Uses provider presets for known cloud providers.

    For Ollama, queries the ``/api/show`` endpoint to check if the model
    template includes tool-calling markers.  This is fast (~5 ms local) and
    avoids a 30-second probe that would hang for unsupported models.
    """
    global _capabilities
    if not _capabilities.probed:
        try:
            from infinidev.config.settings import settings
            provider_id = getattr(settings, "LLM_PROVIDER", "ollama")
            if provider_id in _PROVIDER_PRESETS:
                # Copy the preset so runtime fields (supports_vision) don't
                # pollute the shared dict.
                _capabilities = replace(_PROVIDER_PRESETS[provider_id])
            elif provider_id == "ollama":
                _capabilities = _detect_ollama_capabilities()
            elif provider_id in ("openai_compatible", "llama_cpp", "vllm"):
                _capabilities = _detect_openai_compatible_capabilities()
            # Only fall back to LiteLLM's static table if the provider-specific
            # branch didn't already determine vision support.
            if not _capabilities.supports_vision:
                _capabilities.supports_vision = _detect_vision_support()
        except Exception as exc:
            # One-shot startup probe: a real bug here (renamed settings field,
            # NameError, ImportError) would silently degrade every model to
            # defaults. Log it loudly instead of swallowing.
            logger.warning(
                "capability detection failed; using defaults: %s",
                exc, exc_info=True,
            )
    return _capabilities


def _detect_vision_support() -> bool:
    """Check whether the configured model supports image inputs.

    Uses LiteLLM's static metadata table (``litellm.supports_vision``) — no
    live request. Includes a reviewed fallback for models whose metadata has
    not reached LiteLLM yet.
    """
    try:
        import litellm
        from infinidev.config.settings import settings

        model = settings.LLM_MODEL or ""
        if not model:
            return False
        known_vision_model = _is_reviewed_vision_model(model)
        try:
            return bool(litellm.supports_vision(model=model)) or known_vision_model
        except Exception:
            # Some provider prefixes (ollama_chat/) aren't in the table;
            # retry with common rewrites.
            for prefix_from, prefix_to in (("ollama_chat/", "ollama/"),):
                if model.startswith(prefix_from):
                    try:
                        return (
                            bool(
                                litellm.supports_vision(
                                    model=prefix_to + model[len(prefix_from):]
                                )
                            )
                            or known_vision_model
                        )
                    except Exception:
                        pass
            return known_vision_model
    except Exception:
        return False


# ── Ollama-specific detection ───────────────────────────────────────────

# Markers in the Go template that indicate tool-calling support.
_TOOL_TEMPLATE_MARKERS = (".ToolCalls", "if .Tools", ".tools")


def _detect_ollama_capabilities() -> ModelCapabilities:
    """Detect capabilities for an Ollama model via /api/show.

    Checks whether the model template contains tool-calling markers.
    Falls back to a short live probe if the template check is inconclusive.
    """
    import httpx
    from infinidev.config.settings import settings

    model = settings.LLM_MODEL or ""
    base_url = settings.LLM_BASE_URL or "http://localhost:11434"

    # Strip litellm prefix to get bare model name
    bare_model = model
    for prefix in ("ollama_chat/", "ollama/"):
        if bare_model.startswith(prefix):
            bare_model = bare_model[len(prefix):]
            break

    caps = ModelCapabilities(probed=True)

    try:
        resp = httpx.post(
            f"{base_url}/api/show",
            json={"name": bare_model},
            timeout=5.0,
        )
        if resp.status_code != 200:
            logger.warning(
                "Ollama /api/show returned %d for %s — assuming full capabilities",
                resp.status_code, bare_model,
            )
            return caps

        data = resp.json()
        template = data.get("template", "")

        has_tool_markers = any(m in template for m in _TOOL_TEMPLATE_MARKERS)

        if has_tool_markers:
            logger.info(
                "Ollama model %s: template has tool-calling markers — FC enabled",
                bare_model,
            )
            caps.supports_function_calling = True
            # tool_choice="required" is generally unreliable on Ollama
            caps.supports_tool_choice_required = False
        else:
            logger.info(
                "Ollama model %s: template lacks tool-calling markers — "
                "falling back to manual tool calling",
                bare_model,
            )
            caps.supports_function_calling = False
            caps.supports_tool_choice_required = False

        # JSON mode is generally supported by Ollama
        caps.supports_json_mode = True

        # Check for thinking markers in the template
        template_lower = template.lower()
        if any(m in template_lower for m in ("<thinking>", "<|thinking|>", "<think>")):
            caps.has_thinking_sections = True

        # Vision detection: prefer the explicit `capabilities` array (Ollama
        # 0.4+). Fall back to family names and clip/mllama model_info keys for
        # older daemons that don't emit it.
        capabilities_list = data.get("capabilities") or []
        families = (data.get("details") or {}).get("families") or []
        model_info = data.get("model_info") or {}
        if (
            "vision" in capabilities_list
            or any(fam in ("clip", "mllama") for fam in families)
            or any(
                str(k).startswith(("clip.vision.", "mllama.vision."))
                for k in model_info.keys()
            )
            or bool(data.get("projector_info"))
        ):
            caps.supports_vision = True
            logger.info("Ollama model %s: vision capability detected", bare_model)

    except Exception as exc:
        logger.warning(
            "Failed to query Ollama /api/show for %s: %s — assuming full capabilities",
            bare_model, str(exc)[:200],
        )

    return caps


def _detect_openai_compatible_capabilities() -> ModelCapabilities:
    """Detect capabilities for an OpenAI-compatible server (llama.cpp, vLLM, TGI, etc.).

    For llama_cpp and vLLM: runs a live probe to test FC support. Modern
    llama-server and vLLM both support native function calling for most models.

    For generic openai_compatible: conservative defaults (manual mode).
    """
    from infinidev.config.settings import settings

    provider_id = getattr(settings, "LLM_PROVIDER", "openai_compatible")
    caps: ModelCapabilities

    if provider_id in ("llama_cpp", "vllm"):
        # vLLM has robust FC — probe it
        try:
            from infinidev.config.llm import get_litellm_params
            caps = _run_probes(get_litellm_params())
            caps.probed = True
        except Exception as exc:
            logger.warning("vLLM probe failed: %s — falling back to manual TC", str(exc)[:200])
            caps = ModelCapabilities(
                supports_function_calling=False,
                supports_tool_choice_required=False,
                supports_json_mode=False,
                probed=True,
            )
    else:
        # Generic openai_compatible: conservative defaults
        caps = ModelCapabilities(
            supports_function_calling=False,
            supports_tool_choice_required=False,
            supports_json_mode=False,
            probed=True,
        )

    # llama-server exposes /props with an explicit multimodal flag when an
    # mmproj file is loaded. Cheap local call, fine to attempt for any
    # openai-compatible endpoint.
    if provider_id == "llama_cpp" and _detect_llama_cpp_vision():
        caps.supports_vision = True

    return caps


def _detect_llama_cpp_vision() -> bool:
    """Query llama-server's /props for multimodal support.

    Returns True if the server reports an mmproj (vision projector) is loaded.
    Falls back to scanning the chat template for known image placeholders when
    the explicit flag is absent (older builds).
    """
    import httpx
    from infinidev.config.settings import settings

    base_url = (settings.LLM_BASE_URL or "").rstrip("/")
    if not base_url:
        return False
    # LiteLLM base_urls sometimes include the /v1 suffix — strip it for /props.
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    try:
        resp = httpx.get(f"{base_url}/props", timeout=3.0)
        if resp.status_code != 200:
            return False
        data = resp.json()
    except Exception as exc:
        logger.debug("llama.cpp /props probe failed: %s", str(exc)[:120])
        return False

    # Explicit flags emitted by recent llama-server builds.
    if data.get("has_multimodal") is True:
        return True
    modalities = data.get("modalities") or []
    if isinstance(modalities, list) and "vision" in modalities:
        return True
    if data.get("mmproj") or data.get("mmproj_path"):
        return True

    # Fallback: chat template heuristics for older builds.
    template = str(data.get("chat_template") or "").lower()
    if any(tok in template for tok in ("<|image|>", "<image>", "[img]", "<|vision|>")):
        return True

    return False


def probe_model(llm_params: dict[str, Any]) -> ModelCapabilities:
    """Probe the configured LLM to detect its capabilities.

    Runs 2-3 fast test calls (~3-5s total) with max_tokens=100.
    Updates the module-level singleton and returns it.

    Args:
        llm_params: kwargs dict for litellm.completion (model, api_key, etc.)
    """
    global _capabilities

    caps = ModelCapabilities()
    start = time.monotonic()

    try:
        caps = _run_probes(llm_params)
    except Exception as exc:
        logger.warning(
            "Model probe failed — using defaults: %s", str(exc)[:200]
        )
        caps = ModelCapabilities()

    caps.probe_duration = time.monotonic() - start
    caps.probed = True
    caps.supports_vision = _detect_vision_support()
    _capabilities = caps

    # Log results
    parts = [
        f"function_calling={'yes' if caps.supports_function_calling else 'NO'}",
        f"tool_choice_required={'yes' if caps.supports_tool_choice_required else 'NO'}",
        f"json_mode={'yes' if caps.supports_json_mode else 'NO'}",
        f"vision={'yes' if caps.supports_vision else 'no'}",
        f"thinking={'yes' if caps.has_thinking_sections else 'no'}",
        f"schema_sanitization={'needed' if caps.needs_schema_sanitization else 'no'}",
    ]
    logger.info(
        "Model capabilities probed in %.1fs: %s",
        caps.probe_duration,
        ", ".join(parts),
    )
    return caps


def _run_probes(llm_params: dict[str, Any]) -> ModelCapabilities:
    """Execute the probe sequence."""
    import litellm

    caps = ModelCapabilities()
    probe_params = {**llm_params, "max_tokens": 100, "timeout": _PROBE_TIMEOUT}

    # ── Probe 1: Function calling + tool_choice=required ─────────────
    try:
        resp = litellm.completion(
            **probe_params,
            messages=_PROBE_MESSAGES,
            tools=[_PROBE_TOOL],
            tool_choice="required",
        )
        choice = resp.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)
        content = getattr(choice.message, "content", "") or ""

        if tool_calls:
            # FC works with required — best case
            caps.supports_function_calling = True
            caps.supports_tool_choice_required = True
            # Check for thinking markers in content alongside tool calls
            _check_thinking(caps, content)
        else:
            # Model returned text despite tool_choice=required.
            # Try again with tool_choice=auto to distinguish:
            # "no FC at all" vs "doesn't support required"
            _check_thinking(caps, content)
            caps.supports_tool_choice_required = False
            try:
                resp2 = litellm.completion(
                    **probe_params,
                    messages=_PROBE_MESSAGES,
                    tools=[_PROBE_TOOL],
                    tool_choice="auto",
                )
                tc2 = getattr(resp2.choices[0].message, "tool_calls", None)
                if tc2:
                    caps.supports_function_calling = True
                else:
                    caps.supports_function_calling = False
            except Exception:
                caps.supports_function_calling = False

    except Exception as exc:
        exc_msg = str(exc).lower()
        if "tools" in exc_msg or "function" in exc_msg:
            caps.supports_function_calling = False
            caps.supports_tool_choice_required = False
        else:
            # Any other failure (including servers like llama-server that
            # reject tool_choice="required" with a generic 500 "Failed to
            # parse input" instead of a descriptive error) → treat as
            # "required not supported" and retry with tool_choice="auto".
            caps.supports_tool_choice_required = False
            try:
                resp_fallback = litellm.completion(
                    **probe_params,
                    messages=_PROBE_MESSAGES,
                    tools=[_PROBE_TOOL],
                    tool_choice="auto",
                )
                tc_fb = getattr(resp_fallback.choices[0].message, "tool_calls", None)
                caps.supports_function_calling = bool(tc_fb)
                if tc_fb:
                    _check_thinking(caps, getattr(resp_fallback.choices[0].message, "content", "") or "")
            except Exception:
                caps.supports_function_calling = False

    # ── Probe 2: JSON mode ───────────────────────────────────────────
    try:
        litellm.completion(
            **probe_params,
            messages=[{"role": "user", "content": "Return {\"x\": 1}"}],
            response_format={"type": "json_object"},
        )
        caps.supports_json_mode = True
    except Exception as exc:
        exc_msg = str(exc).lower()
        if "response_format" in exc_msg or "json" in exc_msg:
            caps.supports_json_mode = False
        else:
            # Unrelated error — assume JSON mode is fine
            caps.supports_json_mode = True

    # ── Probe 3: Schema sanitization (anyOf) ─────────────────────────
    if caps.supports_function_calling:
        try:
            litellm.completion(
                **probe_params,
                messages=_PROBE_MESSAGES,
                tools=[_PROBE_TOOL_ANYOF],
                tool_choice="auto",
            )
            caps.needs_schema_sanitization = False
        except Exception as exc:
            exc_msg = str(exc).lower()
            if "anyof" in exc_msg or "schema" in exc_msg or "parameter" in exc_msg:
                caps.needs_schema_sanitization = True
            else:
                # Unrelated error — don't assume sanitization needed
                caps.needs_schema_sanitization = False

    return caps


def _check_thinking(caps: ModelCapabilities, content: str) -> None:
    """Check response content for thinking section markers."""
    content_lower = content.lower()
    if any(marker in content_lower for marker in _THINKING_MARKERS):
        caps.has_thinking_sections = True


def _reset_capabilities() -> None:
    """Reset legacy probes and route-aware snapshots after configuration changes."""
    global _capabilities
    _capabilities = ModelCapabilities()
    capability_resolver.invalidate()
