"""Provider-neutral image-generation requests and the LiteLLM adapter.

This module deliberately stops at provider response normalization. Durable
materialization, operation persistence, tool registration, and UI projection
belong to later layers and consume the source payloads represented here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from infinidev.config.model_capabilities import (
    CapabilitySnapshot,
    ImageGenerationProfile,
    ImageGenerationRoute,
    _credential_id,
    _generation_profile_for_route,
    _generation_route_from_settings,
    get_capability_snapshot,
)
from infinidev.config.settings import settings

_OPERATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_PROMPT_CHARS = 32_000


class ImageGenerationError(RuntimeError):
    """Base error raised before a generation request can be accepted."""


class ImageGenerationConfigurationError(ImageGenerationError):
    """The configured route has no exact, supported generation profile."""


class ImageGenerationValidationError(ImageGenerationError, ValueError):
    """A request violates the selected generation profile."""


class ImageOperationStatus(StrEnum):
    """Lifecycle state of a generation operation."""

    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"
    UNKNOWN_OUTCOME = "unknown_outcome"


class GeneratedImageStatus(StrEnum):
    """Lifecycle state of one expected or returned image."""

    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"
    UNKNOWN_OUTCOME = "unknown_outcome"


class GeneratedImageSourceKind(StrEnum):
    """Provider response representation awaiting durable materialization."""

    BASE64 = "b64_json"
    URL = "url"


@dataclass(frozen=True)
class ImageGenerationRequest:
    """Validated intent for exactly one configured generation route."""

    operation_id: str
    prompt: str
    count: int = 1
    response_format: str = "b64_json"
    size: str | None = None
    quality: str | None = None
    style: str | None = None
    user: str | None = None


@dataclass(frozen=True)
class GeneratedImageItem:
    """One normalized image source returned by the provider."""

    index: int
    status: GeneratedImageStatus
    source_kind: GeneratedImageSourceKind | None = None
    source: str | None = field(default=None, repr=False)
    revised_prompt: str | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ImageGenerationUsage:
    """Provider usage values, when supplied by LiteLLM."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class GeneratedImageResult:
    """Normalized outcome of one provider invocation."""

    operation_id: str
    status: ImageOperationStatus
    items: tuple[GeneratedImageItem, ...]
    route: ImageGenerationRoute
    profile_version: int
    created_at: int | None = None
    usage: ImageGenerationUsage | None = None
    error_code: str | None = None
    error_message: str | None = None
    retry_after_seconds: float | None = None
    provider_request_id: str | None = None
    request_accepted: bool | None = None

    @property
    def retry_safe(self) -> bool:
        """Whether the operation is known not to have been accepted."""
        return self.status is ImageOperationStatus.FAILED and self.request_accepted is False


@runtime_checkable
class ImageGenerationPort(Protocol):
    """Port implemented by image-generation providers."""

    def generate(self, request: ImageGenerationRequest) -> GeneratedImageResult:
        """Submit one operation without internally repeating it."""
        ...


class LiteLLMImageGenerationAdapter:
    """Exact-profile adapter over ``litellm.image_generation``.

    No automatic retry is allowed. A timeout, connection failure, or server
    error can arrive after the provider accepted the request, so those cases
    are represented as ``unknown_outcome`` and require reconciliation rather
    than a second generation.
    """

    def __init__(
        self,
        *,
        snapshot: CapabilitySnapshot | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        image_generation_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._snapshot = snapshot or get_capability_snapshot()
        self._profile, self._route = _require_profile(self._snapshot)
        configured_api_key = settings.IMAGE_GENERATION_API_KEY if api_key is None else api_key
        self._api_key = configured_api_key.strip()
        current_route = _generation_route_from_settings()
        if not self._api_key or _credential_id(self._api_key) != self._route.credential_id:
            raise ImageGenerationConfigurationError(
                "image generation requires the API key bound to the reviewed route"
            )
        if current_route != self._route:
            raise ImageGenerationConfigurationError(
                "image generation configuration changed after capability resolution"
            )
        self._timeout_seconds = (
            float(settings.LLM_TIMEOUT) if timeout_seconds is None else float(timeout_seconds)
        )
        if self._timeout_seconds <= 0:
            raise ImageGenerationConfigurationError(
                "image generation timeout must be greater than zero"
            )
        self._image_generation_fn = image_generation_fn
        self._registry_lock = RLock()
        self._operation_locks: dict[str, RLock] = {}
        self._requests: dict[str, ImageGenerationRequest] = {}
        self._results: dict[str, GeneratedImageResult] = {}

    def generate(self, request: ImageGenerationRequest) -> GeneratedImageResult:
        """Validate and invoke LiteLLM at most once per local operation ID."""
        _validate_request(request, self._profile)
        with self._registry_lock:
            operation_lock = self._operation_locks.setdefault(request.operation_id, RLock())

        with operation_lock:
            previous_request = self._requests.get(request.operation_id)
            if previous_request is not None and previous_request != request:
                raise ImageGenerationValidationError(
                    "operation_id is already bound to a different generation request"
                )
            if previous := self._results.get(request.operation_id):
                return previous
            self._requests[request.operation_id] = request

            fn = self._image_generation_fn
            if fn is None:
                import litellm

                fn = litellm.image_generation

            kwargs: dict[str, Any] = {
                "prompt": request.prompt,
                "model": _litellm_model(self._route),
                "custom_llm_provider": self._route.provider,
                "n": request.count,
                "response_format": request.response_format,
                "timeout": self._timeout_seconds,
                # The OpenAI client interprets zero as no retries. LiteLLM also
                # recognizes this as a transport parameter rather than forwarding
                # it to the generation endpoint.
                "max_retries": 0,
            }
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._route.base_url:
                kwargs["api_base"] = self._route.base_url
            for name in ("size", "quality", "style", "user"):
                if (value := getattr(request, name)) is not None:
                    kwargs[name] = value

            try:
                response = fn(**kwargs)
            except _expected_provider_errors() as exc:
                result = self._normalize_exception(request, exc)
            else:
                result = self._normalize_response(request, response)
            self._results[request.operation_id] = result
            return result

    def _normalize_response(
        self, request: ImageGenerationRequest, response: Any,
    ) -> GeneratedImageResult:
        data = _field(response, "data")
        if not isinstance(data, (list, tuple)) or not data:
            return self._malformed_response(
                request, "Provider returned no image data after request acceptance."
            )

        items: list[GeneratedImageItem] = []
        for index, raw in enumerate(data):
            url = _nonempty_string(_field(raw, "url"))
            b64_json = _nonempty_string(_field(raw, "b64_json"))
            revised_prompt = _nonempty_string(_field(raw, "revised_prompt"))
            if bool(url) == bool(b64_json):
                return self._malformed_response(
                    request,
                    f"Image item {index} must contain exactly one of url or b64_json.",
                )
            source_kind = (
                GeneratedImageSourceKind.URL if url else GeneratedImageSourceKind.BASE64
            )
            if source_kind.value not in self._profile.response_formats:
                return self._malformed_response(
                    request,
                    f"Image item {index} used response format {source_kind.value!r} "
                    "outside the exact profile.",
                )
            if source_kind.value != request.response_format:
                return self._malformed_response(
                    request,
                    f"Image item {index} did not use requested response format "
                    f"{request.response_format!r}.",
                )
            items.append(GeneratedImageItem(
                index=index,
                status=GeneratedImageStatus.COMPLETE,
                source_kind=source_kind,
                source=url or b64_json,
                revised_prompt=revised_prompt,
            ))

        if len(items) != request.count:
            return self._malformed_response(
                request,
                f"Provider returned {len(items)} image(s), expected {request.count}.",
            )

        return GeneratedImageResult(
            operation_id=request.operation_id,
            status=ImageOperationStatus.COMPLETE,
            items=tuple(items),
            route=self._route,
            profile_version=self._profile.version,
            created_at=_optional_nonnegative_int(_field(response, "created")),
            usage=_normalize_usage(_field(response, "usage")),
            provider_request_id=_provider_request_id(response),
            request_accepted=True,
        )

    def _normalize_exception(
        self, request: ImageGenerationRequest, exc: Exception,
    ) -> GeneratedImageResult:
        classification = _classify_exception(exc)
        item_status = (
            GeneratedImageStatus.FAILED
            if classification.status is ImageOperationStatus.FAILED
            else GeneratedImageStatus.UNKNOWN_OUTCOME
        )
        message = _safe_error_message(exc)
        return GeneratedImageResult(
            operation_id=request.operation_id,
            status=classification.status,
            items=tuple(
                GeneratedImageItem(
                    index=index,
                    status=item_status,
                    error_code=classification.code,
                    error_message=message,
                )
                for index in range(request.count)
            ),
            route=self._route,
            profile_version=self._profile.version,
            error_code=classification.code,
            error_message=message,
            retry_after_seconds=_retry_after_seconds(exc),
            provider_request_id=_provider_request_id(exc),
            request_accepted=classification.accepted,
        )

    def _malformed_response(
        self, request: ImageGenerationRequest, message: str,
    ) -> GeneratedImageResult:
        return GeneratedImageResult(
            operation_id=request.operation_id,
            status=ImageOperationStatus.UNKNOWN_OUTCOME,
            items=tuple(
                GeneratedImageItem(
                    index=index,
                    status=GeneratedImageStatus.UNKNOWN_OUTCOME,
                    error_code="malformed_response",
                    error_message=message,
                )
                for index in range(request.count)
            ),
            route=self._route,
            profile_version=self._profile.version,
            error_code="malformed_response",
            error_message=message,
            request_accepted=True,
        )


@dataclass(frozen=True)
class _ExceptionClassification:
    status: ImageOperationStatus
    code: str
    accepted: bool | None


def pending_result(
    request: ImageGenerationRequest,
    route: ImageGenerationRoute,
    profile: ImageGenerationProfile,
) -> GeneratedImageResult:
    """Create the operation state recorded before invoking a provider."""
    _validate_request(request, profile)
    return GeneratedImageResult(
        operation_id=request.operation_id,
        status=ImageOperationStatus.PENDING,
        items=tuple(
            GeneratedImageItem(index=i, status=GeneratedImageStatus.PENDING)
            for i in range(request.count)
        ),
        route=route,
        profile_version=profile.version,
        request_accepted=None,
    )


def _require_profile(
    snapshot: CapabilitySnapshot,
) -> tuple[ImageGenerationProfile, ImageGenerationRoute]:
    profile = snapshot.generation_profile
    route = snapshot.generation_route
    if (
        not snapshot.image_generation.supported
        or profile is None
        or route is None
        or _generation_profile_for_route(route) != profile
    ):
        raise ImageGenerationConfigurationError(
            "image generation requires an explicit route with an exact supported profile"
        )
    if profile.adapter != "litellm.image_generation":
        raise ImageGenerationConfigurationError(
            f"generation profile adapter {profile.adapter!r} is not supported here"
        )
    return profile, route


def _validate_request(
    request: ImageGenerationRequest, profile: ImageGenerationProfile,
) -> None:
    if not _OPERATION_ID_RE.fullmatch(request.operation_id):
        raise ImageGenerationValidationError(
            "operation_id must be 1-128 safe identifier characters"
        )
    if not request.prompt.strip():
        raise ImageGenerationValidationError("prompt must not be empty")
    if len(request.prompt) > _MAX_PROMPT_CHARS:
        raise ImageGenerationValidationError(
            f"prompt exceeds {_MAX_PROMPT_CHARS} characters"
        )
    if type(request.count) is not int or not 1 <= request.count <= profile.max_images:
        raise ImageGenerationValidationError(
            f"count must be between 1 and {profile.max_images} for this profile"
        )
    _validate_choice("response_format", request.response_format, profile.response_formats)
    _validate_optional_choice("size", request.size, profile.sizes)
    _validate_optional_choice("quality", request.quality, profile.qualities)
    _validate_optional_choice("style", request.style, profile.styles)
    if request.user is not None and (not request.user or len(request.user) > 256):
        raise ImageGenerationValidationError("user must contain 1-256 characters")


def _validate_optional_choice(name: str, value: str | None, allowed: tuple[str, ...]) -> None:
    if value is None:
        return
    _validate_choice(name, value, allowed)


def _validate_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ImageGenerationValidationError(
            f"{name}={value!r} is not allowed by the exact profile; allowed={allowed}"
        )


def _litellm_model(route: ImageGenerationRoute) -> str:
    """Return a provider-qualified model without duplicating the prefix."""
    if route.model.startswith(f"{route.provider}/"):
        return route.model
    return f"{route.provider}/{route.model}"


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _nonempty_string(value: Any) -> str | None:
    return value if isinstance(value, str) and bool(value) else None


def _optional_nonnegative_int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _normalize_usage(value: Any) -> ImageGenerationUsage | None:
    if value is None:
        return None
    usage = ImageGenerationUsage(
        input_tokens=_optional_nonnegative_int(_field(value, "input_tokens")),
        output_tokens=_optional_nonnegative_int(_field(value, "output_tokens")),
        total_tokens=_optional_nonnegative_int(_field(value, "total_tokens")),
    )
    if all(item is None for item in (
        usage.input_tokens, usage.output_tokens, usage.total_tokens,
    )):
        return None
    return usage


def _expected_provider_errors() -> tuple[type[Exception], ...]:
    """Return the mapped LiteLLM/OpenAI boundary errors.

    Keeping this import lazy preserves the domain contract without forcing
    LiteLLM import side effects on callers that only use the value objects.
    Unexpected implementation errors deliberately propagate.
    """
    import litellm

    return (
        litellm.BadRequestError,
        litellm.AuthenticationError,
        litellm.PermissionDeniedError,
        litellm.NotFoundError,
        litellm.RateLimitError,
        litellm.Timeout,
        litellm.APIConnectionError,
        litellm.ServiceUnavailableError,
        litellm.InternalServerError,
    )


def _classify_exception(exc: Exception) -> _ExceptionClassification:
    import litellm

    if isinstance(exc, litellm.BadRequestError):
        return _ExceptionClassification(ImageOperationStatus.FAILED, "bad_request", False)
    if isinstance(exc, litellm.AuthenticationError):
        return _ExceptionClassification(ImageOperationStatus.FAILED, "authentication", False)
    if isinstance(exc, litellm.PermissionDeniedError):
        return _ExceptionClassification(ImageOperationStatus.FAILED, "permission_denied", False)
    if isinstance(exc, litellm.NotFoundError):
        return _ExceptionClassification(ImageOperationStatus.FAILED, "not_found", False)
    if isinstance(exc, litellm.RateLimitError):
        return _ExceptionClassification(ImageOperationStatus.FAILED, "rate_limited", False)
    if isinstance(exc, litellm.Timeout):
        return _ExceptionClassification(
            ImageOperationStatus.UNKNOWN_OUTCOME, "timeout", None,
        )
    if isinstance(exc, litellm.APIConnectionError):
        return _ExceptionClassification(
            ImageOperationStatus.UNKNOWN_OUTCOME, "connection_error", None,
        )
    status_code = _status_code(exc)
    if status_code is not None and 400 <= status_code < 500:
        return _ExceptionClassification(ImageOperationStatus.FAILED, f"http_{status_code}", False)
    return _ExceptionClassification(
        ImageOperationStatus.UNKNOWN_OUTCOME,
        f"http_{status_code}" if status_code is not None else "provider_error",
        None,
    )


def _response(value: Any) -> Any:
    return getattr(value, "response", None)


def _status_code(value: Any) -> int | None:
    for source in (value, _response(value)):
        raw = getattr(source, "status_code", None)
        if type(raw) is int:
            return raw
    return None


def _headers(value: Any) -> Mapping[str, Any]:
    for source in (_response(value), value):
        raw = getattr(source, "headers", None)
        if isinstance(raw, Mapping):
            return raw
    return {}


def _retry_after_seconds(exc: Exception) -> float | None:
    raw = next(
        (value for key, value in _headers(exc).items() if str(key).lower() == "retry-after"),
        None,
    )
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _provider_request_id(value: Any) -> str | None:
    for key, raw in _headers(value).items():
        if str(key).lower() in ("x-request-id", "request-id"):
            return raw if isinstance(raw, str) and raw else None
    hidden = getattr(value, "_hidden_params", None)
    if isinstance(hidden, Mapping):
        for key in ("request_id", "id"):
            raw = hidden.get(key)
            if isinstance(raw, str) and raw:
                return raw
    return None


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    return message[:1000]
