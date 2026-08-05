"""Unit tests for the provider-neutral image-generation adapter."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from infinidev.config.model_capabilities import (
    CapabilityAssessment,
    CapabilitySnapshot,
    CapabilityStatus,
    IMAGE_GENERATION_PROFILES,
    ImageGenerationRoute,
    ModelRoute,
    _generation_route_from_settings,
)
from infinidev.config.settings import settings
from infinidev.engine.image_generation import (
    GeneratedImageSourceKind,
    GeneratedImageStatus,
    ImageGenerationConfigurationError,
    ImageGenerationRequest,
    ImageGenerationValidationError,
    ImageOperationStatus,
    LiteLLMImageGenerationAdapter,
    pending_result,
)


@pytest.fixture(autouse=True)
def _exact_openai_images_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind every adapter test to one explicit reviewed OpenAI Images route."""
    values = {
        "IMAGE_GENERATION_PROVIDER": "openai",
        "IMAGE_GENERATION_MODEL": "gpt-image-1",
        "IMAGE_GENERATION_BASE_URL": "",
        "IMAGE_GENERATION_API_KEY": "secret",
        "IMAGE_GENERATION_ACCOUNT_ID": "account-a",
        "IMAGE_GENERATION_PROJECT_ID": "project-a",
        "IMAGE_GENERATION_TRANSPORT": "https",
        "IMAGE_GENERATION_ADAPTER": "litellm.image_generation",
        "IMAGE_GENERATION_MECHANISM": "openai_images_api",
        "IMAGE_GENERATION_OPERATION": "images.generate",
        "IMAGE_GENERATION_REVISION": "2025-04-01",
    }
    for name, value in values.items():
        monkeypatch.setattr(settings, name, value)


def _snapshot() -> CapabilitySnapshot:
    route = _generation_route_from_settings()
    assert route is not None
    profile = IMAGE_GENERATION_PROFILES[(route.provider, route.model)]
    return CapabilitySnapshot(
        route=ModelRoute("anthropic", "claude"),
        image_input=CapabilityAssessment(status=CapabilityStatus.UNSUPPORTED),
        image_generation=CapabilityAssessment(status=CapabilityStatus.SUPPORTED),
        generation_profile=profile,
        generation_route=route,
    )


def _request(**overrides) -> ImageGenerationRequest:
    values = {
        "operation_id": "op-123",
        "prompt": "A small blue robot",
        "response_format": "b64_json",
    }
    values.update(overrides)
    return ImageGenerationRequest(**values)


def test_pending_result_models_each_expected_item() -> None:
    profile = IMAGE_GENERATION_PROFILES[("openai", "gpt-image-1")]
    request = _request(count=2)
    result = pending_result(request, ModelRoute("openai", "gpt-image-1"), profile)

    assert result.status is ImageOperationStatus.PENDING
    assert [item.status for item in result.items] == [
        GeneratedImageStatus.PENDING,
        GeneratedImageStatus.PENDING,
    ]
    assert result.request_accepted is None


def test_adapter_normalizes_base64_and_usage_without_completion_call(monkeypatch) -> None:
    import litellm

    calls: list[dict] = []

    def fake_image_generation(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            created=123,
            data=[SimpleNamespace(
                b64_json="aW1hZ2U=", url=None, revised_prompt="A revised robot",
            )],
            usage=SimpleNamespace(input_tokens=4, output_tokens=8, total_tokens=12),
            _hidden_params={"request_id": "req_1"},
        )

    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: pytest.fail("completion() must not be used for images"),
    )
    adapter = LiteLLMImageGenerationAdapter(
        snapshot=_snapshot(),
        api_key="secret",
        timeout_seconds=9,
        image_generation_fn=fake_image_generation,
    )

    result = adapter.generate(_request())

    assert result.status is ImageOperationStatus.COMPLETE
    assert result.items[0].source_kind is GeneratedImageSourceKind.BASE64
    assert result.items[0].source == "aW1hZ2U="
    assert result.items[0].revised_prompt == "A revised robot"
    assert result.usage is not None and result.usage.total_tokens == 12
    assert result.created_at == 123
    assert result.provider_request_id == "req_1"
    assert result.request_accepted is True
    assert calls == [{
        "prompt": "A small blue robot",
        "model": "openai/gpt-image-1",
        "custom_llm_provider": "openai",
        "n": 1,
        "response_format": "b64_json",
        "timeout": 9.0,
        "max_retries": 0,
        "api_key": "secret",
        "api_base": "https://api.openai.com/v1",
    }]


def test_subscription_snapshot_cannot_dispatch_images() -> None:
    calls = 0

    def fake_image_generation(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("subscription metadata must never reach OpenAI Images")

    snapshot = CapabilitySnapshot(
        route=ModelRoute("openai_subscription", "openai/responses/gpt-5.6-sol"),
        image_input=CapabilityAssessment(status=CapabilityStatus.SUPPORTED),
        image_generation=CapabilityAssessment(status=CapabilityStatus.SUPPORTED),
    )

    with pytest.raises(ImageGenerationConfigurationError):
        LiteLLMImageGenerationAdapter(
            snapshot=snapshot,
            api_key="subscription-token",
            image_generation_fn=fake_image_generation,
        )
    assert calls == 0


def test_adapter_normalizes_url_response() -> None:
    adapter = LiteLLMImageGenerationAdapter(
        snapshot=_snapshot(),
        api_key="secret",
        image_generation_fn=lambda **kwargs: {
            "created": 456,
            "data": [{
                "url": "https://cdn.example/image.png?sig=temporary",
                "b64_json": None,
                "revised_prompt": None,
            }],
        },
    )

    result = adapter.generate(_request(response_format="url"))

    assert result.status is ImageOperationStatus.COMPLETE
    assert result.items[0].source_kind is GeneratedImageSourceKind.URL
    assert result.items[0].source.startswith("https://cdn.example/")


def test_validation_happens_before_provider_call() -> None:
    calls = 0

    def provider(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("must not be called")

    adapter = LiteLLMImageGenerationAdapter(
        snapshot=_snapshot(), image_generation_fn=provider,
    )

    with pytest.raises(ImageGenerationValidationError):
        adapter.generate(_request(size="256x256"))
    with pytest.raises(ImageGenerationValidationError):
        adapter.generate(_request(count=11))
    with pytest.raises(ImageGenerationValidationError):
        adapter.generate(_request(response_format="raw"))
    assert calls == 0


def test_adapter_requires_supported_exact_profile_before_provider_call() -> None:
    calls = 0

    def provider(**kwargs):
        nonlocal calls
        calls += 1

    snapshot = CapabilitySnapshot(
        route=ModelRoute("openai", "gpt-5"),
        image_input=CapabilityAssessment(),
        image_generation=CapabilityAssessment(status=CapabilityStatus.UNKNOWN),
    )
    with pytest.raises(ImageGenerationConfigurationError):
        LiteLLMImageGenerationAdapter(snapshot=snapshot, image_generation_fn=provider)
    assert calls == 0


def test_http_400_is_terminal_and_retry_safe() -> None:
    import litellm

    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.example/images"),
        headers={"x-request-id": "req_bad"},
    )
    error = litellm.BadRequestError(
        message="invalid size",
        model="gpt-image-1",
        llm_provider="openai",
        response=response,
    )
    calls = 0

    def provider(**kwargs):
        nonlocal calls
        calls += 1
        raise error

    result = LiteLLMImageGenerationAdapter(
        snapshot=_snapshot(), image_generation_fn=provider,
    ).generate(_request())

    assert calls == 1
    assert result.status is ImageOperationStatus.FAILED
    assert result.error_code == "bad_request"
    assert result.request_accepted is False
    assert result.retry_safe is True
    assert result.provider_request_id == "req_bad"


def test_rate_limit_preserves_retry_after_without_internal_retry() -> None:
    import litellm

    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://api.example/images"),
        headers={"Retry-After": "17.5"},
    )
    error = litellm.RateLimitError(
        message="slow down",
        model="gpt-image-1",
        llm_provider="openai",
        response=response,
    )
    calls = 0

    def provider(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["max_retries"] == 0
        raise error

    result = LiteLLMImageGenerationAdapter(
        snapshot=_snapshot(), image_generation_fn=provider,
    ).generate(_request())

    assert calls == 1
    assert result.status is ImageOperationStatus.FAILED
    assert result.error_code == "rate_limited"
    assert result.retry_after_seconds == 17.5
    assert result.retry_safe is True


def test_timeout_is_unknown_and_never_retried() -> None:
    import litellm

    calls = 0

    def provider(**kwargs):
        nonlocal calls
        calls += 1
        raise litellm.Timeout(
            message="timed out",
            model="gpt-image-1",
            llm_provider="openai",
        )

    adapter = LiteLLMImageGenerationAdapter(
        snapshot=_snapshot(), image_generation_fn=provider,
    )
    result = adapter.generate(_request(count=2))

    assert calls == 1
    assert result.status is ImageOperationStatus.UNKNOWN_OUTCOME
    assert result.error_code == "timeout"
    assert result.request_accepted is None
    assert result.retry_safe is False
    assert {item.status for item in result.items} == {
        GeneratedImageStatus.UNKNOWN_OUTCOME
    }


def test_malformed_accepted_response_is_unknown_outcome() -> None:
    adapter = LiteLLMImageGenerationAdapter(
        snapshot=_snapshot(),
        image_generation_fn=lambda **kwargs: {
            "data": [{"url": "https://x", "b64_json": "also-present"}],
        },
    )

    result = adapter.generate(_request())

    assert result.status is ImageOperationStatus.UNKNOWN_OUTCOME
    assert result.error_code == "malformed_response"
    assert result.request_accepted is True
    assert result.retry_safe is False


def test_repeated_operation_id_returns_cached_result_without_second_call() -> None:
    calls: list[dict] = []

    def provider(**kwargs):
        calls.append(kwargs)
        return {"data": [{"b64_json": "AA=="}]}

    adapter = LiteLLMImageGenerationAdapter(
        snapshot=_snapshot(), image_generation_fn=provider,
    )
    request = _request(operation_id="stable-op")
    first = adapter.generate(request)
    second = adapter.generate(request)

    assert first is second
    assert first.operation_id == "stable-op"
    assert len(calls) == 1
    assert "operation_id" not in calls[0]
    assert "idempotency_key" not in calls[0]


def test_unknown_outcome_is_cached_and_never_retried() -> None:
    import litellm

    calls = 0

    def provider(**kwargs):
        nonlocal calls
        calls += 1
        raise litellm.Timeout(
            message="late response",
            model="gpt-image-1",
            llm_provider="openai",
        )

    adapter = LiteLLMImageGenerationAdapter(
        snapshot=_snapshot(), image_generation_fn=provider,
    )
    request = _request(operation_id="uncertain-op")

    assert adapter.generate(request).status is ImageOperationStatus.UNKNOWN_OUTCOME
    assert adapter.generate(request).status is ImageOperationStatus.UNKNOWN_OUTCOME
    assert calls == 1


def test_operation_id_cannot_be_reused_for_different_request() -> None:
    adapter = LiteLLMImageGenerationAdapter(
        snapshot=_snapshot(),
        image_generation_fn=lambda **kwargs: {"data": [{"b64_json": "AA=="}]},
    )
    adapter.generate(_request(operation_id="same-id"))

    with pytest.raises(ImageGenerationValidationError):
        adapter.generate(_request(operation_id="same-id", prompt="Different prompt"))


def test_unexpected_adapter_bug_is_not_disguised_as_provider_outcome() -> None:
    def provider(**kwargs):
        raise AssertionError("adapter bug")

    adapter = LiteLLMImageGenerationAdapter(
        snapshot=_snapshot(), image_generation_fn=provider,
    )
    with pytest.raises(AssertionError, match="adapter bug"):
        adapter.generate(_request())
