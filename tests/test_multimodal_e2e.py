"""End-to-end multimodal contract and regression matrix.

The tests keep three boundaries separately observable: model network calls,
durable SQLite state, and published image bytes. No live provider is used.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from infinidev.config.model_capabilities import (
    CapabilityAssessment,
    CapabilitySnapshot,
    CapabilityStatus,
    IMAGE_GENERATION_PROFILES,
    ImageGenerationRoute,
    ModelRoute,
    _credential_id,
)
from infinidev.config.settings import settings
from infinidev.engine.assets import AssetStore, AssetStoreConfig
from infinidev.engine.image_generation import (
    GeneratedImageStatus,
    ImageGenerationRequest,
    ImageOperationStatus,
    LiteLLMImageGenerationAdapter,
)
from infinidev.engine.image_ledger import ImageGenerationService
from infinidev.engine.multimodal import ImageAttachment, build_user_content
from infinidev.tools.base.base_tool import ToolResult
from infinidev.tools.image_generation import GenerateImageTool


@pytest.fixture(autouse=True)
def _exact_openai_images_settings(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _png(*, color: str = "blue") -> bytes:
    stream = BytesIO()
    Image.new("RGB", (3, 2), color=color).save(stream, format="PNG")
    return stream.getvalue()


def _asset_config() -> AssetStoreConfig:
    return AssetStoreConfig(
        max_image_bytes=1024 * 1024,
        max_operation_bytes=2 * 1024 * 1024,
        max_pixels=100_000,
        download_timeout_seconds=2,
        max_redirects=2,
        staging_grace_seconds=60,
    )


def _snapshot(*, vision: CapabilityStatus) -> CapabilitySnapshot:
    generation_route = ImageGenerationRoute(
        provider="openai",
        model="gpt-image-1",
        endpoint="https://api.openai.com/v1",
        transport="https",
        adapter="litellm.image_generation",
        mechanism="openai_images_api",
        operation="images.generate",
        revision="2025-04-01",
        credential_type="api_key",
        account_id="account-a",
        project_id="project-a",
        credential_id=_credential_id("secret"),
    )
    return CapabilitySnapshot(
        route=ModelRoute("anthropic", "claude-test", "https://chat.example"),
        image_input=CapabilityAssessment(status=vision),
        image_generation=CapabilityAssessment(status=CapabilityStatus.SUPPORTED),
        generation_profile=IMAGE_GENERATION_PROFILES[("openai", "gpt-image-1")],
        generation_route=generation_route,
    )


def _unsupported_snapshot(status: CapabilityStatus) -> CapabilitySnapshot:
    return CapabilitySnapshot(
        route=ModelRoute("custom", "chat-only"),
        image_input=CapabilityAssessment(status=status),
        image_generation=CapabilityAssessment(status=CapabilityStatus.UNKNOWN),
    )


def _store(tmp_path: Path) -> AssetStore:
    return AssetStore(root=tmp_path / "assets", config=_asset_config())


def _service(
    tmp_path: Path,
    snapshot: CapabilitySnapshot,
    provider,
    *,
    store: AssetStore | None = None,
) -> ImageGenerationService:
    adapter = LiteLLMImageGenerationAdapter(
        snapshot=snapshot, image_generation_fn=provider, api_key="secret"
    )
    return ImageGenerationService(
        snapshot=snapshot,
        adapter=adapter,
        asset_store=store or _store(tmp_path),
        db_path=settings.DB_PATH,
    )


def test_plain_text_reaches_completion_as_string(monkeypatch) -> None:
    """No-image traffic retains the exact pre-feature content type."""
    captured: list[dict] = []

    def completion(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="done",
                    function=SimpleNamespace(
                        name="respond", arguments='{"message":"ok"}'
                    ),
                )],
            ))]
        )

    import litellm
    import infinidev.engine.orchestration.chat_agent as chat

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(chat, "get_litellm_params_for_behavior", lambda: {"model": "mock"})
    monkeypatch.setattr(chat, "get_tools_for_role", lambda role, **kwargs: [])
    monkeypatch.setattr(
        "infinidev.config.model_capabilities.get_capability_snapshot",
        lambda: _unsupported_snapshot(CapabilityStatus.UNSUPPORTED),
    )

    result = chat.run_chat_agent("hola", max_iterations=1)

    assert result.reply == "ok"
    user_message = next(
        message for message in captured[0]["messages"] if message["role"] == "user"
    )
    assert isinstance(user_message["content"], str)
    assert user_message["content"] == "hola"


@pytest.mark.parametrize(
    "status", [CapabilityStatus.UNSUPPORTED, CapabilityStatus.UNKNOWN]
)
def test_visual_input_fails_closed_before_network_and_preserves_sources(
    monkeypatch, status: CapabilityStatus,
) -> None:
    """Unsupported/unknown never creates image blocks or calls the network."""
    calls = 0
    attachment = ImageAttachment(
        path=Path("/tmp/original.png"),
        mime_type="image/png",
        data_url="https://images.example/original.png?token=keep-me",
        size_bytes=0,
    )
    original_text = "inspect https://docs.example/context"
    original_attachments = [attachment]

    def network(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be reached")

    import litellm
    import infinidev.engine.orchestration.chat_agent as chat

    monkeypatch.setattr(litellm, "completion", network)
    monkeypatch.setattr(
        "infinidev.config.model_capabilities.get_capability_snapshot",
        lambda: _unsupported_snapshot(status),
    )
    result = chat.run_chat_agent(
        original_text, attachments=original_attachments, max_iterations=1
    )

    assert calls == 0
    assert "No request was sent" in result.reply
    assert original_text == "inspect https://docs.example/context"
    assert original_attachments == [attachment]
    assert original_attachments[0].data_url.endswith("token=keep-me")


def test_supported_visual_input_builds_blocks_and_model_change_fails_closed() -> None:
    attachment = ImageAttachment(
        path=Path("<pasted.png>"),
        mime_type="image/png",
        data_url="data:image/png;base64,AAAA",
        size_bytes=3,
    )
    supported = _snapshot(vision=CapabilityStatus.SUPPORTED)
    unknown_after_change = CapabilitySnapshot(
        route=ModelRoute("custom", "changed-model"),
        image_input=CapabilityAssessment(),
        image_generation=supported.image_generation,
        generation_profile=supported.generation_profile,
    )

    assert supported.supports_vision is True
    blocks = build_user_content("describe", [attachment])
    assert isinstance(blocks, list)
    assert blocks[1]["image_url"]["url"] == attachment.data_url
    assert unknown_after_change.supports_vision is False


def test_attachment_added_during_task_is_deferred_and_only_projected_with_vision() -> None:
    from infinidev.engine.loop.tool_runner import ToolRunner

    attachment = ImageAttachment(
        path=Path("later.png"),
        mime_type="image/png",
        data_url="data:image/png;base64,AAAA",
        size_bytes=3,
    )
    tc = SimpleNamespace(id="tc-image", function=SimpleNamespace(name="view_image"))
    ctx = SimpleNamespace()

    import infinidev.config.model_capabilities as capabilities

    original = capabilities.get_capability_snapshot
    try:
        capabilities.get_capability_snapshot = lambda: _snapshot(
            vision=CapabilityStatus.SUPPORTED
        )
        projected = ToolRunner._image_message(ctx, tc, {tc.id: [attachment]})
        assert projected is not None
        assert projected["role"] == "user"
        assert projected["content"][1]["type"] == "image_url"

        capabilities.get_capability_snapshot = lambda: _unsupported_snapshot(
            CapabilityStatus.UNKNOWN
        )
        assert ToolRunner._image_message(ctx, tc, {tc.id: [attachment]}) is None
    finally:
        capabilities.get_capability_snapshot = original


def test_mid_task_injector_keeps_text_and_projects_images_only_when_supported(
    monkeypatch,
) -> None:
    from infinidev.engine.loop.user_message_injector import UserMessageInjector

    attachment = ImageAttachment(
        path=Path("mid-task.png"),
        mime_type="image/png",
        data_url="data:image/png;base64,AAAA",
        size_bytes=3,
    )
    ctx = SimpleNamespace(project_id=1, agent_id="agent")

    monkeypatch.setattr(
        "infinidev.config.model_capabilities.get_capability_snapshot",
        lambda: _snapshot(vision=CapabilityStatus.SUPPORTED),
    )
    injector = UserMessageInjector()
    injector.inject("look at this now", [attachment])
    supported_messages: list[dict] = []
    injector.inject_mid_step(ctx, supported_messages)
    assert isinstance(supported_messages[0]["content"], list)
    assert supported_messages[0]["content"][1]["image_url"]["url"] == attachment.data_url

    monkeypatch.setattr(
        "infinidev.config.model_capabilities.get_capability_snapshot",
        lambda: _unsupported_snapshot(CapabilityStatus.UNKNOWN),
    )
    injector.inject("keep this text", [attachment])
    unknown_messages: list[dict] = []
    injector.inject_mid_step(ctx, unknown_messages)
    assert isinstance(unknown_messages[0]["content"], str)
    assert "keep this text" in unknown_messages[0]["content"]


def test_generation_without_vision_is_durable_downloadable_and_resumable(
    tmp_path: Path, temp_db: str,
) -> None:
    raw = _png()
    encoded = base64.b64encode(raw).decode("ascii")
    calls = 0

    def provider(**kwargs):
        nonlocal calls
        calls += 1
        return {"data": [{"b64_json": encoded}]}

    snapshot = _snapshot(vision=CapabilityStatus.UNSUPPORTED)
    store = _store(tmp_path)
    service = _service(tmp_path, snapshot, provider, store=store)
    tool = GenerateImageTool(snapshot=snapshot, service=service)
    result = tool._run("a durable blue square", operation_id="durable-op")

    assert isinstance(result, ToolResult)
    assert result.attachments == []
    public = json.loads(result.text)
    reference = public["images"][0]["reference"]
    assert reference.startswith("infinidev-image://sha256:")
    assert encoded not in result.text
    assert str(store.root) not in result.text

    resumed = service.get_operation("durable-op")
    assert resumed is not None
    assert resumed.items[0].reference == reference
    assert service.read_asset(reference) == raw
    target = tmp_path / "downloaded.png"
    assert service.export_asset(reference, target) == target
    assert target.read_bytes() == raw
    assert calls == 1

    with sqlite3.connect(temp_db) as conn:
        dumped = "\n".join(conn.iterdump())
    assert encoded not in dumped
    assert str(store.root) not in dumped
    assert "cdn.example" not in dumped


def test_both_capabilities_project_only_materialized_bytes(
    tmp_path: Path, temp_db: str,
) -> None:
    raw = _png(color="green")
    encoded = base64.b64encode(raw).decode("ascii")
    snapshot = _snapshot(vision=CapabilityStatus.SUPPORTED)
    service = _service(
        tmp_path, snapshot, lambda **kwargs: {"data": [{"b64_json": encoded}]}
    )
    result = GenerateImageTool(snapshot=snapshot, service=service)._run(
        "green square", operation_id="vision-generation"
    )

    assert len(result.attachments) == 1
    assert base64.b64decode(result.attachments[0].data_url.split(",", 1)[1]) == raw
    public = json.loads(result.text)
    assert encoded not in result.text
    assert public["images"][0]["reference"].startswith("infinidev-image://")


def test_generation_tool_requires_explicit_exact_route_and_allows_visible_cross_provider(
    monkeypatch,
) -> None:
    import infinidev.config.model_capabilities as capabilities
    import infinidev.tools as tools

    monkeypatch.setattr(tools, "discover_mcp_tool_classes", lambda: [])
    unsupported = _unsupported_snapshot(CapabilityStatus.UNSUPPORTED)
    monkeypatch.setattr(capabilities, "get_capability_snapshot", lambda: unsupported)
    assert "generate_image" not in {
        tool.name for tool in tools.get_tools_for_role("developer", supports_vision=False)
    }

    configured = _snapshot(vision=CapabilityStatus.UNSUPPORTED)
    assert configured.route.provider == "anthropic"
    assert configured.generation_profile is not None
    assert configured.generation_profile.provider == "openai"
    monkeypatch.setattr(capabilities, "get_capability_snapshot", lambda: configured)
    assert "generate_image" in {
        tool.name for tool in tools.get_tools_for_role("developer", supports_vision=False)
    }


def test_late_or_unknown_operation_is_never_generated_or_ingested_twice(
    tmp_path: Path, temp_db: str,
) -> None:
    import litellm

    calls = 0

    def provider(**kwargs):
        nonlocal calls
        calls += 1
        raise litellm.Timeout(
            message="response may arrive late",
            model="gpt-image-1",
            llm_provider="openai",
        )

    snapshot = _snapshot(vision=CapabilityStatus.UNSUPPORTED)
    first_service = _service(tmp_path, snapshot, provider)
    request = ImageGenerationRequest("uncertain-durable", "a fox")
    first = first_service.generate(request)
    assert first.status is ImageOperationStatus.UNKNOWN_OUTCOME

    # Simulate process restart: new adapter/service, same SQLite ledger.
    second_service = _service(tmp_path, snapshot, provider)
    second = second_service.generate(request)
    assert second.status is ImageOperationStatus.UNKNOWN_OUTCOME
    assert calls == 1
    assert list((tmp_path / "assets").glob("blobs/*")) == []


def test_accepted_bad_mime_and_truncated_download_are_terminal_without_reingestion(
    tmp_path: Path, temp_db: str,
) -> None:
    raw = _png()
    signed = "https://cdn.example/generated.png?token=ephemeral"
    requests = 0

    def provider(**kwargs):
        nonlocal requests
        requests += 1
        return {"data": [{"url": signed}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=raw[:-8],
            headers={
                "content-type": "image/jpeg",
                "content-length": str(len(raw) - 8),
            },
        )

    store = AssetStore(
        root=tmp_path / "assets",
        config=_asset_config(),
        resolver=lambda host, port: ("93.184.216.34",),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    snapshot = _snapshot(vision=CapabilityStatus.UNSUPPORTED)
    service = _service(tmp_path, snapshot, provider, store=store)
    request = ImageGenerationRequest(
        "bad-ingestion", "bad provider output", response_format="url"
    )

    first = service.generate(request)
    second = ImageGenerationService(
        snapshot=snapshot,
        adapter=LiteLLMImageGenerationAdapter(
            snapshot=snapshot, image_generation_fn=provider, api_key="secret"
        ),
        asset_store=store,
        db_path=settings.DB_PATH,
    ).generate(request)

    assert first.status is ImageOperationStatus.FAILED
    assert first.error_code == "asset_ingestion_failed"
    assert first.request_accepted is True
    assert second.status is ImageOperationStatus.FAILED
    assert requests == 1
    assert not list((store.root / "blobs").glob("*"))
    with sqlite3.connect(temp_db) as conn:
        dumped = "\n".join(conn.iterdump())
    assert signed not in dumped


def test_legacy_session_and_legacy_tool_result_remain_compatible(temp_db: str) -> None:
    from infinidev.cli.session_resume import resumed_session_state
    from infinidev.db.service import register_session, store_session_message
    from infinidev.tools.base.base_tool import normalize_tool_result

    register_session("legacy-session", "/workspace")
    store_session_message("legacy-session", {"speaker": "User", "message": "old"})
    state = resumed_session_state("legacy-session")

    assert state["messages"][0]["message"] == "old"
    assert normalize_tool_result("legacy string") == ("legacy string", [])
    structured = ToolResult(text="legacy structured", attachments=[])
    assert normalize_tool_result(structured) == ("legacy structured", [])
