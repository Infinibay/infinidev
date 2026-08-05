"""Security and atomicity tests for durable generated-image assets."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image

from infinidev.engine.assets import (
    AssetDownloadError,
    AssetStore,
    AssetStoreConfig,
    AssetValidationError,
    ImageAssetSource,
    UnsafeAssetURLError,
)


def _png(*, width: int = 2, height: int = 2, color: str = "red") -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), color=color).save(output, format="PNG")
    return output.getvalue()


def _config(**overrides: object) -> AssetStoreConfig:
    values: dict[str, object] = {
        "max_image_bytes": 1024 * 1024,
        "max_operation_bytes": 2 * 1024 * 1024,
        "max_pixels": 1_000_000,
        "download_timeout_seconds": 2,
        "max_redirects": 2,
        "staging_grace_seconds": 60,
    }
    values.update(overrides)
    return AssetStoreConfig(**values)


def _public_resolver(host: str, port: int) -> tuple[str, ...]:
    del host, port
    return ("93.184.216.34",)


def test_store_validates_magic_mime_dimensions_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    store = AssetStore(root=tmp_path / "assets", config=_config())
    payload = _png(width=3, height=4)

    asset = store.store_bytes(payload, declared_mime_type="image/png")

    assert asset.asset_id == f"sha256:{asset.sha256}"
    assert asset.mime_type == "image/png"
    assert (asset.width, asset.height) == (3, 4)
    assert asset.byte_count == len(payload)
    assert store.read_bytes(asset) == payload
    blobs = list((store.root / "blobs").iterdir())
    assert len(blobs) == 1
    assert blobs[0].name == f"{asset.sha256}.png"
    assert blobs[0].stat().st_mode & 0o777 == 0o600
    assert list((store.root / "staging").iterdir()) == []


def test_forged_mime_and_corrupt_or_truncated_images_are_rejected(tmp_path: Path) -> None:
    store = AssetStore(root=tmp_path / "assets", config=_config())
    payload = _png()

    with pytest.raises(AssetValidationError, match="does not match"):
        store.store_bytes(payload, declared_mime_type="image/jpeg")
    with pytest.raises(AssetValidationError, match="magic"):
        store.store_bytes(b"not an image", declared_mime_type="image/png")
    with pytest.raises(AssetValidationError, match="truncated|corrupt"):
        store.store_bytes(payload[:-8], declared_mime_type="image/png")

    assert not (store.root / "blobs").exists() or not list((store.root / "blobs").iterdir())


def test_per_image_pixel_and_aggregate_limits_fail_closed(tmp_path: Path) -> None:
    payload = _png(width=10, height=10)
    store = AssetStore(
        root=tmp_path / "assets",
        config=_config(
            max_image_bytes=len(payload) - 1,
            max_operation_bytes=(len(payload) - 1) * 2,
        ),
    )
    with pytest.raises(AssetValidationError, match="per-image"):
        store.store_bytes(payload)

    pixel_store = AssetStore(
        root=tmp_path / "pixels",
        config=_config(max_pixels=99),
    )
    with pytest.raises(AssetValidationError, match="pixel"):
        pixel_store.store_bytes(payload)

    aggregate_store = AssetStore(
        root=tmp_path / "aggregate",
        config=_config(
            max_image_bytes=len(payload),
            max_operation_bytes=len(payload) * 2 - 1,
        ),
    )
    encoded = base64.b64encode(payload).decode("ascii")
    with pytest.raises(AssetValidationError, match="per-operation"):
        aggregate_store.materialize_many([
            ImageAssetSource(encoded, "b64_json", "image/png"),
            ImageAssetSource(encoded, "b64_json", "image/png"),
        ])
    blob_dir = aggregate_store.root / "blobs"
    assert not blob_dir.exists() or not list(blob_dir.iterdir())


def test_content_hash_deduplicates_without_exposing_source_or_path(tmp_path: Path) -> None:
    store = AssetStore(root=tmp_path / "assets", config=_config())
    payload = _png()
    encoded = base64.b64encode(payload).decode("ascii")

    first = store.store_base64(encoded, declared_mime_type="image/png")
    second = store.store_bytes(payload, declared_mime_type="image/png")

    assert first == second
    assert list((store.root / "blobs").glob("*")) == [
        store.root / "blobs" / f"{first.sha256}.png"
    ]
    rendered = repr(first)
    assert encoded not in rendered
    assert str(store.root) not in rendered


def test_url_rejects_private_dns_before_http_request(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    store = AssetStore(
        root=tmp_path / "assets",
        config=_config(),
        resolver=lambda host, port: ("127.0.0.1",),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(UnsafeAssetURLError, match="non-public"):
        store.store_url("https://images.example/private.png")
    assert calls == 0


@pytest.mark.parametrize(
    "url",
    [
        "http://images.example/a.png",
        "file:///tmp/a.png",
        "https://user:secret@images.example/a.png",
        "https://localhost/a.png",
    ],
)
def test_url_rejects_unsafe_schemes_and_authorities(tmp_path: Path, url: str) -> None:
    store = AssetStore(
        root=tmp_path / "assets",
        config=_config(),
        resolver=_public_resolver,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail())),
    )
    with pytest.raises(UnsafeAssetURLError):
        store.store_url(url)


def test_default_transport_connects_only_to_validated_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import infinidev.engine.assets as assets_module

    observed: list[tuple[str, str, int]] = []

    class FakePinnedConnection:
        def __init__(
            self, host: str, port: int, address: str, timeout: float,
        ) -> None:
            del timeout
            observed.append((host, address, port))

        def request(self, method: str, target: str, headers: dict[str, str]) -> None:
            assert method == "GET"
            assert target == "/image?token=signed"
            assert headers["Host"] == "cdn.example"

        def getresponse(self):
            payload = _png()

            class Response:
                status = 200

                @staticmethod
                def getheaders() -> list[tuple[str, str]]:
                    return [
                        ("content-type", "image/png"),
                        ("content-length", str(len(payload))),
                    ]

                def read(self, amount: int) -> bytes:
                    del amount
                    nonlocal payload
                    chunk, payload = payload, b""
                    return chunk

            return Response()

        def close(self) -> None:
            return None

    monkeypatch.setattr(assets_module, "_PinnedHTTPSConnection", FakePinnedConnection)
    store = AssetStore(
        root=tmp_path / "assets",
        config=_config(),
        resolver=lambda host, port: ("93.184.216.34",),
    )

    store.store_url("https://cdn.example/image?token=signed")

    assert observed == [("cdn.example", "93.184.216.34", 443)]


def test_redirect_target_is_revalidated_for_ssrf(tmp_path: Path) -> None:
    calls: list[str] = []

    def resolver(host: str, port: int) -> tuple[str, ...]:
        del port
        return ("127.0.0.1",) if host == "internal.example" else ("93.184.216.34",)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://internal.example/secret"})

    store = AssetStore(
        root=tmp_path / "assets",
        config=_config(),
        resolver=resolver,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(UnsafeAssetURLError):
        store.store_url("https://cdn.example/image")
    assert calls == ["https://cdn.example/image"]


def test_successful_url_is_materialized_without_retaining_signed_url(tmp_path: Path) -> None:
    payload = _png()
    signed_url = "https://cdn.example/image?token=super-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == signed_url
        return httpx.Response(
            200,
            content=payload,
            headers={
                "content-type": "image/png; charset=binary",
                "content-length": str(len(payload)),
            },
        )

    store = AssetStore(
        root=tmp_path / "assets",
        config=_config(),
        resolver=_public_resolver,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    asset = store.store_url(signed_url)

    assert store.read_bytes(asset) == payload
    assert "super-secret" not in repr(asset)
    assert "cdn.example" not in repr(asset)
    assert not any(signed_url.encode() in path.read_bytes() for path in store.root.rglob("*") if path.is_file())


def test_http_error_timeout_and_truncated_download_never_publish(tmp_path: Path) -> None:
    payload = _png()

    def check(handler, expected_exception: type[Exception]) -> None:
        root = tmp_path / secretsafe(handler.__name__)
        store = AssetStore(
            root=root,
            config=_config(),
            resolver=_public_resolver,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(expected_exception):
            store.store_url("https://cdn.example/image")
        assert not (root / "blobs").exists() or not list((root / "blobs").iterdir())

    def status_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"content-type": "image/png"})

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("late", request=request)

    def truncated(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=payload,
            headers={
                "content-type": "image/png",
                "content-length": str(len(payload) + 7),
            },
        )

    check(status_error, AssetDownloadError)
    check(timeout, AssetDownloadError)
    check(truncated, AssetDownloadError)


def secretsafe(value: str) -> str:
    return value.replace("_", "-")


def test_store_rejects_symlinked_managed_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    store = AssetStore(root=linked / "assets", config=_config())

    with pytest.raises(Exception, match="symlink"):
        store.store_bytes(_png())
    assert not (outside / "assets").exists()


def test_staging_cleanup_removes_only_stale_owned_regular_files(tmp_path: Path) -> None:
    store = AssetStore(root=tmp_path / "assets", config=_config(staging_grace_seconds=10))
    # First call creates and validates the managed directory tree.
    store.cleanup_staging(now=100)
    staging = store.root / "staging"
    stale = staging / "1-deadbeef.tmp"
    recent = staging / "2-live.tmp"
    unrelated = staging / "keep.txt"
    stale.write_bytes(b"partial")
    recent.write_bytes(b"partial")
    unrelated.write_bytes(b"unrelated")
    os.utime(stale, (1, 1))
    os.utime(recent, (99, 99))
    os.utime(unrelated, (1, 1))

    removed = store.cleanup_staging(now=100)

    assert removed == 1
    assert not stale.exists()
    assert recent.exists()
    assert unrelated.exists()
