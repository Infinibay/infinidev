"""Private, content-addressed storage for generated image assets.

Provider payloads are untrusted. This module validates their real image format,
size and dimensions before publishing a mode-0600 blob below
``.infinidev/private/image_assets``. Public metadata contains only a SHA-256
identity; provider URLs, base64 payloads, and local paths never become durable
identities.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import http.client
import ipaddress
import os
import secrets
import socket
import ssl
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable, Iterator
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from PIL import Image, UnidentifiedImageError

from infinidev.config.settings import get_base_dir, settings

_SUPPORTED_FORMATS: dict[str, tuple[str, str]] = {
    "PNG": ("image/png", "png"),
    "JPEG": ("image/jpeg", "jpg"),
    "GIF": ("image/gif", "gif"),
    "WEBP": ("image/webp", "webp"),
    "BMP": ("image/bmp", "bmp"),
}
_MIME_TYPES = frozenset(value[0] for value in _SUPPORTED_FORMATS.values())
_CHUNK_SIZE = 64 * 1024


class AssetStoreError(RuntimeError):
    """Base error for image materialization failures."""


class AssetConfigError(AssetStoreError):
    """Configured image limits are absent or internally inconsistent."""


class AssetValidationError(AssetStoreError, ValueError):
    """Provider bytes are not a supported image within configured limits."""


class AssetDownloadError(AssetStoreError):
    """A remote image could not be fetched completely and safely."""


class UnsafeAssetURLError(AssetDownloadError):
    """A URL could reach a local, private, or otherwise non-public address."""


class AssetIntegrityError(AssetStoreError):
    """Published private storage is linked, replaced, or corrupt."""


@dataclass(frozen=True)
class AssetStoreConfig:
    """Finite storage and download bounds required by :class:`AssetStore`."""

    max_image_bytes: int
    max_operation_bytes: int
    max_pixels: int
    download_timeout_seconds: int
    max_redirects: int
    staging_grace_seconds: int
    allowed_url_schemes: tuple[str, ...] = ("https",)

    @classmethod
    def from_settings(cls) -> AssetStoreConfig:
        """Load and fail-closed validate generated-image storage settings."""
        values = {
            "max_image_bytes": settings.IMAGE_ASSET_MAX_BYTES,
            "max_operation_bytes": settings.IMAGE_ASSET_MAX_OPERATION_BYTES,
            "max_pixels": settings.IMAGE_ASSET_MAX_PIXELS,
            "download_timeout_seconds": settings.IMAGE_ASSET_DOWNLOAD_TIMEOUT_SECONDS,
            "max_redirects": settings.IMAGE_ASSET_MAX_REDIRECTS,
            "staging_grace_seconds": settings.IMAGE_ASSET_STAGING_GRACE_SECONDS,
        }
        invalid = [name for name, value in values.items() if type(value) is not int or value <= 0]
        if invalid:
            raise AssetConfigError(
                "image asset storage requires positive integer settings: "
                + ", ".join(invalid)
            )
        if values["max_operation_bytes"] < values["max_image_bytes"]:
            raise AssetConfigError(
                "IMAGE_ASSET_MAX_OPERATION_BYTES must be at least IMAGE_ASSET_MAX_BYTES"
            )
        return cls(**values)


@dataclass(frozen=True)
class ImageAssetSource:
    """One ephemeral provider source awaiting durable materialization."""

    source: str
    kind: str
    declared_mime_type: str | None = None


@dataclass(frozen=True)
class ImageAsset:
    """Path-free durable metadata for one content-addressed image."""

    asset_id: str
    sha256: str
    mime_type: str
    byte_count: int
    width: int
    height: int


Resolver = Callable[[str, int], Iterable[str]]


@dataclass(frozen=True)
class _ValidatedURL:
    url: str
    host: str
    port: int
    addresses: tuple[str, ...]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """TLS connection whose socket target is a previously validated IP."""

    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        super().__init__(
            host=host,
            port=port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._validated_address = address

    def connect(self) -> None:
        """Connect to the pinned address while verifying TLS for the URL host."""
        raw_socket = socket.create_connection(
            (self._validated_address, self.port),
            self.timeout,
            self.source_address,
        )
        try:
            self.sock = self._context.wrap_socket(raw_socket, server_hostname=self.host)
        except Exception:
            raw_socket.close()
            raise


class AssetStore:
    """Validate, download, and atomically publish generated image blobs."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        config: AssetStoreConfig | None = None,
        http_client: httpx.Client | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        self._runtime_base = get_base_dir().absolute()
        self._root = (root or self._runtime_base / "private" / "image_assets").absolute()
        self._enforce_runtime_root = root is None
        self._config = config or AssetStoreConfig.from_settings()
        self._client = http_client
        self._resolver = resolver or _resolve_host

    @property
    def root(self) -> Path:
        """Return the private root for maintenance and tests only."""
        return self._root

    def materialize_many(self, sources: Iterable[ImageAssetSource]) -> tuple[ImageAsset, ...]:
        """Validate a whole operation before publishing any of its images."""
        prepared: list[tuple[bytes, ImageAsset, str]] = []
        total = 0
        for source in sources:
            payload, declared_mime = self._read_source(source)
            total += len(payload)
            if total > self._config.max_operation_bytes:
                raise AssetValidationError("generated images exceed the per-operation byte limit")
            asset, extension = self._validate_payload(
                payload, declared_mime_type=declared_mime
            )
            prepared.append((payload, asset, extension))

        with self._locked():
            self._cleanup_staging_locked(time.time())
            for payload, asset, extension in prepared:
                final_path = self._blob_path(asset.sha256, extension)
                if final_path.exists():
                    self._verify_existing(final_path, asset)
                    continue
                self._atomic_publish(final_path, payload)
                self._verify_existing(final_path, asset)
        return tuple(asset for _, asset, _ in prepared)

    def store_base64(
        self, encoded: str, *, declared_mime_type: str | None = None
    ) -> ImageAsset:
        """Decode and durably store one canonical base64 provider payload."""
        if not isinstance(encoded, str) or not encoded:
            raise AssetValidationError("image base64 payload is empty")
        max_encoded = ((self._config.max_image_bytes + 2) // 3) * 4
        if len(encoded) > max_encoded + 4:
            raise AssetValidationError("image exceeds the per-image byte limit")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise AssetValidationError("invalid image base64 payload") from exc
        return self.store_bytes(payload, declared_mime_type=declared_mime_type)

    def store_url(self, url: str) -> ImageAsset:
        """Safely download and durably store one signed provider URL."""
        payload, content_type = self._download(url)
        return self.store_bytes(payload, declared_mime_type=content_type)

    def store_bytes(
        self, payload: bytes, *, declared_mime_type: str | None = None
    ) -> ImageAsset:
        """Validate bytes, then publish them atomically under their SHA-256."""
        asset, extension = self._validate_payload(
            payload, declared_mime_type=declared_mime_type
        )
        with self._locked():
            self._cleanup_staging_locked(time.time())
            final_path = self._blob_path(asset.sha256, extension)
            if final_path.exists():
                self._verify_existing(final_path, asset)
                return asset
            self._atomic_publish(final_path, payload)
            self._verify_existing(final_path, asset)
        return asset

    def _validate_payload(
        self, payload: bytes, *, declared_mime_type: str | None = None
    ) -> tuple[ImageAsset, str]:
        """Return canonical metadata without writing untrusted bytes."""
        if not isinstance(payload, bytes):
            raise AssetValidationError("image payload must be bytes")
        if not payload:
            raise AssetValidationError("image payload is empty")
        if len(payload) > self._config.max_image_bytes:
            raise AssetValidationError("image exceeds the per-image byte limit")

        mime_type, extension, width, height = self._inspect(payload)
        declared = _normalize_content_type(declared_mime_type)
        if declared is not None and declared != mime_type:
            raise AssetValidationError(
                f"declared MIME {declared!r} does not match image bytes {mime_type!r}"
            )

        digest = hashlib.sha256(payload).hexdigest()
        return ImageAsset(
            asset_id=f"sha256:{digest}",
            sha256=digest,
            mime_type=mime_type,
            byte_count=len(payload),
            width=width,
            height=height,
        ), extension

    def read_bytes(self, asset: ImageAsset) -> bytes:
        """Read and integrity-check a previously returned asset."""
        digest = _validate_digest(asset.sha256)
        extension = _extension_for_mime(asset.mime_type)
        payload = self._read_regular(self._blob_path(digest, extension))
        if len(payload) != asset.byte_count or not secrets.compare_digest(
            hashlib.sha256(payload).hexdigest(), digest
        ):
            raise AssetIntegrityError("image asset length or hash mismatch")
        mime_type, _, width, height = self._inspect(payload)
        if (mime_type, width, height) != (asset.mime_type, asset.width, asset.height):
            raise AssetIntegrityError("image asset metadata mismatch")
        return payload

    def cleanup_staging(self, *, now: float | None = None) -> int:
        """Remove only stale, regular temporary files owned by this store."""
        with self._locked():
            return self._cleanup_staging_locked(time.time() if now is None else now)

    def _read_source(self, source: ImageAssetSource) -> tuple[bytes, str | None]:
        if source.kind == "b64_json":
            # Decode here so aggregate accounting happens before publication.
            if not isinstance(source.source, str) or not source.source:
                raise AssetValidationError("image base64 payload is empty")
            max_encoded = ((self._config.max_image_bytes + 2) // 3) * 4
            if len(source.source) > max_encoded + 4:
                raise AssetValidationError("image exceeds the per-image byte limit")
            try:
                payload = base64.b64decode(source.source, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise AssetValidationError("invalid image base64 payload") from exc
            return payload, source.declared_mime_type
        if source.kind == "url":
            return self._download(source.source)
        raise AssetValidationError(f"unsupported image source kind: {source.kind!r}")

    def _inspect(self, payload: bytes) -> tuple[str, str, int, int]:
        magic_mime = _mime_from_magic(payload)
        if magic_mime is None:
            raise AssetValidationError("unsupported or invalid image magic bytes")
        try:
            with Image.open(BytesIO(payload)) as image:
                image_format = str(image.format or "").upper()
                metadata = _SUPPORTED_FORMATS.get(image_format)
                if metadata is None or metadata[0] != magic_mime:
                    raise AssetValidationError("image decoder and magic bytes disagree")
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise AssetValidationError("image dimensions must be positive")
                if width * height > self._config.max_pixels:
                    raise AssetValidationError("image exceeds the pixel limit")
                image.verify()
        except AssetValidationError:
            raise
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise AssetValidationError("image is truncated or corrupt") from exc
        return metadata[0], metadata[1], width, height

    def _download(self, url: str) -> tuple[bytes, str]:
        current = url
        for redirect_count in range(self._config.max_redirects + 1):
            target = self._validate_public_url(current)
            try:
                response = (
                    self._download_with_client(target)
                    if self._client is not None
                    else self._download_pinned(target)
                )
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise AssetDownloadError("image redirect has no Location header")
                    if redirect_count >= self._config.max_redirects:
                        raise AssetDownloadError("image download exceeded redirect limit")
                    current = urljoin(target.url, location)
                    continue
                if response.status_code < 200 or response.status_code >= 300:
                    raise AssetDownloadError(
                        f"image download returned HTTP {response.status_code}"
                    )
                return self._consume_response(response)
            except AssetStoreError:
                raise
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
                OSError,
                ssl.SSLError,
                http.client.HTTPException,
            ) as exc:
                raise AssetDownloadError("image download failed or timed out") from exc
        raise AssetDownloadError("image download exceeded redirect limit")

    def _download_with_client(self, target: _ValidatedURL) -> httpx.Response:
        """Use an injected transport while retaining bounded streaming reads."""
        assert self._client is not None
        with self._client.stream(
            "GET",
            target.url,
            headers={"Accept": "image/*", "Accept-Encoding": "identity"},
        ) as response:
            if response.status_code < 200 or response.status_code >= 300:
                return httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    request=response.request,
                )
            payload = self._read_httpx_body(response)
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                content=payload,
                request=response.request,
            )

    def _read_httpx_body(self, response: httpx.Response) -> bytes:
        """Read an injected HTTP response without permitting unbounded buffering."""
        expected = _validate_response_headers(response.headers, self._config.max_image_bytes)
        if response.is_stream_consumed:
            payload = response.content
            if len(payload) > self._config.max_image_bytes:
                raise AssetValidationError("image exceeds the per-image byte limit")
            if expected is not None and len(payload) != expected:
                raise AssetDownloadError("image download was truncated")
            return payload

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_raw(_CHUNK_SIZE):
            total += len(chunk)
            if total > self._config.max_image_bytes:
                raise AssetValidationError("image exceeds the per-image byte limit")
            chunks.append(chunk)
        if expected is not None and total != expected:
            raise AssetDownloadError("image download was truncated")
        return b"".join(chunks)

    def _download_pinned(self, target: _ValidatedURL) -> httpx.Response:
        """Fetch from the validated address, preventing DNS rebinding."""
        parsed = urlsplit(target.url)
        request_target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        last_error: OSError | ssl.SSLError | http.client.HTTPException | None = None
        for address in target.addresses:
            connection = _PinnedHTTPSConnection(
                target.host,
                target.port,
                address,
                float(self._config.download_timeout_seconds),
            )
            try:
                connection.request(
                    "GET",
                    request_target,
                    headers={
                        "Accept": "image/*",
                        "Accept-Encoding": "identity",
                        "Host": _host_header(target.host, target.port),
                    },
                )
                raw = connection.getresponse()
                headers = dict(raw.getheaders())
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = raw.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self._config.max_image_bytes:
                        raise AssetValidationError("image exceeds the per-image byte limit")
                    chunks.append(chunk)
                request = httpx.Request("GET", target.url)
                return httpx.Response(
                    raw.status,
                    headers=headers,
                    content=b"".join(chunks),
                    request=request,
                )
            except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
                last_error = exc
            finally:
                connection.close()
        if last_error is not None:
            raise last_error
        raise AssetDownloadError("image URL host resolved to no addresses")

    def _consume_response(self, response: httpx.Response) -> tuple[bytes, str]:
        content_type = _normalize_content_type(response.headers.get("content-type"))
        if content_type not in _MIME_TYPES:
            raise AssetValidationError("image response has an unsupported MIME type")
        expected = _validate_response_headers(
            response.headers, self._config.max_image_bytes
        )
        payload = response.content
        if len(payload) > self._config.max_image_bytes:
            raise AssetValidationError("image exceeds the per-image byte limit")
        if expected is not None and len(payload) != expected:
            raise AssetDownloadError("image download was truncated")
        return payload, content_type

    def _validate_public_url(self, url: str) -> _ValidatedURL:
        if not isinstance(url, str) or not url:
            raise UnsafeAssetURLError("image URL is empty")
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in self._config.allowed_url_schemes:
            raise UnsafeAssetURLError("image URL scheme is not allowed")
        if not parsed.hostname or parsed.username is not None or parsed.password is not None:
            raise UnsafeAssetURLError("image URL authority is invalid")
        if parsed.fragment:
            parsed = parsed._replace(fragment="")
        try:
            host = parsed.hostname.encode("idna").decode("ascii").rstrip(".")
            port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        except (UnicodeError, ValueError) as exc:
            raise UnsafeAssetURLError("image URL host or port is invalid") from exc
        if not host or host.lower() == "localhost":
            raise UnsafeAssetURLError("image URL host is not public")
        try:
            addresses = tuple(self._resolver(host, port))
        except (OSError, socket.gaierror) as exc:
            raise UnsafeAssetURLError("image URL host could not be resolved") from exc
        if not addresses:
            raise UnsafeAssetURLError("image URL host resolved to no addresses")
        for address in addresses:
            try:
                ip = ipaddress.ip_address(str(address).split("%", 1)[0])
            except ValueError as exc:
                raise UnsafeAssetURLError("image URL DNS result is invalid") from exc
            if not ip.is_global:
                raise UnsafeAssetURLError("image URL resolves to a non-public address")
        rendered_host = f"[{host}]" if ":" in host else host
        netloc = rendered_host if parsed.port is None else f"{rendered_host}:{parsed.port}"
        normalized_url = urlunsplit(
            (parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, "")
        )
        return _ValidatedURL(
            url=normalized_url,
            host=host,
            port=port,
            addresses=tuple(str(address) for address in addresses),
        )

    def _atomic_publish(self, final_path: Path, payload: bytes) -> None:
        staging = self._staging_dir()
        temp_path = staging / f"{int(time.time())}-{secrets.token_hex(24)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(temp_path, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write to image asset staging")
                view = view[written:]
            os.fsync(fd)
        except Exception:
            os.close(fd)
            temp_path.unlink(missing_ok=True)
            raise
        else:
            os.close(fd)
        try:
            # Content addressing makes concurrent publication idempotent. The
            # exclusive process lock prevents replacing an existing blob here.
            if final_path.exists() or final_path.is_symlink():
                raise AssetIntegrityError("image asset destination already exists")
            os.replace(temp_path, final_path)
            os.chmod(final_path, 0o600, follow_symlinks=False)
            _fsync_directory(final_path.parent)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _verify_existing(self, path: Path, asset: ImageAsset) -> None:
        payload = self._read_regular(path)
        if len(payload) != asset.byte_count or not secrets.compare_digest(
            hashlib.sha256(payload).hexdigest(), asset.sha256
        ):
            raise AssetIntegrityError("content-addressed image blob is corrupt")

    def _read_regular(self, path: Path) -> bytes:
        if os.path.realpath(path.parent) != os.path.realpath(self._blob_dir()):
            raise AssetIntegrityError("image asset resolves outside its private store")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise AssetIntegrityError("image asset is unavailable") from exc
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o600:
                raise AssetIntegrityError("image asset type or permissions are invalid")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(fd, _CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > self._config.max_image_bytes:
                    raise AssetIntegrityError("stored image exceeds configured limit")
                chunks.append(chunk)
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
            ):
                raise AssetIntegrityError("image asset changed during read")
            return b"".join(chunks)
        finally:
            os.close(fd)

    def _cleanup_staging_locked(self, now: float) -> int:
        removed = 0
        cutoff = now - self._config.staging_grace_seconds
        staging = self._staging_dir()
        for entry in staging.iterdir():
            try:
                info = entry.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode) or entry.is_symlink() or info.st_mtime > cutoff:
                continue
            if not entry.name.endswith(".tmp"):
                continue
            entry.unlink(missing_ok=True)
            removed += 1
        if removed:
            _fsync_directory(staging)
        return removed

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._ensure_directories()
        lock_path = self._root / ".lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(lock_path, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise AssetIntegrityError("image asset lock is not regular")
            try:
                import fcntl
            except ImportError as exc:
                raise AssetStoreError("file locking is unavailable") from exc
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            os.close(fd)

    def _ensure_directories(self) -> None:
        if self._enforce_runtime_root and self._runtime_base not in self._root.parents:
            raise AssetIntegrityError("image asset root is outside runtime data")

        root = self._root
        parts = root.parts
        if not parts or not root.is_absolute():
            raise AssetIntegrityError("image asset root must be absolute")
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(parts[0], flags)
        except OSError as exc:
            raise AssetIntegrityError("image asset directory anchor is unavailable") from exc
        try:
            for index, part in enumerate(parts[1:], start=1):
                created = False
                try:
                    child_fd = os.open(part, flags, dir_fd=fd)
                except FileNotFoundError:
                    created = True
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                        child_fd = os.open(part, flags, dir_fd=fd)
                    except (FileExistsError, OSError) as exc:
                        raise AssetIntegrityError(
                            "image asset directory creation failed"
                        ) from exc
                except OSError as exc:
                    raise AssetIntegrityError("image asset path contains a symlink") from exc
                os.close(fd)
                fd = child_fd
                if not stat.S_ISDIR(os.fstat(fd).st_mode):
                    raise AssetIntegrityError("image asset path is not a directory")
                current = Path(*parts[: index + 1])
                if created or current == root:
                    os.fchmod(fd, 0o700)

            for part in ("blobs", "staging"):
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                try:
                    child_fd = os.open(part, flags, dir_fd=fd)
                except OSError as exc:
                    raise AssetIntegrityError("image asset path contains a symlink") from exc
                try:
                    if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                        raise AssetIntegrityError("image asset path is not a directory")
                    os.fchmod(child_fd, 0o700)
                finally:
                    os.close(child_fd)
        finally:
            os.close(fd)

    def _blob_dir(self) -> Path:
        return self._root / "blobs"

    def _staging_dir(self) -> Path:
        return self._root / "staging"

    def _blob_path(self, digest: str, extension: str) -> Path:
        return self._blob_dir() / f"{_validate_digest(digest)}.{extension}"


def _resolve_host(host: str, port: int) -> tuple[str, ...]:
    """Resolve every TCP address for SSRF validation."""
    results = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return tuple(dict.fromkeys(item[4][0] for item in results))


def _normalize_content_type(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.split(";", 1)[0].strip().lower()
    if not normalized:
        return None
    if normalized == "image/jpg":
        return "image/jpeg"
    return normalized


def _host_header(host: str, port: int) -> str:
    """Render an HTTPS Host header, including non-default ports and IPv6 brackets."""
    rendered_host = f"[{host}]" if ":" in host else host
    return rendered_host if port == 443 else f"{rendered_host}:{port}"


def _validate_response_headers(headers: httpx.Headers, max_bytes: int) -> int | None:
    """Validate representation framing before reading a response body."""
    content_encoding = headers.get("content-encoding", "identity").strip().lower()
    if content_encoding not in {"", "identity"}:
        raise AssetDownloadError("encoded image responses are not accepted")
    expected = _parse_content_length(headers.get("content-length"))
    if expected is not None and expected > max_bytes:
        raise AssetValidationError("image exceeds the per-image byte limit")
    return expected


def _parse_content_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError as exc:
        raise AssetDownloadError("image response has invalid Content-Length") from exc
    if length < 0:
        raise AssetDownloadError("image response has invalid Content-Length")
    return length


def _mime_from_magic(payload: bytes) -> str | None:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(payload) >= 12 and payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return "image/webp"
    if payload.startswith(b"BM"):
        return "image/bmp"
    return None


def _validate_digest(value: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise AssetIntegrityError("invalid image asset digest")
    return value


def _extension_for_mime(mime_type: str) -> str:
    for candidate_mime, extension in _SUPPORTED_FORMATS.values():
        if candidate_mime == mime_type:
            return extension
    raise AssetIntegrityError("unsupported image asset MIME type")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
