"""Durable, idempotent orchestration for generated images.

Provider payloads are materialized before persistence. SQLite stores only
path-free content-addressed references and operation metadata, so signed URLs,
base64 data, and private local paths never become part of the conversation or
ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import RLock

from infinidev.config.model_capabilities import (
    CapabilitySnapshot,
    ImageGenerationRoute,
    _generation_profile_for_route,
)
from infinidev.config.settings import settings
from infinidev.engine.assets import (
    AssetStore,
    AssetStoreError,
    ImageAsset,
    ImageAssetSource,
)
from infinidev.engine.image_generation import (
    GeneratedImageResult,
    GeneratedImageStatus,
    ImageGenerationPort,
    ImageGenerationRequest,
    ImageOperationStatus,
    LiteLLMImageGenerationAdapter,
    pending_result,
)
from infinidev.tools.base.db import execute_with_retry


class ImageLedgerError(RuntimeError):
    """Base error for durable generated-image operations."""


class ImageOperationConflictError(ImageLedgerError, ValueError):
    """An operation ID was previously bound to another immutable request."""


class ImageAssetNotFoundError(ImageLedgerError, LookupError):
    """A durable image reference does not exist in the ledger."""


@dataclass(frozen=True)
class DurableGeneratedImage:
    """One path-free image reference safe for persistence and display."""

    index: int
    status: GeneratedImageStatus
    asset: ImageAsset | None = None
    revised_prompt: str | None = None
    error_code: str | None = None
    error_message: str | None = None

    @property
    def reference(self) -> str | None:
        """Return a stable public reference without exposing a local path."""
        return f"infinidev-image://{self.asset.asset_id}" if self.asset else None


@dataclass(frozen=True)
class DurableImageGenerationResult:
    """Persisted operation outcome containing no ephemeral source material."""

    operation_id: str
    status: ImageOperationStatus
    items: tuple[DurableGeneratedImage, ...]
    route: ImageGenerationRoute
    profile_version: int
    error_code: str | None = None
    error_message: str | None = None
    retry_after_seconds: float | None = None
    provider_request_id: str | None = None
    request_accepted: bool | None = None


class ImageGenerationService:
    """Claim, invoke, materialize, and atomically persist image operations."""

    def __init__(
        self,
        *,
        snapshot: CapabilitySnapshot,
        adapter: ImageGenerationPort | None = None,
        asset_store: AssetStore | None = None,
        db_path: str | None = None,
    ) -> None:
        profile = snapshot.generation_profile
        route = snapshot.generation_route
        if (
            not snapshot.image_generation.supported
            or profile is None
            or route is None
            or _generation_profile_for_route(route) != profile
        ):
            from infinidev.engine.image_generation import ImageGenerationConfigurationError

            raise ImageGenerationConfigurationError(
                "image generation requires an explicit route with an exact supported profile"
            )
        self._snapshot = snapshot
        self._profile = profile
        self._route = route
        self._adapter = adapter or LiteLLMImageGenerationAdapter(snapshot=snapshot)
        self._assets = asset_store or AssetStore()
        self._db_path = db_path
        self._registry_lock = RLock()
        self._operation_locks: dict[str, RLock] = {}

    def generate(
        self,
        request: ImageGenerationRequest,
        *,
        session_id: str | None = None,
        project_id: int | None = None,
    ) -> DurableImageGenerationResult:
        """Run at most one provider request for a durable operation ID."""
        fingerprint, request_json = _request_identity(request, self._route, self._profile.version)
        with self._registry_lock:
            lock = self._operation_locks.setdefault(request.operation_id, RLock())

        with lock:
            claimed, existing = self._claim(
                request,
                fingerprint=fingerprint,
                request_json=request_json,
                session_id=session_id,
                project_id=project_id,
            )
            if not claimed:
                return existing

            provider_result = self._adapter.generate(request)
            durable = self._materialize(provider_result)
            self._finish(durable)
            return durable

    def get_operation(self, operation_id: str) -> DurableImageGenerationResult | None:
        """Load an operation and its durable items for resume/reconciliation."""
        return execute_with_retry(
            lambda conn: _load_operation(conn, operation_id), db_path=self._db_path
        )

    def get_asset(self, asset_id: str) -> ImageAsset:
        """Load path-free metadata for a durable image reference."""
        normalized = _normalize_asset_id(asset_id)

        def _query(conn: sqlite3.Connection) -> ImageAsset | None:
            row = conn.execute(
                "SELECT asset_id, sha256, mime_type, byte_count, width, height "
                "FROM image_assets WHERE asset_id = ?",
                (normalized,),
            ).fetchone()
            return _asset_from_row(row) if row else None

        asset = execute_with_retry(_query, db_path=self._db_path)
        if asset is None:
            raise ImageAssetNotFoundError(f"unknown generated image: {normalized}")
        return asset

    def read_asset(self, asset_id: str) -> bytes:
        """Read and integrity-check bytes for display or download."""
        return self._assets.read_bytes(self.get_asset(asset_id))

    def export_asset(self, asset_id: str, destination: str | Path) -> Path:
        """Copy an asset to an explicit user path without exposing store paths."""
        asset = self.get_asset(asset_id)
        payload = self._assets.read_bytes(asset)
        target = Path(destination).expanduser().resolve()
        if target.exists() and not target.is_file():
            raise ImageLedgerError(f"download destination is not a file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            fd = os.open(target, flags, 0o600)
        except FileExistsError as exc:
            raise ImageLedgerError(f"download destination already exists: {target}") from exc
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            target.unlink(missing_ok=True)
            raise
        return target

    def _claim(
        self,
        request: ImageGenerationRequest,
        *,
        fingerprint: str,
        request_json: str,
        session_id: str | None,
        project_id: int | None,
    ) -> tuple[bool, DurableImageGenerationResult]:
        pending = pending_result(request, self._route, self._profile)

        def _transaction(conn: sqlite3.Connection) -> tuple[bool, DurableImageGenerationResult]:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT request_fingerprint FROM image_generation_operations "
                "WHERE operation_id = ?",
                (request.operation_id,),
            ).fetchone()
            if row is not None:
                if row["request_fingerprint"] != fingerprint:
                    conn.rollback()
                    raise ImageOperationConflictError(
                        "operation_id is already bound to a different generation request"
                    )
                existing = _load_operation(conn, request.operation_id)
                conn.commit()
                if existing is None:
                    raise ImageLedgerError("operation disappeared while loading")
                return False, existing

            conn.execute(
                """INSERT INTO image_generation_operations
                   (operation_id, session_id, project_id, request_json,
                    request_fingerprint, provider, model, base_url,
                    profile_version, status, request_accepted, endpoint,
                    transport, adapter, mechanism, operation, revision,
                    credential_type, account_id, generation_project_id,
                    credential_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request.operation_id,
                    session_id,
                    project_id,
                    request_json,
                    fingerprint,
                    self._route.provider,
                    self._route.model,
                    self._route.base_url,
                    self._profile.version,
                    pending.status.value,
                    None,
                    self._route.endpoint,
                    self._route.transport,
                    self._route.adapter,
                    self._route.mechanism,
                    self._route.operation,
                    self._route.revision,
                    self._route.credential_type,
                    self._route.account_id,
                    self._route.project_id,
                    self._route.credential_id,
                ),
            )
            for item in pending.items:
                conn.execute(
                    "INSERT INTO image_generation_items "
                    "(operation_id, item_index, status) VALUES (?, ?, ?)",
                    (request.operation_id, item.index, item.status.value),
                )
            conn.commit()
            return True, _durable_from_provider(pending, ())

        return execute_with_retry(_transaction, db_path=self._db_path)

    def _materialize(
        self, result: GeneratedImageResult,
    ) -> DurableImageGenerationResult:
        if result.status is not ImageOperationStatus.COMPLETE:
            return _durable_from_provider(result, ())
        sources = tuple(
            ImageAssetSource(source=item.source or "", kind=item.source_kind.value)
            for item in result.items
            if item.source_kind is not None
        )
        try:
            assets = self._assets.materialize_many(sources)
        except AssetStoreError as exc:
            # The provider accepted the request. Never invite an automatic retry
            # just because its output failed local validation or download.
            message = str(exc)[:1000] or type(exc).__name__
            failed = replace(
                result,
                status=ImageOperationStatus.FAILED,
                error_code="asset_ingestion_failed",
                error_message=message,
                request_accepted=True,
                items=tuple(replace(
                    item,
                    status=GeneratedImageStatus.FAILED,
                    source_kind=None,
                    source=None,
                    error_code="asset_ingestion_failed",
                    error_message=message,
                ) for item in result.items),
            )
            return _durable_from_provider(failed, ())
        return _durable_from_provider(result, assets)

    def _finish(self, result: DurableImageGenerationResult) -> None:
        def _transaction(conn: sqlite3.Connection) -> None:
            conn.execute("BEGIN IMMEDIATE")
            for item in result.items:
                if item.asset is not None:
                    asset = item.asset
                    conn.execute(
                        """INSERT INTO image_assets
                           (asset_id, sha256, mime_type, byte_count, width, height)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(asset_id) DO NOTHING""",
                        (
                            asset.asset_id,
                            asset.sha256,
                            asset.mime_type,
                            asset.byte_count,
                            asset.width,
                            asset.height,
                        ),
                    )
                conn.execute(
                    """UPDATE image_generation_items
                       SET status = ?, asset_id = ?, revised_prompt = ?,
                           error_code = ?, error_message = ?
                       WHERE operation_id = ? AND item_index = ?""",
                    (
                        item.status.value,
                        item.asset.asset_id if item.asset else None,
                        item.revised_prompt,
                        item.error_code,
                        item.error_message,
                        result.operation_id,
                        item.index,
                    ),
                )
            conn.execute(
                """UPDATE image_generation_operations
                   SET status = ?, error_code = ?, error_message = ?,
                       retry_after_seconds = ?, provider_request_id = ?,
                       request_accepted = ?,
                       updated_at = strftime('%Y-%m-%d %H:%M:%f','now')
                   WHERE operation_id = ?""",
                (
                    result.status.value,
                    result.error_code,
                    result.error_message,
                    result.retry_after_seconds,
                    result.provider_request_id,
                    _bool_to_db(result.request_accepted),
                    result.operation_id,
                ),
            )
            conn.commit()

        execute_with_retry(_transaction, db_path=self._db_path)


def _request_identity(
    request: ImageGenerationRequest,
    route: ImageGenerationRoute,
    profile_version: int,
) -> tuple[str, str]:
    request_data = asdict(request)
    request_json = json.dumps(request_data, sort_keys=True, separators=(",", ":"))
    identity = json.dumps(
        {
            "request": request_data,
            "route": asdict(route),
            "profile_version": profile_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(identity).hexdigest(), request_json


def _durable_from_provider(
    result: GeneratedImageResult, assets: tuple[ImageAsset, ...],
) -> DurableImageGenerationResult:
    by_index = {index: asset for index, asset in enumerate(assets)}
    return DurableImageGenerationResult(
        operation_id=result.operation_id,
        status=result.status,
        items=tuple(
            DurableGeneratedImage(
                index=item.index,
                status=item.status,
                asset=by_index.get(item.index),
                revised_prompt=item.revised_prompt,
                error_code=item.error_code,
                error_message=item.error_message,
            )
            for item in result.items
        ),
        route=result.route,
        profile_version=result.profile_version,
        error_code=result.error_code,
        error_message=result.error_message,
        retry_after_seconds=result.retry_after_seconds,
        provider_request_id=result.provider_request_id,
        request_accepted=result.request_accepted,
    )


def _load_operation(
    conn: sqlite3.Connection, operation_id: str,
) -> DurableImageGenerationResult | None:
    operation = conn.execute(
        "SELECT * FROM image_generation_operations WHERE operation_id = ?",
        (operation_id,),
    ).fetchone()
    if operation is None:
        return None
    rows = conn.execute(
        """SELECT i.item_index, i.status, i.revised_prompt, i.error_code,
                  i.error_message, a.asset_id, a.sha256, a.mime_type,
                  a.byte_count, a.width, a.height
           FROM image_generation_items AS i
           LEFT JOIN image_assets AS a ON a.asset_id = i.asset_id
           WHERE i.operation_id = ? ORDER BY i.item_index""",
        (operation_id,),
    ).fetchall()
    items = tuple(
        DurableGeneratedImage(
            index=row["item_index"],
            status=GeneratedImageStatus(row["status"]),
            asset=_asset_from_row(row) if row["asset_id"] else None,
            revised_prompt=row["revised_prompt"],
            error_code=row["error_code"],
            error_message=row["error_message"],
        )
        for row in rows
    )
    return DurableImageGenerationResult(
        operation_id=operation["operation_id"],
        status=ImageOperationStatus(operation["status"]),
        items=items,
        route=ImageGenerationRoute(
            provider=operation["provider"],
            model=operation["model"],
            endpoint=operation["endpoint"],
            transport=operation["transport"],
            adapter=operation["adapter"],
            mechanism=operation["mechanism"],
            operation=operation["operation"],
            revision=operation["revision"],
            credential_type=operation["credential_type"],
            account_id=operation["account_id"],
            project_id=operation["generation_project_id"],
            credential_id=operation["credential_id"],
        ),
        profile_version=operation["profile_version"],
        error_code=operation["error_code"],
        error_message=operation["error_message"],
        retry_after_seconds=operation["retry_after_seconds"],
        provider_request_id=operation["provider_request_id"],
        request_accepted=_db_to_bool(operation["request_accepted"]),
    )


def _asset_from_row(row: sqlite3.Row) -> ImageAsset:
    return ImageAsset(
        asset_id=row["asset_id"],
        sha256=row["sha256"],
        mime_type=row["mime_type"],
        byte_count=row["byte_count"],
        width=row["width"],
        height=row["height"],
    )


def _normalize_asset_id(value: str) -> str:
    raw = value.removeprefix("infinidev-image://")
    if not raw.startswith("sha256:") or len(raw) != 71:
        raise ImageAssetNotFoundError("invalid generated-image reference")
    digest = raw[7:]
    if any(char not in "0123456789abcdef" for char in digest):
        raise ImageAssetNotFoundError("invalid generated-image reference")
    return raw


def _bool_to_db(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _db_to_bool(value: int | None) -> bool | None:
    return None if value is None else bool(value)
