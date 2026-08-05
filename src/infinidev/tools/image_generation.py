"""Model-facing tool for explicit, durable image generation."""

from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.model_capabilities import CapabilitySnapshot, get_capability_snapshot
from infinidev.engine.image_generation import ImageGenerationRequest, ImageOperationStatus
from infinidev.engine.image_ledger import ImageGenerationService
from infinidev.engine.multimodal import ImageAttachment
from infinidev.tools.base.base_tool import InfinibayBaseTool, ToolResult


class GenerateImageInput(BaseModel):
    """Arguments accepted by :class:`GenerateImageTool`."""

    prompt: str = Field(..., description="Description of the image to generate.")
    count: int = Field(default=1, description="Number of images allowed by the exact profile.")
    size: str | None = Field(default=None, description="Exact profile image size.")
    quality: str | None = Field(default=None, description="Exact profile quality.")
    style: str | None = Field(default=None, description="Exact profile style.")
    operation_id: str | None = Field(
        default=None,
        description=(
            "Stable operation ID for reconciliation. Reusing it returns the durable "
            "result and never generates twice."
        ),
    )


class GenerateImageTool(InfinibayBaseTool):
    """Generate images only through an exact, separately configured route."""

    name: str = "generate_image"
    description: str = (
        "Generate one or more images with the separately configured image route. "
        "Returns durable infinidev-image:// references; repeated operation IDs never "
        "repeat a provider request."
    )
    args_schema: Type[BaseModel] = GenerateImageInput

    def __init__(
        self,
        *,
        snapshot: CapabilitySnapshot | None = None,
        service: ImageGenerationService | None = None,
        **data,
    ) -> None:
        super().__init__(**data)
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_service", service)

    def _run(
        self,
        prompt: str,
        count: int = 1,
        size: str | None = None,
        quality: str | None = None,
        style: str | None = None,
        operation_id: str | None = None,
    ) -> ToolResult:
        snapshot = self._snapshot or get_capability_snapshot()
        service = self._service or ImageGenerationService(snapshot=snapshot)
        request = ImageGenerationRequest(
            operation_id=operation_id or f"img-{uuid.uuid4().hex}",
            prompt=prompt,
            count=count,
            response_format="b64_json",
            size=size,
            quality=quality,
            style=style,
        )
        try:
            result = service.generate(
                request, session_id=self.session_id, project_id=self.project_id
            )
        except (RuntimeError, ValueError) as exc:
            return ToolResult(text=self._error(str(exc)))

        payload = {
            "operation_id": result.operation_id,
            "status": result.status.value,
            "images": [
                {
                    "index": item.index,
                    "status": item.status.value,
                    "reference": item.reference,
                    "mime_type": item.asset.mime_type if item.asset else None,
                    "byte_count": item.asset.byte_count if item.asset else None,
                    "width": item.asset.width if item.asset else None,
                    "height": item.asset.height if item.asset else None,
                    "revised_prompt": item.revised_prompt,
                    "error_code": item.error_code,
                    "error_message": item.error_message,
                }
                for item in result.items
            ],
            "error_code": result.error_code,
            "error_message": result.error_message,
            "retry_after_seconds": result.retry_after_seconds,
            "request_accepted": result.request_accepted,
        }
        attachments: list[ImageAttachment] = []
        if result.status is ImageOperationStatus.COMPLETE and snapshot.supports_vision:
            for item in result.items:
                if item.asset is None:
                    continue
                raw = service.read_asset(item.asset.asset_id)
                attachments.append(ImageAttachment(
                    path=Path(item.reference or "<generated-image>"),
                    mime_type=item.asset.mime_type,
                    data_url=(
                        f"data:{item.asset.mime_type};base64,"
                        + base64.b64encode(raw).decode("ascii")
                    ),
                    size_bytes=item.asset.byte_count,
                    width=item.asset.width,
                    height=item.asset.height,
                ))
        return ToolResult(
            text=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            attachments=attachments,
        )
