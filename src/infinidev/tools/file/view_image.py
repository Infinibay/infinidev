"""Tool for loading an image file into the LLM context.

Only useful when the configured model supports vision — ``get_tools_for_role``
filters this tool out when ``supports_vision`` is False so it never reaches
the schema that the LLM sees.

The tool itself does not send anything to the model. It returns a
``ToolResult`` whose ``attachments`` list the loop engine detects after the
``role=tool`` message and pushes a follow-up multimodal ``role=user`` message
with an ``image_url`` block per attachment.
"""

from __future__ import annotations

import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool, ToolResult


class ViewImageInput(BaseModel):
    file_path: str = Field(
        ...,
        description=(
            "Absolute or workspace-relative path to an image file "
            "(.png, .jpg, .jpeg, .gif, .webp, .bmp)."
        ),
    )


class ViewImageTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "view_image"
    description: str = (
        "Load an image file so the next model turn can see it. Supported "
        "formats: PNG, JPEG, GIF, WEBP, BMP. Only available when the "
        "configured model supports vision — otherwise this tool is not "
        "exposed. Returns a short description; the image itself is injected "
        "into the conversation automatically."
    )
    args_schema: Type[BaseModel] = ViewImageInput

    def _run(self, file_path: str) -> ToolResult:
        from infinidev.engine.multimodal import AttachmentError, load_image

        resolved = self._resolve_path(os.path.expanduser(file_path))

        sandbox_err = self._validate_sandbox_path(resolved)
        if sandbox_err:
            return ToolResult(text=self._error(sandbox_err))

        try:
            attachment = load_image(resolved)
        except AttachmentError as exc:
            return ToolResult(text=self._error(str(exc)))
        except Exception as exc:  # noqa: BLE001 — surface as tool error
            return ToolResult(text=self._error(f"failed to load image: {exc}"))

        text = (
            f"Loaded image {attachment.path.name} "
            f"({attachment.mime_type}, "
            f"{attachment.size_bytes // 1024} KB"
            + (
                f", {attachment.width}x{attachment.height}"
                if attachment.width and attachment.height
                else ""
            )
            + "). The image is now visible in the next turn."
        )
        return ToolResult(text=text, attachments=[attachment])
