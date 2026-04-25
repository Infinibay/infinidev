"""Multimodal content helpers.

Converts image files to the OpenAI-style ``image_url`` content blocks that
LiteLLM forwards to any vision-capable provider (OpenAI, Anthropic, Gemini,
Ollama llava/qwen-vl, etc.).

Two entry points:

- ``build_user_content(text, attachments)`` — returns a plain string when
  there are no attachments (zero-impact for non-vision providers) or a list
  of content blocks when there are.
- ``mention_paths_as_text(text, attachments)`` — fallback for non-vision
  models: adds human-readable markers so the model at least knows images
  were referenced.

Attachments are loaded eagerly into base64 data URLs. They are *ephemeral*
— callers keep them in memory for the duration of a turn and drop them
after; we deliberately do not persist base64 blobs to the DB.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
)

# Explicit mime map — ``mimetypes`` stdlib is locale-dependent and sometimes
# returns None for .webp on older platforms.
_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

# Hard cap to avoid accidentally sending a 50 MB scan to the model.
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


@dataclass
class ImageAttachment:
    """An image prepared for inclusion in an LLM message."""

    path: Path
    mime_type: str
    data_url: str
    size_bytes: int
    width: int | None = None
    height: int | None = None

    def short_repr(self) -> str:
        """Human-readable summary for log lines and UI chips."""
        dims = f"{self.width}x{self.height}" if self.width and self.height else "?"
        kb = self.size_bytes // 1024
        return f"{self.path.name} ({dims}, {kb} KB)"


class AttachmentError(ValueError):
    """Raised when an attachment cannot be loaded (missing, too large, etc.)."""


def is_image_path(path: str | Path) -> bool:
    """Extension-only check. Cheap — use before touching the filesystem."""
    suffix = Path(path).suffix.lower()
    return suffix in IMAGE_EXTENSIONS


def load_image(path: str | Path) -> ImageAttachment:
    """Load an image file into an ``ImageAttachment`` ready for LLM input.

    Raises ``AttachmentError`` on any validation failure.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise AttachmentError(f"image not found: {p}")
    if not p.is_file():
        raise AttachmentError(f"not a file: {p}")

    suffix = p.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise AttachmentError(
            f"unsupported image extension {suffix!r} "
            f"(supported: {sorted(IMAGE_EXTENSIONS)})"
        )

    size = p.stat().st_size
    if size > _MAX_IMAGE_BYTES:
        raise AttachmentError(
            f"image too large: {size // 1024 // 1024} MB "
            f"(max {_MAX_IMAGE_BYTES // 1024 // 1024} MB)"
        )
    if size == 0:
        raise AttachmentError(f"image is empty: {p}")

    mime = _MIME_BY_EXT.get(suffix) or mimetypes.guess_type(str(p))[0] or "image/png"

    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    width, height = _probe_dimensions(p)

    return ImageAttachment(
        path=p,
        mime_type=mime,
        data_url=data_url,
        size_bytes=size,
        width=width,
        height=height,
    )


# ── URL-sourced attachments ────────────────────────────────────────────────
#
# Supports two remote/pasted forms so drag-drop over SSH becomes workable:
#
# 1. ``data:image/<mime>;base64,<blob>`` — the user pastes a self-contained
#    base64 image. Common recipe on the local side:
#      ``base64 foo.png | xclip -selection clipboard -in``  (+ ``data:image/png;base64,``)
#    We decode to validate and rebuild a canonical data URL so downstream
#    providers see clean input.
#
# 2. ``https://.../pic.png`` — a public image URL. We do NOT download it
#    ourselves; LiteLLM / the vision provider fetches it server-side (the
#    ``image_url`` content block accepts plain URLs just as well as data URLs).

_DATA_URL_RE = re.compile(
    r"data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)",
)


def load_data_url(url: str) -> "ImageAttachment":
    """Parse a ``data:image/...;base64,...`` URL into an ``ImageAttachment``.

    The base64 payload is decoded for validation and to measure size; an
    invalid URL raises ``AttachmentError``. A synthetic ``path`` with a
    placeholder name is used so downstream code that does ``path.name`` etc.
    keeps working.
    """
    m = _DATA_URL_RE.match(url.strip())
    if not m:
        raise AttachmentError(
            f"not a valid data URL "
            f"(expected 'data:image/<mime>;base64,<blob>'): "
            f"{url[:60]}..."
        )
    mime, b64 = m.group(1), m.group(2)
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as exc:
        raise AttachmentError(f"invalid base64 in data URL: {exc}")
    if not raw:
        raise AttachmentError("data URL contains empty image")
    if len(raw) > _MAX_IMAGE_BYTES:
        raise AttachmentError(
            f"data URL image too large: "
            f"{len(raw) // 1024 // 1024} MB (max "
            f"{_MAX_IMAGE_BYTES // 1024 // 1024} MB)"
        )

    # Rebuild the URL from validated bytes so whatever the LLM sees is
    # clean — no stray whitespace, no non-canonical base64 padding.
    clean_b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:{mime};base64,{clean_b64}"

    # Synthetic path: preserves UX (short_repr, mention_paths_as_text) without
    # pretending a real file exists on disk.
    ext = mime.split("/", 1)[1].split("+", 1)[0]
    synthetic = Path(f"<pasted.{ext}>")

    return ImageAttachment(
        path=synthetic,
        mime_type=mime,
        data_url=data_url,
        size_bytes=len(raw),
        width=None,
        height=None,
    )


def load_http_url(url: str) -> "ImageAttachment":
    """Wrap a public ``http(s)://.../image.png`` URL as an ``ImageAttachment``.

    We do NOT fetch the image here — LiteLLM / the provider does it
    server-side. We only validate that the URL *looks* like an image.
    """
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise AttachmentError(f"expected http or https URL, got: {url[:60]}")
    if not parsed.netloc:
        raise AttachmentError(f"URL has no host: {url[:60]}")

    suffix = Path(parsed.path).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise AttachmentError(
            f"URL has no image extension ({sorted(IMAGE_EXTENSIONS)}): "
            f"{url}"
        )
    mime = _MIME_BY_EXT.get(suffix, "image/png")

    return ImageAttachment(
        path=Path(parsed.path or url),
        mime_type=mime,
        data_url=url,  # passed through as-is to the provider
        size_bytes=0,
        width=None,
        height=None,
    )


def _probe_dimensions(path: Path) -> tuple[int | None, int | None]:
    """Return (width, height) if Pillow can open the file; else (None, None).

    Failure to read dimensions is not fatal — we still ship the image.
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size
    except Exception as exc:
        logger.debug("could not probe dimensions for %s: %s", path, exc)
        return (None, None)


def build_user_content(
    text: str, attachments: Iterable[ImageAttachment] | None
) -> str | list[dict[str, Any]]:
    """Build the ``content`` field for a ``role=user`` message.

    - No attachments → plain string (keeps behavior identical for providers
      that only accept string content).
    - Attachments → OpenAI-style content-block list. Text goes first, then
      one ``image_url`` block per attachment.
    """
    att_list = list(attachments or [])
    if not att_list:
        return text

    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for att in att_list:
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": att.data_url},
            }
        )
    return blocks


def mention_paths_as_text(
    text: str, attachments: Iterable[ImageAttachment] | None
) -> str:
    """Fallback rendering when the model does NOT support vision.

    Adds a trailing note listing the attachment paths so the model knows the
    user referenced images even if it cannot see them.
    """
    att_list = list(attachments or [])
    if not att_list:
        return text
    lines = [
        f"- {att.path} ({att.mime_type}, {att.size_bytes // 1024} KB)"
        for att in att_list
    ]
    note = (
        "\n\n[User attached image(s), but the current model does not support "
        "vision — paths listed for reference only:]\n" + "\n".join(lines)
    )
    return (text or "") + note
