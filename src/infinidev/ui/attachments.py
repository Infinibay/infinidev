"""Attachment tray + path parser for the TUI.

Drag-and-drop from terminals (gnome-terminal, iTerm2, kitty) pastes the
dragged file's absolute path into the input buffer as plain text. Instead
of trying to capture a real file-drop event (prompt_toolkit does not
expose one), we parse the submitted input: any whitespace-separated token
that resolves to an existing image file is extracted as an attachment and
removed from the text.

Explicit ``/attach <path>`` is the fallback when auto-detect misses (e.g.
paths containing unusual characters, or images outside the user's message).
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from infinidev.engine.multimodal import (
    IMAGE_EXTENSIONS,
    AttachmentError,
    load_data_url,
    load_http_url,
    load_image,
)

if TYPE_CHECKING:
    from infinidev.engine.multimodal import ImageAttachment

logger = logging.getLogger(__name__)


# Tokens that begin with one of these sigils are clearly NOT paths we want
# to accidentally swallow.
_NON_PATH_PREFIXES = ("-", "--", "!", "/", "#")
# Exception: if it actually resolves to an existing image file, keep it.

# Pasted ``data:image/<mime>;base64,<blob>`` URLs. Boundary is whitespace —
# base64 never contains whitespace, so ``\S+`` is tight enough. Anchoring on
# the scheme prefix avoids false positives on ordinary sentences.
_DATA_URL_PATTERN = re.compile(
    r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
)

# http(s) URLs whose path ends in an image extension (optionally followed by
# a query string or fragment). Matches the common CDN / signed-URL cases
# (``?token=...``) without greedily swallowing trailing punctuation.
_HTTP_IMAGE_PATTERN = re.compile(
    r"https?://[^\s'\"<>]+?"
    r"\.(?:png|jpe?g|gif|webp|bmp)"
    r"(?:\?[^\s'\"<>]*)?"
    r"(?:#[^\s'\"<>]*)?",
    re.IGNORECASE,
)


@dataclass
class AttachmentTray:
    """Holds images the user has queued for their next submit."""

    _items: list["ImageAttachment"] = field(default_factory=list)

    def add(self, attachment: "ImageAttachment") -> None:
        self._items.append(attachment)

    def extend(self, attachments: list["ImageAttachment"]) -> None:
        self._items.extend(attachments)

    def take_all(self) -> list["ImageAttachment"]:
        """Return the current items and clear the tray."""
        out = list(self._items)
        self._items.clear()
        return out

    def clear(self) -> None:
        self._items.clear()

    @property
    def items(self) -> list["ImageAttachment"]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def summary(self) -> str:
        if not self._items:
            return ""
        if len(self._items) == 1:
            return f"📎 1 image attached ({self._items[0].path.name})"
        return f"📎 {len(self._items)} images attached"


def _tokenize(text: str) -> list[str]:
    """Split user text into tokens, respecting quotes and escaped spaces.

    Falls back to whitespace-split on shlex errors (unbalanced quotes).
    """
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        return text.split()


def extract_image_paths(text: str) -> tuple[str, list[Path]]:
    """Scan ``text`` for tokens that resolve to image files.

    Returns ``(cleaned_text, paths)`` where ``cleaned_text`` has the matched
    tokens removed and ``paths`` is the list of paths that resolved.
    Non-image tokens and unresolvable paths are left in place.

    The check is: token's suffix is in ``IMAGE_EXTENSIONS`` AND the path
    exists and is a file. Both conditions are required so ordinary sentences
    mentioning ``.png`` filenames don't get silently swallowed.
    """
    tokens = _tokenize(text)
    paths: list[Path] = []
    removed: set[str] = set()

    for tok in tokens:
        raw = tok
        stripped = raw.strip("\"'")
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in _NON_PATH_PREFIXES) and not Path(
            stripped
        ).expanduser().exists():
            continue
        try:
            p = Path(stripped).expanduser()
        except Exception:
            continue
        suffix = p.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            continue
        try:
            if p.is_file():
                paths.append(p.resolve())
                removed.add(raw)
                # Also match the raw form as it appeared in the original
                # text (shlex may have unquoted it).
                removed.add(stripped)
        except OSError:
            continue

    if not removed:
        return text, paths

    cleaned = text
    for r in sorted(removed, key=len, reverse=True):
        cleaned = _remove_token(cleaned, r)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, paths


def _remove_token(text: str, token: str) -> str:
    """Remove a whole-word occurrence of ``token`` from ``text``.

    Preserves non-path text that happens to contain the same characters by
    requiring either a boundary at each end (whitespace or string edge) or
    a surrounding quote.
    """
    patterns = [
        r"(?:^|(?<=\s))'" + re.escape(token) + r"'(?=\s|$)",
        r'(?:^|(?<=\s))"' + re.escape(token) + r'"(?=\s|$)',
        r"(?:^|(?<=\s))" + re.escape(token) + r"(?=\s|$)",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text)
    return text


def load_attachments_from_paths(paths: list[Path]) -> tuple[list["ImageAttachment"], list[str]]:
    """Load every path into an ``ImageAttachment``.

    Returns ``(loaded, errors)`` — ``errors`` is a list of short strings
    describing any path that failed so the TUI can surface them inline.
    """
    loaded: list["ImageAttachment"] = []
    errors: list[str] = []
    for p in paths:
        try:
            loaded.append(load_image(p))
        except AttachmentError as exc:
            errors.append(f"{p}: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{p}: {exc}")
    return loaded, errors


def extract_urls(text: str) -> tuple[str, list[str], list[str]]:
    """Pull pasted image URLs out of ``text``.

    Returns ``(cleaned_text, data_urls, http_urls)``. Detection order
    matters: data URLs are extracted *first* (they contain ``/`` which
    would also match inside an http URL fragment), then http URLs from
    what remains.
    """
    data_urls = _DATA_URL_PATTERN.findall(text)
    without_data = _DATA_URL_PATTERN.sub(" ", text)
    http_urls = _HTTP_IMAGE_PATTERN.findall(without_data)
    cleaned = _HTTP_IMAGE_PATTERN.sub(" ", without_data)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, data_urls, http_urls


def load_attachment_source(source: str) -> "ImageAttachment":
    """Load a single source (data URL, http URL, or file path).

    Dispatches based on prefix; falls back to ``load_image`` for paths.
    Raises ``AttachmentError`` on any failure.
    """
    stripped = source.strip().strip("\"'")
    if stripped.startswith("data:image/"):
        return load_data_url(stripped)
    if stripped.startswith(("http://", "https://")):
        return load_http_url(stripped)
    return load_image(Path(stripped).expanduser())


def load_attachments_from_sources(
    sources: list[str],
) -> tuple[list["ImageAttachment"], list[str]]:
    """Batch form of ``load_attachment_source`` — collects errors per source."""
    loaded: list["ImageAttachment"] = []
    errors: list[str] = []
    for s in sources:
        try:
            loaded.append(load_attachment_source(s))
        except AttachmentError as exc:
            errors.append(f"{_short(s)}: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{_short(s)}: {exc}")
    return loaded, errors


def _short(s: str, n: int = 60) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 3] + "..."
