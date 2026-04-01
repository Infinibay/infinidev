"""Image viewer control — renders images using half-block characters.

Wraps the existing pixel processing backends (CUDA/NumPy/Python) from
infinidev.ui.widgets.image_viewer but outputs prompt_toolkit FormattedText
instead of Textual Strip objects.
"""

from __future__ import annotations

import logging
import pathlib
import threading

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import UIControl, UIContent
from prompt_toolkit.key_binding import KeyBindings

from typing import Any

from infinidev.ui.theme import TEXT, TEXT_MUTED, IMAGE_VIEWER_BG

logger = logging.getLogger(__name__)

# Half-block character
_HALF_BLOCK = "\u2580"

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _paired_to_lines(paired, width: int) -> list[list[tuple[str, str]]]:
    """Convert paired pixel array to prompt_toolkit formatted text lines.

    Args:
        paired: numpy array of shape (num_rows, width, 6) — [r1,g1,b1, r2,g2,b2]
        width: terminal columns
    Returns:
        List of line fragments: [(style_string, text), ...]
    """
    style_cache: dict[tuple, str] = {}
    lines: list[list[tuple[str, str]]] = []

    num_rows = paired.shape[0]
    for y in range(num_rows):
        row = paired[y]
        fragments: list[tuple[str, str]] = []
        prev_key: tuple | None = None
        run_len = 0

        for x in range(width):
            px = row[x]
            key = (int(px[0]), int(px[1]), int(px[2]),
                   int(px[3]), int(px[4]), int(px[5]))

            if key == prev_key:
                run_len += 1
            else:
                if prev_key is not None and run_len > 0:
                    st = style_cache.get(prev_key)
                    if st is None:
                        fg = f"#{prev_key[0]:02x}{prev_key[1]:02x}{prev_key[2]:02x}"
                        bg = f"#{prev_key[3]:02x}{prev_key[4]:02x}{prev_key[5]:02x}"
                        st = f"{fg} bg:{bg}"
                        style_cache[prev_key] = st
                    fragments.append((st, _HALF_BLOCK * run_len))
                prev_key = key
                run_len = 1

        if prev_key is not None and run_len > 0:
            st = style_cache.get(prev_key)
            if st is None:
                fg = f"#{prev_key[0]:02x}{prev_key[1]:02x}{prev_key[2]:02x}"
                bg = f"#{prev_key[3]:02x}{prev_key[4]:02x}{prev_key[5]:02x}"
                st = f"{fg} bg:{bg}"
                style_cache[prev_key] = st
            fragments.append((st, _HALF_BLOCK * run_len))

        lines.append(fragments)

    return lines


def _render_python_lines(img: "Image.Image", width: int, pixel_h: int) -> list[list[tuple[str, str]]]:
    """Pure Python fallback renderer — outputs prompt_toolkit lines directly."""
    resized = img.resize((width, pixel_h), Image.LANCZOS)
    raw = list(resized.getdata())

    style_cache: dict[tuple, str] = {}
    lines: list[list[tuple[str, str]]] = []

    for y in range(0, pixel_h, 2):
        fragments: list[tuple[str, str]] = []
        row_top = y * width
        row_bot = (y + 1) * width if y + 1 < pixel_h else None
        prev_key: tuple | None = None
        run_len = 0

        for x in range(width):
            r1, g1, b1, a1 = raw[row_top + x]
            if a1 < 255:
                f = a1 / 255.0
                r1, g1, b1 = int(r1 * f), int(g1 * f), int(b1 * f)

            if row_bot is not None:
                r2, g2, b2, a2 = raw[row_bot + x]
                if a2 < 255:
                    f = a2 / 255.0
                    r2, g2, b2 = int(r2 * f), int(g2 * f), int(b2 * f)
            else:
                r2, g2, b2 = 0, 0, 0

            key = (r1 & 0xF0, g1 & 0xF0, b1 & 0xF0,
                   r2 & 0xF0, g2 & 0xF0, b2 & 0xF0)

            if key == prev_key:
                run_len += 1
            else:
                if prev_key is not None and run_len > 0:
                    st = style_cache.get(prev_key)
                    if st is None:
                        fg = f"#{prev_key[0]:02x}{prev_key[1]:02x}{prev_key[2]:02x}"
                        bg = f"#{prev_key[3]:02x}{prev_key[4]:02x}{prev_key[5]:02x}"
                        st = f"{fg} bg:{bg}"
                        style_cache[prev_key] = st
                    fragments.append((st, _HALF_BLOCK * run_len))
                prev_key = key
                run_len = 1

        if prev_key is not None and run_len > 0:
            st = style_cache.get(prev_key)
            if st is None:
                fg = f"#{prev_key[0]:02x}{prev_key[1]:02x}{prev_key[2]:02x}"
                bg = f"#{prev_key[3]:02x}{prev_key[4]:02x}{prev_key[5]:02x}"
                st = f"{fg} bg:{bg}"
                style_cache[prev_key] = st
            fragments.append((st, _HALF_BLOCK * run_len))

        lines.append(fragments)

    return lines


class ImageViewerControl(UIControl):
    """UIControl that renders an image using half-block characters.

    Uses the best available backend (CUDA > NumPy > Python).
    Supports zoom via +/- keys.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._zoom: float = 1.0
        self._lines: list[list[tuple[str, str]]] = []
        self._loading = True
        self._error: str | None = None
        self._img: Any = None
        self._img_width: int = 0
        self._img_height: int = 0
        self._render_lock = threading.Lock()

        if not HAS_PIL:
            self._loading = False
            self._error = "Pillow not installed"
            return

        # Load image
        try:
            self._img = Image.open(file_path).convert("RGBA")
            self._img_width, self._img_height = self._img.size
            self._loading = False
        except Exception as e:
            self._loading = False
            self._error = str(e)

    @property
    def zoom(self) -> float:
        return self._zoom

    def zoom_in(self) -> None:
        self._zoom = min(self._zoom * 1.5, 10.0)
        self._lines.clear()

    def zoom_out(self) -> None:
        self._zoom = max(self._zoom / 1.5, 0.1)
        self._lines.clear()

    def zoom_reset(self) -> None:
        self._zoom = 1.0
        self._lines.clear()

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        if self._error:
            return UIContent(
                get_line=lambda i: [(f"{TEXT_MUTED}", f" Error: {self._error}")] if i == 0 else [],
                line_count=1,
            )

        if self._loading or self._img is None:
            return UIContent(
                get_line=lambda i: [(f"{TEXT_MUTED}", " Loading image...")] if i == 0 else [],
                line_count=1,
            )

        # Render if cache is empty
        if not self._lines:
            self._render(width)

        lines = self._lines
        # Info line at top
        backend = "cuda" if HAS_NUMPY and hasattr(self, '_used_cuda') else ("numpy" if HAS_NUMPY else "python")
        info = f" {pathlib.Path(self.file_path).name} | {self._img_width}x{self._img_height} | zoom: {self._zoom:.1f}x"
        info_line = [(f"{TEXT_MUTED}", info)]

        all_lines = [info_line] + lines

        def get_line(i: int) -> list[tuple[str, str]]:
            if 0 <= i < len(all_lines):
                return all_lines[i]
            return []

        return UIContent(
            get_line=get_line,
            line_count=len(all_lines),
            cursor_position=None,
            show_cursor=False,
        )

    def _render(self, term_width: int) -> None:
        """Render the image to formatted text lines."""
        if self._img is None:
            return

        with self._render_lock:
            render_width = max(10, int(term_width * self._zoom))
            aspect = self._img_height / max(self._img_width, 1)
            pixel_h = int(render_width * aspect)
            if pixel_h % 2 != 0:
                pixel_h += 1
            pixel_h = max(2, pixel_h)

            if HAS_NUMPY:
                from infinidev.ui.widgets.image_viewer import (
                    _process_pixels_numpy, _process_pixels_cuda,
                    HAS_CUPY,
                )
                try:
                    if HAS_CUPY:
                        paired = _process_pixels_cuda(self._img, render_width, pixel_h)
                        self._used_cuda = True
                    else:
                        paired = _process_pixels_numpy(self._img, render_width, pixel_h)
                    self._lines = _paired_to_lines(paired, render_width)
                except Exception:
                    self._lines = _render_python_lines(self._img, render_width, pixel_h)
            else:
                self._lines = _render_python_lines(self._img, render_width, pixel_h)


def create_image_keybindings(viewer: ImageViewerControl) -> KeyBindings:
    """Create key bindings for the image viewer."""
    kb = KeyBindings()

    @kb.add("+")
    @kb.add("=")
    def zoom_in(event):
        viewer.zoom_in()

    @kb.add("-")
    def zoom_out(event):
        viewer.zoom_out()

    @kb.add("0")
    def zoom_reset(event):
        viewer.zoom_reset()

    return kb
