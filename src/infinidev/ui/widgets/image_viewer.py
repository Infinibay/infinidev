"""TUI Image Viewer using half-block characters (▀▄).

Each terminal character cell represents 2 vertical pixels by using the
upper-half-block character (▀) with foreground = top pixel color and
background = bottom pixel color.

Rendering pipeline (auto-detected, best available):
  1. CUDA (cupy)  — GPU-accelerated resize + pixel math
  2. NumPy        — vectorized CPU (10-50x faster than pure Python)
  3. Pure Python  — fallback, no extra dependencies

Requires: Pillow (PIL)
Optional: numpy (recommended), cupy-cuda* (for GPU acceleration)
"""

from __future__ import annotations

import logging
import pathlib
import threading
from typing import Optional

from rich.color import Color
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

from textual.binding import Binding
from textual.reactive import reactive
from textual.strip import Strip
from textual.widget import Widget

logger = logging.getLogger(__name__)

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

try:
    import cupy as cp
    _test = cp.zeros(1)  # Force CUDA init to verify it actually works
    HAS_CUPY = True
    logger.info("CUDA image rendering available via CuPy")
except Exception:
    HAS_CUPY = False

# Upper half block — fg = top pixel, bg = bottom pixel
_HALF_BLOCK = "\u2580"

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    ".ico", ".tiff", ".tif",
}


def is_image_file(path: str) -> bool:
    """Check if a file path is a supported image."""
    return pathlib.Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _get_backend() -> str:
    """Return the best available rendering backend."""
    if HAS_CUPY:
        return "cuda"
    if HAS_NUMPY:
        return "numpy"
    return "python"


# ── Pixel processing backends ───────────────────────────────────────────


def _process_pixels_cuda(
    img: "Image.Image", width: int, pixel_h: int
) -> "np.ndarray":
    """CUDA-accelerated pixel processing via CuPy.

    Returns a numpy array of shape (rows, width, 6) with quantized
    RGB values for top and bottom pixels of each half-block row.
    """
    resized = img.resize((width, pixel_h), Image.LANCZOS)
    # Transfer to GPU
    arr = cp.asarray(np.array(resized))  # (pixel_h, width, 4) uint8 RGBA

    # Separate channels
    rgb = arr[:, :, :3].astype(cp.float32)
    alpha = arr[:, :, 3:4].astype(cp.float32) / 255.0

    # Alpha premultiply
    rgb = (rgb * alpha).astype(cp.uint8)

    # Quantize to 4-bit (& 0xF0)
    rgb = rgb & 0xF0

    # Pair top/bottom rows
    top_rows = rgb[0::2]  # even rows
    if pixel_h % 2 == 0:
        bot_rows = rgb[1::2]
    else:
        bot_rows = cp.zeros_like(top_rows)
        bot_rows[:-1] = rgb[1::2]

    # Stack: (num_char_rows, width, 6) = [r1,g1,b1, r2,g2,b2]
    paired = cp.concatenate([top_rows, bot_rows], axis=2)

    # Transfer back to CPU
    return cp.asnumpy(paired)


def _process_pixels_numpy(
    img: "Image.Image", width: int, pixel_h: int
) -> "np.ndarray":
    """NumPy-vectorized pixel processing.

    Returns a numpy array of shape (rows, width, 6) with quantized
    RGB values for top and bottom pixels of each half-block row.
    """
    resized = img.resize((width, pixel_h), Image.LANCZOS)
    arr = np.array(resized)  # (pixel_h, width, 4) uint8 RGBA

    # Alpha premultiply (vectorized)
    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    rgb = (rgb * alpha).astype(np.uint8)

    # Quantize
    rgb = rgb & 0xF0

    # Pair rows
    top_rows = rgb[0::2]  # (char_rows, width, 3)
    num_char_rows = top_rows.shape[0]

    bot_rows = np.zeros_like(top_rows)
    actual_bot = rgb[1::2]
    bot_rows[:actual_bot.shape[0]] = actual_bot

    # Concatenate: (char_rows, width, 6)
    return np.concatenate([top_rows, bot_rows], axis=2)


# ── Strip builder (shared by all backends) ──────────────────────────────


def _paired_to_strips(paired: "np.ndarray", width: int) -> list[Strip]:
    """Convert paired pixel array to Textual Strips with run-length encoding.

    Args:
        paired: shape (num_rows, width, 6) — [r1,g1,b1, r2,g2,b2] per cell
        width: terminal columns
    """
    style_cache: dict[tuple, Style] = {}
    strips: list[Strip] = []

    num_rows = paired.shape[0]
    for y in range(num_rows):
        row = paired[y]  # (width, 6)
        segments: list[Segment] = []
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
                        st = Style(
                            color=Color.from_rgb(prev_key[0], prev_key[1], prev_key[2]),
                            bgcolor=Color.from_rgb(prev_key[3], prev_key[4], prev_key[5]),
                        )
                        style_cache[prev_key] = st
                    segments.append(Segment(_HALF_BLOCK * run_len, st))
                prev_key = key
                run_len = 1

        # Flush last run
        if prev_key is not None and run_len > 0:
            st = style_cache.get(prev_key)
            if st is None:
                st = Style(
                    color=Color.from_rgb(prev_key[0], prev_key[1], prev_key[2]),
                    bgcolor=Color.from_rgb(prev_key[3], prev_key[4], prev_key[5]),
                )
                style_cache[prev_key] = st
            segments.append(Segment(_HALF_BLOCK * run_len, st))

        strips.append(Strip(segments, width))

    return strips


def _render_strips_python(
    img: "Image.Image", width: int, pixel_h: int
) -> list[Strip]:
    """Pure Python fallback — no numpy/cupy needed."""
    resized = img.resize((width, pixel_h), Image.LANCZOS)
    raw = list(resized.getdata())

    style_cache: dict[tuple, Style] = {}
    strips: list[Strip] = []

    for y in range(0, pixel_h, 2):
        segments: list[Segment] = []
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
                        st = Style(
                            color=Color.from_rgb(prev_key[0], prev_key[1], prev_key[2]),
                            bgcolor=Color.from_rgb(prev_key[3], prev_key[4], prev_key[5]),
                        )
                        style_cache[prev_key] = st
                    segments.append(Segment(_HALF_BLOCK * run_len, st))
                prev_key = key
                run_len = 1

        if prev_key is not None and run_len > 0:
            st = style_cache.get(prev_key)
            if st is None:
                st = Style(
                    color=Color.from_rgb(prev_key[0], prev_key[1], prev_key[2]),
                    bgcolor=Color.from_rgb(prev_key[3], prev_key[4], prev_key[5]),
                )
                style_cache[prev_key] = st
            segments.append(Segment(_HALF_BLOCK * run_len, st))

        strips.append(Strip(segments, width))

    return strips


# ── Main entry point ────────────────────────────────────────────────────


def _render_strips(
    img: "Image.Image",
    width: int,
    height_chars: int | None = None,
) -> list[Strip]:
    """Convert a PIL Image to Textual Strips using the best available backend."""
    img = img.convert("RGBA")
    orig_w, orig_h = img.size
    if orig_w == 0 or orig_h == 0:
        return [Strip([Segment("(empty image)")])]

    scale = width / orig_w
    pixel_h = int(orig_h * scale)
    if height_chars is not None:
        pixel_h = min(pixel_h, height_chars * 2)
    pixel_h = max(2, pixel_h - (pixel_h % 2))

    backend = _get_backend()

    if backend == "cuda":
        paired = _process_pixels_cuda(img, width, pixel_h)
        return _paired_to_strips(paired, width)
    elif backend == "numpy":
        paired = _process_pixels_numpy(img, width, pixel_h)
        return _paired_to_strips(paired, width)
    else:
        return _render_strips_python(img, width, pixel_h)


# ── Widget ──────────────────────────────────────────────────────────────


class ImageViewer(Widget):
    """Textual widget that displays an image using half-block characters.

    The image is rendered once in a background thread and cached.
    Zoom changes trigger a re-render.
    """

    BINDINGS = [
        Binding("plus", "zoom_in", "+Zoom", show=True),
        Binding("equals", "zoom_in", "+Zoom", show=False),
        Binding("minus", "zoom_out", "-Zoom", show=True),
        Binding("0", "zoom_reset", "Reset", show=True),
        Binding("f", "zoom_fit", "Fit", show=True),
    ]

    DEFAULT_CSS = """
    ImageViewer {
        height: 100%;
        width: 100%;
        overflow-y: auto;
        overflow-x: auto;
    }
    """

    zoom_level: reactive[float] = reactive(1.0)

    def __init__(
        self,
        image_path: str,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._image_path = image_path
        self._img: Optional["Image.Image"] = None
        self._cached_strips: list[Strip] = []
        self._cache_key: tuple[int, int, float] = (0, 0, 0.0)
        self._info_strip: Strip | None = None
        self._error: str = ""
        self._loading = True
        self.can_focus = True

    def on_mount(self) -> None:
        self._load_image()

    def _load_image(self) -> None:
        """Load the image from disk."""
        if not HAS_PIL:
            self._error = "Pillow not installed. Run: pip install Pillow"
            self._loading = False
            self.refresh()
            return
        try:
            self._img = Image.open(self._image_path)
            self._img.load()
            self._loading = False
            self._rebuild_cache()
        except Exception as e:
            self._error = f"Cannot load image: {e}"
            self._loading = False
            self.refresh()

    def _rebuild_cache(self) -> None:
        """Rebuild the strip cache in a background thread."""
        if self._img is None:
            return

        avail_w = max(20, self.size.width)
        avail_h = max(10, self.size.height - 1)
        new_key = (avail_w, avail_h, self.zoom_level)

        if new_key == self._cache_key and self._cached_strips:
            return

        self._cache_key = new_key
        img = self._img
        zoom = self.zoom_level

        render_w = min(int(avail_w * zoom), 500)
        render_h = int(avail_h * zoom)

        def _bg_render():
            backend = _get_backend()
            strips = _render_strips(img, render_w, render_h)
            # Build info strip
            w, h = img.size
            fmt = img.format or "?"
            mode = img.mode
            try:
                size_kb = pathlib.Path(self._image_path).stat().st_size / 1024
            except OSError:
                size_kb = 0
            info = (
                f" {pathlib.Path(self._image_path).name}  |  "
                f"{w}\u00d7{h}  |  {fmt}/{mode}  |  "
                f"{size_kb:.1f} KB  |  "
                f"Zoom {zoom:.0%}  |  "
                f"[{backend}]  |  "
                f"+/- zoom  0 reset  F fit"
            )
            info_seg = [Segment(info, Style(bold=True, dim=True))]
            self._info_strip = Strip(info_seg, len(info))
            self._cached_strips = strips
            try:
                self.app.call_from_thread(self.refresh)
            except Exception:
                pass

        thread = threading.Thread(target=_bg_render, daemon=True)
        thread.start()

    def get_content_height(self, container, viewport, width):
        """Tell Textual how many rows we need for scrolling."""
        if self._cached_strips:
            return len(self._cached_strips) + 1
        return self.size.height

    def render_line(self, y: int) -> Strip:
        """Render a single line — Textual calls this for visible lines only."""
        if self._error:
            if y == 0:
                return Strip([Segment(self._error, Style(color="red", bold=True))])
            return Strip.blank(self.size.width)

        if self._loading:
            if y == 0:
                return Strip([Segment("Loading image...", Style(dim=True))])
            return Strip.blank(self.size.width)

        if not self._cached_strips:
            if y == 0:
                return Strip([Segment("Rendering...", Style(dim=True))])
            return Strip.blank(self.size.width)

        if y == 0:
            return self._info_strip or Strip.blank(self.size.width)

        img_row = y - 1
        if img_row < len(self._cached_strips):
            return self._cached_strips[img_row]

        return Strip.blank(self.size.width)

    def watch_zoom_level(self, new_zoom: float) -> None:
        self._rebuild_cache()

    def on_resize(self, event) -> None:
        self._rebuild_cache()

    def action_zoom_in(self) -> None:
        self.zoom_level = min(5.0, self.zoom_level + 0.25)

    def action_zoom_out(self) -> None:
        self.zoom_level = max(0.25, self.zoom_level - 0.25)

    def action_zoom_reset(self) -> None:
        self.zoom_level = 1.0

    def action_zoom_fit(self) -> None:
        if self._img is None:
            return
        w, h = self._img.size
        avail_w = max(20, self.size.width)
        avail_h = max(10, (self.size.height - 1) * 2)
        scale_w = avail_w / w
        scale_h = avail_h / h
        self.zoom_level = round(min(scale_w, scale_h), 2)
