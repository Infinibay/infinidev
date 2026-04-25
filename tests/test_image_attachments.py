"""Tests for the image / vision support feature.

Covers:
- Vision capability detection (mocked via ``litellm.supports_vision``).
- TUI auto-detect parser ``extract_image_paths``.
- ``load_image`` → data URL roundtrip.
- ``build_user_content`` / ``mention_paths_as_text`` branching.
- ``view_image`` tool returns a ``ToolResult`` with one attachment.
- Tool-role filter hides ``view_image`` when vision is unavailable.

These are all unit-level — no live LLM calls.
"""

from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────


def _write_tiny_png(path: Path, width: int = 2, height: int = 2) -> None:
    """Write a valid 2x2 RGBA PNG so ``PIL.Image.open`` succeeds."""
    # Hand-rolled minimal PNG — avoids depending on Pillow in the producer.
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    # One filter byte per row, then RGBA pixels.
    raw = b"".join(b"\x00" + b"\xff\x00\x00\xff" * width for _ in range(height))
    idat = zlib.compress(raw)
    path.write_bytes(sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b""))


# ── multimodal module ─────────────────────────────────────────────────────


def test_load_image_roundtrip(tmp_path: Path) -> None:
    from infinidev.engine.multimodal import load_image

    img = tmp_path / "sample.png"
    _write_tiny_png(img)

    att = load_image(img)

    assert att.mime_type == "image/png"
    assert att.data_url.startswith("data:image/png;base64,")
    assert att.size_bytes > 0
    # Decoding the data URL must yield the same bytes.
    b64_payload = att.data_url.split(",", 1)[1]
    assert base64.b64decode(b64_payload) == img.read_bytes()


def test_load_image_rejects_non_image(tmp_path: Path) -> None:
    from infinidev.engine.multimodal import AttachmentError, load_image

    not_img = tmp_path / "hello.txt"
    not_img.write_text("not an image")
    with pytest.raises(AttachmentError):
        load_image(not_img)


def test_build_user_content_no_attachments_returns_string() -> None:
    from infinidev.engine.multimodal import build_user_content

    # Zero-impact on providers that only accept string content.
    assert build_user_content("hola", None) == "hola"
    assert build_user_content("hola", []) == "hola"


def test_build_user_content_with_attachments(tmp_path: Path) -> None:
    from infinidev.engine.multimodal import build_user_content, load_image

    img = tmp_path / "a.png"
    _write_tiny_png(img)
    att = load_image(img)

    content = build_user_content("describe", [att])
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_mention_paths_as_text_adds_footnote(tmp_path: Path) -> None:
    from infinidev.engine.multimodal import load_image, mention_paths_as_text

    img = tmp_path / "b.jpg"
    _write_tiny_png(img)  # content bytes don't have to be valid JPEG for this
    att = load_image.__wrapped__(img) if hasattr(load_image, "__wrapped__") else None
    # Build attachment manually since we can't load an invalid .jpg via Pillow.
    from infinidev.engine.multimodal import ImageAttachment

    manual = ImageAttachment(
        path=img, mime_type="image/jpeg",
        data_url="data:image/jpeg;base64,AA==",
        size_bytes=1, width=None, height=None,
    )
    out = mention_paths_as_text("hey", [manual])
    assert "hey" in out
    assert str(img) in out
    assert "does not support vision" in out


# ── TUI attachment parser ─────────────────────────────────────────────────


def test_extract_image_paths_picks_existing_image(tmp_path: Path) -> None:
    from infinidev.ui.attachments import extract_image_paths

    img = tmp_path / "pic.png"
    _write_tiny_png(img)

    text = f"describe this {img} please"
    cleaned, paths = extract_image_paths(text)

    assert paths == [img.resolve()]
    assert str(img) not in cleaned
    assert "describe this" in cleaned
    assert "please" in cleaned


def test_extract_image_paths_ignores_mentioned_nonexistent_paths() -> None:
    from infinidev.ui.attachments import extract_image_paths

    # A sentence mentioning a .png filename that doesn't exist on disk must
    # NOT be silently swallowed.
    text = "add logic to handle missing.png gracefully"
    cleaned, paths = extract_image_paths(text)
    assert paths == []
    assert "missing.png" in cleaned


def test_extract_image_paths_ignores_non_image_files(tmp_path: Path) -> None:
    from infinidev.ui.attachments import extract_image_paths

    txt = tmp_path / "readme.txt"
    txt.write_text("hi")
    text = f"look at {txt}"
    cleaned, paths = extract_image_paths(text)
    assert paths == []
    assert str(txt) in cleaned


def test_extract_image_paths_handles_quoted_path(tmp_path: Path) -> None:
    from infinidev.ui.attachments import extract_image_paths

    img = tmp_path / "my pic.png"
    _write_tiny_png(img)
    text = f"check '{img}' now"
    cleaned, paths = extract_image_paths(text)
    assert paths == [img.resolve()]
    assert "check" in cleaned and "now" in cleaned


# ── URL-sourced attachments (data URLs + http URLs) ──────────────────────


def test_load_data_url_roundtrip(tmp_path: Path) -> None:
    from infinidev.engine.multimodal import load_data_url, load_image

    img = tmp_path / "a.png"
    _write_tiny_png(img)
    original = load_image(img)

    # Parse back the canonical URL the loader generated.
    att = load_data_url(original.data_url)
    assert att.mime_type == "image/png"
    assert att.size_bytes == original.size_bytes
    # Canonical re-encoding should yield the same bytes.
    assert att.data_url == original.data_url


def test_load_data_url_rejects_non_image() -> None:
    from infinidev.engine.multimodal import AttachmentError, load_data_url

    with pytest.raises(AttachmentError):
        load_data_url("not a data url at all")
    with pytest.raises(AttachmentError):
        load_data_url("data:text/plain;base64,aGVsbG8=")  # text, not image
    with pytest.raises(AttachmentError):
        load_data_url("data:image/png;base64,@@@invalid@@@")  # bad b64


def test_load_http_url_wraps_url_without_fetching() -> None:
    from infinidev.engine.multimodal import load_http_url

    url = "https://example.com/path/pic.png?token=abc"
    att = load_http_url(url)
    assert att.mime_type == "image/png"
    assert att.data_url == url  # passed through as-is
    assert att.size_bytes == 0   # unknown — provider fetches server-side


def test_load_http_url_rejects_non_image_url() -> None:
    from infinidev.engine.multimodal import AttachmentError, load_http_url

    with pytest.raises(AttachmentError):
        load_http_url("https://example.com/page.html")
    with pytest.raises(AttachmentError):
        load_http_url("ftp://example.com/pic.png")  # wrong scheme
    with pytest.raises(AttachmentError):
        load_http_url("https:///pic.png")  # no host


def test_extract_urls_pulls_out_data_and_http() -> None:
    from infinidev.ui.attachments import extract_urls

    text = (
        "check this data:image/png;base64,AAAA and also "
        "https://example.com/cat.jpg thanks"
    )
    cleaned, data_urls, http_urls = extract_urls(text)
    assert data_urls == ["data:image/png;base64,AAAA"]
    assert http_urls == ["https://example.com/cat.jpg"]
    assert "check this" in cleaned
    assert "thanks" in cleaned
    assert "data:" not in cleaned
    assert "https://" not in cleaned


def test_extract_urls_ignores_non_image_http() -> None:
    from infinidev.ui.attachments import extract_urls

    text = "docs at https://example.com/readme and https://example.com/pic.png"
    cleaned, data_urls, http_urls = extract_urls(text)
    assert data_urls == []
    assert http_urls == ["https://example.com/pic.png"]
    assert "https://example.com/readme" in cleaned  # not an image — left alone


def test_load_attachment_source_dispatches(tmp_path: Path) -> None:
    from infinidev.ui.attachments import load_attachment_source

    img = tmp_path / "z.png"
    _write_tiny_png(img)

    # File path
    a = load_attachment_source(str(img))
    assert a.mime_type == "image/png"

    # Data URL
    b = load_attachment_source(a.data_url)
    assert b.mime_type == "image/png"
    assert str(b.path).startswith("<pasted")

    # http URL
    c = load_attachment_source("https://example.com/cat.webp")
    assert c.mime_type == "image/webp"
    assert c.data_url == "https://example.com/cat.webp"


# ── Tool gating + ToolResult ──────────────────────────────────────────────


def test_get_tools_for_role_filters_vision_only(monkeypatch) -> None:
    from infinidev.tools import get_tools_for_role

    no_vision = [t.name for t in get_tools_for_role("chat_agent", supports_vision=False)]
    with_vision = [t.name for t in get_tools_for_role("chat_agent", supports_vision=True)]

    assert "view_image" not in no_vision
    assert "view_image" in with_vision


def test_view_image_tool_returns_tool_result(tmp_path: Path) -> None:
    from infinidev.tools.base.base_tool import ToolResult
    from infinidev.tools.file.view_image import ViewImageTool

    img = tmp_path / "x.png"
    _write_tiny_png(img)
    tool = ViewImageTool()
    result = tool._run(file_path=str(img))
    assert isinstance(result, ToolResult)
    assert len(result.attachments) == 1
    assert result.attachments[0].mime_type == "image/png"
    assert "Loaded" in result.text


def test_view_image_tool_errors_on_missing(tmp_path: Path) -> None:
    from infinidev.tools.base.base_tool import ToolResult
    from infinidev.tools.file.view_image import ViewImageTool

    tool = ViewImageTool()
    result = tool._run(file_path=str(tmp_path / "missing.png"))
    assert isinstance(result, ToolResult)
    assert result.attachments == []
    assert "error" in result.text.lower()


def test_normalize_tool_result_handles_both_paths() -> None:
    from infinidev.tools.base.base_tool import ToolResult, normalize_tool_result

    text, atts = normalize_tool_result("just a string")
    assert text == "just a string" and atts == []

    tr = ToolResult(text="hi", attachments=[])
    text, atts = normalize_tool_result(tr)
    assert text == "hi" and atts == []


# ── Capability detection ──────────────────────────────────────────────────


def test_supports_vision_uses_litellm_metadata(monkeypatch) -> None:
    import infinidev.config.model_capabilities as mc

    calls: list[str] = []

    def fake_supports_vision(model: str) -> bool:
        calls.append(model)
        return model.startswith("openai/")

    import litellm
    from infinidev.config.settings import settings
    monkeypatch.setattr(litellm, "supports_vision", fake_supports_vision)
    monkeypatch.setattr(settings, "LLM_MODEL", "openai/gpt-4o", raising=False)
    assert mc._detect_vision_support() is True

    monkeypatch.setattr(settings, "LLM_MODEL", "ollama/qwen2.5-coder:7b", raising=False)
    assert mc._detect_vision_support() is False
    assert calls  # both calls happened


# ── Provider-native vision detection ──────────────────────────────────────


class _FakeResp:
    """Minimal stand-in for httpx.Response used by the capability probes."""

    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


def test_ollama_detects_vision_via_capabilities_field(monkeypatch) -> None:
    """Ollama 0.4+ reports `capabilities: [..., 'vision']`."""
    import httpx
    import infinidev.config.model_capabilities as mc
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "LLM_MODEL", "ollama_chat/llama3.2-vision", raising=False)
    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://localhost:11434", raising=False)

    payload = {
        "template": "{{ if .Tools }}...{{ end }}",
        "capabilities": ["completion", "tools", "vision"],
        "details": {"families": ["mllama"]},
    }
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(payload))

    caps = mc._detect_ollama_capabilities()
    assert caps.supports_vision is True
    assert caps.supports_function_calling is True  # template has the markers


def test_ollama_detects_vision_via_family_fallback(monkeypatch) -> None:
    """Older Ollama without `capabilities` → fall back to family/model_info."""
    import httpx
    import infinidev.config.model_capabilities as mc
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "LLM_MODEL", "ollama_chat/my-custom-llava:q4", raising=False)
    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://localhost:11434", raising=False)

    payload = {
        "template": "{{ .Prompt }}",  # no tool markers, no capabilities key
        "details": {"families": ["llama", "clip"]},
        "model_info": {"clip.vision.image_size": 336},
    }
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(payload))

    caps = mc._detect_ollama_capabilities()
    assert caps.supports_vision is True


def test_ollama_text_only_model_has_no_vision(monkeypatch) -> None:
    import httpx
    import infinidev.config.model_capabilities as mc
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "LLM_MODEL", "ollama_chat/qwen2.5-coder:7b", raising=False)
    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://localhost:11434", raising=False)

    payload = {
        "template": "{{ if .Tools }}...{{ end }}",
        "capabilities": ["completion", "tools"],
        "details": {"families": ["qwen2"]},
        "model_info": {},
    }
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(payload))

    caps = mc._detect_ollama_capabilities()
    assert caps.supports_vision is False


def test_llama_cpp_detects_vision_via_props_flag(monkeypatch) -> None:
    """`has_multimodal: true` in /props → vision enabled."""
    import httpx
    import infinidev.config.model_capabilities as mc
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://localhost:8080/v1", raising=False)

    captured: list[str] = []

    def fake_get(url: str, timeout: float = 0):
        captured.append(url)
        return _FakeResp({"has_multimodal": True, "chat_template": ""})

    monkeypatch.setattr(httpx, "get", fake_get)

    assert mc._detect_llama_cpp_vision() is True
    # /v1 suffix must be stripped — /props lives at the root.
    assert captured == ["http://localhost:8080/props"]


def test_llama_cpp_template_heuristic_fallback(monkeypatch) -> None:
    """Older builds without `has_multimodal` → chat template heuristic."""
    import httpx
    import infinidev.config.model_capabilities as mc
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://localhost:8080", raising=False)
    payload = {"chat_template": "USER: <|image|> describe ASSISTANT:"}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp(payload))

    assert mc._detect_llama_cpp_vision() is True


def test_llama_cpp_no_vision_when_props_empty(monkeypatch) -> None:
    import httpx
    import infinidev.config.model_capabilities as mc
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://localhost:8080", raising=False)
    monkeypatch.setattr(
        httpx, "get", lambda *a, **k: _FakeResp({"chat_template": "plain text"})
    )

    assert mc._detect_llama_cpp_vision() is False


def test_llama_cpp_props_unreachable_returns_false(monkeypatch) -> None:
    import httpx
    import infinidev.config.model_capabilities as mc
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "LLM_BASE_URL", "http://localhost:8080", raising=False)

    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", boom)

    assert mc._detect_llama_cpp_vision() is False
