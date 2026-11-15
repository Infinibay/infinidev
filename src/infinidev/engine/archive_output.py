"""Exact archived data codecs; no database, settings, engine or model bootstrap."""

from __future__ import annotations

import hashlib
import json
from typing import Any
import uuid

CHUNK_CHARACTERS = 8000
MAX_PARTS = 128
MAX_MANIFEST_BYTES = 32768
MAX_RESPONSE_BYTES = 32768
MANIFEST_KIND = "tool_archive_manifest"
_MANIFEST_FIELDS = {"version", "reference", "session_id", "title", "offset_unit",
                    "total_characters", "total_utf8_bytes", "sha256", "parts"}
_PART_FIELDS = {"index", "record_id", "start", "length", "sha256"}


class ArchiveError(ValueError):
    """A fixed public error code, without storage or payload details."""


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def parse_reference(reference: str) -> tuple[str, str]:
    if not isinstance(reference, str):
        raise ArchiveError("invalid_reference")
    prefix, separator, identifier = reference.partition(":")
    try:
        parsed = uuid.UUID(identifier)
    except (ValueError, AttributeError):
        raise ArchiveError("invalid_reference") from None
    if separator and prefix == "wm-archive" and parsed.hex == identifier:
        return "archive", identifier
    if separator and prefix == "wm-record" and str(parsed) == identifier:
        return "memory_record", identifier
    raise ArchiveError("invalid_reference")


def failure(reference: str | None, status: str) -> str:
    return encode({"version": 1, "reference": reference, "status": status,
                   "archive_complete": False, "range_complete": False,
                   "fully_returned": False, "artifact_fully_returned": False})


def validate_range(start: int, length: int) -> None:
    if type(start) is not int or type(length) is not int or start < 0 or not 0 <= length <= 8000:
        raise ArchiveError("range_error")


def split_output(text: str) -> list[str]:
    if not isinstance(text, str) or len(text) > CHUNK_CHARACTERS * MAX_PARTS:
        raise ArchiveError("limit_exceeded")
    text.encode("utf-8")
    return [text[i:i + CHUNK_CHARACTERS] for i in range(0, len(text), CHUNK_CHARACTERS)]


def make_manifest(reference: str, session_id: str, title: str, text: str,
                  parts: list[dict[str, Any]]) -> str:
    body = encode({"version": 1, "reference": reference, "session_id": session_id,
                   "title": title[:200], "offset_unit": "unicode_codepoint",
                   "total_characters": len(text), "total_utf8_bytes": len(text.encode("utf-8")),
                   "sha256": digest(text), "parts": parts})
    if len(body.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ArchiveError("limit_exceeded")
    return body


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def decode_manifest(body: str, content_hash: str, reference: str, session_id: str) -> dict:
    try:
        if not isinstance(body, str) or len(body.encode("utf-8")) > MAX_MANIFEST_BYTES:
            raise ArchiveError("limit_exceeded")
        if not _hash(content_hash) or digest(body) != content_hash:
            raise ArchiveError("incomplete")
        data = json.loads(body)
    except ArchiveError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        raise ArchiveError("incomplete") from None
    if (not isinstance(data, dict) or set(data) != _MANIFEST_FIELDS
            or type(data["version"]) is not int or data["version"] != 1
            or data["reference"] != reference or data["session_id"] != session_id
            or not isinstance(data["title"], str) or len(data["title"]) > 200
            or data["offset_unit"] != "unicode_codepoint" or not _hash(data["sha256"])
            or type(data["total_characters"]) is not int
            or not 0 <= data["total_characters"] <= CHUNK_CHARACTERS * MAX_PARTS
            or type(data["total_utf8_bytes"]) is not int
            or not 0 <= data["total_utf8_bytes"] <= 4 * CHUNK_CHARACTERS * MAX_PARTS
            or not isinstance(data["parts"], list) or len(data["parts"]) > MAX_PARTS):
        raise ArchiveError("incomplete")
    offset = 0
    for index, part in enumerate(data["parts"]):
        if (not isinstance(part, dict) or set(part) != _PART_FIELDS
                or type(part["index"]) is not int or part["index"] != index
                or type(part["start"]) is not int or part["start"] != offset
                or type(part["length"]) is not int or not 1 <= part["length"] <= CHUNK_CHARACTERS
                or not _hash(part["sha256"])):
            raise ArchiveError("incomplete")
        try:
            parse_reference("wm-record:" + part["record_id"])
        except (ArchiveError, TypeError):
            raise ArchiveError("incomplete") from None
        offset += part["length"]
    if offset != data["total_characters"]:
        raise ArchiveError("incomplete")
    return data


def verify_record(content: str, content_hash: str) -> str:
    try:
        if (not isinstance(content, str) or len(content) > CHUNK_CHARACTERS
                or not _hash(content_hash) or digest(content) != content_hash):
            raise ArchiveError("incomplete")
    except UnicodeError:
        raise ArchiveError("incomplete") from None
    return content


def reconstruct(data: dict, rows: dict[str, tuple[str, str]]) -> str:
    pieces = []
    for part in data["parts"]:
        row = rows.get(part["record_id"])
        if row is None:
            raise ArchiveError("incomplete")
        text = verify_record(*row)
        if len(text) != part["length"] or digest(text) != part["sha256"]:
            raise ArchiveError("incomplete")
        pieces.append(text)
    text = "".join(pieces)
    if (len(text) != data["total_characters"]
            or len(text.encode("utf-8")) != data["total_utf8_bytes"]
            or digest(text) != data["sha256"]):
        raise ArchiveError("incomplete")
    return text


def page(reference: str, scope: str, text: str, start: int, length: int) -> str:
    """Serialize the verified snapshot, never re-fetching a body after validation."""
    validate_range(start, length)
    if start > len(text):
        return failure(reference, "range_error")
    end = min(start + length, len(text))
    data = {"version": 1, "status": "ok", "reference": reference, "scope": scope,
            "offset_unit": "unicode_codepoint", "start": start, "end": end,
            "requested_length": length, "next_start": end if end < len(text) else None,
            "total_characters": len(text), "total_utf8_bytes": len(text.encode("utf-8")),
            "sha256": digest(text), "archive_complete": True if scope == "archive" else None,
            "range_complete": True, "fully_returned": True,
            "artifact_fully_returned": scope == "archive" and start == 0 and end == len(text),
            "authority": "historical_data", "is_instruction": False, "content": text[start:end]}
    serialized = encode(data)
    if len(serialized.encode("utf-8")) > MAX_RESPONSE_BYTES:
        data.pop("content")
        data.update(status="response_budget_exceeded", fully_returned=False,
                    artifact_fully_returned=False)
        serialized = encode(data)
    return serialized
